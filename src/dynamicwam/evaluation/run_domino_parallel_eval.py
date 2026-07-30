"""Run one DynamicWAM DOMINO task across process-isolated GPU slots."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _episode_summaries(log_path: Path, prefix: str) -> list[dict[str, Any]]:
    summaries = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if prefix not in line:
            continue
        try:
            value = ast.literal_eval(line.split(prefix, 1)[1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            summaries.append(value)
    return summaries


def _policy_summaries(log_path: Path) -> list[dict[str, Any]]:
    """Read synchronous native episode telemetry emitted by the adapter."""

    return _episode_summaries(log_path, "SYNC-EPISODE-SUMMARY ")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not safe:
        raise ValueError(f"label has no path-safe characters: {value!r}")
    return safe


def _partition_episode_counts(total_episodes: int, slot_count: int) -> list[int]:
    """Split a requested episode total across all GPU slots without dropping work."""
    if isinstance(total_episodes, bool) or total_episodes <= 0:
        raise ValueError(f"total_episodes must be positive, got {total_episodes!r}")
    if isinstance(slot_count, bool) or slot_count <= 0:
        raise ValueError(f"slot_count must be positive, got {slot_count!r}")
    if total_episodes < slot_count:
        raise ValueError(
            "total_episodes must be at least the number of GPU slots so every "
            f"slot has work: total_episodes={total_episodes}, slot_count={slot_count}"
        )
    base, remainder = divmod(int(total_episodes), int(slot_count))
    return [base + (1 if slot < remainder else 0) for slot in range(slot_count)]


def _validated_curobo_root(value: Any) -> Path:
    root = Path(str(value)).expanduser().resolve()
    if not (root / "curobo").is_dir():
        raise FileNotFoundError(
            f"CuRobo root must contain a curobo package directory: {root}"
        )
    return root


def _child_env(
    *,
    gpu: str,
    runtime_root: Path,
    domino_root: Path,
    curobo_root: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Must precede the editable CuRobo checkout referenced by the RoboTwin
    # environment. On H100 the latter can contain an incompatible custom CUDA
    # binary even though both packages have the same Python version.
    pythonpath = [str(curobo_root)]
    pythonpath.extend([str(runtime_root), str(domino_root / "script")])
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _probe_curobo_extension(
    *,
    python: Path,
    domino_root: Path,
    env: dict[str, str],
    expected_root: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError(
            "runtime.curobo_line_search_sha256 must be a 64-character hex digest"
        )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; import torch; "
                "import curobo.curobolib.line_search_cu as module; "
                "print(Path(module.__file__).resolve())"
            ),
        ],
        cwd=domino_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120.0,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "CuRobo extension preflight failed before evaluation:\n"
            f"{probe.stderr[-4000:]}"
        )
    output_lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("CuRobo extension preflight returned no module path")
    extension = Path(output_lines[-1]).resolve()
    try:
        extension.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            "CuRobo import resolved outside the configured runtime root: "
            f"expected under {expected_root}, loaded {extension}"
        ) from exc
    if not extension.is_file():
        raise FileNotFoundError(extension)
    actual_sha256 = _sha256(extension)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            "CuRobo line_search extension hash mismatch: "
            f"expected {expected_sha256.lower()}, got {actual_sha256} ({extension})"
        )
    return {
        "root": str(expected_root),
        "line_search_extension": str(extension),
        "line_search_sha256": actual_sha256,
    }


def _reported_int(value: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        candidate = value.get(key)
        if candidate is None or isinstance(candidate, bool):
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _reported_float(value: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        candidate = value.get(key)
        if candidate is None or isinstance(candidate, bool):
            continue
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            return number
    return None


def _telemetry_steps(value: dict[str, Any]) -> int | None:
    direct = _reported_int(value, ("executed_steps", "steps", "step_count"))
    if direct is not None:
        return direct
    for key in ("execution", "protocol"):
        nested = value.get(key)
        if isinstance(nested, dict):
            found = _reported_int(
                nested,
                ("executed_steps", "steps", "step_count"),
            )
            if found is not None:
                return found
    return None


def _compact_episodes(
    report: list[dict[str, Any]],
    *,
    telemetry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(telemetry) != len(report):
        raise RuntimeError(
            "DynamicWAM requires one SYNC-EPISODE-SUMMARY per episode: "
            f"episodes={len(report)}, telemetry={len(telemetry)}"
        )
    episodes = []
    for index, raw in enumerate(report):
        reported_seed = _reported_int(raw, ("seed", "episode_seed", "env_seed"))
        if reported_seed is None:
            raise RuntimeError(f"DOMINO episode {index} did not report its seed")
        success = raw.get("success")
        if not isinstance(success, bool):
            raise RuntimeError(f"DOMINO episode {index} did not report boolean success")
        steps = _reported_int(
            raw,
            ("steps", "step", "step_count", "executed_steps", "take_action_cnt"),
        )
        policy_telemetry = telemetry[index]
        if steps is None:
            steps = _telemetry_steps(policy_telemetry)
        episodes.append(
            {
                "seed": reported_seed,
                "seed_source": "report",
                "success": success,
                "steps": steps,
                "manipulation_score": _reported_float(
                    raw, ("manipulation_score", "ms")
                ),
                "route_completion": _reported_float(
                    raw, ("route_completion", "completion")
                ),
                "fail_reason": raw.get("fail_reason"),
                "policy_telemetry": policy_telemetry,
            }
        )
    return episodes


def _terminate(processes: list[subprocess.Popen], grace_s: float = 10.0) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while live and time.monotonic() < deadline:
        live = [process for process in live if process.poll() is None]
        time.sleep(0.2)
    for process in live:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--domino-root", required=True)
    parser.add_argument("--eval-policy", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument(
        "--curobo-root",
        required=True,
        help="CuRobo source root prepended to PYTHONPATH and verified at launch.",
    )
    parser.add_argument(
        "--curobo-extension-sha256",
        required=True,
        help="Expected SHA256 for curobolib/line_search_cu*.so",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument(
        "--total-episodes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        required=True,
        help="Raw seed for slot 0; later slots use --slot-seed-stride.",
    )
    parser.add_argument("--slot-seed-stride", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--stagger-seconds", type=float, required=True)
    args = parser.parse_args()
    import yaml  # type: ignore[import-untyped]

    python = Path(args.python).expanduser().absolute()
    domino_root = Path(args.domino_root).resolve()
    eval_policy = Path(args.eval_policy).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    run_root = Path(args.run_root).resolve()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list must be non-empty and unique: {gpus}")
    for required in (python, domino_root, eval_policy, runtime_root):
        if not required.exists():
            raise FileNotFoundError(required)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run root is not empty: {run_root}")
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    curobo_root = _validated_curobo_root(args.curobo_root)
    probe_env = _child_env(
        gpu=gpus[0],
        runtime_root=runtime_root,
        domino_root=domino_root,
        curobo_root=curobo_root,
    )
    curobo_probe = _probe_curobo_extension(
        python=python,
        domino_root=domino_root,
        env=probe_env,
        expected_root=curobo_root,
        expected_sha256=args.curobo_extension_sha256,
    )

    logs_root = run_root / "logs"
    results_root = run_root / "results"
    snapshots_root = run_root / "config_snapshots"
    logs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    snapshots_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, snapshots_root / config_path.name)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"eval config must be a mapping: {config_path}")
    task_name = str(config["task_name"])
    policy_name = str(config["policy_name"])
    task_config = str(config["task_config"])
    total_episodes = int(args.total_episodes)
    episode_counts = _partition_episode_counts(total_episodes, len(gpus))
    raw_base_start_seed = args.start_seed
    if (
        isinstance(raw_base_start_seed, bool)
        or not isinstance(raw_base_start_seed, int)
        or raw_base_start_seed < 0
    ):
        raise ValueError(
            f"start seed must be a non-negative integer, got {raw_base_start_seed!r}"
        )
    base_start_seed = int(raw_base_start_seed)
    if args.slot_seed_stride <= 0:
        raise ValueError("slot-seed-stride must be positive")
    launch: dict[str, Any] = {
        "started_unix": time.time(),
        "python": str(python),
        "domino_root": str(domino_root),
        "eval_policy": str(eval_policy),
        "eval_policy_sha256": _sha256(eval_policy),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runtime_root": str(runtime_root),
        "curobo": curobo_probe,
        "gpus": gpus,
        "requested_episodes": total_episodes,
        "base_start_seed": base_start_seed,
        "slot_seed_stride": int(args.slot_seed_stride),
        "episodes_per_slot": episode_counts,
        "slots": [],
    }

    processes: list[subprocess.Popen] = []
    handles = []
    slot_records: list[dict[str, Any]] = []
    try:
        for slot, (gpu, requested_episodes) in enumerate(
            zip(gpus, episode_counts, strict=True)
        ):
            seed_index = slot
            start_seed = base_start_seed + int(args.slot_seed_stride) * seed_index
            label_prefix = _safe_name(str(config["ckpt_setting"]))
            label = f"{label_prefix}-slot{slot}"
            log_path = logs_root / f"slot{slot}_gpu{gpu}.log"
            command = [
                str(python),
                "-u",
                str(eval_policy),
                "--config",
                str(config_path),
                "--overrides",
                "--seed",
                str(seed_index),
                "--start_seed",
                str(start_seed),
                "--episode_num",
                str(requested_episodes),
                "--ckpt_setting",
                label,
                "--eval_output_root",
                str(results_root),
            ]
            env = _child_env(
                gpu=gpu,
                runtime_root=runtime_root,
                domino_root=domino_root,
                curobo_root=curobo_root,
            )
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(
                command,
                cwd=domino_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            record = {
                "slot": slot,
                "gpu": gpu,
                "seed_index": seed_index,
                "start_seed": start_seed,
                "label": label,
                "pid": process.pid,
                "log": str(log_path),
                "command": command,
                "started_unix": time.time(),
                "requested_episodes": requested_episodes,
            }
            slot_records.append(record)
            launch["slots"].append(record.copy())
            print(
                f"launched slot={slot} gpu={gpu} pid={process.pid} "
                f"start_seed={start_seed} episodes={requested_episodes}",
                flush=True,
            )
            if args.stagger_seconds > 0 and slot + 1 < len(gpus):
                time.sleep(args.stagger_seconds)

        _write_json(run_root / "launch.json", launch)
        deadline = time.monotonic() + args.timeout_seconds
        pending = set(range(len(processes)))
        while pending:
            for index in list(pending):
                return_code = processes[index].poll()
                if return_code is not None:
                    pending.remove(index)
                    slot_records[index]["return_code"] = return_code
                    slot_records[index]["finished_unix"] = time.time()
                    print(
                        f"finished slot={index} return_code={return_code} "
                        f"remaining={len(pending)}",
                        flush=True,
                    )
            if pending and time.monotonic() >= deadline:
                raise TimeoutError(f"evaluation timed out with slots {sorted(pending)}")
            if pending:
                time.sleep(5.0)
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for handle in handles:
            handle.close()

    episodes = []
    for record in slot_records:
        label_root = (
            results_root / task_name / policy_name / task_config / record["label"]
        )
        reports = sorted(label_root.glob("*/_episodes_detail.json"))
        record["report_candidates"] = [str(path) for path in reports]
        if record.get("return_code") == 0 and len(reports) == 1:
            report_path = reports[0]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                isinstance(report, list)
                and len(report) == record["requested_episodes"]
                and all(isinstance(episode, dict) for episode in report)
            ):
                run_dir = report_path.parent
                telemetry = _policy_summaries(Path(record["log"]))
                compact = _compact_episodes(
                    report,
                    telemetry=telemetry,
                )
                videos = [
                    run_dir / f"episode{index}.mp4" for index in range(len(report))
                ]
                record.update(
                    {
                        "status": "complete",
                        "run_dir": str(run_dir),
                        "report": str(report_path),
                        "metrics": str(run_dir / "_metrics.json"),
                        "raw_videos": [str(path) for path in videos],
                        "raw_video_count": sum(
                            path.is_file() and path.stat().st_size > 0
                            for path in videos
                        ),
                        "episodes": compact,
                        "policy_summaries": telemetry,
                        "policy_summary": telemetry[-1] if telemetry else None,
                        "seeds_reported": True,
                        "seed_sequence": [episode["seed"] for episode in compact],
                    }
                )
                if len(report) == 1:
                    record["episode"] = report[0]
                    record["raw_video"] = str(videos[0])
                    record["raw_video_exists"] = record["raw_video_count"] == 1
                for local_index, episode in enumerate(compact):
                    episodes.append(
                        {
                            "slot": record["slot"],
                            "gpu": record["gpu"],
                            "local_episode_index": local_index,
                            "run_dir": str(run_dir),
                            "raw_video": str(videos[local_index]),
                            **episode,
                        }
                    )
            else:
                record["status"] = "invalid_report"
        else:
            record["status"] = "failed"

    success_count = sum(bool(episode.get("success")) for episode in episodes)
    seeds_reported = len(episodes) == total_episodes and all(
        episode.get("seed_source") == "report" for episode in episodes
    )
    aggregate = {
        "finished_unix": time.time(),
        "requested_episodes": total_episodes,
        "reported_episodes": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / total_episodes,
        "success_rate_denominator": "requested_episodes",
        "base_start_seed": base_start_seed,
        "seeds_reported": seeds_reported,
        "seed_sequence": [episode["seed"] for episode in episodes],
        "episodes": episodes,
        "slots": slot_records,
    }
    _write_json(run_root / "aggregate.json", aggregate)
    print(
        f"aggregate reported={len(episodes)}/{total_episodes} "
        f"success={success_count}/{len(episodes)} run_root={run_root}",
        flush=True,
    )
    complete_slots = sum(record.get("status") == "complete" for record in slot_records)
    return (
        0
        if len(episodes) == total_episodes
        and complete_slots == len(gpus)
        and seeds_reported
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
