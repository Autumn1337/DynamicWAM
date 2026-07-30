#!/usr/bin/env python3
"""Run the official 35-task DOMINO Level 1 suite with resumable GPU scheduling."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO

from dynamicwam.config import load_profile, write_config_snapshot
from dynamicwam.external_assets import (
    verify_checkpoint_artifact,
    verify_robotwin_asset_trees,
    verify_wan_assets,
)
from dynamicwam.external_setup import (
    inspect_curobo_runtime,
    verify_curobo_runtime,
    verify_curobo_source,
    verify_domino_python_runtime,
    verify_robotwin_asset_links,
)
from dynamicwam.integrity import (
    DOMINO_RUNTIME_ASSET_PREFIXES,
    sha256_file,
    sha256_tree,
)


def _worker_env(extra_pythonpath: tuple[Path, ...]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(path) for path in extra_pythonpath]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _integrity_artifacts(
    *,
    profile,
    benchmark: dict[str, Any],
    curobo_runtime: dict[str, dict[str, Any]],
    domino_python_runtime: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}

    def add(label: str, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"integrity artifact does not exist: {path}")
        artifacts[label] = {
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file",
            "sha256": sha256_tree(path) if path.is_dir() else sha256_file(path),
        }

    add("profile", profile.path)
    deploy_config = profile.inference_config()
    checkpoint_path = Path(deploy_config["checkpoint_path"]).expanduser().resolve()
    checkpoint_records = verify_checkpoint_artifact(
        root=checkpoint_path.parent,
        manifest_path=Path(deploy_config["checkpoint_manifest"]),
        artifact_id=str(deploy_config["checkpoint_artifact_id"]),
    )
    verified_checkpoint = Path(checkpoint_records["checkpoint"]["path"]).resolve()
    if verified_checkpoint != checkpoint_path:
        raise RuntimeError(
            "checkpoint manifest resolves to a different file than the profile: "
            f"{verified_checkpoint} != {checkpoint_path}"
        )
    artifacts.update(checkpoint_records)
    artifacts.update(curobo_runtime)
    artifacts.update(domino_python_runtime)

    wan_root = Path(deploy_config["wan_root"]).expanduser()
    artifacts.update(
        verify_wan_assets(
            root=wan_root,
            manifest_path=Path(deploy_config["external_assets_manifest"]),
            purpose="inference",
        )
    )
    raw_integrity_paths = benchmark["integrity_paths"]
    for index, raw_path in enumerate(raw_integrity_paths):
        add(
            f"source_{index}",
            Path(str(raw_path)).expanduser().resolve(),
        )
    return artifacts


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_task_aggregate(
    path: Path,
    requested: int,
    *,
    expected_start_seed: int | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    episodes = value.get("episodes")
    if (
        value.get("requested_episodes") != requested
        or value.get("reported_episodes") != requested
        or value.get("success_rate_denominator") != "requested_episodes"
        or value.get("seeds_reported") is not True
        or not isinstance(episodes, list)
        or len(episodes) != requested
        or not all(isinstance(episode, dict) for episode in episodes)
    ):
        return None
    raw_seeds = [episode.get("seed") for episode in episodes]
    if (
        len(raw_seeds) != requested
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds
        )
        or any(episode.get("seed_source") != "report" for episode in episodes)
    ):
        return None
    seeds = [int(seed) for seed in raw_seeds]
    if len(set(seeds)) != requested or any(
        left >= right for left, right in pairwise(seeds)
    ):
        return None
    if expected_start_seed is not None:
        if value.get("base_start_seed") != expected_start_seed:
            return None
        if seeds and seeds[0] < expected_start_seed:
            return None
    slots = value.get("slots")
    if (
        not isinstance(slots, list)
        or len(slots) != 1
        or slots[0].get("status") != "complete"
    ):
        return None
    for episode in episodes:
        telemetry = episode.get("policy_telemetry")
        if not isinstance(telemetry, dict):
            return None
        steps = telemetry.get("steps")
        kinds = telemetry.get("kinds")
        overrun_steps = telemetry.get("overrun_steps")
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
            or not isinstance(kinds, dict)
            or set(kinds) != {"action"}
            or kinds.get("action") != steps
            or isinstance(overrun_steps, bool)
            or not isinstance(overrun_steps, int)
            or not 0 <= overrun_steps <= steps
        ):
            return None
    return value


def archive_incomplete_task(task_root: Path, archive_root: Path) -> None:
    if not task_root.exists() or not any(task_root.iterdir()):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = archive_root / f"{task_root.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{task_root.name}_{stamp}_{suffix}"
        suffix += 1
    archive_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_root), str(destination))


def task_record(
    task: str, aggregate_path: Path, aggregate: dict[str, Any]
) -> dict[str, Any]:
    manipulation_scores = [
        episode.get("manipulation_score")
        for episode in aggregate["episodes"]
        if isinstance(episode.get("manipulation_score"), (int, float))
        and math.isfinite(float(episode["manipulation_score"]))
    ]
    return {
        "task": task,
        "status": "complete",
        "aggregate": str(aggregate_path),
        "requested_episodes": aggregate["requested_episodes"],
        "reported_episodes": aggregate["reported_episodes"],
        "success_count": aggregate["success_count"],
        "success_rate": aggregate["success_rate"],
        "manipulation_score_mean": (
            sum(float(value) for value in manipulation_scores)
            / len(manipulation_scores)
            if manipulation_scores
            else None
        ),
        "seed_first": aggregate["episodes"][0]["seed"],
        "seed_last": aggregate["episodes"][-1]["seed"],
    }


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stratified_bootstrap_ci(
    successes_by_task: list[list[bool]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    if not successes_by_task or any(not values for values in successes_by_task):
        return None
    rng = random.Random(seed)
    rates: list[float] = []
    total = sum(len(values) for values in successes_by_task)
    for _ in range(replicates):
        success = 0
        for values in successes_by_task:
            success += sum(rng.choice(values) for _ in range(len(values)))
        rates.append(success / total)
    return {
        "method": "stratified_task_episode_bootstrap",
        "confidence": 0.95,
        "replicates": replicates,
        "seed": seed,
        "low": _quantile(rates, 0.025),
        "high": _quantile(rates, 0.975),
    }


def summarize(
    *,
    run_root: Path,
    tasks: list[str],
    episodes_per_task: int,
    failures: dict[str, dict[str, Any]],
    start_seed: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_reported = 0
    total_success = 0
    manipulation_scores: list[float] = []
    total_native_steps = 0
    total_overrun_steps = 0
    telemetry_episode_count = 0
    successes_by_task: list[list[bool]] = []
    for task in tasks:
        aggregate_path = run_root / "tasks" / task / "aggregate.json"
        aggregate = valid_task_aggregate(
            aggregate_path,
            episodes_per_task,
            expected_start_seed=start_seed,
        )
        if aggregate is None:
            record = {
                "task": task,
                "status": "incomplete",
                "aggregate": str(aggregate_path),
            }
            record.update(failures.get(task, {}))
        else:
            record = task_record(task, aggregate_path, aggregate)
            total_reported += aggregate["reported_episodes"]
            total_success += aggregate["success_count"]
            task_successes: list[bool] = []
            for episode in aggregate["episodes"]:
                task_successes.append(bool(episode.get("success", False)))
                score = episode.get("manipulation_score")
                if isinstance(score, (int, float)) and math.isfinite(float(score)):
                    manipulation_scores.append(float(score))
                telemetry = episode.get("policy_telemetry")
                if not isinstance(telemetry, dict):
                    continue
                telemetry_episode_count += 1
                total_native_steps += int(telemetry["steps"])
                total_overrun_steps += int(telemetry["overrun_steps"])
            successes_by_task.append(task_successes)
        records.append(record)
    total_requested = len(tasks) * episodes_per_task
    completed_tasks = sum(record["status"] == "complete" for record in records)
    return {
        "finished_unix": time.time(),
        "benchmark": "DOMINO",
        "level": 1,
        "task_count": len(tasks),
        "completed_tasks": completed_tasks,
        "episodes_per_task": episodes_per_task,
        "requested_episodes": total_requested,
        "reported_episodes": total_reported,
        "success_count": total_success,
        "success_rate": total_success / total_requested,
        "success_rate_denominator": "requested_episodes",
        "start_seed": start_seed,
        "manipulation_score_mean": (
            sum(manipulation_scores) / len(manipulation_scores)
            if manipulation_scores
            else None
        ),
        "success_rate_ci": (
            _stratified_bootstrap_ci(
                successes_by_task,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            if completed_tasks == len(tasks)
            else None
        ),
        "telemetry": {
            "reported_episodes": telemetry_episode_count,
            "native_action_steps": total_native_steps,
            "overrun_steps": total_overrun_steps,
        },
        "invalid_episodes": total_requested - total_reported,
        "tasks": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--python")
    parser.add_argument("--parallel-runner")
    parser.add_argument("--domino-root")
    parser.add_argument("--eval-policy")
    parser.add_argument(
        "--runtime-root",
        help="Directory whose child package is ciwam.",
    )
    parser.add_argument(
        "--extra-pythonpath",
        action="append",
        default=[],
        help="Additional source root forwarded to every evaluation worker.",
    )
    parser.add_argument("--curobo-root")
    parser.add_argument("--curobo-extension-sha256")
    parser.add_argument("--run-root")
    parser.add_argument("--gpus")
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument(
        "--start-seed",
        type=int,
        help="Raw seed at the start of every task's paired episode sequence.",
    )
    parser.add_argument("--task-timeout-seconds", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import yaml  # type: ignore[import-untyped]

    profile = load_profile(args.config)
    benchmark = profile.benchmark_config()
    episodes_per_task = int(
        args.episodes_per_task
        if args.episodes_per_task is not None
        else benchmark["episodes_per_task"]
    )
    if episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")

    python = Path(args.python or benchmark["python"]).expanduser().absolute()
    parallel_runner = Path(
        args.parallel_runner or Path(__file__).with_name("run_domino_parallel_eval.py")
    ).resolve()
    domino_root = Path(args.domino_root or benchmark["domino_root"]).resolve()
    eval_policy = Path(args.eval_policy or benchmark["eval_policy"]).resolve()
    runtime_root = Path(args.runtime_root or benchmark["runtime_root"]).resolve()
    configured_pythonpath = [
        *benchmark["extra_pythonpath"],
        *args.extra_pythonpath,
    ]
    extra_pythonpath = tuple(
        Path(value).expanduser().resolve() for value in configured_pythonpath
    )
    curobo_root = Path(args.curobo_root or benchmark["curobo_root"]).resolve()
    curobo_extension_sha256 = str(
        args.curobo_extension_sha256 or benchmark["curobo_extension_sha256"]
    ).lower()
    run_root = Path(args.run_root or benchmark["run_root"]).resolve()
    for path in (
        python,
        parallel_runner,
        domino_root,
        eval_policy,
        runtime_root,
        curobo_root,
        *extra_pythonpath,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(set(extra_pythonpath)) != len(extra_pythonpath):
        raise ValueError(f"extra PYTHONPATH entries must be unique: {extra_pythonpath}")
    if not all(path.is_dir() for path in extra_pythonpath):
        raise ValueError(
            f"extra PYTHONPATH entries must be directories: {extra_pythonpath}"
        )
    external_manifest = Path(profile.raw["paths"]["external_assets_manifest"])
    robotwin_asset_root = (
        Path(profile.raw["paths"]["project_root"]) / "external" / "robotwin-assets"
    )
    verify_robotwin_asset_trees(
        root=robotwin_asset_root,
        manifest_path=external_manifest,
    )
    verify_robotwin_asset_links(
        asset_root=robotwin_asset_root,
        domino_root=domino_root,
    )
    domino_python_runtime = verify_domino_python_runtime(
        python=python,
        manifest_path=external_manifest,
    )
    verify_curobo_source(
        destination=curobo_root.parent,
        manifest_path=external_manifest,
    )
    if args.curobo_extension_sha256 is None:
        curobo_runtime = verify_curobo_runtime(
            destination=curobo_root.parent,
            manifest_path=external_manifest,
        )
    else:
        curobo_runtime = inspect_curobo_runtime(
            destination=curobo_root.parent,
            manifest_path=external_manifest,
        )
    actual_line_search_sha256 = str(
        curobo_runtime["curobo_extension_line_search_cu"]["sha256"]
    )
    if actual_line_search_sha256 != curobo_extension_sha256:
        raise RuntimeError(
            "CuRobo line_search extension differs from the launch contract: "
            f"expected {curobo_extension_sha256}, got {actual_line_search_sha256}"
        )
    generated_collection_configs = tuple(
        f"task_config/{name}.yml"
        for name in (
            profile.raw["collection"]["clean_config_name"],
            profile.raw["collection"]["randomized_config_name"],
        )
    )
    actual_domino_source_sha256 = sha256_tree(
        domino_root,
        excluded_relative_paths=generated_collection_configs,
        excluded_relative_prefixes=DOMINO_RUNTIME_ASSET_PREFIXES,
    )
    if actual_domino_source_sha256 != benchmark["domino_source_sha256"]:
        raise RuntimeError(
            "DOMINO source differs from the pinned commit "
            f"{benchmark['domino_commit']}: {actual_domino_source_sha256}"
        )
    actual_eval_policy_sha256 = sha256_file(eval_policy)
    if actual_eval_policy_sha256 != benchmark["eval_policy_sha256"]:
        raise RuntimeError(
            "DOMINO evaluator differs from the pinned source: "
            f"{actual_eval_policy_sha256}"
        )

    raw_gpus = args.gpus or ",".join(benchmark["gpus"])
    gpus = [item.strip() for item in raw_gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU list must be non-empty and unique: {gpus}")

    tasks = list(benchmark["tasks"])
    base_config = profile.domino_base_config()
    raw_start_seed = (
        args.start_seed if args.start_seed is not None else benchmark["start_seed"]
    )
    if (
        isinstance(raw_start_seed, bool)
        or not isinstance(raw_start_seed, int)
        or raw_start_seed < 0
    ):
        raise ValueError(
            "suite start seed must be an explicit non-negative integer in the "
            f"CLI or base config, got {raw_start_seed!r}"
        )
    start_seed = int(raw_start_seed)
    base_config["start_seed"] = start_seed
    task_timeout_seconds = float(
        args.task_timeout_seconds
        if args.task_timeout_seconds is not None
        else benchmark["task_timeout_seconds"]
    )
    if task_timeout_seconds <= 0.0:
        raise ValueError("task-timeout-seconds must be positive")
    run_root.mkdir(parents=True, exist_ok=True)
    tasks_root = run_root / "tasks"
    configs_root = run_root / "configs"
    logs_root = run_root / "logs"
    archive_root = run_root / "archive"
    for path in (tasks_root, configs_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)
    base_config_path = run_root / "base_config.yml"
    base_config_path.write_text(
        yaml.safe_dump(base_config, sort_keys=False),
        encoding="utf-8",
    )
    resolved_launch = {
        "benchmark": benchmark,
        "base_config": base_config,
        "launch": {
            "python": str(python),
            "parallel_runner": str(parallel_runner),
            "domino_root": str(domino_root),
            "eval_policy": str(eval_policy),
            "runtime_root": str(runtime_root),
            "extra_pythonpath": [str(path) for path in extra_pythonpath],
            "curobo_root": str(curobo_root),
            "curobo_extension_sha256": curobo_extension_sha256,
            "run_root": str(run_root),
            "gpus": gpus,
            "episodes_per_task": episodes_per_task,
            "start_seed": start_seed,
            "task_timeout_seconds": task_timeout_seconds,
        },
    }
    write_config_snapshot(
        run_root / "config_audit",
        profile=profile,
        label="benchmark",
        resolved_config=resolved_launch,
    )

    contract = {
        "version": 2,
        "benchmark": "DOMINO",
        "level": 1,
        "profile_sha256": profile.sha256,
        "base_config_sha256": sha256_file(base_config_path),
        "suite_runner_sha256": sha256_file(Path(__file__).resolve()),
        "eval_policy_sha256": sha256_file(eval_policy),
        "parallel_runner_sha256": sha256_file(parallel_runner),
        "python": str(python),
        "curobo_root": str(curobo_root),
        "curobo_extension_sha256": curobo_extension_sha256,
        "extra_pythonpath": [str(path) for path in extra_pythonpath],
        "episodes_per_task": episodes_per_task,
        "start_seed": start_seed,
        "gpus": gpus,
        "scheduling": "one_complete_task_per_gpu",
        "task_config": base_config["task_config"],
        "dynamic_coefficient": benchmark["dynamic_coefficient"],
        "protocol": (f"native_sync_execute{int(benchmark['execute_steps'])}"),
        "action_interval_ms": profile.raw["inference"]["action_interval_ms"],
        "tasks": tasks,
        "integrity_artifacts": _integrity_artifacts(
            profile=profile,
            benchmark=benchmark,
            curobo_runtime=curobo_runtime,
            domino_python_runtime=domino_python_runtime,
        ),
    }
    contract_path = run_root / "run_contract.json"
    if contract_path.is_file():
        existing_contract = load_json(contract_path)
        if existing_contract != contract:
            raise RuntimeError(
                "run root contract differs from this launch; refusing to mix "
                f"checkpoints, seeds, or runner versions: {contract_path}"
            )
    else:
        write_json(contract_path, contract)

    pending: deque[str] = deque()
    failures: dict[str, dict[str, Any]] = {}
    for task in tasks:
        aggregate_path = tasks_root / task / "aggregate.json"
        if (
            valid_task_aggregate(
                aggregate_path,
                episodes_per_task,
                expected_start_seed=start_seed,
            )
            is None
        ):
            archive_incomplete_task(tasks_root / task, archive_root)
            pending.append(task)

    running: dict[str, dict[str, Any]] = {}
    free_gpus: deque[str] = deque(gpus)
    status_path = run_root / "STATUS.jsonl"

    while pending or running:
        while pending and free_gpus:
            task = pending.popleft()
            gpu = free_gpus.popleft()
            task_root = tasks_root / task
            task_config = dict(base_config)
            task_config["task_name"] = task
            checkpoint_label = str(base_config["ckpt_setting"])
            task_config["ckpt_setting"] = f"{checkpoint_label}-{task}"
            config_path = configs_root / f"{task}.yml"
            config_path.write_text(
                yaml.safe_dump(task_config, sort_keys=False), encoding="utf-8"
            )
            log_path = logs_root / f"{task}.launcher.log"
            handle: TextIO = log_path.open("w", encoding="utf-8")
            command = [
                str(python),
                "-u",
                str(parallel_runner),
                "--python",
                str(python),
                "--domino-root",
                str(domino_root),
                "--eval-policy",
                str(eval_policy),
                "--config",
                str(config_path),
                "--runtime-root",
                str(runtime_root),
                "--curobo-root",
                str(curobo_root),
                "--curobo-extension-sha256",
                curobo_extension_sha256,
                "--run-root",
                str(task_root),
                "--gpus",
                gpu,
                "--total-episodes",
                str(episodes_per_task),
                "--start-seed",
                str(start_seed),
                "--slot-seed-stride",
                str(benchmark["slot_seed_stride"]),
                "--timeout-seconds",
                str(task_timeout_seconds),
                "--stagger-seconds",
                "0",
            ]
            process = subprocess.Popen(
                command,
                cwd=domino_root,
                env=_worker_env(extra_pythonpath),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[task] = {
                "gpu": gpu,
                "process": process,
                "handle": handle,
                "log": str(log_path),
                "started_unix": time.time(),
            }
            with status_path.open("a", encoding="utf-8") as status_stream:
                status_stream.write(
                    json.dumps(
                        {
                            "event": "started",
                            "task": task,
                            "gpu": gpu,
                            "pid": process.pid,
                        }
                    )
                    + "\n"
                )
            print(f"started task={task} gpu={gpu} pid={process.pid}", flush=True)

        finished: list[str] = []
        for task, record in running.items():
            running_process: subprocess.Popen[Any] = record["process"]
            return_code = running_process.poll()
            if return_code is None:
                continue
            record["handle"].close()
            aggregate_path = tasks_root / task / "aggregate.json"
            aggregate = valid_task_aggregate(
                aggregate_path,
                episodes_per_task,
                expected_start_seed=start_seed,
            )
            task_status = (
                "complete" if return_code == 0 and aggregate is not None else "failed"
            )
            if task_status != "complete":
                failures[task] = {
                    "return_code": return_code,
                    "gpu": record["gpu"],
                    "log": record["log"],
                }
            with status_path.open("a", encoding="utf-8") as status_file:
                status_file.write(
                    json.dumps(
                        {
                            "event": task_status,
                            "task": task,
                            "gpu": record["gpu"],
                            "return_code": return_code,
                        }
                    )
                    + "\n"
                )
            print(
                f"finished task={task} gpu={record['gpu']} "
                f"status={task_status} rc={return_code}",
                flush=True,
            )
            free_gpus.append(record["gpu"])
            finished.append(task)
        for task in finished:
            del running[task]

        summary = summarize(
            run_root=run_root,
            tasks=tasks,
            episodes_per_task=episodes_per_task,
            failures=failures,
            start_seed=start_seed,
            bootstrap_replicates=int(benchmark["bootstrap_replicates"]),
            bootstrap_seed=int(benchmark["bootstrap_seed"]),
        )
        write_json(run_root / "aggregate.json", summary)
        print(
            f"progress tasks={summary['completed_tasks']}/{summary['task_count']} "
            f"episodes={summary['reported_episodes']}/{summary['requested_episodes']}",
            flush=True,
        )
        if pending or running:
            time.sleep(5)

    summary = summarize(
        run_root=run_root,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
        failures=failures,
        start_seed=start_seed,
        bootstrap_replicates=int(benchmark["bootstrap_replicates"]),
        bootstrap_seed=int(benchmark["bootstrap_seed"]),
    )
    write_json(run_root / "aggregate.json", summary)
    print(
        f"suite complete tasks={summary['completed_tasks']}/{summary['task_count']} "
        f"episodes={summary['reported_episodes']}/{summary['requested_episodes']} "
        f"success={summary['success_count']}/{summary['requested_episodes']}",
        flush=True,
    )
    return 0 if summary["completed_tasks"] == len(tasks) else 2


if __name__ == "__main__":
    sys.exit(main())
