"""Parallel exact-time DOMINO data collection launcher."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from dynamicwam.config import load_profile
from dynamicwam.config.schema import OFFICIAL_LEVEL1_TASKS
from dynamicwam.external_assets import verify_robotwin_asset_trees
from dynamicwam.external_setup import (
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
from dynamicwam.training.domino_stream_queue import StreamQueue


def _curobo_environment(
    *,
    gpu: str,
    curobo_root: Path,
    extra_pythonpath: tuple[Path, ...] = (),
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    pythonpath = [str(curobo_root)]
    pythonpath.extend(str(path) for path in extra_pythonpath)
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return environment


def _probe_curobo_extension(
    *,
    domino_root: Path,
    python: str,
    gpu: str,
    curobo_root: Path,
    expected_sha256: str,
) -> None:
    if not (curobo_root / "curobo").is_dir():
        raise FileNotFoundError(f"CuRobo root has no package directory: {curobo_root}")
    completed = subprocess.run(
        [
            python,
            "-c",
            (
                "from pathlib import Path; import torch; "
                "import curobo.curobolib.line_search_cu as module; "
                "print(Path(module.__file__).resolve())"
            ),
        ],
        cwd=domino_root,
        env=_curobo_environment(gpu=gpu, curobo_root=curobo_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CuRobo SM90 extension preflight failed before collection:\n"
            f"{completed.stderr[-4000:]}"
        )
    output = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not output:
        raise RuntimeError("CuRobo extension preflight returned no path")
    extension = Path(output[-1]).resolve()
    try:
        extension.relative_to(curobo_root)
    except ValueError as exc:
        raise RuntimeError(
            f"CuRobo import resolved outside the pinned SM90 runtime: {extension}"
        ) from exc
    if not extension.is_file() or sha256_file(extension) != str(expected_sha256):
        raise RuntimeError(
            f"CuRobo line_search extension differs from its pinned SHA256: {extension}"
        )


def _install_collection_configs(
    *,
    domino_root: Path,
    project_root: Path,
    config_source_root: Path,
    config_names: tuple[str, str],
    expected_episodes: tuple[int, int],
    expected_language_prompts_per_task: int,
    expected_save_roots: tuple[Path, Path],
    expected_dynamic_coefficient: float,
) -> None:
    import yaml

    task_config_root = domino_root / "task_config"
    for index, name in enumerate(config_names):
        source = config_source_root / f"{name}.yml"
        if not source.is_file():
            raise FileNotFoundError(
                f"absolute-motion collection config is missing: {source}"
            )
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"collection config must be a mapping: {source}")
        configured_save_root = Path(str(payload.get("save_path", "")))
        if not configured_save_root.is_absolute():
            configured_save_root = (project_root / configured_save_root).resolve()
        randomized = index == 1
        domain = payload.get("domain_randomization", {})
        required_true = (
            payload.get("use_dynamic") is True
            and payload.get("collect_data") is True
            and payload.get("check_render_success") is True
            and payload.get("use_seed") is False
            and payload.get("camera", {}).get("collect_head_camera") is True
            and payload.get("camera", {}).get("collect_wrist_camera") is True
            and payload.get("data_type", {}).get("rgb") is True
            and payload.get("data_type", {}).get("qpos") is True
            and payload.get("data_type", {}).get("interception") is True
        )
        if (
            not required_true
            or payload.get("save_failed_cases", False) is not False
            or payload.get("embodiment") != ["aloha-agilex"]
            or payload.get("camera", {}).get("head_camera_type") != "D435"
            or payload.get("camera", {}).get("wrist_camera_type") != "D435"
            or domain.get("random_background") is not randomized
            or domain.get("cluttered_table") is not randomized
            or domain.get("random_light") is not randomized
            or int(payload.get("dynamic_level", -1)) != 1
            or float(payload.get("dynamic_coefficient", -1.0))
            != float(expected_dynamic_coefficient)
            or int(payload.get("episode_num", -1)) != int(expected_episodes[index])
            or int(payload.get("language_num", -1))
            != int(expected_language_prompts_per_task)
            or int(payload.get("max_seed_attempts", -1)) < int(expected_episodes[index])
            or int(payload.get("max_regeneration_attempts", -1))
            < int(expected_episodes[index])
            or configured_save_root != expected_save_roots[index]
        ):
            raise RuntimeError(
                f"collection config differs from the production contract: {source}"
            )
        payload["save_path"] = str(configured_save_root)
        target_content = yaml.safe_dump(
            payload,
            sort_keys=False,
        ).encode("utf-8")
        target = task_config_root / f"{name}.yml"
        if target.is_file() and target.read_bytes() == target_content:
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            temporary.write_bytes(target_content)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        if target.read_bytes() != target_content:
            raise RuntimeError(f"collection config copy failed: {target}")


def _stream_worker_command(
    *,
    python: str,
    role: str,
    domino_root: Path,
    job_root: Path,
    task: str,
    split: str,
    task_config: str,
    target_episodes: int,
    worker_id: str,
    ready_buffer_episodes: int,
    max_new_attempts: int,
    poll_seconds: float,
    renderer_recycle_attempts: int,
) -> list[str]:
    return [
        python,
        "-m",
        "dynamicwam.training.domino_streaming_worker",
        "--role",
        role,
        "--domino-root",
        str(domino_root),
        "--job-root",
        str(job_root),
        "--task",
        task,
        "--split",
        split,
        "--task-config",
        task_config,
        "--target-episodes",
        str(target_episodes),
        "--worker-id",
        worker_id,
        "--ready-buffer-episodes",
        str(ready_buffer_episodes),
        "--max-new-attempts",
        str(max_new_attempts),
        "--poll-seconds",
        str(poll_seconds),
        "--renderer-recycle-attempts",
        str(renderer_recycle_attempts),
    ]


def _terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 30.0
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        process.wait()


def _run_stream_job(
    *,
    domino_root: Path,
    dynamicwam_src: Path,
    python: str,
    task: str,
    task_config: str,
    split: str,
    gpu: str,
    stream_log_root: Path,
    curobo_root: Path,
    output_root: Path,
    target_episodes: int,
    planner_workers: int,
    renderer_workers: int,
    ready_buffer_episodes: int,
    max_new_attempts: int,
    poll_seconds: float,
    renderer_recycle_attempts: int,
) -> None:
    if renderer_workers != 1:
        raise ValueError(
            "Ordered exact-time publication requires one renderer per task"
        )
    job_root = output_root / split / task / task_config
    job_root.mkdir(parents=True, exist_ok=True)
    stream_root = job_root / ".absolute_motion_stream_v2"
    stream_root.mkdir(parents=True, exist_ok=True)
    supervisor_lock = stream_root / "supervisor.lock"
    log_directory = stream_log_root / split / task
    log_directory.mkdir(parents=True, exist_ok=True)
    environment = _curobo_environment(
        gpu=gpu,
        curobo_root=curobo_root,
        extra_pythonpath=(dynamicwam_src, domino_root),
    )

    with supervisor_lock.open("a+b", buffering=0) as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"streaming collection is already active for {split}/{task}"
            ) from exc

        common = {
            "python": python,
            "domino_root": domino_root,
            "job_root": job_root,
            "task": task,
            "split": split,
            "task_config": task_config,
            "target_episodes": target_episodes,
            "ready_buffer_episodes": ready_buffer_episodes,
            "max_new_attempts": max_new_attempts,
            "poll_seconds": poll_seconds,
            "renderer_recycle_attempts": renderer_recycle_attempts,
        }
        with (log_directory / "initialize.log").open(
            "ab",
            buffering=0,
        ) as initialize_log:
            completed = subprocess.run(
                _stream_worker_command(
                    role="initialize",
                    worker_id="initialize",
                    **common,
                ),
                cwd=domino_root,
                env=environment,
                stdout=initialize_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stream initialization failed for {split}/{task}; "
                f"see {log_directory / 'initialize.log'}"
            )

        processes: list[subprocess.Popen[bytes]] = []
        logs: list[object] = []
        try:
            for index in range(renderer_workers):
                log = (log_directory / f"renderer_{index}.log").open(
                    "ab",
                    buffering=0,
                )
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        _stream_worker_command(
                            role="renderer",
                            worker_id=f"renderer-{index}",
                            **common,
                        ),
                        cwd=domino_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            for index in range(planner_workers):
                log = (log_directory / f"planner_{index}.log").open(
                    "ab",
                    buffering=0,
                )
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        _stream_worker_command(
                            role="planner",
                            worker_id=f"planner-{index}",
                            **common,
                        ),
                        cwd=domino_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )

            while True:
                return_codes = [process.poll() for process in processes]
                failures = [
                    code for code in return_codes if code is not None and code != 0
                ]
                if failures:
                    _terminate_processes(processes)
                    raise RuntimeError(
                        f"stream worker failed for {split}/{task} "
                        f"on GPU {gpu}; see {log_directory}"
                    )
                if all(code == 0 for code in return_codes):
                    break
                time.sleep(0.5)
        except Exception:
            _terminate_processes(processes)
            raise
        finally:
            for log in logs:
                log.close()

        with (log_directory / "validate.log").open(
            "ab",
            buffering=0,
        ) as validate_log:
            completed = subprocess.run(
                _stream_worker_command(
                    role="validate",
                    worker_id="validate",
                    **common,
                ),
                cwd=domino_root,
                env=environment,
                stdout=validate_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stream validation failed for {split}/{task}; "
                f"see {log_directory / 'validate.log'}"
            )


def _jobs(
    *,
    selected_split: str,
    clean_config: str,
    randomized_config: str,
    tasks: Iterable[str],
) -> list[tuple[str, str, str]]:
    splits = (
        ("clean", clean_config),
        ("randomized", randomized_config),
    )
    return [
        (split, task, task_config)
        for split, task_config in splits
        if selected_split == "all" or selected_split == split
        for task in tasks
    ]


def _job_is_complete(
    *,
    output_root: Path,
    split: str,
    task: str,
    task_config: str,
    expected_episodes: int,
) -> bool:
    job_root = output_root / split / task / task_config
    if not job_root.exists():
        return False
    queue = StreamQueue(
        job_root=job_root,
        task=task,
        split=split,
        task_config=task_config,
        target_episodes=expected_episodes,
    )
    if not queue.database_path.is_file():
        return False
    status = queue.open_existing()
    if not status.complete:
        return False
    queue.validate_complete()
    return True


def _run_stream_gpu_queue(
    *,
    gpu: str,
    jobs: list[tuple[str, str, str]],
    domino_root: Path,
    dynamicwam_src: Path,
    python: str,
    stream_log_root: Path,
    curobo_root: Path,
    output_root: Path,
    expected_by_split: dict[str, int],
    planner_workers: int,
    renderer_workers: int,
    ready_buffer_episodes: int,
    max_attempts_by_split: dict[str, int],
    poll_seconds: float,
    renderer_recycle_attempts: int,
) -> list[tuple[str, str]]:
    completed: list[tuple[str, str]] = []
    for split, task, task_config in jobs:
        _run_stream_job(
            domino_root=domino_root,
            dynamicwam_src=dynamicwam_src,
            python=python,
            task=task,
            task_config=task_config,
            split=split,
            gpu=gpu,
            stream_log_root=stream_log_root,
            curobo_root=curobo_root,
            output_root=output_root,
            target_episodes=expected_by_split[split],
            planner_workers=planner_workers,
            renderer_workers=renderer_workers,
            ready_buffer_episodes=ready_buffer_episodes,
            max_new_attempts=max_attempts_by_split[split],
            poll_seconds=poll_seconds,
            renderer_recycle_attempts=renderer_recycle_attempts,
        )
        completed.append((split, task))
        print(
            f"stream collection complete {split}/{task} GPU={gpu}",
            flush=True,
        )
    return completed


def run(
    *,
    config_path: str,
    selected_split: str,
    tasks: list[str],
    gpus: list[str],
    output_root_override: str | None = None,
    log_root_override: str | None = None,
    target_episodes_override: int | None = None,
) -> None:
    profile = load_profile(config_path)
    raw = profile.raw
    collection = raw["collection"]
    benchmark = raw["benchmark"]
    project_root = Path(raw["paths"]["project_root"])
    config_source_root = project_root / "configs" / "domino"
    domino_root = Path(benchmark["domino_root"])
    external_manifest = Path(raw["paths"]["external_assets_manifest"])
    verify_robotwin_asset_trees(
        root=project_root / "external" / "robotwin-assets",
        manifest_path=external_manifest,
    )
    verify_robotwin_asset_links(
        asset_root=project_root / "external" / "robotwin-assets",
        domino_root=domino_root,
    )
    if not (domino_root / "script" / "collect_data.py").is_file():
        raise FileNotFoundError(f"pinned DOMINO source is missing: {domino_root}")
    generated_configs = tuple(
        f"task_config/{name}.yml"
        for name in (
            str(collection["clean_config_name"]),
            str(collection["randomized_config_name"]),
        )
    )
    actual_source_sha256 = sha256_tree(
        domino_root,
        excluded_relative_paths=generated_configs,
        excluded_relative_prefixes=DOMINO_RUNTIME_ASSET_PREFIXES,
    )
    if actual_source_sha256 != benchmark["domino_source_sha256"]:
        raise RuntimeError(
            "DOMINO source differs from the pinned commit "
            f"{benchmark['domino_commit']}: {actual_source_sha256}"
        )
    _install_collection_configs(
        domino_root=domino_root,
        project_root=project_root,
        config_source_root=config_source_root,
        config_names=(
            str(collection["clean_config_name"]),
            str(collection["randomized_config_name"]),
        ),
        expected_episodes=(
            int(collection["clean_episodes_per_task"]),
            int(collection["randomized_episodes_per_task"]),
        ),
        expected_language_prompts_per_task=int(
            raw["data"]["language_prompts_per_task"]
        ),
        expected_save_roots=(
            Path(raw["paths"]["raw_dataset"]) / "clean",
            Path(raw["paths"]["raw_dataset"]) / "randomized",
        ),
        expected_dynamic_coefficient=float(benchmark["dynamic_coefficient"]),
    )
    invalid_tasks = sorted(set(tasks) - set(OFFICIAL_LEVEL1_TASKS))
    if invalid_tasks:
        raise ValueError(f"unknown Level-1 tasks: {invalid_tasks}")
    if (
        selected_split not in {"clean", "randomized", "all"}
        or not tasks
        or len(tasks) != len(set(tasks))
        or not gpus
        or len(gpus) != len(set(gpus))
    ):
        raise ValueError("collection requires tasks and GPUs")
    curobo_root = Path(str(benchmark["curobo_root"])).resolve()
    verify_curobo_source(
        destination=curobo_root.parent,
        manifest_path=external_manifest,
    )
    verify_domino_python_runtime(
        python=Path(str(benchmark["python"])),
        manifest_path=external_manifest,
    )
    verify_curobo_runtime(
        destination=curobo_root.parent,
        manifest_path=external_manifest,
    )
    _probe_curobo_extension(
        domino_root=domino_root,
        python=str(benchmark["python"]),
        gpu=gpus[0],
        curobo_root=curobo_root,
        expected_sha256=str(benchmark["curobo_extension_sha256"]),
    )
    collection_log_root = Path(raw["paths"]["collection_logs"])
    stream_log_root = (
        Path(log_root_override).resolve()
        if log_root_override is not None
        else collection_log_root / "stream"
    )
    output_root = (
        Path(output_root_override).resolve()
        if output_root_override is not None
        else Path(raw["paths"]["raw_dataset"])
    )
    jobs = _jobs(
        selected_split=selected_split,
        clean_config=collection["clean_config_name"],
        randomized_config=collection["randomized_config_name"],
        tasks=tasks,
    )
    expected_by_split = {
        "clean": int(collection["clean_episodes_per_task"]),
        "randomized": int(collection["randomized_episodes_per_task"]),
    }
    if target_episodes_override is not None:
        if output_root_override is None or int(target_episodes_override) <= 0:
            raise ValueError(
                "target episode overrides require a positive isolated --output-root"
            )
        expected_by_split = {
            split: int(target_episodes_override) for split in expected_by_split
        }
    import yaml

    max_attempts_by_split = {
        split: int(
            yaml.safe_load(
                (config_source_root / f"{task_config}.yml").read_text(encoding="utf-8")
            )["max_seed_attempts"]
        )
        for split, task_config in (
            ("clean", str(collection["clean_config_name"])),
            ("randomized", str(collection["randomized_config_name"])),
        )
    }
    pending_jobs = []
    for split, task, task_config in jobs:
        if _job_is_complete(
            output_root=output_root,
            split=split,
            task=task,
            task_config=task_config,
            expected_episodes=expected_by_split[split],
        ):
            print(f"collection already complete {split}/{task}", flush=True)
            continue
        pending_jobs.append((split, task, task_config))

    workers_per_gpu = int(collection["workers_per_gpu"])
    planner_workers = int(collection["planner_workers_per_task"])
    renderer_workers = int(collection["renderer_workers_per_task"])
    ready_buffer_episodes = int(collection["ready_buffer_episodes"])
    poll_seconds = float(collection["worker_poll_seconds"])
    renderer_recycle_attempts = int(collection["renderer_recycle_attempts"])
    if collection["collection_raw_mp4"] is not False:
        raise RuntimeError(
            "absolute-motion streaming collection forbids raw MP4 encoding"
        )
    dynamicwam_src = Path(__file__).resolve().parents[2]
    gpu_slots = [gpu for _ in range(workers_per_gpu) for gpu in gpus]
    gpu_queues = [
        pending_jobs[index :: len(gpu_slots)] for index in range(len(gpu_slots))
    ]
    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as executor:
        futures = {
            executor.submit(
                _run_stream_gpu_queue,
                gpu=gpu,
                jobs=gpu_jobs,
                domino_root=domino_root,
                dynamicwam_src=dynamicwam_src,
                python=str(benchmark["python"]),
                stream_log_root=stream_log_root,
                curobo_root=curobo_root,
                output_root=output_root,
                expected_by_split=expected_by_split,
                planner_workers=planner_workers,
                renderer_workers=renderer_workers,
                ready_buffer_episodes=ready_buffer_episodes,
                max_attempts_by_split=max_attempts_by_split,
                poll_seconds=poll_seconds,
                renderer_recycle_attempts=renderer_recycle_attempts,
            ): gpu
            for gpu, gpu_jobs in zip(gpu_slots, gpu_queues, strict=True)
            if gpu_jobs
        }
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        choices=("clean", "randomized", "all"),
        default="all",
    )
    parser.add_argument("--tasks", nargs="+", default=list(OFFICIAL_LEVEL1_TASKS))
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[str(index) for index in range(8)],
    )
    parser.add_argument("--output-root")
    parser.add_argument("--log-root")
    parser.add_argument("--target-episodes", type=int)
    arguments = parser.parse_args()
    run(
        config_path=arguments.config,
        selected_split=arguments.split,
        tasks=list(arguments.tasks),
        gpus=list(arguments.gpus),
        output_root_override=arguments.output_root,
        log_root_override=arguments.log_root,
        target_episodes_override=arguments.target_episodes,
    )


if __name__ == "__main__":
    main()
