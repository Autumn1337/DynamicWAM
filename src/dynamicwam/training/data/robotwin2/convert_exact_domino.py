"""Strict DOMINO schema-v2 to DynamicWAM source conversion."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from dynamicwam.absolute_motion import (
    build_flow_cache_parameters,
    build_motion_features,
    compute_flow_observation,
    load_exact_flow_cache,
    write_exact_flow_cache,
)
from dynamicwam.config import load_profile
from dynamicwam.config.schema import OFFICIAL_LEVEL1_TASKS
from dynamicwam.training.data.robotwin2.aloha_qpos import (
    ALOHA_QPOS_DIM,
    aloha_qpos_is_valid,
)

CAMERA_PATHS = (
    "observation/head_camera/rgb",
    "observation/left_camera/rgb",
    "observation/right_camera/rgb",
)
INTERCEPTION_FIELDS = (
    "schema_version",
    "frame_index",
    "sim_step_index",
    "sim_time_seconds",
    "sim_timestep_seconds",
    "motion_start_sim_step_index",
    "motion_start_sim_time_seconds",
    "planned_intercept_sim_time_seconds",
    "planned_intercept_position",
    "target_pose_wxyz",
    "target_contact_poses_wxyz",
    "ee_poses_wxyz",
    "camera_poses_wxyz",
    "gripper_positions",
    "target_gripper_contact_mask",
    "target_gripper_contact_centroids_world",
    "target_gripper_contact_point_counts",
    "first_target_gripper_contact_valid",
    "first_target_gripper_contact_step_index",
    "first_target_gripper_contact_time_seconds",
    "first_target_gripper_contact_centroids_world",
    "first_target_gripper_contact_point_counts",
    "first_target_gripper_contact_ee_poses_wxyz",
    "first_target_gripper_contact_target_poses_wxyz",
    "stable_grasp_succeeded",
)


def _decode_image(encoded: Any) -> np.ndarray:
    import cv2

    image = cv2.imdecode(
        np.frombuffer(encoded, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("DOMINO contains an undecodable RGB frame")
    return image


def _combined_frame(
    head_bgr: np.ndarray,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
) -> np.ndarray:
    import cv2

    height, width = head_bgr.shape[:2]
    if left_bgr.shape[:2] != (height, width) or right_bgr.shape[:2] != (
        height,
        width,
    ):
        raise ValueError("DOMINO camera resolutions differ within an episode")
    half = (width // 2, height // 2)
    bottom = np.hstack(
        (
            cv2.resize(left_bgr, half, interpolation=cv2.INTER_AREA),
            cv2.resize(right_bgr, half, interpolation=cv2.INTER_AREA),
        )
    )
    return np.vstack((head_bgr, bottom))


def _read_episode(
    source_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[Any, Any, Any]]:
    import h5py

    with h5py.File(source_path, "r") as source:
        if "joint_action/vector" not in source:
            raise ValueError(f"joint_action/vector is missing: {source_path}")
        qpos = np.asarray(source["joint_action/vector"][()], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != ALOHA_QPOS_DIM or len(qpos) == 0:
            raise ValueError(f"invalid qpos shape in {source_path}: {qpos.shape}")
        if not aloha_qpos_is_valid(qpos):
            raise ValueError(f"invalid qpos values in {source_path}")
        if "interception" not in source:
            raise ValueError(f"schema-v2 interception group is missing: {source_path}")
        group = source["interception"]
        if set(group.keys()) != set(INTERCEPTION_FIELDS):
            raise ValueError(
                f"interception fields differ from pinned schema v2 in {source_path}"
            )
        interception = {key: np.asarray(group[key][()]) for key in INTERCEPTION_FIELDS}
        frame_count = len(qpos)
        for key, value in interception.items():
            if value.ndim == 0 or int(value.shape[0]) != frame_count:
                raise ValueError(
                    f"interception field {key} is not frame-synchronous in "
                    f"{source_path}: {value.shape}"
                )
        versions = np.asarray(interception["schema_version"], dtype=np.int64)
        frame_indices = np.asarray(interception["frame_index"], dtype=np.int64)
        simulator_steps = np.asarray(
            interception["sim_step_index"],
            dtype=np.int64,
        )
        times = np.asarray(
            interception["sim_time_seconds"],
            dtype=np.float64,
        )
        timesteps = np.asarray(
            interception["sim_timestep_seconds"],
            dtype=np.float64,
        )
        if (
            not np.all(versions == 2)
            or not np.array_equal(frame_indices, np.arange(frame_count))
            or not np.isfinite(times).all()
            or not np.isfinite(timesteps).all()
            or np.any(timesteps <= 0.0)
            or np.any(np.diff(simulator_steps) < 0)
            or np.any(np.diff(times) < -1e-12)
        ):
            raise ValueError(f"invalid schema-v2 timeline in {source_path}")
        if not np.allclose(
            np.diff(times),
            np.diff(simulator_steps).astype(np.float64) * timesteps[1:],
            atol=1e-9,
            rtol=1e-9,
        ):
            raise ValueError(
                f"inconsistent simulator step/time timeline in {source_path}"
            )
        cameras = tuple(source[path][()] for path in CAMERA_PATHS)
        if any(len(camera) != frame_count for camera in cameras):
            raise ValueError(f"camera/qpos frame mismatch in {source_path}")
    return qpos, interception, cameras


def _existing_output_is_valid(
    video_path: Path,
    qpos_path: Path,
    interception_path: Path,
    cache_path: Path,
    params_json: str,
) -> bool:
    import cv2
    import torch

    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in (video_path, qpos_path, interception_path)
    ):
        return False
    qpos = torch.load(qpos_path, map_location="cpu", weights_only=True)
    interception = torch.load(
        interception_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(qpos, torch.Tensor) or not isinstance(interception, dict):
        return False
    if not aloha_qpos_is_valid(qpos.numpy()) or set(interception) != set(
        INTERCEPTION_FIELDS
    ):
        return False
    frame_count_from_qpos = int(qpos.shape[0])
    for value in interception.values():
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError):
            return False
        if tensor.ndim == 0 or int(tensor.shape[0]) != frame_count_from_qpos:
            return False
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            return False
    try:
        versions = torch.as_tensor(
            interception["schema_version"],
            dtype=torch.int64,
        )
        frame_indices = torch.as_tensor(
            interception["frame_index"],
            dtype=torch.int64,
        )
        simulator_steps = torch.as_tensor(
            interception["sim_step_index"],
            dtype=torch.int64,
        )
        times = torch.as_tensor(
            interception["sim_time_seconds"],
            dtype=torch.float64,
        )
        timesteps = torch.as_tensor(
            interception["sim_timestep_seconds"],
            dtype=torch.float64,
        )
    except (KeyError, TypeError, ValueError):
        return False
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    readable = capture.isOpened()
    capture.release()
    artifacts_valid = (
        readable
        and frame_count == frame_count_from_qpos
        and abs(fps - 30.0) <= 1e-3
        and versions.shape == (frame_count,)
        and frame_indices.shape == (frame_count,)
        and simulator_steps.shape == (frame_count,)
        and times.shape == (frame_count,)
        and timesteps.shape == (frame_count,)
        and bool((versions == 2).all())
        and torch.equal(
            frame_indices,
            torch.arange(frame_count, dtype=torch.int64),
        )
        and bool(torch.isfinite(times).all())
        and bool(torch.isfinite(timesteps).all())
        and bool((timesteps > 0.0).all())
        and bool((torch.diff(simulator_steps) >= 0).all())
        and bool((torch.diff(times) >= -1e-12).all())
        and torch.allclose(
            torch.diff(times),
            torch.diff(simulator_steps).to(dtype=torch.float64) * timesteps[1:],
            atol=1e-9,
            rtol=1e-9,
        )
    )
    if not artifacts_valid or not cache_path.is_file():
        return False
    try:
        cache = load_exact_flow_cache(
            cache_path,
            expected_params_json=params_json,
        )
    except (OSError, TypeError, ValueError):
        return False
    return int(cache["flow_rgb"].shape[0]) == frame_count


def _convert_episode(job: tuple[str, str, str, str]) -> str:
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    source_path = Path(job[0])
    episode_root = Path(job[1])
    cache_path = Path(job[2])
    params_json = job[3]
    params = json.loads(params_json)
    episode_id = int(source_path.stem.removeprefix("episode"))
    video_path = episode_root / "videos" / f"{episode_id}.mp4"
    qpos_path = episode_root / "qpos" / f"{episode_id}.pt"
    interception_path = episode_root / "interception" / f"{episode_id}.pt"
    if _existing_output_is_valid(
        video_path,
        qpos_path,
        interception_path,
        cache_path,
        params_json,
    ):
        return str(source_path)

    qpos, interception, cameras = _read_episode(source_path)
    for directory in (
        video_path.parent,
        qpos_path.parent,
        interception_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    token = f"tmp.{os.getpid()}"
    temporary_video = video_path.with_name(f"{episode_id}.{token}.mp4")
    temporary_qpos = qpos_path.with_name(f"{episode_id}.{token}.pt")
    temporary_interception = interception_path.with_name(f"{episode_id}.{token}.pt")

    try:
        frame_count = len(qpos)
        timestamps = np.asarray(
            interception["sim_time_seconds"],
            dtype=np.float64,
        )
        interval_stride = int(params["raw_stride"])
        compute_size = tuple(int(value) for value in params["size"])
        flow_rgb = np.zeros(
            (frame_count, *compute_size, 3),
            dtype=np.uint8,
        )
        displacement = np.zeros((frame_count, 4), dtype=np.float32)
        starts = np.empty(frame_count, dtype=np.float64)
        temporal_valid = np.zeros(frame_count, dtype=np.bool_)
        flow_quality_valid = np.zeros(frame_count, dtype=np.bool_)
        flow_reliable_fraction = np.zeros(frame_count, dtype=np.float32)
        interval_valid = np.zeros(frame_count, dtype=np.bool_)
        head_history: deque[np.ndarray] = deque(maxlen=interval_stride)
        first_head_rgb: np.ndarray | None = None
        writer = None
        try:
            for index in range(frame_count):
                head_bgr, left_bgr, right_bgr = (
                    _decode_image(camera[index]) for camera in cameras
                )
                combined = _combined_frame(
                    head_bgr,
                    left_bgr,
                    right_bgr,
                )
                if writer is None:
                    height, width = combined.shape[:2]
                    writer = cv2.VideoWriter(
                        str(temporary_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(params["container_fps"]),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot create video: {temporary_video}")
                writer.write(combined)

                head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
                if first_head_rgb is None:
                    first_head_rgb = head_rgb
                previous_index = max(0, index - interval_stride)
                previous_head = (
                    first_head_rgb if index < interval_stride else head_history[0]
                )
                starts[index] = timestamps[previous_index]
                temporal_valid[index] = (
                    index - previous_index == interval_stride
                    and timestamps[index] > timestamps[previous_index]
                )
                if temporal_valid[index]:
                    (
                        rgb,
                        statistics,
                        reliable_fraction,
                        quality_valid,
                    ) = compute_flow_observation(
                        previous_head,
                        head_rgb,
                        compute_size=compute_size,
                        normalization_percentile=float(
                            params["rgb_normalization"]["percentile"]
                        ),
                        farneback=dict(params["farneback"]),
                        quality=dict(params["quality"]),
                    )
                    flow_reliable_fraction[index] = reliable_fraction
                    flow_quality_valid[index] = quality_valid
                    interval_valid[index] = quality_valid
                    if quality_valid:
                        flow_rgb[index] = rgb
                        displacement[index] = statistics
                head_history.append(head_rgb)
        finally:
            if writer is not None:
                writer.release()
        previous = np.arange(frame_count, dtype=np.int64) - interval_stride
        features, interval_valid, acceleration_valid = build_motion_features(
            displacement,
            starts,
            timestamps,
            interval_valid,
            previous_interval_indices=previous,
        )
        write_exact_flow_cache(
            cache_path,
            arrays={
                "flow_rgb": flow_rgb,
                "motion_features": features,
                "temporal_valid": temporal_valid,
                "flow_quality_valid": flow_quality_valid,
                "flow_reliable_fraction": flow_reliable_fraction,
                "interval_valid": interval_valid,
                "acceleration_valid": acceleration_valid,
                "interval_start_time_seconds": starts,
                "interval_end_time_seconds": timestamps,
            },
            params_json=params_json,
        )
        capture = cv2.VideoCapture(str(temporary_video))
        encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
        readable = capture.isOpened()
        capture.release()
        if (
            not readable
            or encoded_frames != frame_count
            or abs(encoded_fps - float(params["container_fps"])) > 1e-3
        ):
            raise RuntimeError(f"encoded video failed validation: {source_path}")

        torch.save(torch.from_numpy(qpos), temporary_qpos)
        torch.save(
            {
                key: torch.from_numpy(np.ascontiguousarray(value))
                for key, value in interception.items()
            },
            temporary_interception,
        )
        temporary_video.replace(video_path)
        temporary_qpos.replace(qpos_path)
        temporary_interception.replace(interception_path)
    except BaseException:
        temporary_video.unlink(missing_ok=True)
        temporary_qpos.unlink(missing_ok=True)
        temporary_interception.unlink(missing_ok=True)
        raise
    return str(source_path)


def _copy_languages(
    *,
    language_embeddings_root: Path,
    output_root: Path,
) -> None:
    from dynamicwam.language import (
        assert_language_embeddings_equal,
        load_language_embeddings,
    )

    target_root = output_root / "language"
    target_root.mkdir(parents=True, exist_ok=True)
    for task in OFFICIAL_LEVEL1_TASKS:
        source = language_embeddings_root / f"{task}.pt"
        values = load_language_embeddings(source)
        target = target_root / f"{task}.pt"
        if target.is_file():
            assert_language_embeddings_equal(
                load_language_embeddings(target),
                values,
                label=str(target),
            )
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def _verify_converted_file_set(
    *,
    output_root: Path,
    cache_root: Path,
    split_contracts: tuple[tuple[str, str, int], ...],
) -> None:
    for split, _config_name, expected_count in split_contracts:
        expected_names = {str(index) for index in range(expected_count)}
        for task in OFFICIAL_LEVEL1_TASKS:
            task_root = output_root / split / task
            for directory, suffix in (
                ("videos", ".mp4"),
                ("qpos", ".pt"),
                ("interception", ".pt"),
            ):
                actual = {
                    path.stem for path in (task_root / directory).glob(f"*{suffix}")
                }
                if actual != expected_names:
                    raise RuntimeError(
                        "converted file set differs from the exact collection "
                        f"contract: {split}/{task}/{directory}"
                    )
            actual_caches = {
                path.name.removesuffix(".flow.npz")
                for path in (cache_root / split / task / "videos").glob("*.flow.npz")
            }
            if actual_caches != expected_names:
                raise RuntimeError(
                    "motion cache set differs from the exact collection "
                    f"contract: {split}/{task}"
                )
    language_names = {path.stem for path in (output_root / "language").glob("*.pt")}
    if language_names != set(OFFICIAL_LEVEL1_TASKS):
        raise RuntimeError("converted language bank differs from the official task set")


def _select_episode_prefix(
    *,
    source_dir: Path,
    expected_count: int,
    split: str,
    task: str,
) -> list[Path]:
    episode_paths = sorted(
        source_dir.glob("episode*.hdf5"),
        key=lambda path: int(path.stem.removeprefix("episode")),
    )
    if len(episode_paths) < expected_count:
        raise RuntimeError(
            f"{split}/{task} has {len(episode_paths)} raw episodes; "
            f"expected at least {expected_count}"
        )
    actual_ids = [int(path.stem.removeprefix("episode")) for path in episode_paths]
    if actual_ids != list(range(len(episode_paths))):
        raise RuntimeError(f"{split}/{task} episode ids are not contiguous")
    return episode_paths[:expected_count]


def run(*, config_path: str, workers: int) -> None:
    profile = load_profile(config_path)
    raw = profile.raw
    raw_root = Path(raw["paths"]["raw_dataset"])
    output_root = Path(raw["paths"]["source_dataset"])
    cache_root = Path(raw["paths"]["head_flow_cache"])
    collection = raw["collection"]
    params = build_flow_cache_parameters(
        head_flow_config=raw["method"]["head_flow"],
        global_downsample_rate=int(raw["method"]["video"]["global_downsample_rate"]),
    )
    params_json = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
    )
    split_contracts = (
        (
            "clean",
            collection["clean_config_name"],
            int(collection["clean_episodes_per_task"]),
        ),
        (
            "randomized",
            collection["randomized_config_name"],
            int(collection["randomized_episodes_per_task"]),
        ),
    )
    jobs: list[tuple[str, str, str, str]] = []
    for split, config_name, expected_count in split_contracts:
        for task in OFFICIAL_LEVEL1_TASKS:
            source_dir = raw_root / split / task / config_name / "data"
            episode_paths = _select_episode_prefix(
                source_dir=source_dir,
                expected_count=expected_count,
                split=split,
                task=task,
            )
            target = output_root / split / task
            jobs.extend(
                (
                    str(path),
                    str(target),
                    str(
                        cache_root
                        / split
                        / task
                        / "videos"
                        / (path.stem.removeprefix("episode") + ".flow.npz")
                    ),
                    params_json,
                )
                for path in episode_paths
            )

    with mp.get_context("spawn").Pool(max(1, int(workers))) as pool:
        for index, _source in enumerate(
            pool.imap_unordered(_convert_episode, jobs, chunksize=1),
            start=1,
        ):
            if index % 100 == 0 or index == len(jobs):
                print(f"exact DOMINO conversion {index}/{len(jobs)}", flush=True)
    _copy_languages(
        language_embeddings_root=Path(raw["paths"]["language_embeddings"]),
        output_root=output_root,
    )
    _verify_converted_file_set(
        output_root=output_root,
        cache_root=cache_root,
        split_contracts=split_contracts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    profile = load_profile(arguments.config)
    workers = (
        int(arguments.workers)
        if arguments.workers is not None
        else int(profile.raw["collection"]["converter_workers"])
    )
    if workers <= 0:
        parser.error("workers must be positive")
    run(config_path=arguments.config, workers=workers)


if __name__ == "__main__":
    main()
