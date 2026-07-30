"""Precompute exact-time flow RGB, numeric motion, and feature statistics."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from dynamicwam.absolute_motion import (
    MOTION_FEATURE_DIM,
    MOTION_FEATURE_NAMES,
    MOTION_STATISTICS_VERSION,
    TEMPORAL_CONTRACT,
    build_flow_cache_parameters,
    load_exact_flow_cache,
)
from dynamicwam.config import load_profile
from dynamicwam.config.schema import OFFICIAL_LEVEL1_TASKS


def _statistics_terms(
    features: np.ndarray,
    interval_valid: np.ndarray,
    acceleration_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.repeat(interval_valid[:, None], MOTION_FEATURE_DIM, axis=1)
    valid[:, 9:12] = acceleration_valid[:, None]
    values = np.asarray(features, dtype=np.float64)
    count = valid.sum(axis=0, dtype=np.int64)
    total = np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
    total_square = np.where(valid, values * values, 0.0).sum(
        axis=0,
        dtype=np.float64,
    )
    return count, total, total_square


def _process_one(
    job: tuple[str, str, str],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    video_value, cache_value, params_json = job
    video_path = Path(video_value)
    cache_path = Path(cache_value)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"conversion did not create the exact motion cache: {cache_path}"
        )
    arrays = load_exact_flow_cache(
        cache_path,
        expected_params_json=params_json,
    )
    return str(video_path), *_statistics_terms(
        arrays["motion_features"],
        arrays["interval_valid"],
        arrays["acceleration_valid"],
    )


def _write_statistics(
    path: Path,
    *,
    count: np.ndarray,
    total: np.ndarray,
    total_square: np.ndarray,
    minimum_scale: float,
) -> None:
    if np.any(count <= 0):
        raise RuntimeError(
            f"some motion features have no valid observations: {count.tolist()}"
        )
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    standard_deviation = np.sqrt(variance)
    scale = np.maximum(standard_deviation, float(minimum_scale))
    payload = {
        "schema_version": MOTION_STATISTICS_VERSION,
        "feature_names": list(MOTION_FEATURE_NAMES),
        "count": count.astype(np.int64).tolist(),
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
        "scale": scale.tolist(),
        "minimum_scale": float(minimum_scale),
        "temporal_contract": TEMPORAL_CONTRACT,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    *,
    config_path: str,
    workers: int,
) -> None:
    profile = load_profile(config_path)
    raw = profile.raw
    dataset_root = Path(raw["paths"]["source_dataset"])
    cache_root = Path(raw["paths"]["head_flow_cache"])
    method = raw["method"]
    flow = method["head_flow"]
    video = method["video"]
    params = build_flow_cache_parameters(
        head_flow_config=flow,
        global_downsample_rate=int(video["global_downsample_rate"]),
    )
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    collection = raw["collection"]
    split_counts = {
        "clean": int(collection["clean_episodes_per_task"]),
        "randomized": int(collection["randomized_episodes_per_task"]),
    }
    expected_videos = {
        dataset_root / split / task / "videos" / f"{episode}.mp4"
        for split, episode_count in split_counts.items()
        for task in OFFICIAL_LEVEL1_TASKS
        for episode in range(episode_count)
    }
    actual_videos = set(dataset_root.rglob("*.mp4"))
    if actual_videos != expected_videos:
        missing = sorted(str(path) for path in expected_videos - actual_videos)
        extra = sorted(str(path) for path in actual_videos - expected_videos)
        raise RuntimeError(
            "converted video set differs from the collection contract: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    videos = sorted(expected_videos)
    jobs = [
        (
            str(video_path),
            str(
                cache_root
                / video_path.relative_to(dataset_root).with_suffix(".flow.npz")
            ),
            params_json,
        )
        for video_path in videos
    ]
    total_count = np.zeros(MOTION_FEATURE_DIM, dtype=np.int64)
    total = np.zeros(MOTION_FEATURE_DIM, dtype=np.float64)
    total_square = np.zeros(MOTION_FEATURE_DIM, dtype=np.float64)
    with mp.get_context("spawn").Pool(max(1, int(workers))) as pool:
        for index, (_video, count, partial, partial_square) in enumerate(
            pool.imap_unordered(_process_one, jobs, chunksize=1),
            start=1,
        ):
            total_count += count
            total += partial
            total_square += partial_square
            if index % 100 == 0 or index == len(jobs):
                print(f"absolute-motion cache {index}/{len(jobs)}", flush=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    _write_statistics(
        cache_root / "motion_stats.json",
        count=total_count,
        total=total,
        total_square=total_square,
        minimum_scale=1e-6,
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
        else int(profile.raw["collection"]["motion_statistics_workers"])
    )
    if workers <= 0:
        parser.error("workers must be positive")
    run(
        config_path=arguments.config,
        workers=workers,
    )


if __name__ == "__main__":
    main()
