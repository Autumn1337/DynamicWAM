"""Shared exact-time image-plane motion contract.

Flow RGB intentionally keeps the original per-map percentile normalization for
spatial shape.  The parallel numeric features preserve cross-sample magnitude
in pixels and exact simulator seconds.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from dynamicwam.motion_contract import (
    ENDPOINT_RULE,
    FARNEBACK_DEFAULTS,
    SPATIAL_UNIT,
    TEMPORAL_CONTRACT,
    TIMESTAMP_SOURCE,
    flow_compute_contract,
    validate_flow_quality_config,
)

ABSOLUTE_MOTION_CHECKPOINT_VERSION = 2
FLOW_CACHE_VERSION = 2
MOTION_STATISTICS_VERSION = 2
FLOW_CACHE_ARRAY_KEYS = frozenset(
    {
        "flow_rgb",
        "motion_features",
        "temporal_valid",
        "flow_quality_valid",
        "flow_reliable_fraction",
        "interval_valid",
        "acceleration_valid",
        "interval_start_time_seconds",
        "interval_end_time_seconds",
        "params",
    }
)
MOTION_FEATURE_NAMES = (
    "mean_displacement_x_pixels",
    "mean_displacement_y_pixels",
    "mean_displacement_magnitude_pixels",
    "p99_displacement_magnitude_pixels",
    "delta_t_seconds",
    "mean_velocity_x_pixels_per_second",
    "mean_velocity_y_pixels_per_second",
    "mean_speed_pixels_per_second",
    "p99_speed_pixels_per_second",
    "mean_acceleration_x_pixels_per_second2",
    "mean_acceleration_y_pixels_per_second2",
    "mean_acceleration_magnitude_pixels_per_second2",
)
MOTION_FEATURE_DIM = len(MOTION_FEATURE_NAMES)
CHECKPOINT_MOTION_METADATA_KEYS = frozenset(
    {
        "history_count",
        "feature_names",
        "feature_mean",
        "feature_scale",
        "temporal_contract",
        "timestamp_source",
        "spatial_unit",
        "statistics_sha256",
        "flow_contract",
    }
)


def build_checkpoint_motion_metadata(
    *,
    history_count: int,
    statistics: Any,
    statistics_sha256: str,
    head_flow_config: Any,
) -> dict[str, Any]:
    statistics = validate_motion_statistics(statistics)
    metadata = {
        "history_count": int(history_count),
        "feature_names": list(statistics["feature_names"]),
        "feature_mean": [float(value) for value in statistics["mean"]],
        "feature_scale": [float(value) for value in statistics["scale"]],
        "temporal_contract": statistics["temporal_contract"],
        "timestamp_source": TIMESTAMP_SOURCE,
        "spatial_unit": SPATIAL_UNIT,
        "statistics_sha256": str(statistics_sha256),
        "flow_contract": flow_compute_contract(head_flow_config),
    }
    return validate_checkpoint_motion_metadata(metadata)


def validate_checkpoint_motion_metadata(payload: Any) -> dict[str, Any]:
    """Validate checkpoint metadata that makes absolute speed comparable."""

    if not isinstance(payload, dict):
        raise TypeError("checkpoint absolute-motion metadata must be a mapping")
    if set(payload) != CHECKPOINT_MOTION_METADATA_KEYS:
        raise ValueError(
            "checkpoint absolute-motion metadata keys differ from v2: "
            f"missing={sorted(CHECKPOINT_MOTION_METADATA_KEYS - set(payload))}, "
            f"unknown={sorted(set(payload) - CHECKPOINT_MOTION_METADATA_KEYS)}"
        )
    history_count = int(payload["history_count"])
    if history_count <= 0:
        raise ValueError("checkpoint motion history_count must be positive")
    if tuple(payload["feature_names"]) != MOTION_FEATURE_NAMES:
        raise ValueError("checkpoint motion feature order differs from v2")
    if (
        payload["temporal_contract"] != TEMPORAL_CONTRACT
        or payload["timestamp_source"] != TIMESTAMP_SOURCE
        or payload["spatial_unit"] != SPATIAL_UNIT
    ):
        raise ValueError("checkpoint motion units are not exact-time v2")
    mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    scale = np.asarray(payload["feature_scale"], dtype=np.float64)
    if (
        mean.shape != (MOTION_FEATURE_DIM,)
        or scale.shape != (MOTION_FEATURE_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("checkpoint motion normalization is invalid")
    digest = payload["statistics_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("checkpoint motion statistics_sha256 is invalid")
    flow_contract = flow_compute_contract(
        {
            "count": history_count,
            **dict(payload["flow_contract"]),
        }
    )
    normalized = dict(payload)
    normalized["history_count"] = history_count
    normalized["feature_names"] = list(MOTION_FEATURE_NAMES)
    normalized["feature_mean"] = mean.tolist()
    normalized["feature_scale"] = scale.tolist()
    normalized["flow_contract"] = flow_contract
    return normalized


def raw_stride(*, policy_stride: int, global_downsample_rate: int) -> int:
    policy_stride = int(policy_stride)
    global_downsample_rate = int(global_downsample_rate)
    if policy_stride <= 0 or global_downsample_rate <= 0:
        raise ValueError(
            "policy_stride and global_downsample_rate must be positive, got "
            f"{policy_stride}, {global_downsample_rate}"
        )
    return policy_stride * global_downsample_rate


def raw_offsets(
    *,
    history_count: int,
    policy_stride: int,
    global_downsample_rate: int,
) -> list[int]:
    history_count = int(history_count)
    if history_count <= 0:
        raise ValueError(f"history_count must be positive, got {history_count}")
    stride = raw_stride(
        policy_stride=policy_stride,
        global_downsample_rate=global_downsample_rate,
    )
    return [index * stride for index in range(history_count, -1, -1)]


def raw_pairs(
    condition_index: int,
    *,
    history_count: int,
    policy_stride: int,
    global_downsample_rate: int,
) -> list[tuple[int, int]]:
    offsets = raw_offsets(
        history_count=history_count,
        policy_stride=policy_stride,
        global_downsample_rate=global_downsample_rate,
    )
    endpoints = [max(0, int(condition_index) - offset) for offset in offsets]
    return list(pairwise(endpoints))


def build_flow_cache_parameters(
    *,
    head_flow_config: dict[str, Any],
    global_downsample_rate: int,
) -> dict[str, Any]:
    """Resolve the one exact cache identity shared by conversion and packing."""

    contract = flow_compute_contract(head_flow_config)
    history_count = int(head_flow_config["count"])
    policy_stride = int(contract["policy_stride"])
    downsample = int(global_downsample_rate)
    if downsample <= 0:
        raise ValueError("global_downsample_rate must be positive")
    container_fps = float(head_flow_config["container_fps"])
    if not np.isfinite(container_fps) or container_fps <= 0.0:
        raise ValueError("head-flow container_fps must be positive")
    return {
        "version": FLOW_CACHE_VERSION,
        "temporal_contract": TEMPORAL_CONTRACT,
        "history_count": history_count,
        "policy_stride": policy_stride,
        "global_downsample_rate": downsample,
        "raw_stride": raw_stride(
            policy_stride=policy_stride,
            global_downsample_rate=downsample,
        ),
        "endpoint_rule": ENDPOINT_RULE,
        "raw_offsets": raw_offsets(
            history_count=history_count,
            policy_stride=policy_stride,
            global_downsample_rate=downsample,
        ),
        "raw_index_unit": "converted_video_frame",
        "size": list(contract["compute_size"]),
        "container_fps": container_fps,
        "physical_timestamps_available": True,
        "timestamp_source": TIMESTAMP_SOURCE,
        "spatial_unit": SPATIAL_UNIT,
        "motion_feature_names": list(MOTION_FEATURE_NAMES),
        "source_view": contract["source_view"],
        "rgb_normalization": {
            "type": "per_map_percentile",
            "percentile": contract["normalization_percentile"],
        },
        "farneback": dict(contract["farneback"]),
        "quality": dict(contract["quality"]),
    }


def load_exact_flow_cache(
    path: str | Path,
    *,
    expected_params_json: str,
) -> dict[str, np.ndarray]:
    """Load and fully validate one exact-time cache without repair."""

    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != FLOW_CACHE_ARRAY_KEYS:
            raise ValueError(f"cache arrays differ from exact-time v2: {cache_path}")
        if str(payload["params"].item()) != expected_params_json:
            raise ValueError(f"cache parameters differ from this run: {cache_path}")
        arrays = {
            "flow_rgb": np.asarray(payload["flow_rgb"]),
            "motion_features": np.asarray(payload["motion_features"]),
            "temporal_valid": np.asarray(payload["temporal_valid"]),
            "flow_quality_valid": np.asarray(payload["flow_quality_valid"]),
            "flow_reliable_fraction": np.asarray(payload["flow_reliable_fraction"]),
            "interval_valid": np.asarray(payload["interval_valid"]),
            "acceleration_valid": np.asarray(payload["acceleration_valid"]),
            "interval_start_time_seconds": np.asarray(
                payload["interval_start_time_seconds"]
            ),
            "interval_end_time_seconds": np.asarray(
                payload["interval_end_time_seconds"]
            ),
        }
    flow_rgb = arrays["flow_rgb"]
    features = arrays["motion_features"]
    temporal_valid = arrays["temporal_valid"]
    flow_quality_valid = arrays["flow_quality_valid"]
    reliable_fraction = arrays["flow_reliable_fraction"]
    interval_valid = arrays["interval_valid"]
    acceleration_valid = arrays["acceleration_valid"]
    starts = arrays["interval_start_time_seconds"]
    ends = arrays["interval_end_time_seconds"]
    frame_count = int(flow_rgb.shape[0]) if flow_rgb.ndim == 4 else 0
    parameters = json.loads(expected_params_json)
    expected_size = tuple(int(value) for value in parameters["size"])
    raw_interval_stride = int(parameters["raw_stride"])
    if (
        frame_count <= 0
        or flow_rgb.dtype != np.uint8
        or flow_rgb.ndim != 4
        or tuple(flow_rgb.shape[1:3]) != expected_size
        or flow_rgb.shape[-1] != 3
        or features.dtype != np.float32
        or features.shape != (frame_count, MOTION_FEATURE_DIM)
        or temporal_valid.dtype != np.bool_
        or temporal_valid.shape != (frame_count,)
        or flow_quality_valid.dtype != np.bool_
        or flow_quality_valid.shape != (frame_count,)
        or reliable_fraction.dtype != np.float32
        or reliable_fraction.shape != (frame_count,)
        or interval_valid.dtype != np.bool_
        or interval_valid.shape != (frame_count,)
        or acceleration_valid.dtype != np.bool_
        or acceleration_valid.shape != (frame_count,)
        or starts.dtype != np.float64
        or starts.shape != (frame_count,)
        or ends.dtype != np.float64
        or ends.shape != (frame_count,)
        or not np.isfinite(features).all()
        or not np.isfinite(reliable_fraction).all()
        or not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
    ):
        raise ValueError(f"cache arrays violate exact-time v2: {cache_path}")

    indices = np.arange(frame_count, dtype=np.int64)
    previous_indices = np.maximum(0, indices - raw_interval_stride)
    expected_starts = ends[previous_indices]
    expected_temporal = (indices >= raw_interval_stride) & (ends > starts)
    minimum_reliable_fraction = float(
        parameters["quality"]["minimum_reliable_fraction"]
    )
    expected_quality = expected_temporal & (
        reliable_fraction >= minimum_reliable_fraction
    )
    centers = (starts + ends) * 0.5
    previous = indices - raw_interval_stride
    expected_acceleration = np.zeros(frame_count, dtype=np.bool_)
    eligible = np.flatnonzero(previous >= 0)
    if eligible.size:
        expected_acceleration[eligible] = (
            interval_valid[eligible]
            & interval_valid[previous[eligible]]
            & (centers[eligible] > centers[previous[eligible]])
        )
    if (
        not np.array_equal(starts, expected_starts)
        or np.any(reliable_fraction < 0.0)
        or np.any(reliable_fraction > 1.0)
        or np.any(reliable_fraction[~expected_temporal] != 0.0)
        or np.any(temporal_valid != expected_temporal)
        or np.any(flow_quality_valid != expected_quality)
        or np.any(interval_valid != (temporal_valid & flow_quality_valid))
        or np.any(acceleration_valid != expected_acceleration)
        or np.any(acceleration_valid & ~interval_valid)
        or np.any(flow_rgb[~interval_valid] != 0)
        or np.any(features[~interval_valid] != 0.0)
        or np.any(features[~acceleration_valid, 9:12] != 0.0)
    ):
        raise ValueError(f"cache arrays violate exact-time v2: {cache_path}")
    return arrays


def write_exact_flow_cache(
    path: str | Path,
    *,
    arrays: dict[str, np.ndarray],
    params_json: str,
) -> None:
    """Atomically write and revalidate one exact-time cache."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez(temporary, **arrays, params=params_json)
        load_exact_flow_cache(
            temporary,
            expected_params_json=params_json,
        )
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def _opencv():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise ImportError("opencv-python is required for absolute motion") from exc
    return cv2


def compute_dense_flow(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    compute_size: tuple[int, int],
    farneback: dict[str, Any],
) -> np.ndarray:
    """Return signed Farneback displacement at the configured image size."""

    cv2 = _opencv()
    height, width = (int(value) for value in compute_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"compute_size must be positive, got {compute_size}")
    for name, frame in (
        ("previous_rgb", previous_rgb),
        ("current_rgb", current_rgb),
    ):
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[-1] != 3
        ):
            raise ValueError(
                f"{name} must be HWC uint8 RGB, got "
                f"{getattr(frame, 'dtype', None)} {getattr(frame, 'shape', None)}"
            )
    previous = cv2.resize(
        previous_rgb,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    current = cv2.resize(
        current_rgb,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    config = {**FARNEBACK_DEFAULTS, **dict(farneback)}
    flow = cv2.calcOpticalFlowFarneback(
        prev=cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY),
        next=cv2.cvtColor(current, cv2.COLOR_RGB2GRAY),
        flow=None,
        pyr_scale=float(config["pyr_scale"]),
        levels=int(config["levels"]),
        winsize=int(config["winsize"]),
        iterations=int(config["iterations"]),
        poly_n=int(config["poly_n"]),
        poly_sigma=float(config["poly_sigma"]),
        flags=int(config["flags"]),
    )
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
        raise RuntimeError(f"invalid Farneback output: {flow.shape}")
    return flow


def filter_flow_by_forward_backward_consistency(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    quality: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Remove flow vectors without an in-frame, cycle-consistent match."""

    cv2 = _opencv()
    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if (
        forward.ndim != 3
        or forward.shape[-1] != 2
        or backward.shape != forward.shape
        or not np.isfinite(forward).all()
        or not np.isfinite(backward).all()
    ):
        raise ValueError(
            "forward and backward flow must be finite matching [H,W,2] arrays"
        )
    quality = validate_flow_quality_config(quality)
    height, width = forward.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    mapped_x = grid_x + forward[..., 0]
    mapped_y = grid_y + forward[..., 1]
    in_bounds = (
        (mapped_x >= 0.0)
        & (mapped_x <= float(width - 1))
        & (mapped_y >= 0.0)
        & (mapped_y <= float(height - 1))
    )
    backward_x = cv2.remap(
        backward[..., 0],
        mapped_x,
        mapped_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    backward_y = cv2.remap(
        backward[..., 1],
        mapped_x,
        mapped_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    cycle_error_squared = (forward[..., 0] + backward_x) ** 2 + (
        forward[..., 1] + backward_y
    ) ** 2
    reference_squared = (
        quality["relative_error"]
        * (
            np.square(forward).sum(axis=-1)
            + np.square(backward_x)
            + np.square(backward_y)
        )
        + quality["absolute_error_squared"]
    )
    reliable = in_bounds & (cycle_error_squared <= reference_squared)
    reliable_fraction = float(reliable.mean())
    quality_valid = reliable_fraction >= quality["minimum_reliable_fraction"]
    filtered = np.where(reliable[..., None], forward, 0.0).astype(
        np.float32,
        copy=False,
    )
    if not quality_valid:
        filtered.fill(0.0)
    return filtered, reliable, reliable_fraction, quality_valid


def flow_to_rgb(
    flow_xy: np.ndarray,
    *,
    normalization_percentile: float,
) -> np.ndarray:
    """Encode direction and within-map relative magnitude as RGB."""

    cv2 = _opencv()
    percentile = float(normalization_percentile)
    if not 0.0 < percentile <= 100.0:
        raise ValueError(
            f"normalization_percentile must be in (0, 100], got {percentile}"
        )
    magnitude, angle = cv2.cartToPolar(
        flow_xy[..., 0],
        flow_xy[..., 1],
        angleInDegrees=True,
    )
    hsv = np.zeros((*flow_xy.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 0.5, 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    scale = float(np.percentile(magnitude, percentile))
    if scale >= 1e-6:
        hsv[..., 2] = np.clip(magnitude / scale * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def displacement_statistics(
    flow_xy: np.ndarray,
    *,
    magnitude_percentile: float,
) -> np.ndarray:
    """Return signed mean xy, mean magnitude, and the HSV scale magnitude."""

    flow_xy = np.asarray(flow_xy, dtype=np.float32)
    if flow_xy.ndim != 3 or flow_xy.shape[-1] != 2:
        raise ValueError(f"flow_xy must be [H,W,2], got {flow_xy.shape}")
    if not np.isfinite(flow_xy).all():
        raise ValueError("flow_xy contains non-finite values")
    percentile = float(magnitude_percentile)
    if percentile != 99.0:
        raise ValueError("absolute-motion v2 requires the exact p99 HSV scale")
    magnitude = np.linalg.norm(flow_xy, axis=-1)
    return np.asarray(
        (
            float(flow_xy[..., 0].mean()),
            float(flow_xy[..., 1].mean()),
            float(magnitude.mean()),
            float(np.percentile(magnitude, percentile)),
        ),
        dtype=np.float32,
    )


def compute_flow_observation(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    *,
    compute_size: tuple[int, int],
    normalization_percentile: float,
    farneback: dict[str, Any],
    quality: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    forward_flow = compute_dense_flow(
        previous_rgb,
        current_rgb,
        compute_size=compute_size,
        farneback=farneback,
    )
    backward_flow = compute_dense_flow(
        current_rgb,
        previous_rgb,
        compute_size=compute_size,
        farneback=farneback,
    )
    flow_xy, _reliable, reliable_fraction, quality_valid = (
        filter_flow_by_forward_backward_consistency(
            forward_flow,
            backward_flow,
            quality=quality,
        )
    )
    return (
        flow_to_rgb(
            flow_xy,
            normalization_percentile=normalization_percentile,
        ),
        displacement_statistics(
            flow_xy,
            magnitude_percentile=normalization_percentile,
        ),
        reliable_fraction,
        quality_valid,
    )


def build_motion_features(
    displacement_stats: np.ndarray,
    interval_start_times: np.ndarray,
    interval_end_times: np.ndarray,
    interval_valid_mask: np.ndarray,
    *,
    previous_interval_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build displacement, exact-time velocity, and center-time acceleration."""

    stats = np.asarray(displacement_stats, dtype=np.float64)
    starts = np.asarray(interval_start_times, dtype=np.float64)
    ends = np.asarray(interval_end_times, dtype=np.float64)
    interval_valid = np.asarray(interval_valid_mask, dtype=np.bool_)
    if stats.ndim != 2 or stats.shape[1] != 4:
        raise ValueError(f"displacement_stats must be [N,4], got {stats.shape}")
    count = int(stats.shape[0])
    if (
        starts.shape != (count,)
        or ends.shape != (count,)
        or interval_valid.shape != (count,)
    ):
        raise ValueError("all motion interval arrays must have the same length")
    if (
        not np.isfinite(stats).all()
        or not np.isfinite(starts).all()
        or not np.isfinite(ends).all()
    ):
        raise ValueError("motion interval arrays contain non-finite values")

    dt = ends - starts
    interval_valid = interval_valid & (dt > 0.0)
    features = np.zeros((count, MOTION_FEATURE_DIM), dtype=np.float32)
    acceleration_valid = np.zeros(count, dtype=np.bool_)
    velocity = np.zeros((count, 4), dtype=np.float64)
    velocity[interval_valid] = stats[interval_valid] / dt[interval_valid, None]
    features[interval_valid, :4] = stats[interval_valid].astype(np.float32)
    features[interval_valid, 4] = dt[interval_valid].astype(np.float32)
    features[interval_valid, 5:9] = velocity[interval_valid].astype(np.float32)

    if previous_interval_indices is None:
        previous = np.arange(count, dtype=np.int64) - 1
    else:
        previous = np.asarray(previous_interval_indices, dtype=np.int64)
        if previous.shape != (count,):
            raise ValueError("previous_interval_indices must match the interval count")
    centers = (starts + ends) * 0.5
    for index, previous_index in enumerate(previous.tolist()):
        if previous_index < 0 or previous_index >= count:
            continue
        center_dt = centers[index] - centers[previous_index]
        if (
            not interval_valid[index]
            or not interval_valid[previous_index]
            or center_dt <= 0.0
        ):
            continue
        acceleration_xy = (
            velocity[index, :2] - velocity[previous_index, :2]
        ) / center_dt
        features[index, 9:11] = acceleration_xy.astype(np.float32)
        features[index, 11] = np.float32(np.linalg.norm(acceleration_xy))
        acceleration_valid[index] = True
    return features, interval_valid, acceleration_valid


@dataclass(frozen=True)
class MotionObservationBatch:
    flow_rgb: np.ndarray
    motion_features: np.ndarray
    interval_valid_mask: np.ndarray
    acceleration_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        flow_rgb = np.asarray(self.flow_rgb)
        features = np.asarray(self.motion_features)
        interval_valid = np.asarray(self.interval_valid_mask)
        acceleration_valid = np.asarray(self.acceleration_valid_mask)
        if flow_rgb.dtype != np.uint8 or flow_rgb.ndim != 4 or flow_rgb.shape[-1] != 3:
            raise ValueError(
                f"flow_rgb must be [K,H,W,3] uint8, got "
                f"{flow_rgb.dtype} {flow_rgb.shape}"
            )
        history_count = int(flow_rgb.shape[0])
        if features.shape != (history_count, MOTION_FEATURE_DIM):
            raise ValueError(
                f"motion_features shape differs from flow history: {features.shape}"
            )
        if (
            interval_valid.shape != (history_count,)
            or acceleration_valid.shape != (history_count,)
            or interval_valid.dtype != np.bool_
            or acceleration_valid.dtype != np.bool_
        ):
            raise ValueError("motion validity masks differ from flow history")
        if not np.isfinite(features).all():
            raise ValueError("motion_features contains non-finite values")
        if np.any(acceleration_valid & ~interval_valid):
            raise ValueError("acceleration cannot be valid for an invalid interval")


def validate_motion_statistics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("motion statistics must be a mapping")
    expected = {
        "schema_version",
        "feature_names",
        "count",
        "mean",
        "standard_deviation",
        "scale",
        "minimum_scale",
        "temporal_contract",
    }
    if set(payload) != expected:
        raise ValueError(
            "motion statistics keys differ from the contract: "
            f"missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    if payload["schema_version"] != MOTION_STATISTICS_VERSION:
        raise ValueError(
            f"motion statistics schema_version must be {MOTION_STATISTICS_VERSION}"
        )
    if tuple(payload["feature_names"]) != MOTION_FEATURE_NAMES:
        raise ValueError("motion statistics feature order differs from the model")
    if payload["temporal_contract"] != TEMPORAL_CONTRACT:
        raise ValueError("motion statistics temporal contract is not exact-time v2")
    arrays = {}
    for key in ("count", "mean", "standard_deviation", "scale"):
        array = np.asarray(payload[key])
        if array.shape != (MOTION_FEATURE_DIM,):
            raise ValueError(f"motion statistics {key} must have 12 values")
        arrays[key] = array
    if np.any(arrays["count"].astype(np.int64) <= 0):
        raise ValueError("every motion feature requires at least one valid sample")
    for key in ("mean", "standard_deviation", "scale"):
        array = arrays[key].astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"motion statistics {key} contains non-finite values")
    minimum_scale = float(payload["minimum_scale"])
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("motion statistics minimum_scale must be positive")
    if np.any(arrays["scale"].astype(np.float64) < minimum_scale):
        raise ValueError("motion feature scales fall below minimum_scale")
    return dict(payload)
