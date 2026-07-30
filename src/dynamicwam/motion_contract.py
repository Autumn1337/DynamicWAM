"""Dependency-light validation for the exact-time motion contract."""

from __future__ import annotations

import math
from typing import Any

TEMPORAL_CONTRACT = "exact_simulator_time_absolute_motion_v2"
ENDPOINT_RULE = "full_fixed_model_policy_stride_v2"
TIMESTAMP_SOURCE = "domino_schema_v2.sim_time_seconds"
SPATIAL_UNIT = "configured_flow_compute_grid_pixels"

FARNEBACK_DEFAULTS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}
FLOW_QUALITY_METHOD = "forward_backward_consistency_v1"
FLOW_QUALITY_KEYS = frozenset(
    {
        "method",
        "relative_error",
        "absolute_error_squared",
        "minimum_reliable_fraction",
    }
)


def validate_flow_quality_config(quality: Any) -> dict[str, Any]:
    if not isinstance(quality, dict) or set(quality) != FLOW_QUALITY_KEYS:
        raise ValueError(f"invalid flow quality contract: {quality!r}")
    normalized = {
        "method": str(quality["method"]),
        "relative_error": float(quality["relative_error"]),
        "absolute_error_squared": float(quality["absolute_error_squared"]),
        "minimum_reliable_fraction": float(quality["minimum_reliable_fraction"]),
    }
    if (
        normalized["method"] != FLOW_QUALITY_METHOD
        or not math.isfinite(normalized["relative_error"])
        or normalized["relative_error"] < 0.0
        or not math.isfinite(normalized["absolute_error_squared"])
        or normalized["absolute_error_squared"] <= 0.0
        or not math.isfinite(normalized["minimum_reliable_fraction"])
        or not 0.0 < normalized["minimum_reliable_fraction"] <= 1.0
    ):
        raise ValueError(f"invalid flow quality contract: {quality!r}")
    return normalized


def flow_compute_contract(config: Any) -> dict[str, Any]:
    """Return the exact image-plane computation contract used by the model."""

    if not isinstance(config, dict):
        raise TypeError("head-flow configuration must be a mapping")
    required = {
        "count",
        "policy_stride",
        "compute_size",
        "normalization_percentile",
        "farneback",
        "quality",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"head-flow configuration is missing {sorted(missing)}")
    raw_history_count = config["count"]
    raw_policy_stride = config["policy_stride"]
    history_count = int(raw_history_count)
    policy_stride = int(raw_policy_stride)
    compute_size = config["compute_size"]
    percentile = float(config["normalization_percentile"])
    farneback = config["farneback"]
    quality = config["quality"]
    if (
        ("source_view" in config and config["source_view"] != "head")
        or isinstance(raw_history_count, bool)
        or not isinstance(raw_history_count, int)
        or isinstance(raw_policy_stride, bool)
        or not isinstance(raw_policy_stride, int)
        or history_count <= 0
        or policy_stride <= 0
        or not isinstance(compute_size, (list, tuple))
        or len(compute_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in compute_size
        )
        or any(int(value) <= 0 for value in compute_size)
        or not math.isfinite(percentile)
        or percentile != 99.0
        or not isinstance(farneback, dict)
        or set(farneback) != set(FARNEBACK_DEFAULTS)
        or not isinstance(quality, dict)
        or set(quality) != FLOW_QUALITY_KEYS
    ):
        raise ValueError(f"invalid head-flow computation contract: {config!r}")
    normalized_farneback = {
        "pyr_scale": float(farneback["pyr_scale"]),
        "levels": int(farneback["levels"]),
        "winsize": int(farneback["winsize"]),
        "iterations": int(farneback["iterations"]),
        "poly_n": int(farneback["poly_n"]),
        "poly_sigma": float(farneback["poly_sigma"]),
        "flags": int(farneback["flags"]),
    }
    if (
        any(
            isinstance(farneback[key], bool) or not isinstance(farneback[key], int)
            for key in ("levels", "winsize", "iterations", "poly_n", "flags")
        )
        or not math.isfinite(normalized_farneback["pyr_scale"])
        or not math.isfinite(normalized_farneback["poly_sigma"])
        or normalized_farneback["pyr_scale"] <= 0.0
        or normalized_farneback["pyr_scale"] > 1.0
        or normalized_farneback["poly_sigma"] <= 0.0
        or any(
            normalized_farneback[key] <= 0
            for key in ("levels", "winsize", "iterations", "poly_n")
        )
        or normalized_farneback["winsize"] % 2 == 0
        or normalized_farneback["poly_n"] not in {5, 7}
        or normalized_farneback["flags"] < 0
    ):
        raise ValueError(f"invalid Farneback contract: {farneback!r}")
    normalized_quality = validate_flow_quality_config(quality)
    return {
        "source_view": "head",
        "compute_size": [int(value) for value in compute_size],
        "policy_stride": policy_stride,
        "normalization_percentile": percentile,
        "farneback": normalized_farneback,
        "quality": normalized_quality,
    }
