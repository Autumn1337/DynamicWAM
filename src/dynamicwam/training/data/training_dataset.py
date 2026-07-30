"""Dataset construction shared by the DynamicWAM training stages."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping

from torch.utils.data import Sampler

from dynamicwam.training.data.packed_dataset import (
    PackedAbsoluteMotionDataset,
)


def build_packed_training_dataset(
    config: Mapping[str, Any],
) -> PackedAbsoluteMotionDataset:
    return PackedAbsoluteMotionDataset(
        root=str(config["root"]),
        max_open_shards=int(config["max_open_shards"]),
    )


def make_packed_training_sampler(
    dataset: PackedAbsoluteMotionDataset,
    config: Mapping[str, Any],
) -> Sampler[int]:
    return dataset.make_sampler(
        samples_per_episode=int(config["samples_per_episode"]),
        seed=int(config["sampler_seed"]),
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def validate_dataset_identity(identity: Any) -> Dict[str, Any]:
    if not isinstance(identity, dict):
        raise TypeError("training dataset identity must be a mapping")
    required = {
        "format",
        "version",
        "dataset_fingerprint",
        "action_stats_sha256",
        "motion_statistics_sha256",
    }
    # Early checkpoints recorded the local dataset root. It is not part of
    # artifact identity, so accept it only for backward compatibility and
    # remove it from the canonical comparison.
    unknown = set(identity) - required - {"root"}
    missing = required - set(identity)
    if missing or unknown:
        raise ValueError(
            "training dataset identity keys differ from v2: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if identity["format"] != "dynamicwam_absolute_motion_dataset":
        raise ValueError("unsupported training dataset identity format")
    if isinstance(identity["version"], bool) or identity["version"] != 2:
        raise ValueError("unsupported training dataset identity version")
    dataset_fingerprint = _require_sha256(
        identity.get("dataset_fingerprint"),
        "dataset fingerprint",
    )
    action_stats_sha256 = _require_sha256(
        identity.get("action_stats_sha256"),
        "action statistics",
    )
    motion_statistics_sha256 = _require_sha256(
        identity.get("motion_statistics_sha256"),
        "motion statistics",
    )
    return {
        "format": "dynamicwam_absolute_motion_dataset",
        "version": 2,
        "dataset_fingerprint": dataset_fingerprint,
        "action_stats_sha256": action_stats_sha256,
        "motion_statistics_sha256": motion_statistics_sha256,
    }
