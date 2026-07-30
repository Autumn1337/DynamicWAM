"""Strict state composition for absolute-motion Stage-2 initialization."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from dynamicwam.training.data.training_dataset import (
    validate_dataset_identity,
)

COMPACT_WAN_TRAINING_CONTRACT_KEYS = (
    "precision",
    "dim",
    "ffn_dim",
    "num_heads",
    "num_layers",
    "head_dim",
    "future_video_size",
    "hidden_anchor_layers",
    "motion_anchor_layers",
    "teacher_layer_mapping",
)
COMPACT_WAN_MACHINE_PATH_KEYS = {
    "checkpoint_path",
    "config_path",
}


def compact_wan_training_contract(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Return the architecture contract without machine-local asset paths."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    keys = set(value)
    missing = set(COMPACT_WAN_TRAINING_CONTRACT_KEYS) - keys
    unknown = (
        keys - set(COMPACT_WAN_TRAINING_CONTRACT_KEYS) - COMPACT_WAN_MACHINE_PATH_KEYS
    )
    if missing or unknown:
        raise ValueError(
            f"{label} keys differ from the compact WAN contract: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    normalized = {
        key: deepcopy(value[key]) for key in COMPACT_WAN_TRAINING_CONTRACT_KEYS
    }
    normalized["future_video_size"] = [
        int(item) for item in normalized["future_video_size"]
    ]
    for key in (
        "hidden_anchor_layers",
        "motion_anchor_layers",
        "teacher_layer_mapping",
    ):
        normalized[key] = [int(item) for item in normalized[key]]
    for key in (
        "dim",
        "ffn_dim",
        "num_heads",
        "num_layers",
        "head_dim",
    ):
        normalized[key] = int(normalized[key])
    normalized["precision"] = str(normalized["precision"])
    return normalized


def stage1_training_contract(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Canonicalize a Stage-1 export for relocation-safe comparison."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = deepcopy(dict(value))
    for key in ("checkpoint_dir", "resume_from"):
        normalized.pop(key, None)

    dataset = normalized.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError(f"{label}.dataset must be a mapping")
    dataset.pop("root", None)

    teacher = normalized.get("teacher")
    if not isinstance(teacher, dict):
        raise TypeError(f"{label}.teacher must be a mapping")
    for key in ("checkpoint_path", "config_path", "pca_stats_path"):
        teacher.pop(key, None)

    normalized["student"] = compact_wan_training_contract(
        normalized.get("student"),
        label=f"{label}.student",
    )
    normalized["dataset_identity"] = validate_dataset_identity(
        normalized.get("dataset_identity")
    )
    return normalized


def merge_stage1_compact_with_base_action(
    *,
    base_state: dict[str, Any],
    stage1_payload: Any,
    expected_stage1_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Combine the Stage-1 video expert with the history-flow action expert."""

    if (
        not isinstance(stage1_payload, dict)
        or set(stage1_payload)
        != {"model", "global_step", "config", "compact_wan_config"}
        or int(stage1_payload.get("global_step", -1))
        != int(expected_stage1_config["max_steps"])
    ):
        raise RuntimeError(
            "Stage 1 checkpoint does not match the completed v2 training contract"
        )
    exported_config = stage1_payload["config"]
    if (
        not isinstance(exported_config, dict)
        or set(exported_config) != set(expected_stage1_config)
        or stage1_training_contract(
            exported_config,
            label="Stage 1 checkpoint config",
        )
        != stage1_training_contract(
            expected_stage1_config,
            label="Stage 1 launch config",
        )
    ):
        raise RuntimeError(
            "Stage 1 checkpoint does not match the completed v2 training contract"
        )

    exported_metadata = stage1_payload["compact_wan_config"]
    if compact_wan_training_contract(
        exported_metadata,
        label="Stage 1 compact WAN metadata",
    ) != compact_wan_training_contract(
        expected_stage1_config["student"],
        label="Stage 1 launch compact WAN",
    ):
        raise RuntimeError(
            "Stage 1 compact WAN metadata differs from the launch contract"
        )

    stage1_state = stage1_payload["model"]
    if not isinstance(stage1_state, dict) or not stage1_state:
        raise TypeError("Stage 1 checkpoint model state must be a non-empty mapping")
    base_compact_state = {
        key.removeprefix("compact_wan."): value
        for key, value in base_state.items()
        if key.startswith("compact_wan.")
    }
    base_action_state = {
        key: value
        for key, value in base_state.items()
        if key.startswith("action_expert.")
    }
    if not base_compact_state or not base_action_state:
        raise RuntimeError(
            "history-flow base state must contain compact WAN and action expert tensors"
        )
    if set(stage1_state) != set(base_compact_state):
        raise RuntimeError(
            "Stage 1 compact WAN tensor keys differ from the history-flow architecture"
        )
    shape_mismatches = [
        key
        for key, value in stage1_state.items()
        if not isinstance(value, torch.Tensor)
        or not isinstance(base_compact_state[key], torch.Tensor)
        or tuple(value.shape) != tuple(base_compact_state[key].shape)
    ]
    non_tensor_actions = [
        key
        for key, value in base_action_state.items()
        if not isinstance(value, torch.Tensor)
    ]
    if shape_mismatches or non_tensor_actions:
        raise RuntimeError(
            "Stage 1/history-flow tensor contract mismatch: "
            f"compact={shape_mismatches[:10]}, "
            f"action={non_tensor_actions[:10]}"
        )

    merged = {key: value.detach().cpu() for key, value in base_action_state.items()}
    merged.update(
        {
            f"compact_wan.{key}": value.detach().cpu()
            for key, value in stage1_state.items()
        }
    )
    return merged
