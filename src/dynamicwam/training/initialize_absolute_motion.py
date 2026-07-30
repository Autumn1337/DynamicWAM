"""Create the single strict full-model absolute-motion initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from dynamicwam.absolute_motion import (
    ABSOLUTE_MOTION_CHECKPOINT_VERSION,
    build_checkpoint_motion_metadata,
    validate_checkpoint_motion_metadata,
    validate_motion_statistics,
)
from dynamicwam.config import load_profile
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
from dynamicwam.training.checkpoint_merge import (
    merge_stage1_compact_with_base_action,
)
from dynamicwam.training.data.packed_dataset import (
    TRAIN_DATASET_ACTION_STATS,
    TRAIN_DATASET_FORMAT,
    TRAIN_DATASET_METADATA,
    TRAIN_DATASET_MOTION_STATS,
    TRAIN_DATASET_VERSION,
)
from dynamicwam.training.data.training_dataset import (
    build_packed_training_dataset,
    validate_dataset_identity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return payload


def _qpos_stats(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    stats = payload.get("robotwin_qpos")
    if set(payload) != {"robotwin_qpos"} or not isinstance(stats, dict):
        raise ValueError(
            f"history-flow action stats must contain only robotwin_qpos: {path}"
        )
    return stats


def _validate_base_model_config(
    payload: dict[str, Any],
    *,
    expected_compact: dict[str, Any],
    expected_action: dict[str, Any],
) -> None:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("history-flow base checkpoint has no config mapping")
    model = require_exact_keys(
        config.get("model"),
        {
            "compact_wan_checkpoint",
            "action_init_checkpoint",
            "compact_wan",
            "action_expert",
            "wan_frozen",
        },
        "history-flow base checkpoint model",
    )
    compact = require_exact_keys(
        model["compact_wan"],
        {
            "checkpoint_path",
            "config_path",
            "vae_path",
            "precision",
            "dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "head_dim",
            "future_video_size",
        },
        "history-flow base compact_wan",
    )
    comparable_keys = {
        "precision",
        "dim",
        "ffn_dim",
        "num_heads",
        "num_layers",
        "head_dim",
        "future_video_size",
    }
    if (
        {key: compact[key] for key in comparable_keys}
        != {key: expected_compact[key] for key in comparable_keys}
        or model["action_expert"] != expected_action
        or model["wan_frozen"] is not False
    ):
        raise RuntimeError(
            "pinned base checkpoint architecture differs from absolute_motion_v2"
        )


def _validate_base_action_normalization(
    payload: dict[str, Any],
    *,
    pinned_stats: dict[str, Any],
) -> None:
    config = payload["config"]
    normalization = require_exact_keys(
        config.get("action_normalization"),
        {"enabled", "type", "stats_file", "stats"},
        "history-flow base action_normalization",
    )
    if (
        normalization["enabled"] is not True
        or normalization["type"] != "mean_std"
        or normalization["stats_file"] != TRAIN_DATASET_ACTION_STATS
        or normalization["stats"] != pinned_stats
    ):
        raise RuntimeError(
            "pinned base checkpoint action normalization differs from "
            "the pinned history-flow statistics"
        )


def run(*, config_path: str) -> Path:
    profile = load_profile(config_path)
    raw = profile.raw
    paths = raw["paths"]
    initialization = raw["initialization"]
    base_path = Path(paths["base_checkpoint"])
    stage1_path = Path(paths["stage1_checkpoint"])
    base_stats_path = Path(paths["base_action_stats"])
    packed_root = Path(paths["packed_dataset"])
    packed_stats_path = packed_root / TRAIN_DATASET_ACTION_STATS
    motion_stats_path = packed_root / TRAIN_DATASET_MOTION_STATS
    metadata_path = packed_root / TRAIN_DATASET_METADATA
    output_path = Path(paths["motion_init_checkpoint"])

    required_files = (
        base_path,
        stage1_path,
        base_stats_path,
        packed_stats_path,
        motion_stats_path,
        metadata_path,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"initialization input is missing: {path}")
    if output_path.exists():
        raise FileExistsError(
            f"initialization checkpoint already exists: {output_path}"
        )

    expected_base_sha256 = initialization["base_checkpoint_sha256"]
    actual_base_sha256 = _sha256(base_path)
    if actual_base_sha256 != expected_base_sha256:
        raise RuntimeError(
            "base checkpoint SHA256 mismatch: "
            f"expected {expected_base_sha256}, got {actual_base_sha256}"
        )
    expected_action_sha256 = initialization["base_action_stats_sha256"]
    actual_action_sha256 = _sha256(base_stats_path)
    if actual_action_sha256 != expected_action_sha256:
        raise RuntimeError(
            "base action-statistics SHA256 mismatch: "
            f"expected {expected_action_sha256}, got {actual_action_sha256}"
        )
    for path in (packed_stats_path,):
        actual = _sha256(path)
        if actual != expected_action_sha256:
            raise RuntimeError(
                f"pinned action stats SHA256 mismatch at {path}: "
                f"expected {expected_action_sha256}, got {actual}"
            )
    base_stats_payload = _read_json(base_stats_path)
    packed_stats_payload = _read_json(packed_stats_path)
    if base_stats_payload != packed_stats_payload:
        raise RuntimeError(
            "packed action statistics differ from the history-flow source"
        )
    pinned_qpos_stats = _qpos_stats(
        base_stats_payload,
        path=base_stats_path,
    )

    metadata = _read_json(metadata_path)
    if (
        metadata.get("format") != TRAIN_DATASET_FORMAT
        or metadata.get("version") != TRAIN_DATASET_VERSION
    ):
        raise RuntimeError(f"packed dataset is not absolute-motion v2: {metadata_path}")
    motion_contract = dict(metadata.get("absolute_motion") or {})
    if motion_contract.pop("statistics_file", None) != TRAIN_DATASET_MOTION_STATS:
        raise RuntimeError(
            "packed dataset does not identify its motion statistics file"
        )
    motion_contract = validate_checkpoint_motion_metadata(motion_contract)
    motion_statistics = validate_motion_statistics(_read_json(motion_stats_path))
    expected_motion = build_checkpoint_motion_metadata(
        history_count=int(raw["method"]["head_flow"]["count"]),
        statistics=motion_statistics,
        statistics_sha256=_canonical_json_sha256(motion_statistics),
        head_flow_config=raw["method"]["head_flow"],
    )
    if motion_contract != expected_motion:
        raise RuntimeError(
            "packed motion contract differs from the profile or statistics"
        )

    payload = torch.load(
        base_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "config",
        "global_step",
        "model",
    }:
        raise TypeError("base checkpoint must be the exact history-flow Stage-3 export")
    state = payload["model"]
    if not isinstance(state, dict):
        raise TypeError("base checkpoint model state must be a mapping")
    prefixes = {
        key.split(".", 1)[0] for key in state if isinstance(key, str) and "." in key
    }
    if (
        prefixes != {"compact_wan", "action_expert"}
        or not state
        or any(not isinstance(key, str) for key in state)
    ):
        raise RuntimeError(
            "base checkpoint must contain only compact_wan.* and "
            "action_expert.* tensors"
        )
    expected_stage1_config = profile.training_config("stage1")
    identity_dataset = build_packed_training_dataset(expected_stage1_config["dataset"])
    dataset_identity = validate_dataset_identity(identity_dataset.dataset_identity)
    del identity_dataset
    expected_stage1_config = dict(expected_stage1_config)
    expected_stage1_config["dataset_identity"] = dataset_identity
    stage1_payload = torch.load(
        stage1_path,
        map_location="cpu",
        weights_only=True,
    )
    output_state = merge_stage1_compact_with_base_action(
        base_state=state,
        stage1_payload=stage1_payload,
        expected_stage1_config=expected_stage1_config,
    )
    training_config = profile.training_config("stage2")
    _validate_base_model_config(
        payload,
        expected_compact=training_config["model"]["compact_wan"],
        expected_action=training_config["model"]["action_expert"],
    )
    _validate_base_action_normalization(
        payload,
        pinned_stats=pinned_qpos_stats,
    )

    seed = int(initialization["torch_seed"])
    torch.manual_seed(seed)
    motion = AbsoluteMotionTokenModule(
        dim=int(raw["method"]["action_expert"]["dim"]),
        history_count=int(motion_contract["history_count"]),
        feature_mean=tuple(motion_contract["feature_mean"]),
        feature_scale=tuple(motion_contract["feature_scale"]),
    )
    for key, value in motion.state_dict().items():
        output_state[f"absolute_motion_tokens.{key}"] = value.cpu()

    training_config["training_stage"] = "absolute_motion_initialization"
    training_config["model"] = dict(training_config["model"])
    training_config["model"]["absolute_motion"] = motion_contract
    training_config["action_normalization"] = {
        "enabled": True,
        "type": "mean_std",
        "epsilon": float(raw["method"]["normalization_epsilon"]),
        "stats_file": TRAIN_DATASET_ACTION_STATS,
        "stats_sha256": expected_action_sha256,
        "stats": pinned_qpos_stats,
    }
    training_config["dataset_identity"] = dataset_identity
    output = {
        "format": "dynamicwam_absolute_motion_checkpoint",
        "version": ABSOLUTE_MOTION_CHECKPOINT_VERSION,
        "model": output_state,
        "global_step": 0,
        "config": training_config,
        "initialization": {
            "source_checkpoint": str(base_path),
            "source_checkpoint_sha256": actual_base_sha256,
            "source_stage1_checkpoint": str(stage1_path),
            "source_stage1_checkpoint_sha256": _sha256(stage1_path),
            "source_action_stats": str(base_stats_path),
            "source_action_stats_sha256": expected_action_sha256,
            "packed_dataset_fingerprint": metadata["dataset_fingerprint"],
            "training_dataset_identity": dataset_identity,
            "motion_statistics": str(motion_stats_path),
            "motion_statistics_sha256": motion_contract["statistics_sha256"],
            "torch_seed": seed,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        torch.save(output, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    output_path = run(config_path=arguments.config)
    print(output_path)


if __name__ == "__main__":
    main()
