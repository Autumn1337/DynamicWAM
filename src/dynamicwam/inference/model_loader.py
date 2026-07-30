from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from dynamicwam.absolute_motion import (
    ABSOLUTE_MOTION_CHECKPOINT_VERSION,
    TEMPORAL_CONTRACT,
    validate_checkpoint_motion_metadata,
)
from dynamicwam.action_normalization import (
    RoboTwinQposNormalizer,
    validate_action_normalization_config,
)
from dynamicwam.config import load_profile
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.inference.models.compact_wan import CompactWANConfig, CompactWANModel
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel

if TYPE_CHECKING:
    from dynamicwam.vendor.wan.modules.t5 import T5EncoderModel


@dataclass(frozen=True)
class DynamicWAMRuntime:
    model: SmallWAMActionModel
    t5_encoder: T5EncoderModel
    action_normalizer: RoboTwinQposNormalizer
    device: str
    num_inference_steps: int
    num_video_inference_steps: int
    video_refresh_steps: tuple[int, ...]
    chunk_size: int
    num_video_frames: int
    video_size: tuple[int, int]
    scene_prefix: str
    head_flow_config: dict[str, Any]
    observation_config: dict[str, Any]
    composite_frame_size: tuple[int, int]
    action_interval_seconds: float
    flow_match_shift: float
    flow_match_sigma_min: float
    flow_match_extra_one_step: bool


COMPACT_WAN_CHECKPOINT_KEYS = (
    "precision",
    "dim",
    "ffn_dim",
    "num_heads",
    "num_layers",
    "head_dim",
)


def load_deploy_config(config_path: str | Path) -> dict[str, Any]:
    return load_profile(config_path).inference_config()


def _load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"DynamicWAM checkpoint must contain a mapping: {checkpoint_path}"
        )
    if not isinstance(payload.get("model"), dict):
        raise TypeError(
            f"DynamicWAM checkpoint is missing its model state: {checkpoint_path}"
        )
    if not isinstance(payload.get("config"), dict):
        raise RuntimeError(
            f"DynamicWAM checkpoint is missing exported config metadata: "
            f"{checkpoint_path}"
        )
    if (
        payload.get("format") != "dynamicwam_absolute_motion_checkpoint"
        or payload.get("version") != ABSOLUTE_MOTION_CHECKPOINT_VERSION
    ):
        raise RuntimeError(
            f"deployment accepts only absolute-motion checkpoint v2: {checkpoint_path}"
        )
    expected_keys = {
        "format",
        "version",
        "model",
        "global_step",
        "config",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "deployment checkpoint keys differ from the Stage-3 v2 export: "
            f"{checkpoint_path}"
        )
    return payload


def _select_keys(config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: config.get(key) for key in keys}


def _validate_model_config(
    runtime_config: dict[str, Any],
    checkpoint_config: dict[str, Any],
) -> None:
    expected_model = runtime_config["model"]
    checkpoint_model = checkpoint_config.get("model")
    if not isinstance(checkpoint_model, dict):
        raise RuntimeError("checkpoint metadata is missing model configuration")
    checkpoint_model = require_exact_keys(
        checkpoint_model,
        {
            "initial_checkpoint",
            "compact_wan",
            "action_expert",
            "absolute_motion",
        },
        "checkpoint model",
    )
    expected = {
        "compact_wan": _select_keys(
            expected_model["compact_wan"],
            COMPACT_WAN_CHECKPOINT_KEYS,
        ),
        "action_expert": expected_model["action_expert"],
    }
    checkpoint_motion = validate_checkpoint_motion_metadata(
        checkpoint_model.get("absolute_motion")
    )
    actual = {
        "compact_wan": _select_keys(
            checkpoint_model.get("compact_wan") or {},
            COMPACT_WAN_CHECKPOINT_KEYS,
        ),
        "action_expert": checkpoint_model.get("action_expert"),
    }
    mismatches = [
        f"{key}: expected {expected[key]!r}, got {actual[key]!r}"
        for key in expected
        if expected[key] != actual[key]
    ]
    if mismatches:
        raise RuntimeError(
            "deploy config does not match the checkpoint: " + "; ".join(mismatches)
        )
    runtime_motion = expected_model["absolute_motion"]
    if (
        int(runtime_motion["history_count"]) != int(checkpoint_motion["history_count"])
        or runtime_motion["temporal_contract"] != checkpoint_motion["temporal_contract"]
        or runtime_motion["timestamp_source"] != checkpoint_motion["timestamp_source"]
        or runtime_motion["spatial_unit"] != checkpoint_motion["spatial_unit"]
        or runtime_motion["flow_contract"] != checkpoint_motion["flow_contract"]
    ):
        raise RuntimeError("deploy motion computation does not match the checkpoint")


def _normalization_config(config: dict[str, Any], *, owner: str) -> dict[str, Any]:
    value = config.get("action_normalization")
    if not isinstance(value, dict):
        raise RuntimeError(f"{owner} is missing top-level action_normalization")
    require_exact_keys(
        value,
        {
            "enabled",
            "type",
            "epsilon",
            "stats_file",
            "stats_sha256",
            "stats",
        },
        f"{owner} action_normalization",
    )
    try:
        return validate_action_normalization_config(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{owner} must use DynamicWAM mean/std action normalization"
        ) from exc


def _build_action_normalizer(
    runtime_config: dict[str, Any],
    checkpoint_config: dict[str, Any],
) -> RoboTwinQposNormalizer:
    checkpoint_value = _normalization_config(
        checkpoint_config,
        owner="checkpoint metadata",
    )
    checkpoint_stats = checkpoint_value.get("stats")
    if not isinstance(checkpoint_stats, dict):
        raise RuntimeError("checkpoint metadata does not contain qpos statistics")
    epsilon = float(runtime_config["normalization_epsilon"])
    checkpoint_epsilon = checkpoint_value.get("epsilon")
    if checkpoint_epsilon is not None and float(checkpoint_epsilon) != epsilon:
        raise RuntimeError(
            "Checkpoint normalization epsilon differs from the DynamicWAM profile: "
            f"{checkpoint_epsilon!r} != {epsilon!r}"
        )
    return RoboTwinQposNormalizer.from_stats(
        checkpoint_stats,
        epsilon=epsilon,
    )


def _compact_config(
    runtime_config: dict[str, Any],
) -> CompactWANConfig:
    wan_root = Path(str(runtime_config["wan_root"]))
    model_config = runtime_config["model"]["compact_wan"]
    future_video_size = model_config["future_video_size"]
    if not isinstance(future_video_size, (list, tuple)) or len(future_video_size) != 2:
        raise ValueError(
            "resolved compact_wan.future_video_size must contain two integers"
        )
    return CompactWANConfig(
        checkpoint_path=str(wan_root),
        config_path=str(wan_root),
        vae_path=str(wan_root / "Wan2.2_VAE.pth"),
        precision=str(model_config["precision"]),
        dim=int(model_config["dim"]),
        ffn_dim=int(model_config["ffn_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        head_dim=int(model_config["head_dim"]),
        future_video_size=(
            int(future_video_size[0]),
            int(future_video_size[1]),
        ),
    )


def _build_model(
    runtime_config: dict[str, Any],
    payload: dict[str, Any],
    *,
    device: str,
) -> SmallWAMActionModel:
    checkpoint_config = payload["config"]
    _validate_model_config(runtime_config, checkpoint_config)
    compact_config = _compact_config(runtime_config)
    compact_wan = CompactWANModel.from_config(compact_config, device=device)
    action = runtime_config["model"]["action_expert"]
    absolute_motion = checkpoint_config["model"]["absolute_motion"]
    model_config = SmallWAMActionConfig(
        compact_wan=compact_wan.config,
        action_dim=int(action["action_dim"]),
        state_dim=int(action["state_dim"]),
        chunk_size=int(action["chunk_size"]),
        ae_dim=int(action["dim"]),
        ae_ffn_dim=int(action["ffn_dim"]),
        ae_num_layers=int(action["num_layers"]),
        wan_frozen=True,
        motion_history_count=int(absolute_motion["history_count"]),
        motion_feature_mean=tuple(
            float(value) for value in absolute_motion["feature_mean"]
        ),
        motion_feature_scale=tuple(
            float(value) for value in absolute_motion["feature_scale"]
        ),
    )
    model = SmallWAMActionModel(model_config, compact_wan)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _build_t5_encoder(config: dict[str, Any], device: str) -> T5EncoderModel:
    from dynamicwam.vendor.wan.modules.t5 import T5EncoderModel

    wan_root = Path(str(config["wan_root"]))
    return T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(wan_root / "google" / "umt5-xxl"),
    )


def _video_refresh_steps(
    inference: dict[str, Any],
    action_steps: int,
) -> tuple[int, ...]:
    raw = inference.get("video_refresh_steps")
    if not isinstance(raw, list) or not raw:
        raise ValueError("inference.video_refresh_steps must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError("inference.video_refresh_steps must contain integers")
    steps = tuple(int(value) for value in raw)
    if steps[0] != 0:
        raise ValueError("inference.video_refresh_steps must start at 0")
    if steps != tuple(sorted(set(steps))):
        raise ValueError("inference.video_refresh_steps must be sorted and unique")
    if steps[-1] >= action_steps:
        raise ValueError(
            "inference.video_refresh_steps must be smaller than "
            "inference.num_inference_steps"
        )
    return steps


def build_runtime_from_config(
    config: dict[str, Any],
) -> DynamicWAMRuntime:
    require_exact_keys(
        config,
        {
            "checkpoint_path",
            "wan_root",
            "normalization_epsilon",
            "device",
            "flow_matching",
            "observation",
            "common",
            "model",
            "head_flow",
            "inference",
        },
        "resolved inference config",
    )
    inference = config["inference"]
    device = str(config["device"])
    if device != "cuda":
        raise ValueError("DynamicWAM inference requires device=cuda")
    flow_matching = require_exact_keys(
        config["flow_matching"],
        {"shift", "sigma_min", "extra_one_step"},
        "resolved inference flow_matching",
    )
    observation = require_exact_keys(
        config["observation"],
        {"head_size", "wrist_size"},
        "resolved inference observation",
    )
    head_size = tuple(int(value) for value in observation["head_size"])
    wrist_size = tuple(int(value) for value in observation["wrist_size"])
    if len(head_size) != 2 or len(wrist_size) != 2:
        raise ValueError("resolved observation sizes must be height-width pairs")
    composite_frame_size = (
        head_size[0] + wrist_size[0],
        head_size[1],
    )
    common = config["common"]
    require_exact_keys(
        common,
        {"num_video_frames", "video_height", "video_width"},
        "resolved inference common",
    )
    model_config = require_exact_keys(
        config["model"],
        {"compact_wan", "action_expert", "absolute_motion"},
        "resolved inference model",
    )
    require_exact_keys(
        model_config["compact_wan"],
        {
            "precision",
            "dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "head_dim",
            "future_video_size",
        },
        "resolved inference compact_wan",
    )
    require_exact_keys(
        model_config["action_expert"],
        {"dim", "ffn_dim", "num_layers", "chunk_size", "state_dim", "action_dim"},
        "resolved inference action_expert",
    )
    require_exact_keys(
        model_config["absolute_motion"],
        {
            "history_count",
            "temporal_contract",
            "timestamp_source",
            "spatial_unit",
            "flow_contract",
        },
        "resolved inference absolute_motion",
    )
    if model_config["absolute_motion"]["temporal_contract"] != TEMPORAL_CONTRACT:
        raise ValueError("inference requires exact simulator-time motion v2")
    require_exact_keys(
        inference,
        {
            "num_inference_steps",
            "video_refresh_steps",
            "action_interval_ms",
            "scene_prefix",
        },
        "resolved inference settings",
    )
    if not isinstance(config["head_flow"], dict):
        raise ValueError("resolved head_flow config must be a mapping")
    if int(config["head_flow"].get("count", -1)) != int(
        model_config["absolute_motion"]["history_count"]
    ):
        raise ValueError("head-flow history count differs from motion tokens")
    action_steps = int(inference["num_inference_steps"])
    refresh_steps = _video_refresh_steps(inference, action_steps)
    video_shape = (
        int(common["num_video_frames"]),
        int(common["video_height"]),
        int(common["video_width"]),
    )
    if any(value <= 0 for value in video_shape):
        raise ValueError(f"inference video shape must be positive: {video_shape}")
    action_interval_seconds = float(inference["action_interval_ms"]) / 1000.0
    if action_interval_seconds <= 0.0:
        raise ValueError("inference.action_interval_ms must be positive")
    payload = _load_checkpoint_payload(config["checkpoint_path"])
    checkpoint_config = payload["config"]
    action = config["model"]["action_expert"]
    return DynamicWAMRuntime(
        model=_build_model(config, payload, device=device),
        t5_encoder=_build_t5_encoder(config, device),
        action_normalizer=_build_action_normalizer(config, checkpoint_config),
        device=device,
        num_inference_steps=action_steps,
        num_video_inference_steps=len(refresh_steps),
        video_refresh_steps=refresh_steps,
        chunk_size=int(action["chunk_size"]),
        num_video_frames=video_shape[0],
        video_size=(
            video_shape[1],
            video_shape[2],
        ),
        scene_prefix=str(inference["scene_prefix"]),
        head_flow_config=dict(config["head_flow"]),
        observation_config=dict(observation),
        composite_frame_size=composite_frame_size,
        action_interval_seconds=action_interval_seconds,
        flow_match_shift=float(flow_matching["shift"]),
        flow_match_sigma_min=float(flow_matching["sigma_min"]),
        flow_match_extra_one_step=bool(flow_matching["extra_one_step"]),
    )
