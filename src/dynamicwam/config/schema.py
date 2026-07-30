"""Strict schema checks shared by every DynamicWAM entrypoint."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dynamicwam.motion_contract import FLOW_QUALITY_METHOD

OFFICIAL_LEVEL1_TASKS = (
    "adjust_bottle",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "move_can_pot",
    "move_playingcard_away",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_skillet",
    "place_can_basket",
    "place_object_basket",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "move_pillbottle_pad",
    "move_stapler_pad",
    "place_bread_basket",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "stamp_seal",
)


def require_exact_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ from the schema: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return dict(value)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return number


def _pair(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{label} must contain exactly two integers")
    pair = tuple(
        _positive_int(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    return pair[0], pair[1]


def _int_list(
    value: Any,
    label: str,
    *,
    nonempty: bool = True,
    allow_zero: bool = False,
) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be a non-empty integer list")
    return [
        _positive_int(item, f"{label}[{index}]", allow_zero=allow_zero)
        for index, item in enumerate(value)
    ]


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_paths(raw: Any) -> dict[str, Any]:
    expected_keys = {
        "project_root",
        "raw_dataset",
        "source_dataset",
        "packed_dataset",
        "head_flow_cache",
        "language_embeddings",
        "wan_root",
        "external_assets_manifest",
        "checkpoint_manifest",
        "training_runs",
        "collection_logs",
        "pca_artifacts",
        "stage1_checkpoint",
        "base_checkpoint",
        "base_action_stats",
        "motion_init_checkpoint",
        "stage2_checkpoint",
        "stage3_checkpoint",
        "action_stats",
        "evaluation_runs",
    }
    paths = require_exact_keys(
        raw,
        expected_keys,
        "paths",
    )
    for key, value in paths.items():
        _nonempty_string(value, f"paths.{key}")
        if not Path(value).is_absolute():
            raise ValueError(f"paths.{key} must be absolute, got {value!r}")
    return paths


def _validate_method(raw: Any) -> dict[str, Any]:
    method = require_exact_keys(
        raw,
        {
            "flow_matching",
            "observation",
            "video",
            "head_flow",
            "compact_wan",
            "action_expert",
            "normalization_epsilon",
        },
        "method",
    )
    flow_matching = require_exact_keys(
        method["flow_matching"],
        {
            "shift",
            "sigma_min",
            "extra_one_step",
            "training_timesteps",
        },
        "method.flow_matching",
    )
    _finite_number(
        flow_matching["shift"],
        "method.flow_matching.shift",
        positive=True,
    )
    sigma_min = _finite_number(
        flow_matching["sigma_min"],
        "method.flow_matching.sigma_min",
    )
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError("method.flow_matching.sigma_min must be in [0, 1)")
    if not isinstance(flow_matching["extra_one_step"], bool):
        raise ValueError("method.flow_matching.extra_one_step must be boolean")
    _positive_int(
        flow_matching["training_timesteps"],
        "method.flow_matching.training_timesteps",
    )

    observation = require_exact_keys(
        method["observation"],
        {"head_size", "wrist_size"},
        "method.observation",
    )
    head_size = _pair(
        observation["head_size"],
        "method.observation.head_size",
    )
    wrist_size = _pair(
        observation["wrist_size"],
        "method.observation.wrist_size",
    )
    if head_size != (2 * wrist_size[0], 2 * wrist_size[1]):
        raise ValueError(
            "method.observation.head_size must be exactly twice wrist_size"
        )

    video = require_exact_keys(
        method["video"],
        {
            "num_frames",
            "size",
            "future_size",
            "global_downsample_rate",
            "video_action_frequency_ratio",
        },
        "method.video",
    )
    num_video_frames = _positive_int(
        video["num_frames"],
        "method.video.num_frames",
    )
    if num_video_frames % 4:
        raise ValueError(
            "method.video.num_frames must be divisible by the Wan temporal "
            "compression factor 4"
        )
    _pair(video["size"], "method.video.size")
    future_size = _pair(video["future_size"], "method.video.future_size")
    if any(value % 32 for value in future_size):
        raise ValueError("method.video.future_size values must be divisible by 32")
    _positive_int(
        video["global_downsample_rate"],
        "method.video.global_downsample_rate",
    )
    _positive_int(
        video["video_action_frequency_ratio"],
        "method.video.video_action_frequency_ratio",
    )
    if tuple(2 * value for value in future_size) != _pair(
        video["size"],
        "method.video.size",
    ):
        raise ValueError("method.video.size must be twice future_size")

    head_flow = require_exact_keys(
        method["head_flow"],
        {
            "count",
            "policy_stride",
            "compute_size",
            "container_fps",
            "normalization_percentile",
            "farneback",
            "quality",
        },
        "method.head_flow",
    )
    _positive_int(head_flow["count"], "method.head_flow.count")
    _positive_int(head_flow["policy_stride"], "method.head_flow.policy_stride")
    _pair(head_flow["compute_size"], "method.head_flow.compute_size")
    _finite_number(
        head_flow["container_fps"],
        "method.head_flow.container_fps",
        positive=True,
    )
    percentile = _finite_number(
        head_flow["normalization_percentile"],
        "method.head_flow.normalization_percentile",
        positive=True,
    )
    if percentile != 99.0:
        raise ValueError(
            "absolute-motion v2 requires method.head_flow.normalization_percentile=99"
        )
    farneback = require_exact_keys(
        head_flow["farneback"],
        {
            "pyr_scale",
            "levels",
            "winsize",
            "iterations",
            "poly_n",
            "poly_sigma",
            "flags",
        },
        "method.head_flow.farneback",
    )
    _finite_number(
        farneback["pyr_scale"],
        "method.head_flow.farneback.pyr_scale",
        positive=True,
    )
    if float(farneback["pyr_scale"]) > 1.0:
        raise ValueError("method.head_flow.farneback.pyr_scale must be <= 1")
    for key in ("levels", "winsize", "iterations", "poly_n"):
        _positive_int(farneback[key], f"method.head_flow.farneback.{key}")
    if int(farneback["winsize"]) % 2 == 0:
        raise ValueError("method.head_flow.farneback.winsize must be odd")
    if int(farneback["poly_n"]) not in {5, 7}:
        raise ValueError("method.head_flow.farneback.poly_n must be 5 or 7")
    _finite_number(
        farneback["poly_sigma"],
        "method.head_flow.farneback.poly_sigma",
        positive=True,
    )
    _positive_int(
        farneback["flags"],
        "method.head_flow.farneback.flags",
        allow_zero=True,
    )
    quality = require_exact_keys(
        head_flow["quality"],
        {
            "method",
            "relative_error",
            "absolute_error_squared",
            "minimum_reliable_fraction",
        },
        "method.head_flow.quality",
    )
    if quality["method"] != FLOW_QUALITY_METHOD:
        raise ValueError(
            f"method.head_flow.quality.method must be {FLOW_QUALITY_METHOD!r}"
        )
    relative_error = _finite_number(
        quality["relative_error"],
        "method.head_flow.quality.relative_error",
    )
    if relative_error < 0.0:
        raise ValueError("method.head_flow.quality.relative_error must be non-negative")
    _finite_number(
        quality["absolute_error_squared"],
        "method.head_flow.quality.absolute_error_squared",
        positive=True,
    )
    minimum_reliable_fraction = _finite_number(
        quality["minimum_reliable_fraction"],
        "method.head_flow.quality.minimum_reliable_fraction",
        positive=True,
    )
    if minimum_reliable_fraction > 1.0:
        raise ValueError(
            "method.head_flow.quality.minimum_reliable_fraction must be <= 1"
        )

    compact = require_exact_keys(
        method["compact_wan"],
        {
            "precision",
            "dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "head_dim",
            "hidden_anchor_layers",
            "motion_anchor_layers",
            "teacher_layer_mapping",
        },
        "method.compact_wan",
    )
    if compact["precision"] != "bfloat16":
        raise ValueError("DynamicWAM currently implements only bfloat16 compact WAN")
    for key in ("dim", "ffn_dim", "num_heads", "num_layers", "head_dim"):
        _positive_int(compact[key], f"method.compact_wan.{key}")
    if int(compact["dim"]) != int(compact["num_heads"]) * int(compact["head_dim"]):
        raise ValueError("method.compact_wan.dim must equal num_heads * head_dim")
    hidden = _int_list(
        compact["hidden_anchor_layers"],
        "method.compact_wan.hidden_anchor_layers",
    )
    motion = _int_list(
        compact["motion_anchor_layers"],
        "method.compact_wan.motion_anchor_layers",
    )
    mapping = _int_list(
        compact["teacher_layer_mapping"],
        "method.compact_wan.teacher_layer_mapping",
    )
    num_layers = int(compact["num_layers"])
    for label, layers in (("hidden", hidden), ("motion", motion)):
        if layers != sorted(set(layers)) or layers[-1] > num_layers:
            raise ValueError(
                f"method.compact_wan.{label}_anchor_layers must be unique, "
                f"sorted, and within 1..{num_layers}"
            )
    if len(mapping) != num_layers or mapping != sorted(set(mapping)):
        raise ValueError(
            "method.compact_wan.teacher_layer_mapping must contain one unique, "
            "sorted teacher layer per compact layer"
        )

    action = require_exact_keys(
        method["action_expert"],
        {"dim", "ffn_dim", "num_layers", "chunk_size", "state_dim", "action_dim"},
        "method.action_expert",
    )
    for key, value in action.items():
        _positive_int(value, f"method.action_expert.{key}")
    expected_chunk = int(video["num_frames"]) * int(
        video["video_action_frequency_ratio"]
    )
    if int(action["chunk_size"]) != expected_chunk:
        raise ValueError(
            "method.action_expert.chunk_size must equal "
            "video.num_frames * video_action_frequency_ratio"
        )
    if int(head_flow["count"]) * int(head_flow["policy_stride"]) != int(
        action["chunk_size"]
    ):
        raise ValueError(
            "method.head_flow.count * policy_stride must equal "
            "method.action_expert.chunk_size"
        )
    _finite_number(
        method["normalization_epsilon"],
        "method.normalization_epsilon",
        positive=True,
    )
    return method


def _validate_data(raw: Any) -> dict[str, Any]:
    expected_keys = {
        "splits",
        "samples_per_episode",
        "sampler_seed",
        "language_prompts_per_task",
        "language_prompt_seed",
        "max_open_shards",
    }
    data = require_exact_keys(
        raw,
        expected_keys,
        "data",
    )
    splits = data["splits"]
    if (
        not isinstance(splits, list)
        or not splits
        or len(set(splits)) != len(splits)
        or any(not isinstance(value, str) or not value for value in splits)
    ):
        raise ValueError("data.splits must contain unique non-empty names")
    _positive_int(data["samples_per_episode"], "data.samples_per_episode")
    _positive_int(data["sampler_seed"], "data.sampler_seed", allow_zero=True)
    if (
        _positive_int(
            data["language_prompts_per_task"],
            "data.language_prompts_per_task",
        )
        != 100
    ):
        raise ValueError("data.language_prompts_per_task must equal 100")
    _positive_int(
        data["language_prompt_seed"],
        "data.language_prompt_seed",
        allow_zero=True,
    )
    _positive_int(data["max_open_shards"], "data.max_open_shards")
    return data


def _validate_collection(raw: Any) -> dict[str, Any]:
    collection = require_exact_keys(
        raw,
        {
            "domino_commit",
            "clean_config_name",
            "randomized_config_name",
            "clean_episodes_per_task",
            "randomized_episodes_per_task",
            "workers_per_gpu",
            "planner_workers_per_task",
            "renderer_workers_per_task",
            "renderer_recycle_attempts",
            "ready_buffer_episodes",
            "worker_poll_seconds",
            "collection_raw_mp4",
            "converter_workers",
            "motion_statistics_workers",
        },
        "collection",
    )
    if re.fullmatch(r"[0-9a-f]{40}", collection["domino_commit"]) is None:
        raise ValueError("collection.domino_commit must be a full Git SHA")
    for key in ("clean_config_name", "randomized_config_name"):
        _nonempty_string(collection[key], f"collection.{key}")
    for key in (
        "clean_episodes_per_task",
        "randomized_episodes_per_task",
        "workers_per_gpu",
        "planner_workers_per_task",
        "renderer_workers_per_task",
        "renderer_recycle_attempts",
        "converter_workers",
        "motion_statistics_workers",
    ):
        _positive_int(collection[key], f"collection.{key}")
    _positive_int(
        collection["ready_buffer_episodes"],
        "collection.ready_buffer_episodes",
        allow_zero=True,
    )
    _finite_number(
        collection["worker_poll_seconds"],
        "collection.worker_poll_seconds",
        positive=True,
    )
    if collection["renderer_workers_per_task"] != 1:
        raise ValueError("collection.renderer_workers_per_task must equal 1")
    if collection["collection_raw_mp4"] is not False:
        raise ValueError("collection.collection_raw_mp4 must be false")
    return collection


def _validate_initialization(raw: Any) -> dict[str, Any]:
    initialization = require_exact_keys(
        raw,
        {
            "base_checkpoint_sha256",
            "base_action_stats_sha256",
            "torch_seed",
        },
        "initialization",
    )
    for key in (
        "base_checkpoint_sha256",
        "base_action_stats_sha256",
    ):
        digest = initialization[key]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"initialization.{key} must be a lowercase SHA256")
    _positive_int(
        initialization["torch_seed"],
        "initialization.torch_seed",
        allow_zero=True,
    )
    return initialization


def _validate_packing(raw: Any) -> dict[str, Any]:
    packing = require_exact_keys(
        raw,
        {
            "shard_size",
            "language_items_per_shard",
            "max_open_shards",
            "device",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "qpos_cache_size",
            "progress_interval_seconds",
            "log_level",
            "distributed_timeout_minutes",
        },
        "packing",
    )
    for key in (
        "shard_size",
        "language_items_per_shard",
        "max_open_shards",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "qpos_cache_size",
        "distributed_timeout_minutes",
    ):
        _positive_int(packing[key], f"packing.{key}")
    _finite_number(
        packing["progress_interval_seconds"],
        "packing.progress_interval_seconds",
        positive=True,
    )
    for key in ("pin_memory", "persistent_workers"):
        if not isinstance(packing[key], bool):
            raise ValueError(f"packing.{key} must be boolean")
    _nonempty_string(packing["device"], "packing.device")
    _nonempty_string(packing["log_level"], "packing.log_level")
    return packing


def _validate_training(raw: Any) -> dict[str, Any]:
    training = require_exact_keys(
        raw,
        {"common", "distributed", "stage1_pca", "stage1", "stage2", "stage3"},
        "training",
    )
    common = require_exact_keys(
        training["common"],
        {
            "device",
            "log_level",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "weight_decay",
            "grad_clip_norm",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "log_interval",
            "checkpoint_interval",
            "checkpoint_total_limit",
            "resume_from",
        },
        "training.common",
    )
    if common["device"] != "cuda":
        raise ValueError("DynamicWAM training requires device=cuda")
    _nonempty_string(common["log_level"], "training.common.log_level")
    for key in (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "prefetch_factor",
        "log_interval",
        "checkpoint_interval",
        "checkpoint_total_limit",
    ):
        _positive_int(common[key], f"training.common.{key}")
    for key in ("weight_decay", "grad_clip_norm"):
        _finite_number(common[key], f"training.common.{key}", positive=True)
    for key in ("pin_memory", "persistent_workers"):
        if not isinstance(common[key], bool):
            raise ValueError(f"training.common.{key} must be boolean")
    if not isinstance(common["resume_from"], str):
        raise ValueError("training.common.resume_from must be a string")

    distributed = require_exact_keys(
        training["distributed"],
        {"zero_stage", "bf16", "steps_per_print"},
        "training.distributed",
    )
    if distributed["zero_stage"] != 0 or distributed["bf16"] is not True:
        raise ValueError("DynamicWAM requires DeepSpeed ZeRO-0 with bf16")
    _positive_int(
        distributed["steps_per_print"],
        "training.distributed.steps_per_print",
    )

    pca_keys = {
        "episodes",
        "subclips_per_episode",
        "states_per_subclip",
        "max_tokens_per_layer",
        "seed",
    }
    stage1_pca = require_exact_keys(
        training["stage1_pca"],
        pca_keys,
        "training.stage1_pca",
    )
    for key, value in stage1_pca.items():
        _positive_int(
            value,
            f"training.stage1_pca.{key}",
            allow_zero=key == "seed",
        )

    stage1 = require_exact_keys(
        training["stage1"],
        {
            "allow_tf32",
            "cudnn_benchmark",
            "max_steps",
            "learning_rate",
            "min_lr_ratio",
            "warmup_steps",
            "projection_dim",
            "hidden_teacher_layers",
            "motion_teacher_layers",
            "lambda_gt",
            "lambda_hidden_schedule",
            "lambda_motion_schedule",
            "schedule_boundaries",
            "timestep_sampler",
            "sigma_loss_weights",
        },
        "training.stage1",
    )
    for key in ("allow_tf32", "cudnn_benchmark"):
        if not isinstance(stage1[key], bool):
            raise ValueError(f"training.stage1.{key} must be boolean")
    _positive_int(stage1["max_steps"], "training.stage1.max_steps")
    _positive_int(
        stage1["warmup_steps"], "training.stage1.warmup_steps", allow_zero=True
    )
    _positive_int(stage1["projection_dim"], "training.stage1.projection_dim")
    for key in ("learning_rate", "min_lr_ratio", "lambda_gt"):
        _finite_number(stage1[key], f"training.stage1.{key}", positive=True)
    if float(stage1["min_lr_ratio"]) > 1.0:
        raise ValueError("training.stage1.min_lr_ratio must be <= 1")
    for key in ("hidden_teacher_layers", "motion_teacher_layers"):
        layers = _int_list(stage1[key], f"training.stage1.{key}")
        if layers != sorted(set(layers)):
            raise ValueError(f"training.stage1.{key} must be sorted and unique")
    boundaries = stage1["schedule_boundaries"]
    hidden_schedule = stage1["lambda_hidden_schedule"]
    motion_schedule = stage1["lambda_motion_schedule"]
    if not all(
        isinstance(value, list)
        for value in (boundaries, hidden_schedule, motion_schedule)
    ):
        raise ValueError("Stage 1 schedules must be lists")
    if (
        not boundaries
        or len(hidden_schedule) != len(boundaries)
        or len(motion_schedule) != len(boundaries)
    ):
        raise ValueError("Stage 1 schedules must be non-empty and have equal lengths")
    for label, values in (
        ("schedule_boundaries", boundaries),
        ("lambda_hidden_schedule", hidden_schedule),
        ("lambda_motion_schedule", motion_schedule),
    ):
        for index, value in enumerate(values):
            _finite_number(value, f"training.stage1.{label}[{index}]")
    if (
        any(not 0.0 < float(value) <= 1.0 for value in boundaries)
        or boundaries != sorted(set(boundaries))
        or float(boundaries[-1]) != 1.0
    ):
        raise ValueError(
            "training.stage1.schedule_boundaries must increase uniquely to 1.0"
        )
    for label, values in (
        ("lambda_hidden_schedule", hidden_schedule),
        ("lambda_motion_schedule", motion_schedule),
    ):
        if any(float(value) < 0.0 for value in values):
            raise ValueError(f"training.stage1.{label} must be non-negative")
    timestep_sampler = require_exact_keys(
        stage1["timestep_sampler"],
        {
            "uniform_weight",
            "low_center",
            "low_weight",
            "mid_center",
            "mid_weight",
            "high_center",
            "high_weight",
            "width",
        },
        "training.stage1.timestep_sampler",
    )
    for key in ("uniform_weight", "low_weight", "mid_weight", "high_weight", "width"):
        _finite_number(
            timestep_sampler[key],
            f"training.stage1.timestep_sampler.{key}",
            positive=True,
        )
    for key in ("low_center", "mid_center", "high_center"):
        center = _finite_number(
            timestep_sampler[key],
            f"training.stage1.timestep_sampler.{key}",
        )
        if not 0.0 <= center <= 1.0:
            raise ValueError(
                f"training.stage1.timestep_sampler.{key} must be in [0, 1]"
            )
    sigma = require_exact_keys(
        stage1["sigma_loss_weights"],
        {"gt", "hidden", "motion"},
        "training.stage1.sigma_loss_weights",
    )
    sigma_gt = require_exact_keys(
        sigma["gt"],
        {"max_sigma", "softness", "floor"},
        "training.stage1.sigma_loss_weights.gt",
    )
    for key in ("max_sigma", "softness"):
        _finite_number(
            sigma_gt[key],
            f"training.stage1.sigma_loss_weights.gt.{key}",
            positive=True,
        )
    gt_floor = _finite_number(
        sigma_gt["floor"],
        "training.stage1.sigma_loss_weights.gt.floor",
    )
    if not 0.0 <= gt_floor <= 1.0:
        raise ValueError("training.stage1 sigma floor values must be in [0, 1]")
    for key in ("hidden", "motion"):
        sigma_term = require_exact_keys(
            sigma[key],
            {"center", "width", "floor"},
            f"training.stage1.sigma_loss_weights.{key}",
        )
        for field in ("center", "width"):
            _finite_number(
                sigma_term[field],
                f"training.stage1.sigma_loss_weights.{key}.{field}",
                positive=True,
            )
        floor = _finite_number(
            sigma_term["floor"],
            f"training.stage1.sigma_loss_weights.{key}.floor",
        )
        if not 0.0 <= floor <= 1.0:
            raise ValueError("training.stage1 sigma floor values must be in [0, 1]")

    stage2 = require_exact_keys(
        training["stage2"],
        {
            "max_steps",
            "learning_rate",
            "min_lr_ratio",
            "warmup_steps",
            "action_weight",
        },
        "training.stage2",
    )
    stage3 = require_exact_keys(
        training["stage3"],
        {
            "max_steps",
            "action_learning_rate",
            "video_learning_rate",
            "min_lr_ratio",
            "warmup_steps",
            "action_weight",
            "video_weight_initial",
            "video_weight",
            "video_weight_anneal_steps",
        },
        "training.stage3",
    )
    for label, stage in (("stage2", stage2), ("stage3", stage3)):
        _positive_int(stage["max_steps"], f"training.{label}.max_steps")
        _positive_int(
            stage["warmup_steps"],
            f"training.{label}.warmup_steps",
            allow_zero=True,
        )
        min_lr_ratio = _finite_number(
            stage["min_lr_ratio"],
            f"training.{label}.min_lr_ratio",
            positive=True,
        )
        if min_lr_ratio > 1.0:
            raise ValueError(f"training.{label}.min_lr_ratio must be <= 1")
    for key in ("learning_rate", "action_weight"):
        _finite_number(stage2[key], f"training.stage2.{key}", positive=True)
    for key in (
        "action_learning_rate",
        "video_learning_rate",
        "action_weight",
        "video_weight_initial",
        "video_weight",
    ):
        _finite_number(stage3[key], f"training.stage3.{key}", positive=True)
    _positive_int(
        stage3["video_weight_anneal_steps"],
        "training.stage3.video_weight_anneal_steps",
    )
    return training


def _validate_inference(raw: Any, action_chunk_size: int) -> dict[str, Any]:
    inference = require_exact_keys(
        raw,
        {
            "device",
            "checkpoint_artifact_id",
            "num_inference_steps",
            "video_refresh_steps",
            "action_interval_ms",
            "scene_prefix",
        },
        "inference",
    )
    if inference["device"] != "cuda":
        raise ValueError("DynamicWAM inference requires device=cuda")
    artifact_id = _nonempty_string(
        inference["checkpoint_artifact_id"],
        "inference.checkpoint_artifact_id",
    )
    if artifact_id != "full":
        raise ValueError(
            "canonical DynamicWAM inference requires checkpoint_artifact_id=full"
        )
    steps = _positive_int(
        inference["num_inference_steps"],
        "inference.num_inference_steps",
    )
    refresh = _int_list(
        inference["video_refresh_steps"],
        "inference.video_refresh_steps",
        allow_zero=True,
    )
    if refresh != sorted(set(refresh)) or refresh[0] != 0 or refresh[-1] >= steps:
        raise ValueError(
            "inference.video_refresh_steps must be sorted, unique, start at 0, "
            "and remain below num_inference_steps"
        )
    _positive_int(inference["action_interval_ms"], "inference.action_interval_ms")
    prefix = _nonempty_string(inference["scene_prefix"], "inference.scene_prefix")
    if not prefix.endswith(": "):
        raise ValueError("inference.scene_prefix must retain the trailing ': '")
    if action_chunk_size <= 0:
        raise ValueError("method action chunk size must be positive")
    return inference


def _validate_benchmark(raw: Any, action_chunk_size: int) -> dict[str, Any]:
    benchmark = require_exact_keys(
        raw,
        {
            "policy_registration_name",
            "task_config",
            "instruction_type",
            "checkpoint_label",
            "dynamic_coefficient",
            "episodes_per_task",
            "start_seed",
            "slot_seed_stride",
            "execute_steps",
            "gpus",
            "task_timeout_seconds",
            "bootstrap_replicates",
            "bootstrap_seed",
            "python",
            "domino_root",
            "eval_policy",
            "curobo_root",
            "tasks",
        },
        "benchmark",
    )
    for key in (
        "policy_registration_name",
        "task_config",
        "instruction_type",
        "checkpoint_label",
        "python",
        "domino_root",
        "eval_policy",
        "curobo_root",
    ):
        _nonempty_string(benchmark[key], f"benchmark.{key}")
    for key in ("python", "domino_root", "eval_policy", "curobo_root"):
        if not Path(benchmark[key]).is_absolute():
            raise ValueError(f"benchmark.{key} must be absolute")
    expected_policy = "ciwam.adapters.domino.deploy_policy_sync_flow"
    if benchmark["policy_registration_name"] != expected_policy:
        raise ValueError(
            "absolute-motion evaluation requires policy_registration_name="
            f"{expected_policy}"
        )
    _finite_number(
        benchmark["dynamic_coefficient"],
        "benchmark.dynamic_coefficient",
        positive=True,
    )
    for key in (
        "episodes_per_task",
        "start_seed",
        "slot_seed_stride",
        "execute_steps",
        "task_timeout_seconds",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        _positive_int(
            benchmark[key],
            f"benchmark.{key}",
            allow_zero=key in {"start_seed", "bootstrap_seed"},
        )
    if int(benchmark["execute_steps"]) != action_chunk_size:
        raise ValueError(
            "benchmark.execute_steps must equal method.action_expert.chunk_size"
        )
    gpus = benchmark["gpus"]
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(not isinstance(value, str) or not value for value in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("benchmark.gpus must contain unique non-empty strings")
    tasks = benchmark["tasks"]
    if tasks != list(OFFICIAL_LEVEL1_TASKS):
        raise ValueError(
            "benchmark.tasks must match the ordered official 35-task Level 1 suite"
        )
    return benchmark


def validate_absolute_motion_profile(raw: Any) -> dict[str, Any]:
    profile = require_exact_keys(
        raw,
        {
            "schema_version",
            "baseline",
            "paths",
            "method",
            "collection",
            "initialization",
            "data",
            "packing",
            "training",
            "inference",
            "benchmark",
        },
        "absolute-motion profile",
    )
    if profile["schema_version"] != 2 or profile["baseline"] != "absolute_motion_v2":
        raise ValueError("Expected an absolute-motion v2 profile with schema_version=2")
    _validate_paths(profile["paths"])
    method = _validate_method(profile["method"])
    _validate_collection(profile["collection"])
    _validate_initialization(profile["initialization"])
    _validate_data(profile["data"])
    _validate_packing(profile["packing"])
    _validate_training(profile["training"])
    action_chunk_size = int(method["action_expert"]["chunk_size"])
    _validate_inference(profile["inference"], action_chunk_size)
    _validate_benchmark(profile["benchmark"], action_chunk_size)
    return profile
