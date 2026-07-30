from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.training.data.robotwin2.robotwin_agilex_dataset import (
    RoboTwinSourceDataset,
)


def output_root(
    config: Mapping[str, Any],
    override: str | None = None,
) -> str:
    value = override or config.get("output_root")
    if not value:
        raise ValueError("pack config requires output_root")
    return str(value)


def source_dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("source")
    if not isinstance(value, Mapping):
        raise ValueError("pack config requires a source mapping")
    return require_exact_keys(
        value,
        {
            "root",
            "splits",
            "num_video_frames",
            "video_size",
            "global_downsample_rate",
            "video_action_freq_ratio",
            "normalization_epsilon",
        },
        "DynamicWAM pack source",
    )


def training_dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("dataset")
    if not isinstance(value, Mapping):
        raise ValueError("pack config requires a dataset mapping")
    return require_exact_keys(
        value,
        {
            "max_open_shards",
            "samples_per_episode",
            "sampler_seed",
        },
        "DynamicWAM pack dataset",
    )


def _build_robotwin_dataset(
    config: Mapping[str, Any],
    *,
    normalizer: RoboTwinQposNormalizer,
) -> RoboTwinSourceDataset:
    source = source_dataset_config(config)
    raw_video_size = source["video_size"]
    if not isinstance(raw_video_size, (list, tuple)) or len(raw_video_size) != 2:
        raise ValueError("source.video_size must contain height and width")
    return RoboTwinSourceDataset(
        dataset_dir=source["root"],
        splits=list(source["splits"]),
        num_video_frames=int(source["num_video_frames"]),
        video_size=(int(raw_video_size[0]), int(raw_video_size[1])),
        global_downsample_rate=int(source["global_downsample_rate"]),
        video_action_freq_ratio=int(source["video_action_freq_ratio"]),
        normalization_epsilon=float(source["normalization_epsilon"]),
        action_normalizer=normalizer,
    )


def build_robotwin_packing_dataset(
    config: Mapping[str, Any],
    *,
    action_stats_path: str,
) -> RoboTwinSourceDataset:
    source = source_dataset_config(config)
    normalizer = RoboTwinQposNormalizer.from_stats_path(
        action_stats_path,
        epsilon=float(source["normalization_epsilon"]),
    )
    return _build_robotwin_dataset(
        config,
        normalizer=normalizer,
    )


def compact_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("latent")
    if not isinstance(value, Mapping):
        raise ValueError("pack config requires a latent mapping")
    return require_exact_keys(
        value,
        {
            "vae_path",
            "precision",
            "shard_size",
            "future_video_size",
        },
        "DynamicWAM pack latent",
    )


def vae_path_from_config(config: Mapping[str, Any]) -> str:
    return str(compact_vae_config(config)["vae_path"])


def dtype_from_config(config: Mapping[str, Any]) -> torch.dtype:
    precision = str(compact_vae_config(config)["precision"])
    if precision != "bfloat16":
        raise ValueError(
            f"DynamicWAM latent.precision must be bfloat16, got {precision!r}"
        )
    return torch.bfloat16


def future_video_size_from_config(
    config: Mapping[str, Any],
) -> tuple[int, int]:
    raw = compact_vae_config(config)["future_video_size"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"latent.future_video_size must be [height, width], got {raw!r}"
        )
    size = tuple(int(value) for value in raw)
    if any(value <= 0 or value % 32 for value in size):
        raise ValueError(
            "latent.future_video_size values must be positive and divisible "
            f"by 32, got {size}"
        )
    return size[0], size[1]
