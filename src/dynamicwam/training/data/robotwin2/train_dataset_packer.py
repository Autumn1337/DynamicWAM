"""Build the DynamicWAM dataset from preprocessed RoboTwin data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dynamicwam.absolute_motion import (
    ENDPOINT_RULE,
    MOTION_FEATURE_NAMES,
    SPATIAL_UNIT,
    TEMPORAL_CONTRACT,
    TIMESTAMP_SOURCE,
    build_checkpoint_motion_metadata,
    build_flow_cache_parameters,
    load_exact_flow_cache,
    raw_offsets,
    raw_pairs,
    raw_stride,
    validate_motion_statistics,
)
from dynamicwam.config import load_profile, write_config_snapshot
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.external_assets import verify_wan_assets
from dynamicwam.image import load_video_frames
from dynamicwam.training.data.packed_dataset import (
    TRAIN_DATASET_ACTION_STATS,
    TRAIN_DATASET_FORMAT,
    TRAIN_DATASET_LANG_DIR,
    TRAIN_DATASET_METADATA,
    TRAIN_DATASET_MOTION_STATS,
    TRAIN_DATASET_SHARD_DIR,
    TRAIN_DATASET_STATS,
    TRAIN_DATASET_VERSION,
)
from dynamicwam.training.data.robotwin2.robotwin_agilex_dataset import (
    RoboTwinSourceDataset,
)
from dynamicwam.training.data.robotwin2.train_dataset_config import (
    build_robotwin_packing_dataset,
    compact_vae_config,
    dtype_from_config,
    future_video_size_from_config,
    training_dataset_config,
    vae_path_from_config,
)
from dynamicwam.training.data.robotwin2.train_dataset_config import (
    output_root as output_root_from_config,
)
from dynamicwam.training.train.common import setup_logging

logger = logging.getLogger(__name__)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _phase(rank: int, index: int, total: int, message: str) -> None:
    if rank == 0:
        logger.info("[%d/%d] %s", index, total, message)


def _distributed_info(device: str, timeout_minutes: int) -> tuple[int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_device = device

    if world_size > 1:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            resolved_device = f"cuda:{local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                timeout=timedelta(minutes=max(1, int(timeout_minutes))),
            )
    return rank, world_size, resolved_device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cuda_synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resize_video_frames(
    frames: torch.Tensor, size_hw: tuple[int, int]
) -> torch.Tensor:
    batch, frames_per_clip, channels, height, width = frames.shape
    if (height, width) == tuple(size_hw):
        return frames
    resized = F.interpolate(
        frames.reshape(batch * frames_per_clip, channels, height, width).float(),
        size=tuple(size_hw),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).to(dtype=frames.dtype)
    return resized.reshape(batch, frames_per_clip, channels, *size_hw)


def _build_low_resolution_full_video(
    first_frame: torch.Tensor,
    video_frames: torch.Tensor,
    size_hw: tuple[int, int],
) -> torch.Tensor:
    first_low = _resize_video_frames(first_frame.unsqueeze(1), size_hw)
    future_low = _resize_video_frames(video_frames, size_hw)
    full_low = torch.cat([first_low, future_low], dim=1)
    return (full_low * 2.0 - 1.0).permute(0, 2, 1, 3, 4)


def _stats_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = payload.get("robotwin_qpos", payload)
    if not isinstance(stats, dict):
        raise ValueError("qpos stats JSON must contain a mapping payload")
    return stats


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_action_stats(
    contract: Dict[str, Any],
    output_root: Path,
) -> Path:
    contract = require_exact_keys(
        contract,
        {"source_path", "sha256", "qpos_dim"},
        "absolute-motion action stats",
    )
    source = Path(str(contract["source_path"])).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"base action stats do not exist: {source}")
    actual_sha256 = _sha256_file(source)
    expected_sha256 = str(contract["sha256"])
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "base action stats SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    stats = _stats_payload(payload)
    qpos_dim = int(contract["qpos_dim"])
    mean = torch.as_tensor(stats.get("mean"), dtype=torch.float64)
    standard_deviation = torch.as_tensor(
        stats.get("std"),
        dtype=torch.float64,
    )
    if (
        stats.get("type") != "mean_std"
        or int(stats.get("qpos_dim", -1)) != qpos_dim
        or mean.shape != (qpos_dim,)
        or standard_deviation.shape != (qpos_dim,)
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(standard_deviation).all())
        or not bool((standard_deviation > 0.0).all())
    ):
        raise ValueError(f"invalid base action stats contract: {source}")
    target = output_root / TRAIN_DATASET_ACTION_STATS
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Copied pinned action statistics: %s", target)
    return target


def _prepare_motion_stats(
    source_path: str | Path,
    output_root: Path,
) -> Path:
    source = Path(source_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"absolute motion statistics do not exist: {source}")
    with source.open("r", encoding="utf-8") as stream:
        payload = validate_motion_statistics(json.load(stream))
    target = output_root / TRAIN_DATASET_MOTION_STATS
    _write_json_atomic(target, payload)
    return target


def _action_normalization_metadata(
    epsilon: float,
    *,
    stats_sha256: str,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "type": "mean_std",
        "epsilon": float(epsilon),
        "stats_file": TRAIN_DATASET_ACTION_STATS,
        "stats_sha256": str(stats_sha256),
    }


def _path_independent_training_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(config)
    payload.pop("root", None)
    payload.pop("source", None)
    payload.pop("source_dataset", None)
    return payload


def _hash_json_payload(payload: Dict[str, Any]) -> str:
    data = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_language_list(path: str) -> List[torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected non-empty list of language embeddings in {path}")

    embeddings: List[torch.Tensor] = []
    for value in payload:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                f"Expected language embedding [tokens, dim], got {tuple(tensor.shape)} in {path}"
            )
        embeddings.append(tensor.detach().cpu().contiguous())
    return embeddings


def _write_language_shard(
    lang_root: Path,
    shard_index: int,
    embeddings: List[torch.Tensor],
) -> Dict[str, Any]:
    if not embeddings:
        raise ValueError("Cannot write an empty language shard")
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "Packing train dataset requires safetensors to be installed"
        ) from exc

    token_offsets = [0]
    for embedding in embeddings:
        token_offsets.append(token_offsets[-1] + int(embedding.shape[0]))
    token_data = torch.cat(embeddings, dim=0).contiguous()
    offsets_tensor = torch.tensor(token_offsets, dtype=torch.long)

    shard_path = (
        lang_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.safetensors"
    )
    shard_meta_path = (
        lang_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_shard_path = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
    try:
        save_file(
            {
                "token_data": token_data,
                "token_offsets": offsets_tensor,
            },
            str(tmp_shard_path),
        )
        tmp_shard_path.replace(shard_path)
    finally:
        tmp_shard_path.unlink(missing_ok=True)
    metadata = {
        "shard_index": int(shard_index),
        "item_count": len(embeddings),
        "token_count": int(token_data.shape[0]),
        "embedding_dim": int(token_data.shape[1]),
        "dtype": str(token_data.dtype).replace("torch.", ""),
    }
    _write_json_atomic(shard_meta_path, metadata)
    return metadata


def _language_group_ids(
    dataset: RoboTwinSourceDataset,
) -> Dict[int, int]:
    groups_by_path: Dict[str, int] = {}
    episode_groups: Dict[int, int] = {}
    for episode_index, episode_data in enumerate(dataset.all_episodes):
        language_path = str(Path(episode_data["lang_path"]).resolve())
        if language_path not in groups_by_path:
            groups_by_path[language_path] = len(groups_by_path)
        episode_groups[episode_index] = groups_by_path[language_path]
    if len(episode_groups) != len(dataset.all_episodes):
        raise RuntimeError("language group mapping is incomplete")
    return episode_groups


def _write_language_bank(
    dataset: RoboTwinSourceDataset,
    output_root: Path,
    *,
    lang_items_per_shard: int,
    lang_group_ids: Dict[int, int],
) -> tuple[Dict[int, int], str]:
    lang_root = output_root / TRAIN_DATASET_LANG_DIR
    groups: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    current_embeddings: List[torch.Tensor] = []
    current_shard_index = 0
    embedding_dim: Optional[int] = None
    target_dtype: Optional[torch.dtype] = None
    groups_by_language_path: Dict[str, int] = {}

    def flush_language_shard() -> None:
        nonlocal current_embeddings, current_shard_index
        if not current_embeddings:
            return
        _write_language_shard(lang_root, current_shard_index, current_embeddings)
        current_shard_index += 1
        current_embeddings = []

    iterator = tqdm(
        dataset.all_episodes, desc="[3/6] packing language bank", unit="episode"
    )
    for episode_index, episode_data in enumerate(iterator):
        language_path = str(Path(episode_data["lang_path"]).resolve())
        existing_group = groups_by_language_path.get(language_path)
        if existing_group is not None:
            if lang_group_ids[episode_index] != existing_group:
                raise RuntimeError("language mapping changed during bank construction")
            continue

        embeddings = _load_language_list(language_path)
        group_lang_ids: List[int] = []
        lang_group_id = len(groups)
        if lang_group_ids[episode_index] != lang_group_id:
            raise RuntimeError("language group order differs from the dataset mapping")
        groups_by_language_path[language_path] = lang_group_id

        for instruction_idx, embedding in enumerate(embeddings):
            if embedding_dim is None:
                embedding_dim = int(embedding.shape[1])
                target_dtype = embedding.dtype
            if int(embedding.shape[1]) != int(embedding_dim):
                raise ValueError(
                    f"Language embedding dim mismatch in {episode_data['lang_path']}: "
                    f"expected {embedding_dim}, got {embedding.shape[1]}"
                )
            if embedding.dtype != target_dtype:
                raise ValueError(
                    f"Language embedding dtype mismatch in {episode_data['lang_path']}: "
                    f"expected {target_dtype}, got {embedding.dtype}"
                )
            if current_embeddings and len(current_embeddings) >= lang_items_per_shard:
                flush_language_shard()

            lang_id = len(items)
            local_index = len(current_embeddings)
            current_embeddings.append(embedding.contiguous())
            group_lang_ids.append(lang_id)
            items.append(
                {
                    "lang_id": int(lang_id),
                    "lang_group_id": int(lang_group_id),
                    "shard_index": int(current_shard_index),
                    "local_index": int(local_index),
                    "instruction_idx": int(instruction_idx),
                }
            )

        groups.append(
            {
                "lang_group_id": int(lang_group_id),
                "split": episode_data.get("split", ""),
                "task_name": episode_data["task_name"],
                "lang_ids": group_lang_ids,
            }
        )

    flush_language_shard()
    if target_dtype is None:
        raise ValueError("DynamicWAM source dataset contains no language embeddings")
    dtype_name = str(target_dtype).replace("torch.", "")
    _write_json_atomic(
        lang_root / "lang.json",
        {
            "format": "dynamicwam_language_bank",
            "version": 1,
            "storage": "flat_token_shards",
            "dtype": dtype_name,
            "embedding_dim": int(embedding_dim or 0),
            "item_count": len(items),
            "group_count": len(groups),
            "groups": groups,
            "items": items,
        },
    )
    return lang_group_ids, dtype_name


def _episode_sample_ranges(dataset: RoboTwinSourceDataset) -> Dict[int, Dict[str, int]]:
    ranges: Dict[int, Dict[str, int]] = {}
    for sample_id, (episode_index, _condition_frame_idx) in enumerate(
        dataset.sample_index
    ):
        entry = ranges.setdefault(
            int(episode_index),
            {
                "first_sample_id": int(sample_id),
                "sample_count": 0,
            },
        )
        entry["sample_count"] += 1
    return ranges


def _summarize_numbers(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _build_dataset_statistics(
    dataset: RoboTwinSourceDataset,
    ranges: Dict[int, Dict[str, int]],
) -> Dict[str, Any]:
    task_stats: Dict[str, Dict[str, Any]] = {}
    split_counts: Dict[str, int] = {}
    all_lengths: List[int] = []
    all_sample_counts: List[int] = []
    for episode_index, episode_data in enumerate(dataset.all_episodes):
        task_name = str(episode_data["task_name"])
        split = str(episode_data.get("split", ""))
        sample_count = int(
            ranges.get(episode_index, {"sample_count": 0})["sample_count"]
        )
        try:
            effective_frame_count = int(
                dataset._effective_episode_frame_count(episode_data)
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to determine the effective frame count for "
                f"episode {episode_index} "
                f"({episode_data.get('task_name')}/"
                f"{episode_data.get('episode_name')})"
            ) from exc
        all_lengths.append(effective_frame_count)
        all_sample_counts.append(sample_count)
        split_counts[split] = split_counts.get(split, 0) + 1

        entry = task_stats.setdefault(
            task_name,
            {
                "episode_count": 0,
                "sample_count": 0,
                "splits": {},
                "_effective_frame_counts": [],
                "_sample_counts": [],
            },
        )
        entry["episode_count"] += 1
        entry["sample_count"] += sample_count
        entry["splits"][split] = int(entry["splits"].get(split, 0)) + 1
        entry["_effective_frame_counts"].append(effective_frame_count)
        entry["_sample_counts"].append(sample_count)

    tasks: Dict[str, Dict[str, Any]] = {}
    for task_name, entry in sorted(task_stats.items()):
        lengths = entry.pop("_effective_frame_counts")
        sample_counts = entry.pop("_sample_counts")
        tasks[task_name] = {
            **entry,
            "effective_frame_count": _summarize_numbers(lengths),
            "sample_count_per_episode": _summarize_numbers(sample_counts),
        }

    return {
        "task_count": len(tasks),
        "episode_count": len(dataset.all_episodes),
        "sample_count": len(dataset.sample_index),
        "error_count": 0,
        "splits": split_counts,
        "effective_frame_count": _summarize_numbers(all_lengths),
        "sample_count_per_episode": _summarize_numbers(all_sample_counts),
        "tasks": tasks,
    }


def _write_manifests_and_stats(
    dataset: RoboTwinSourceDataset,
    output_root: Path,
    *,
    lang_group_ids: Dict[int, int],
) -> Dict[str, Any]:
    ranges = _episode_sample_ranges(dataset)
    statistics = _build_dataset_statistics(dataset, ranges)
    episodes_path = output_root / "episodes.jsonl"
    samples_path = output_root / "samples.jsonl"

    fingerprint_hasher = hashlib.sha256()
    with episodes_path.open("w", encoding="utf-8") as episodes_file:
        for episode_index, episode_data in enumerate(dataset.all_episodes):
            sample_range = ranges.get(
                episode_index, {"first_sample_id": -1, "sample_count": 0}
            )
            payload = {
                "episode_id": int(episode_index),
                "split": episode_data.get("split", ""),
                "task_name": episode_data["task_name"],
                "episode_name": episode_data["episode_name"],
                "first_sample_id": int(sample_range["first_sample_id"]),
                "sample_count": int(sample_range["sample_count"]),
                "lang_group_id": int(lang_group_ids[episode_index]),
            }
            line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
            fingerprint_hasher.update(line.encode("utf-8"))
            fingerprint_hasher.update(b"\n")
            episodes_file.write(line + "\n")

    iterator = tqdm(
        range(len(dataset.sample_index)),
        desc="[3/6] writing sample manifest",
        unit="sample",
    )
    with samples_path.open("w", encoding="utf-8") as samples_file:
        for sample_id in iterator:
            episode_index, condition_frame_idx = dataset.sample_index[sample_id]
            episode_data = dataset.get_episode(episode_index)
            payload = {
                "sample_id": int(sample_id),
                "full_sample_id": int(sample_id),
                "episode_index": int(episode_index),
                "split": episode_data.get("split", ""),
                "task_name": episode_data["task_name"],
                "episode_name": episode_data["episode_name"],
                "condition_frame_idx": int(condition_frame_idx),
                "lang_group_id": int(lang_group_ids[episode_index]),
            }
            line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
            fingerprint_hasher.update(line.encode("utf-8"))
            fingerprint_hasher.update(b"\n")
            samples_file.write(line + "\n")

    _write_json_atomic(output_root / TRAIN_DATASET_STATS, statistics)
    return {
        "ranges": ranges,
        "statistics": statistics,
        "manifest_sha256": fingerprint_hasher.hexdigest(),
    }


def _load_qpos_tensor(path: str) -> torch.Tensor:
    qpos_data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(qpos_data, torch.Tensor):
        qpos_data = torch.as_tensor(qpos_data)
    return qpos_data


def _load_absolute_motion(
    cache_root: str,
    source_root: str,
    video_path: str,
    condition_idx: int,
    k: int,
    policy_stride: int,
    global_downsample_rate: int,
    params_json: str,
    expected_frame_count: int,
    video_size: tuple[int, int],
    episode_cache: OrderedDict[str, Dict[str, np.ndarray]],
    max_cached_episodes: int,
) -> Dict[str, torch.Tensor]:
    """Load exact-time RGB flow and numeric motion for one packed sample."""
    import cv2

    rel = Path(video_path).resolve().relative_to(Path(source_root).resolve())
    npz_path = Path(cache_root) / rel.with_suffix(".flow.npz")
    cache_key = str(npz_path)
    arrays = episode_cache.get(cache_key)
    if arrays is None:
        arrays = load_exact_flow_cache(
            npz_path,
            expected_params_json=params_json,
        )
        episode_cache[cache_key] = arrays
        episode_cache.move_to_end(cache_key)
        while len(episode_cache) > max(1, int(max_cached_episodes)):
            episode_cache.popitem(last=False)
    else:
        episode_cache.move_to_end(cache_key)
    flow = arrays["flow_rgb"]
    motion_features = arrays["motion_features"]
    interval_valid = arrays["interval_valid"]
    acceleration_valid = arrays["acceleration_valid"]
    if len(flow) != int(expected_frame_count):
        raise ValueError(
            f"flow cache frame count mismatch for {npz_path}: "
            f"expected {expected_frame_count}, got {len(flow)}"
        )
    selected_pairs = raw_pairs(
        condition_idx,
        history_count=k,
        policy_stride=policy_stride,
        global_downsample_rate=global_downsample_rate,
    )
    selected = np.stack(
        [flow[current] for _previous, current in selected_pairs],
        axis=0,
    )

    height, width = int(video_size[0]), int(video_size[1])
    frames = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        for frame in selected
    ]
    return {
        "frames": torch.from_numpy(np.stack(frames, axis=0)),
        "features": torch.from_numpy(
            np.stack(
                [motion_features[current] for _previous, current in selected_pairs]
            )
        ),
        "interval_valid_mask": torch.from_numpy(
            np.asarray(
                [interval_valid[current] for _previous, current in selected_pairs],
                dtype=np.bool_,
            )
        ),
        "acceleration_valid_mask": torch.from_numpy(
            np.asarray(
                [acceleration_valid[current] for _previous, current in selected_pairs],
                dtype=np.bool_,
            )
        ),
    }


def _select_robot_data(
    dataset: RoboTwinSourceDataset,
    qpos_data: torch.Tensor,
    action_indices: List[int],
    initial_state_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_state_idx >= len(qpos_data):
        initial_state_idx = len(qpos_data) - 1
    initial_state = qpos_data[initial_state_idx].float()
    actions = []
    for idx in action_indices:
        if idx >= len(qpos_data):
            raise IndexError(
                f"Action index {idx} out of bounds for qpos data length {len(qpos_data)}"
            )
        actions.append(qpos_data[idx])
    action_sequence = torch.stack(actions).float()
    if dataset.action_normalizer is None:
        raise RuntimeError("DynamicWAM packing dataset has no action normalizer")
    initial_state = dataset.action_normalizer.normalize(initial_state)
    action_sequence = dataset.action_normalizer.normalize(action_sequence)
    return initial_state.contiguous(), action_sequence.contiguous()


class _TrainSamplePrepDataset(Dataset):
    """Loads raw frames and robot tensors for final shard construction."""

    def __init__(
        self,
        dataset: RoboTwinSourceDataset,
        *,
        max_qpos_cache: int = 8,
        head_flow: Dict[str, Any],
        lang_group_ids: Dict[int, int],
    ):
        self.dataset = dataset
        self.max_qpos_cache = max(1, int(max_qpos_cache))
        self._qpos_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._motion_cache: OrderedDict[
            str,
            Dict[str, np.ndarray],
        ] = OrderedDict()
        self.head_flow = dict(head_flow)
        self.lang_group_ids = dict(lang_group_ids)
        if set(self.lang_group_ids) != set(range(len(self.dataset.all_episodes))):
            raise ValueError("language group mapping must cover every source episode")

    def __len__(self) -> int:
        return len(self.dataset.sample_index)

    def _qpos(self, path: str) -> torch.Tensor:
        cached = self._qpos_cache.get(path)
        if cached is not None:
            self._qpos_cache.move_to_end(path)
            return cached
        tensor = _load_qpos_tensor(path)
        self._qpos_cache[path] = tensor
        self._qpos_cache.move_to_end(path)
        while len(self._qpos_cache) > self.max_qpos_cache:
            self._qpos_cache.popitem(last=False)
        return tensor

    def __getitem__(self, sample_id: int) -> Dict[str, Any]:
        sample_id = int(sample_id)
        episode_index, condition_frame_idx = self.dataset.sample_index[sample_id]
        episode_data = self.dataset.get_episode(episode_index)
        total_frames = self.dataset._effective_episode_frame_count(episode_data)
        resolved_condition_idx, video_indices, action_indices = (
            self.dataset._calculate_sampling_indices(
                total_frames=total_frames,
                condition_frame_idx=condition_frame_idx,
            )
        )
        sampled_frames = load_video_frames(
            episode_data["video_path"],
            [resolved_condition_idx, *video_indices],
            self.dataset.video_size,
        )
        initial_state, action_sequence = _select_robot_data(
            self.dataset,
            self._qpos(episode_data["qpos_path"]),
            action_indices,
            resolved_condition_idx,
        )
        absolute_motion = _load_absolute_motion(
            cache_root=self.head_flow["cache_root"],
            source_root=self.head_flow["source_root"],
            video_path=episode_data["video_path"],
            condition_idx=resolved_condition_idx,
            k=int(self.head_flow["k"]),
            policy_stride=int(self.head_flow["policy_stride"]),
            global_downsample_rate=int(self.head_flow["global_downsample_rate"]),
            params_json=str(self.head_flow["params_json"]),
            expected_frame_count=self.dataset._get_video_frame_count_cached(
                episode_data["video_path"]
            ),
            video_size=self.dataset.video_size,
            episode_cache=self._motion_cache,
            max_cached_episodes=self.max_qpos_cache,
        )
        return {
            "head_flow": absolute_motion["frames"],
            "absolute_motion_features": absolute_motion["features"],
            "absolute_motion_interval_valid_mask": absolute_motion[
                "interval_valid_mask"
            ],
            "absolute_motion_acceleration_valid_mask": absolute_motion[
                "acceleration_valid_mask"
            ],
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "episode_index": torch.tensor(int(episode_index), dtype=torch.long),
            "condition_frame_idx": torch.tensor(
                int(resolved_condition_idx), dtype=torch.long
            ),
            "video_indices": torch.tensor(
                [int(value) for value in video_indices], dtype=torch.long
            ),
            "action_indices": torch.tensor(
                [int(value) for value in action_indices], dtype=torch.long
            ),
            "lang_group_id": torch.tensor(
                int(self.lang_group_ids[episode_index]),
                dtype=torch.long,
            ),
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "first_frame": sampled_frames[0],
            "video_frames": sampled_frames[1:],
            "task_name": episode_data["task_name"],
        }


def _collate_prep_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _sample_ids_sha256(sample_ids: torch.Tensor) -> str:
    ids = sample_ids.detach().cpu().to(dtype=torch.int64).contiguous()
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _validate_shard_tensors(
    shard_index: int, sample_start: int, tensors: Dict[str, torch.Tensor]
) -> None:
    if not tensors:
        raise ValueError(f"Shard {shard_index} has no tensors")
    sample_count = int(next(iter(tensors.values())).shape[0])
    if sample_count <= 0:
        raise ValueError(f"Shard {shard_index} is empty")
    for key, value in tensors.items():
        if int(value.shape[0]) != sample_count:
            raise ValueError(
                f"Shard {shard_index} tensor {key!r} has first dim {value.shape[0]}, expected {sample_count}"
            )
    sample_ids = tensors["sample_ids"].to(dtype=torch.long)
    expected = torch.arange(sample_start, sample_start + sample_count, dtype=torch.long)
    if not torch.equal(sample_ids.cpu(), expected):
        raise ValueError(
            f"Shard {shard_index} sample_ids are not contiguous from {sample_start}"
        )


def _write_train_shard(
    output_root: Path,
    shard_index: int,
    sample_start: int,
    tensors: Dict[str, torch.Tensor],
) -> int:
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - environment dependency
        raise RuntimeError(
            "Packing train dataset requires safetensors to be installed"
        ) from exc

    _validate_shard_tensors(shard_index, sample_start, tensors)
    shard_path = (
        output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.safetensors"
    )
    shard_meta_path = (
        output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_shard_path = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
    try:
        save_file(
            {key: value.contiguous() for key, value in tensors.items()},
            str(tmp_shard_path),
        )
        tmp_shard_path.replace(shard_path)
    finally:
        tmp_shard_path.unlink(missing_ok=True)
    size_bytes = int(shard_path.stat().st_size)
    sample_count = int(tensors["sample_ids"].shape[0])
    _write_json_atomic(
        shard_meta_path,
        {
            "shard_index": int(shard_index),
            "sample_start": int(sample_start),
            "sample_count": int(sample_count),
            "sample_ids_sha256": _sample_ids_sha256(tensors["sample_ids"]),
            "size_bytes": int(size_bytes),
            "tensor_bytes": int(
                sum(_tensor_bytes(value) for value in tensors.values())
            ),
            "keys": {
                key: {
                    "shape": [int(dim) for dim in value.shape],
                    "dtype": str(value.dtype).replace("torch.", ""),
                }
                for key, value in tensors.items()
            },
        },
    )
    return size_bytes


def _progress_path(output_root: Path, rank: int) -> Path:
    return output_root / f".build_progress_rank{int(rank):03d}.json"


def _write_progress(
    output_root: Path,
    rank: int,
    *,
    samples_done: int,
    shards_done: int,
    written_bytes: int,
    current_task: str = "",
) -> None:
    _write_json_atomic(
        _progress_path(output_root, rank),
        {
            "rank": int(rank),
            "samples_done": int(samples_done),
            "shards_done": int(shards_done),
            "written_bytes": int(written_bytes),
            "current_task": str(current_task),
            "updated_at_unix": time.time(),
        },
    )


def _read_progress_totals(output_root: Path, world_size: int) -> Dict[str, Any]:
    totals: Dict[str, Any] = {
        "samples_done": 0,
        "shards_done": 0,
        "written_bytes": 0,
        "current_tasks": [],
    }
    for rank in range(world_size):
        path = _progress_path(output_root, rank)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        totals["samples_done"] += int(payload.get("samples_done", 0))
        totals["shards_done"] += int(payload.get("shards_done", 0))
        totals["written_bytes"] += int(payload.get("written_bytes", 0))
        current_task = str(payload.get("current_task", ""))
        if current_task:
            totals["current_tasks"].append(current_task)
    return totals


def _cleanup_progress_files(output_root: Path, world_size: int) -> None:
    for rank in range(world_size):
        path = _progress_path(output_root, rank)
        if path.exists():
            path.unlink()
        for tmp_path in output_root.glob(f"{path.name}.tmp.*"):
            tmp_path.unlink()


def _owned_sample_indices(
    sample_count: int, shard_size: int, rank: int, world_size: int
) -> List[int]:
    indices: List[int] = []
    for shard_start in range(0, sample_count, shard_size):
        shard_index = shard_start // shard_size
        if shard_index % world_size != rank:
            continue
        shard_end = min(shard_start + shard_size, sample_count)
        indices.extend(range(shard_start, shard_end))
    return indices


def _write_main_shards(
    dataset: RoboTwinSourceDataset,
    output_root: Path,
    *,
    shard_size: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    lang_group_ids: Dict[int, int],
) -> Dict[str, int]:
    sample_count = len(dataset.sample_index)
    shard_count = (sample_count + shard_size - 1) // shard_size
    owned_indices = _owned_sample_indices(sample_count, shard_size, rank, world_size)
    _write_progress(output_root, rank, samples_done=0, shards_done=0, written_bytes=0)

    progress_bar = None
    last_global_done = 0
    if rank == 0:
        progress_bar = tqdm(
            total=sample_count,
            desc="[4/6] encoding videos + packing shards",
            unit="sample",
        )

    def refresh_progress(force: bool = False) -> None:
        nonlocal last_global_done
        if rank != 0 or progress_bar is None:
            return
        totals = _read_progress_totals(output_root, world_size)
        global_done = min(int(totals["samples_done"]), sample_count)
        delta = global_done - last_global_done
        if delta > 0:
            progress_bar.update(delta)
            last_global_done = global_done
        if force or delta > 0:
            progress_bar.set_postfix(
                {
                    "shards": f"{int(totals['shards_done'])}/{shard_count}",
                    "GB": f"{float(totals['written_bytes']) / (1024**3):.2f}",
                    "rank": f"0/{world_size}",
                }
            )

    encoded = 0
    written_shards = 0
    written_bytes = 0
    current_task = ""

    if owned_indices:
        logger.info(
            "Rank %d/%d packing %d samples across shard_size=%d on %s",
            rank,
            world_size,
            len(owned_indices),
            shard_size,
            args.device,
        )
        head_flow_cfg = {
            "cache_root": str(args.head_flow_cache_root),
            "source_root": str(args.head_flow_source_root),
            "k": int(args.head_flow_k),
            "policy_stride": int(args.head_flow_policy_stride),
            "global_downsample_rate": int(dataset.global_downsample_rate),
            "compute_size": tuple(args.head_flow_compute_size),
            "container_fps": float(args.head_flow_container_fps),
            "normalization_percentile": float(args.head_flow_normalization_percentile),
            "farneback": dict(args.head_flow_farneback),
            "quality": dict(args.head_flow_quality),
        }
        head_flow_cfg["params_json"] = json.dumps(
            build_flow_cache_parameters(
                head_flow_config={
                    "count": head_flow_cfg["k"],
                    "policy_stride": head_flow_cfg["policy_stride"],
                    "compute_size": list(head_flow_cfg["compute_size"]),
                    "container_fps": head_flow_cfg["container_fps"],
                    "normalization_percentile": head_flow_cfg[
                        "normalization_percentile"
                    ],
                    "farneback": head_flow_cfg["farneback"],
                    "quality": head_flow_cfg["quality"],
                },
                global_downsample_rate=head_flow_cfg["global_downsample_rate"],
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        prep_dataset = _TrainSamplePrepDataset(
            dataset,
            max_qpos_cache=int(args.qpos_cache_size),
            head_flow=head_flow_cfg,
            lang_group_ids=lang_group_ids,
        )
        loader_kwargs: Dict[str, Any] = {
            "batch_size": int(args.batch_size),
            "shuffle": False,
            "num_workers": int(args.num_workers),
            "pin_memory": bool(args.pin_memory),
            "collate_fn": _collate_prep_batch,
        }
        if int(args.num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
        loader = DataLoader(
            Subset(prep_dataset, owned_indices),
            **loader_kwargs,
        )
        from dynamicwam.vendor.wan.modules.vae2_2 import Wan2_2_VAE

        vae = Wan2_2_VAE(
            vae_pth=vae_path_from_config(args.config_payload), device=args.device
        )
        dtype = dtype_from_config(args.config_payload)
        future_video_size = future_video_size_from_config(args.config_payload)

        current_shard_index: Optional[int] = None
        current_sample_start: Optional[int] = None
        current: Dict[str, List[torch.Tensor]] = {}
        last_progress_write = time.monotonic()

        def reset_current() -> None:
            nonlocal current
            current = {
                "sample_ids": [],
                "episode_indices": [],
                "condition_frame_indices": [],
                "video_indices": [],
                "action_indices": [],
                "lang_group_ids": [],
                "initial_states": [],
                "action_sequences": [],
                "condition_latents": [],
                "absolute_motion_features": [],
                "absolute_motion_interval_valid_masks": [],
                "absolute_motion_acceleration_valid_masks": [],
            }
            current["future_latents"] = []

        def flush_current() -> None:
            nonlocal \
                current_shard_index, \
                current_sample_start, \
                written_shards, \
                written_bytes
            if current_shard_index is None or current_sample_start is None:
                return
            if not current["sample_ids"]:
                return
            tensors = {
                key: torch.stack(values, dim=0).contiguous()
                for key, values in current.items()
            }
            size_bytes = _write_train_shard(
                output_root,
                current_shard_index,
                current_sample_start,
                tensors,
            )
            written_shards += 1
            written_bytes += size_bytes
            current_shard_index = None
            current_sample_start = None
            reset_current()

        reset_current()
        for batch in loader:
            batch_count = int(batch["first_frame"].shape[0])
            current_task = str(batch.get("task_name", [""])[-1])
            first_frame = batch["first_frame"].to(
                args.device, dtype=dtype, non_blocking=True
            )
            video_frames = batch["video_frames"].to(
                args.device, dtype=dtype, non_blocking=True
            )
            first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
            head_flow_pixels = (
                batch["head_flow"].to(args.device, non_blocking=True).float() / 255.0
            ).permute(0, 4, 1, 2, 3).to(dtype=dtype) * 2.0 - 1.0

            _cuda_synchronize(args.device)
            with torch.no_grad():
                cond_pixels = torch.cat(
                    [head_flow_pixels, first_frame_norm],
                    dim=2,
                )
                condition_latent = vae.encode(cond_pixels)
                future_full_video = _build_low_resolution_full_video(
                    first_frame,
                    video_frames,
                    future_video_size,
                )
                future_full_latent = vae.encode(future_full_video)
                future_latent = future_full_latent[:, :, 1:].contiguous()
            _cuda_synchronize(args.device)

            condition_latent_cpu = condition_latent.detach().cpu()
            latent_cpu = future_latent.detach().cpu()
            for row in range(batch_count):
                sample_id = int(batch["sample_id"][row].item())
                shard_index = sample_id // shard_size
                shard_sample_start = shard_index * shard_size
                if (
                    current_shard_index is not None
                    and shard_index != current_shard_index
                ):
                    flush_current()
                if current_shard_index is None:
                    current_shard_index = shard_index
                    current_sample_start = shard_sample_start

                current["sample_ids"].append(
                    batch["sample_id"][row].to(dtype=torch.long).cpu()
                )
                current["episode_indices"].append(
                    batch["episode_index"][row].to(dtype=torch.long).cpu()
                )
                current["condition_frame_indices"].append(
                    batch["condition_frame_idx"][row].to(dtype=torch.long).cpu()
                )
                current["video_indices"].append(
                    batch["video_indices"][row].to(dtype=torch.long).cpu()
                )
                current["action_indices"].append(
                    batch["action_indices"][row].to(dtype=torch.long).cpu()
                )
                current["lang_group_ids"].append(
                    batch["lang_group_id"][row].to(dtype=torch.long).cpu()
                )
                current["initial_states"].append(batch["initial_state"][row].cpu())
                current["action_sequences"].append(batch["action_sequence"][row].cpu())
                current["future_latents"].append(latent_cpu[row])
                current["condition_latents"].append(condition_latent_cpu[row])
                current["absolute_motion_features"].append(
                    batch["absolute_motion_features"][row].to(dtype=torch.float32).cpu()
                )
                current["absolute_motion_interval_valid_masks"].append(
                    batch["absolute_motion_interval_valid_mask"][row]
                    .to(dtype=torch.bool)
                    .cpu()
                )
                current["absolute_motion_acceleration_valid_masks"].append(
                    batch["absolute_motion_acceleration_valid_mask"][row]
                    .to(dtype=torch.bool)
                    .cpu()
                )

            encoded += batch_count
            now = time.monotonic()
            if now - last_progress_write >= max(
                0.5, float(args.progress_interval_seconds)
            ):
                _write_progress(
                    output_root,
                    rank,
                    samples_done=encoded,
                    shards_done=written_shards,
                    written_bytes=written_bytes,
                    current_task=current_task,
                )
                refresh_progress()
                last_progress_write = now
        flush_current()

    _write_progress(
        output_root,
        rank,
        samples_done=encoded,
        shards_done=written_shards,
        written_bytes=written_bytes,
        current_task=current_task,
    )
    refresh_progress(force=True)

    if rank == 0:
        while True:
            totals = _read_progress_totals(output_root, world_size)
            refresh_progress(force=True)
            if int(totals["samples_done"]) >= sample_count:
                break
            time.sleep(max(0.5, float(args.progress_interval_seconds)))
        if progress_bar is not None:
            progress_bar.close()

    _barrier()
    if rank == 0:
        totals = _read_progress_totals(output_root, world_size)
        return {
            "sample_count": int(sample_count),
            "shard_count": int(shard_count),
            "written_bytes": int(totals["written_bytes"]),
        }
    return {
        "sample_count": int(sample_count),
        "shard_count": int(shard_count),
        "written_bytes": int(written_bytes),
    }


def _verify_generated_shards(
    output_root: Path, *, sample_count: int, shard_count: int, shard_size: int
) -> None:
    required_keys = {
        "sample_ids",
        "episode_indices",
        "condition_frame_indices",
        "video_indices",
        "action_indices",
        "lang_group_ids",
        "initial_states",
        "action_sequences",
        "condition_latents",
        "absolute_motion_features",
        "absolute_motion_interval_valid_masks",
        "absolute_motion_acceleration_valid_masks",
    }
    latent_key = "future_latents"
    covered = 0
    for shard_index in tqdm(
        range(shard_count), desc="[5/6] verifying shards", unit="shard"
    ):
        meta_path = (
            output_root / TRAIN_DATASET_SHARD_DIR / f"shard_{shard_index:06d}.json"
        )
        data_path = (
            output_root
            / TRAIN_DATASET_SHARD_DIR
            / f"shard_{shard_index:06d}.safetensors"
        )
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Missing generated train shard: {data_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_start = int(meta["sample_start"])
        shard_sample_count = int(meta["sample_count"])
        expected_start = shard_index * shard_size
        expected_count = min(shard_size, sample_count - expected_start)
        if sample_start != expected_start or shard_sample_count != expected_count:
            raise RuntimeError(
                f"Shard {shard_index} range mismatch: got start={sample_start} count={shard_sample_count}, "
                f"expected start={expected_start} count={expected_count}"
            )
        keys = set(meta.get("keys", {}).keys())
        missing = required_keys - keys
        if missing:
            raise RuntimeError(
                f"Shard {shard_index} metadata missing keys: {sorted(missing)}"
            )
        if latent_key not in keys:
            raise RuntimeError(f"Shard {shard_index} metadata missing {latent_key}")
        for key in required_keys | {latent_key}:
            shape = meta["keys"][key]["shape"]
            if int(shape[0]) != shard_sample_count:
                raise RuntimeError(
                    f"Shard {shard_index} key {key} first dim {shape[0]} != {shard_sample_count}"
                )
        covered += shard_sample_count
    if covered != sample_count:
        raise RuntimeError(
            f"Generated shards cover {covered} samples, expected {sample_count}"
        )


def _prepare_output_root(output_root: Path, rank: int, *, overwrite: bool) -> Path:
    temp_root = output_root.with_name(f"{output_root.name}.tmp_build")
    if rank == 0:
        if output_root.exists() and not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
    _barrier()
    return temp_root


def _finalize_output_root(
    temp_root: Path, output_root: Path, rank: int, *, overwrite: bool
) -> None:
    _barrier()
    if rank == 0:
        if output_root.exists():
            if not overwrite:
                raise FileExistsError(f"Output root already exists: {output_root}")
            shutil.rmtree(output_root)
        temp_root.replace(output_root)
    _barrier()


def run(args: argparse.Namespace) -> None:
    profile = load_profile(args.config)
    config = profile.packing_config()
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.device is not None:
        config["build"]["device"] = str(args.device)
    if args.batch_size is not None:
        config["build"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config["build"]["num_workers"] = int(args.num_workers)
    args.config_payload = config
    require_exact_keys(
        config,
        {
            "output_root",
            "external_assets_manifest",
            "source",
            "action_stats",
            "head_flow",
            "latent",
            "language",
            "dataset",
            "build",
        },
        "DynamicWAM pack config",
    )
    action_stats_cfg = require_exact_keys(
        config["action_stats"],
        {"source_path", "sha256", "qpos_dim"},
        "absolute-motion pack action_stats",
    )
    config["action_stats"] = action_stats_cfg
    build_cfg = require_exact_keys(
        config["build"],
        {
            "device",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "qpos_cache_size",
            "progress_interval_seconds",
            "log_level",
            "dist_timeout_minutes",
        },
        "DynamicWAM pack build",
    )
    language_cfg = require_exact_keys(
        config["language"],
        {"items_per_shard"},
        "DynamicWAM pack language",
    )
    latent_cfg = require_exact_keys(
        config["latent"],
        {"vae_path", "precision", "shard_size", "future_video_size"},
        "DynamicWAM pack latent",
    )
    if latent_cfg["precision"] != "bfloat16":
        raise ValueError("DynamicWAM pack latent.precision must be bfloat16")
    head_flow_config = require_exact_keys(
        config["head_flow"],
        {
            "cache_root",
            "source_root",
            "motion_stats_path",
            "count",
            "policy_stride",
            "compute_size",
            "container_fps",
            "normalization_percentile",
            "farneback",
            "quality",
        },
        "DynamicWAM pack head_flow",
    )
    args.head_flow_cache_root = str(head_flow_config["cache_root"])
    args.head_flow_source_root = str(head_flow_config["source_root"])
    args.motion_stats_path = str(head_flow_config["motion_stats_path"])
    args.head_flow_k = int(head_flow_config["count"])
    args.head_flow_policy_stride = int(head_flow_config["policy_stride"])
    args.head_flow_compute_size = tuple(
        int(value) for value in head_flow_config["compute_size"]
    )
    args.head_flow_container_fps = float(head_flow_config["container_fps"])
    args.head_flow_normalization_percentile = float(
        head_flow_config["normalization_percentile"]
    )
    args.head_flow_farneback = dict(head_flow_config["farneback"])
    args.head_flow_quality = dict(head_flow_config["quality"])

    args.device = str(build_cfg["device"])
    args.batch_size = int(build_cfg["batch_size"])
    args.num_workers = int(build_cfg["num_workers"])
    args.pin_memory = bool(build_cfg["pin_memory"])
    args.persistent_workers = bool(build_cfg["persistent_workers"])
    args.prefetch_factor = int(build_cfg["prefetch_factor"])
    args.qpos_cache_size = int(build_cfg["qpos_cache_size"])
    args.shard_size = int(latent_cfg["shard_size"])
    args.lang_items_per_shard = int(language_cfg["items_per_shard"])
    args.log_level = str(build_cfg["log_level"])
    args.dist_timeout_minutes = int(build_cfg["dist_timeout_minutes"])
    args.progress_interval_seconds = float(build_cfg["progress_interval_seconds"])

    rank, world_size, device = _distributed_info(
        str(args.device), args.dist_timeout_minutes
    )
    args.device = device
    setup_logging(args.log_level, rank=rank)
    if rank == 0:
        verify_wan_assets(
            root=Path(latent_cfg["vae_path"]).parent,
            manifest_path=Path(config["external_assets_manifest"]),
            purpose="packing",
        )
    _barrier()

    output_root_value = output_root_from_config(config)
    if not output_root_value:
        raise ValueError(
            "Provide --output-root or set output_root in the dataset build config"
        )
    output_root = Path(output_root_value).expanduser()
    temp_root = _prepare_output_root(output_root, rank, overwrite=bool(args.overwrite))
    if rank == 0:
        write_config_snapshot(
            temp_root / "config_audit",
            profile=profile,
            label="packing",
            resolved_config={
                "packing": config,
                "launch": {"overwrite": bool(args.overwrite)},
            },
        )
    _barrier()

    _phase(rank, 1, 6, "preparing output and action stats")
    if rank == 0:
        _prepare_action_stats(config["action_stats"], temp_root)
        _prepare_motion_stats(args.motion_stats_path, temp_root)
    _barrier()

    action_stats_path = temp_root / TRAIN_DATASET_ACTION_STATS
    if not action_stats_path.is_file():
        raise FileNotFoundError(
            f"DynamicWAM action stats were not created: {action_stats_path}"
        )
    action_stats_sha256 = _sha256_file(action_stats_path)
    if action_stats_sha256 != str(config["action_stats"]["sha256"]):
        raise RuntimeError("packed action statistics differ from the pinned source")
    motion_stats_path = temp_root / TRAIN_DATASET_MOTION_STATS
    if not motion_stats_path.is_file():
        raise FileNotFoundError(
            f"absolute motion stats were not created: {motion_stats_path}"
        )

    _phase(rank, 2, 6, "scanning preprocessed RoboTwin episodes")
    dataset = build_robotwin_packing_dataset(
        config,
        action_stats_path=str(action_stats_path),
    )
    if args.shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {args.shard_size}")
    raw_offsets(
        history_count=args.head_flow_k,
        policy_stride=args.head_flow_policy_stride,
        global_downsample_rate=dataset.global_downsample_rate,
    )

    if rank == 0:
        logger.info("Final train dataset output: %s", output_root)
        logger.info("Temporary build root      : %s", temp_root)
        logger.info(
            "Samples=%d episodes=%d shard_size=%d world_size=%d",
            len(dataset),
            len(dataset.all_episodes),
            args.shard_size,
            world_size,
        )
        logger.info("VAE: %s", vae_path_from_config(config))
        logger.info("VAE config: %s", compact_vae_config(config))

    _phase(rank, 3, 6, "packing language bank and manifests")
    lang_group_ids = _language_group_ids(dataset)
    if rank == 0:
        written_group_ids, language_dtype = _write_language_bank(
            dataset,
            temp_root,
            lang_items_per_shard=max(1, int(args.lang_items_per_shard)),
            lang_group_ids=lang_group_ids,
        )
        if written_group_ids != lang_group_ids:
            raise RuntimeError("language bank returned a different group mapping")
        manifest_info = _write_manifests_and_stats(
            dataset,
            temp_root,
            lang_group_ids=lang_group_ids,
        )
    else:
        manifest_info = {}
    _barrier()

    _phase(rank, 4, 6, "encoding videos and writing final train shards")
    shard_info = _write_main_shards(
        dataset,
        temp_root,
        shard_size=int(args.shard_size),
        args=args,
        rank=rank,
        world_size=world_size,
        lang_group_ids=lang_group_ids,
    )
    _barrier()

    if rank == 0:
        _phase(rank, 5, 6, "verifying generated dataset")
        _verify_generated_shards(
            temp_root,
            sample_count=int(shard_info["sample_count"]),
            shard_count=int(shard_info["shard_count"]),
            shard_size=int(args.shard_size),
        )
        statistics = manifest_info["statistics"]
        train_dataset_cfg = _path_independent_training_dataset_config(
            training_dataset_config(config)
        )
        with motion_stats_path.open("r", encoding="utf-8") as stream:
            motion_stats_payload = validate_motion_statistics(json.load(stream))
        absolute_motion_metadata = build_checkpoint_motion_metadata(
            history_count=int(args.head_flow_k),
            statistics=motion_stats_payload,
            statistics_sha256=_hash_json_payload(motion_stats_payload),
            head_flow_config={
                "count": int(args.head_flow_k),
                "policy_stride": int(args.head_flow_policy_stride),
                "compute_size": [int(value) for value in args.head_flow_compute_size],
                "normalization_percentile": float(
                    args.head_flow_normalization_percentile
                ),
                "farneback": dict(args.head_flow_farneback),
                "quality": dict(args.head_flow_quality),
            },
        )
        absolute_motion_metadata["statistics_file"] = TRAIN_DATASET_MOTION_STATS
        sampling_contract = {
            "strategy": "episode_balanced_without_replacement",
            "samples_per_episode": int(train_dataset_cfg["samples_per_episode"]),
            "seed": int(train_dataset_cfg["sampler_seed"]),
        }
        fingerprint_payload = {
            "format": TRAIN_DATASET_FORMAT,
            "version": TRAIN_DATASET_VERSION,
            "manifest_sha256": manifest_info["manifest_sha256"],
            "sample_count": int(shard_info["sample_count"]),
            "episode_count": len(dataset.all_episodes),
            "shard_size": int(args.shard_size),
            "data": {
                "num_video_frames": int(dataset.num_video_frames),
                "video_size": [int(v) for v in dataset.video_size],
                "global_downsample_rate": int(dataset.global_downsample_rate),
                "video_action_freq_ratio": int(dataset.video_action_freq_ratio),
                "action_chunk_size": int(dataset.action_chunk_size),
                "action_normalization": _action_normalization_metadata(
                    dataset.normalization_epsilon,
                    stats_sha256=action_stats_sha256,
                ),
            },
            "latent": {
                "precision": str(compact_vae_config(config)["precision"]),
                "condition_latent_source": (
                    f"head_flow_{int(args.head_flow_k)}_plus_current_frame"
                ),
                "future_video_size": [
                    int(value) for value in future_video_size_from_config(config)
                ],
                "head_flow": (
                    {
                        "k": int(args.head_flow_k),
                        "temporal_contract": TEMPORAL_CONTRACT,
                        "policy_stride": int(args.head_flow_policy_stride),
                        "global_downsample_rate": int(dataset.global_downsample_rate),
                        "raw_stride": raw_stride(
                            policy_stride=args.head_flow_policy_stride,
                            global_downsample_rate=dataset.global_downsample_rate,
                        ),
                        "endpoint_rule": ENDPOINT_RULE,
                        "raw_offsets": raw_offsets(
                            history_count=args.head_flow_k,
                            policy_stride=args.head_flow_policy_stride,
                            global_downsample_rate=dataset.global_downsample_rate,
                        ),
                        "raw_index_unit": "converted_video_frame",
                        "source_view": "head",
                        "compute_size": [int(v) for v in args.head_flow_compute_size],
                        "container_fps": float(args.head_flow_container_fps),
                        "physical_timestamps_available": True,
                        "timestamp_source": TIMESTAMP_SOURCE,
                        "spatial_unit": SPATIAL_UNIT,
                        "motion_feature_names": list(MOTION_FEATURE_NAMES),
                        "farneback": dict(args.head_flow_farneback),
                        "quality": dict(args.head_flow_quality),
                        "rgb_normalization": {
                            "type": "per_map_percentile",
                            "percentile": float(
                                args.head_flow_normalization_percentile
                            ),
                        },
                    }
                ),
            },
            "language": {
                "dtype": language_dtype,
                "items_per_shard": int(args.lang_items_per_shard),
            },
            "sampling": sampling_contract,
            "absolute_motion": absolute_motion_metadata,
        }
        metadata = {
            "format": TRAIN_DATASET_FORMAT,
            "version": TRAIN_DATASET_VERSION,
            "sample_layout": "exhaustive",
            "sample_count": int(shard_info["sample_count"]),
            "episode_count": len(dataset.all_episodes),
            "shard_count": int(shard_info["shard_count"]),
            "shard_size": int(args.shard_size),
            "dataset_fingerprint": _hash_json_payload(fingerprint_payload),
            "manifest_sha256": manifest_info["manifest_sha256"],
            "created_at_unix": time.time(),
            "written_bytes": int(shard_info["written_bytes"]),
            "statistics": {
                "path": TRAIN_DATASET_STATS,
                "task_count": int(statistics["task_count"]),
                "episode_count": int(statistics["episode_count"]),
                "sample_count": int(statistics["sample_count"]),
                "error_count": int(statistics["error_count"]),
                "effective_frame_count": statistics["effective_frame_count"],
                "sample_count_per_episode": statistics["sample_count_per_episode"],
            },
            "storage": {
                "shard_dir": TRAIN_DATASET_SHARD_DIR,
                "shard_size": int(args.shard_size),
            },
            "training_dataset": train_dataset_cfg,
            "data": fingerprint_payload["data"],
            "latent": fingerprint_payload["latent"],
            "language": {
                "storage": "bank",
                "policy": "deterministic",
                "dtype": language_dtype,
            },
            "sampling": sampling_contract,
            "absolute_motion": absolute_motion_metadata,
        }
        _phase(rank, 6, 6, "writing dataset metadata")
        _write_json_atomic(temp_root / TRAIN_DATASET_METADATA, metadata)
        _cleanup_progress_files(temp_root, world_size)
        logger.info(
            "Packed train dataset: samples=%d shards=%d bytes=%.2f GB",
            shard_info["sample_count"],
            shard_info["shard_count"],
            shard_info["written_bytes"] / (1024**3),
        )

    _finalize_output_root(temp_root, output_root, rank, overwrite=bool(args.overwrite))
    if rank == 0:
        logger.info("Final dataset is ready: %s", output_root)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the DynamicWAM packed dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS
    )
    return parser


def main() -> None:
    run(_build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
