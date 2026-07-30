from __future__ import annotations

import hashlib
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import Dataset, Sampler

from dynamicwam.absolute_motion import (
    CHECKPOINT_MOTION_METADATA_KEYS,
    validate_checkpoint_motion_metadata,
    validate_motion_statistics,
)

TRAIN_DATASET_FORMAT = "dynamicwam_absolute_motion_dataset"
TRAIN_DATASET_VERSION = 2
TRAIN_DATASET_METADATA = "dataset.json"
TRAIN_DATASET_STATS = "stats.json"
TRAIN_DATASET_ACTION_STATS = "action_stats.json"
TRAIN_DATASET_MOTION_STATS = "motion_stats.json"
TRAIN_DATASET_SHARD_DIR = "shards"
TRAIN_DATASET_LANG_DIR = "lang"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json_payload(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class _SafetensorShardCache:
    """Small per-worker LRU cache for safetensors shard handles."""

    def __init__(self, max_open_shards: int):
        self.max_open_shards = max(1, int(max_open_shards))
        self._handles: OrderedDict[int, Any] = OrderedDict()

    def get(self, shard_index: int, path: Path):
        shard_index = int(shard_index)
        handle = self._handles.get(shard_index)
        if handle is not None:
            self._handles.move_to_end(shard_index)
            return handle

        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - environment dependency
            raise RuntimeError(
                "DynamicWAM packed dataset requires safetensors"
            ) from exc

        handle = safe_open(str(path), framework="pt", device="cpu")
        self._handles[shard_index] = handle
        self._handles.move_to_end(shard_index)
        while len(self._handles) > self.max_open_shards:
            self._handles.popitem(last=False)
        return handle


class AbsoluteMotionEpisodeSampler(Sampler[int]):
    """Draw the exact configured number of unique samples from every episode."""

    def __init__(
        self,
        episodes: List[Dict[str, int]],
        *,
        samples_per_episode: int,
        seed: int = 0,
    ):
        self.episodes = [
            {
                "first_sample_id": int(episode["first_sample_id"]),
                "sample_count": int(episode["sample_count"]),
            }
            for episode in episodes
            if int(episode.get("sample_count", 0)) > 0
        ]
        if not self.episodes:
            raise ValueError(
                "absolute-motion sampling requires at least one usable episode"
            )
        self.samples_per_episode = int(samples_per_episode)
        if self.samples_per_episode <= 0:
            raise ValueError(
                f"samples_per_episode must be positive, got {samples_per_episode}"
            )
        too_short = [
            episode["sample_count"]
            for episode in self.episodes
            if episode["sample_count"] < self.samples_per_episode
        ]
        if too_short:
            raise ValueError(
                "absolute-motion sampling requires at least "
                f"{self.samples_per_episode} unique samples per episode; "
                f"minimum={min(too_short)}"
            )
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        sample_ids: List[int] = []
        for episode in self.episodes:
            first_sample_id = int(episode["first_sample_id"])
            sample_count = int(episode["sample_count"])
            offsets = rng.sample(
                range(sample_count),
                self.samples_per_episode,
            )
            sample_ids.extend(first_sample_id + int(offset) for offset in offsets)
        rng.shuffle(sample_ids)
        yield from sample_ids


class PackedAbsoluteMotionDataset(Dataset):
    """Exact-time training dataset backed by one aligned shard stream.

    Conversion-time checks guarantee that each sample's latents, actions, robot
    state, frame indices, and language group id were written together. Runtime
    reading stays lean: no raw RoboTwin scan, no sidecar join, and no per-sample
    joins remain in the training process.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_open_shards: int,
    ):
        self.root = Path(root).expanduser()
        self.metadata = _read_json(self.root / TRAIN_DATASET_METADATA)
        if self.metadata.get("format") != TRAIN_DATASET_FORMAT:
            raise ValueError(
                f"Unsupported train dataset format at {self.root}: {self.metadata.get('format')!r}"
            )
        version = int(self.metadata.get("version", -1))
        if version != TRAIN_DATASET_VERSION:
            raise ValueError(
                f"Unsupported train dataset version at {self.root}: {version}; "
                f"expected {TRAIN_DATASET_VERSION}"
            )
        sampling = self.metadata.get("sampling")
        if (
            not isinstance(sampling, dict)
            or set(sampling) != {"strategy", "samples_per_episode", "seed"}
            or sampling.get("strategy") != "episode_balanced_without_replacement"
            or int(sampling.get("samples_per_episode", 0)) <= 0
        ):
            raise ValueError(
                "packed dataset does not declare the v2 unique episode sampler"
            )
        self.sampling_contract = {
            "strategy": sampling["strategy"],
            "samples_per_episode": int(sampling["samples_per_episode"]),
            "seed": int(sampling["seed"]),
        }
        motion_contract = self.metadata.get("absolute_motion")
        if not isinstance(motion_contract, dict):
            raise ValueError(f"dataset has no absolute_motion contract: {self.root}")
        expected_contract_keys = set(CHECKPOINT_MOTION_METADATA_KEYS) | {
            "statistics_file"
        }
        if set(motion_contract) != expected_contract_keys:
            raise ValueError(
                "absolute_motion metadata differs from the production schema"
            )
        checkpoint_motion = validate_checkpoint_motion_metadata(
            {
                key: value
                for key, value in motion_contract.items()
                if key != "statistics_file"
            }
        )
        if motion_contract["statistics_file"] != TRAIN_DATASET_MOTION_STATS:
            raise ValueError(
                "packed dataset does not use exact DOMINO simulator-time motion"
            )
        self.motion_checkpoint_metadata = checkpoint_motion
        self.motion_history_count = int(checkpoint_motion["history_count"])
        motion_stats_path = self.root / TRAIN_DATASET_MOTION_STATS
        if not motion_stats_path.is_file():
            raise FileNotFoundError(
                f"absolute motion statistics are missing: {motion_stats_path}"
            )
        self.motion_statistics = validate_motion_statistics(
            _read_json(motion_stats_path)
        )
        if _hash_json_payload(self.motion_statistics) != str(
            motion_contract["statistics_sha256"]
        ):
            raise RuntimeError("motion statistics differ from packed dataset metadata")
        if [
            float(value) for value in self.motion_statistics["mean"]
        ] != checkpoint_motion["feature_mean"] or [
            float(value) for value in self.motion_statistics["scale"]
        ] != checkpoint_motion["feature_scale"]:
            raise RuntimeError(
                "motion normalization differs from packed dataset metadata"
            )
        action_stats_path = self.root / TRAIN_DATASET_ACTION_STATS
        action_normalization = self.metadata.get("data", {}).get("action_normalization")
        if (
            not isinstance(action_normalization, dict)
            or action_normalization.get("stats_file") != TRAIN_DATASET_ACTION_STATS
            or not action_stats_path.is_file()
            or _sha256_file(action_stats_path)
            != action_normalization.get("stats_sha256")
        ):
            raise RuntimeError(
                "packed action statistics differ from their pinned identity"
            )
        self.action_stats_sha256 = _sha256_file(action_stats_path)

        dataset_fingerprint = self.metadata.get("dataset_fingerprint")
        if (
            not isinstance(dataset_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", dataset_fingerprint) is None
        ):
            raise ValueError(f"packed dataset has no valid fingerprint: {self.root}")
        self.dataset_identity = {
            "format": TRAIN_DATASET_FORMAT,
            "version": TRAIN_DATASET_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "action_stats_sha256": self.action_stats_sha256,
            "motion_statistics_sha256": checkpoint_motion["statistics_sha256"],
        }

        self.sample_count = int(self.metadata["sample_count"])
        self.shard_size = int(self.metadata["shard_size"])
        self.storage = dict(self.metadata["storage"])
        self.shard_dir = str(self.storage["shard_dir"])
        stats_path = self.root / TRAIN_DATASET_STATS
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"DynamicWAM packed dataset statistics are missing: {stats_path}"
            )
        self.statistics = _read_json(stats_path)
        self.episodes = list(_iter_jsonl(self.root / "episodes.jsonl"))
        self.episodes_with_samples: List[Dict[str, int]] = []
        for episode in self.episodes:
            sample_count = int(episode.get("sample_count", 0))
            if sample_count <= 0:
                continue
            first_sample_id = episode.get("first_sample_id")
            if first_sample_id is None:
                raise ValueError(f"Episode row is missing first_sample_id: {episode}")
            self.episodes_with_samples.append(
                {
                    "episode_id": int(episode["episode_id"]),
                    "first_sample_id": int(first_sample_id),
                    "sample_count": sample_count,
                }
            )

        lang_root = self.root / TRAIN_DATASET_LANG_DIR
        self.lang_metadata = _read_json(lang_root / "lang.json")
        self.lang_groups: Dict[int, List[int]] = {
            int(group["lang_group_id"]): [int(lang_id) for lang_id in group["lang_ids"]]
            for group in self.lang_metadata.get("groups", [])
        }
        self.lang_items: Dict[int, Dict[str, int]] = {
            int(item["lang_id"]): {
                "shard_index": int(item["shard_index"]),
                "local_index": int(item["local_index"]),
            }
            for item in self.lang_metadata.get("items", [])
        }

        self._data_shards = _SafetensorShardCache(max_open_shards=max_open_shards)
        self._lang_shards = _SafetensorShardCache(max_open_shards=max_open_shards)
        self._lang_offsets: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.sample_count

    def make_sampler(
        self,
        *,
        samples_per_episode: int,
        seed: int,
    ) -> Sampler[int]:
        requested = {
            "strategy": "episode_balanced_without_replacement",
            "samples_per_episode": int(samples_per_episode),
            "seed": int(seed),
        }
        if requested != self.sampling_contract:
            raise ValueError(
                "training sampler differs from the packed v2 sampling contract"
            )
        return AbsoluteMotionEpisodeSampler(
            self.episodes_with_samples,
            samples_per_episode=samples_per_episode,
            seed=seed,
        )

    def _data_shard_path(self, shard_index: int) -> Path:
        return self.root / self.shard_dir / f"shard_{int(shard_index):06d}.safetensors"

    def _lang_shard_path(self, shard_index: int) -> Path:
        return (
            self.root
            / TRAIN_DATASET_LANG_DIR
            / TRAIN_DATASET_SHARD_DIR
            / f"shard_{int(shard_index):06d}.safetensors"
        )

    def _load_language_embedding(
        self, lang_group_id: int, sample_id: int
    ) -> torch.Tensor:
        lang_ids = self.lang_groups.get(int(lang_group_id))
        if not lang_ids:
            raise KeyError(f"Missing language group {lang_group_id}")

        lang_id = lang_ids[int(sample_id) % len(lang_ids)]

        item = self.lang_items[int(lang_id)]
        shard_index = int(item["shard_index"])
        local_index = int(item["local_index"])
        shard = self._lang_shards.get(shard_index, self._lang_shard_path(shard_index))
        offsets = self._lang_offsets.get(shard_index)
        if offsets is None:
            offsets = shard.get_tensor("token_offsets").to(dtype=torch.long)
            self._lang_offsets[shard_index] = offsets
        start = int(offsets[local_index].item())
        end = int(offsets[local_index + 1].item())
        return shard.get_slice("token_data")[start:end].clone()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = int(index)
        if sample_id < 0 or sample_id >= self.sample_count:
            raise IndexError(
                f"sample_id {sample_id} out of range for {self.sample_count}"
            )

        shard_index = sample_id // self.shard_size
        local_index = sample_id % self.shard_size
        shard = self._data_shards.get(shard_index, self._data_shard_path(shard_index))

        lang_group_id = int(shard.get_slice("lang_group_ids")[local_index].item())
        sample: Dict[str, Any] = {
            "condition_latent": shard.get_slice("condition_latents")[
                local_index
            ].clone(),
            "future_latent": shard.get_slice("future_latents")[local_index].clone(),
            "initial_state": shard.get_slice("initial_states")[local_index].clone(),
            "action_sequence": shard.get_slice("action_sequences")[local_index].clone(),
            "absolute_motion_features": shard.get_slice("absolute_motion_features")[
                local_index
            ].clone(),
            "absolute_motion_interval_valid_mask": shard.get_slice(
                "absolute_motion_interval_valid_masks"
            )[local_index].clone(),
            "absolute_motion_acceleration_valid_mask": shard.get_slice(
                "absolute_motion_acceleration_valid_masks"
            )[local_index].clone(),
            "episode_index": int(
                shard.get_slice("episode_indices")[local_index].item()
            ),
        }
        sample["language_embedding"] = self._load_language_embedding(
            lang_group_id, sample_id
        )
        return sample


def packed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate the exact production tensor surface."""

    return {
        "condition_latent": torch.stack(
            [sample["condition_latent"] for sample in batch],
            dim=0,
        ),
        "future_latent": torch.stack(
            [sample["future_latent"] for sample in batch],
            dim=0,
        ),
        "initial_state": torch.stack(
            [sample["initial_state"] for sample in batch],
            dim=0,
        ),
        "action_sequence": torch.stack(
            [sample["action_sequence"] for sample in batch],
            dim=0,
        ),
        "absolute_motion_features": torch.stack(
            [sample["absolute_motion_features"] for sample in batch],
            dim=0,
        ),
        "absolute_motion_interval_valid_mask": torch.stack(
            [sample["absolute_motion_interval_valid_mask"] for sample in batch],
            dim=0,
        ),
        "absolute_motion_acceleration_valid_mask": torch.stack(
            [sample["absolute_motion_acceleration_valid_mask"] for sample in batch],
            dim=0,
        ),
        "text_embeddings": [sample["language_embedding"] for sample in batch],
        "episode_index": [int(sample["episode_index"]) for sample in batch],
    }
