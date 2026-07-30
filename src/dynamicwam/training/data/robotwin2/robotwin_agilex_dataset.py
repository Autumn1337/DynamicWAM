"""Deterministic index over the exact-time converted DOMINO source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.image import get_video_frame_count

logger = logging.getLogger(__name__)


class RoboTwinSourceDataset:
    """Index the strict clean+randomized absolute-motion source."""

    def __init__(
        self,
        *,
        dataset_dir: str,
        splits: list[str],
        global_downsample_rate: int,
        video_action_freq_ratio: int,
        num_video_frames: int,
        video_size: tuple[int, int],
        normalization_epsilon: float,
        action_normalizer: RoboTwinQposNormalizer | None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.splits = tuple(str(value) for value in splits)
        self.global_downsample_rate = int(global_downsample_rate)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.num_video_frames = int(num_video_frames)
        self.video_size: tuple[int, int] = (
            int(video_size[0]),
            int(video_size[1]),
        )
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        self.normalization_epsilon = float(normalization_epsilon)
        self._validate_geometry()
        self.action_normalizer = action_normalizer
        self._frame_count_cache: dict[str, int] = {}
        self._qpos_frame_count_cache: dict[str, int] = {}
        self.all_episodes = self._scan_all_episodes()
        self.total_episodes = len(self.all_episodes)
        self.sample_index: list[tuple[int, int]] = []
        self._build_sample_index()

        logger.info(
            "Indexed DynamicWAM RoboTwin source: episodes=%d samples=%d",
            self.total_episodes,
            len(self.sample_index),
        )

    def _validate_geometry(self) -> None:
        if not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("RoboTwin source splits must be non-empty and unique")
        values = (
            self.global_downsample_rate,
            self.video_action_freq_ratio,
            self.num_video_frames,
            *self.video_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError(f"RoboTwin geometry must be positive, got {values}")
        if len(self.video_size) != 2:
            raise ValueError(f"video_size must contain two values: {self.video_size}")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")

    @staticmethod
    def _episode_sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return 0, int(path.stem)
        except ValueError:
            return 1, path.stem

    def _scan_task_folder(
        self,
        task_path: Path,
        split: str,
    ) -> list[dict[str, Any]]:
        qpos_dir = task_path / "qpos"
        videos_dir = task_path / "videos"
        interception_dir = task_path / "interception"
        language_path = self.dataset_dir / "language" / f"{task_path.name}.pt"
        if (
            not all(path.is_dir() for path in (qpos_dir, videos_dir, interception_dir))
            or not language_path.is_file()
        ):
            raise FileNotFoundError(
                f"exact-time converted task is incomplete: {task_path}"
            )

        episodes: list[dict[str, Any]] = []
        for qpos_path in sorted(
            qpos_dir.glob("*.pt"),
            key=self._episode_sort_key,
        ):
            episode_name = qpos_path.stem
            video_path = videos_dir / f"{episode_name}.mp4"
            interception_path = interception_dir / f"{episode_name}.pt"
            if not video_path.is_file() or not interception_path.is_file():
                raise FileNotFoundError(
                    "exact-time converted episode is incomplete: "
                    f"{split}/{task_path.name}/{episode_name}"
                )
            episodes.append(
                {
                    "episode_name": episode_name,
                    "task_name": task_path.name,
                    "split": split,
                    "qpos_path": str(qpos_path),
                    "video_path": str(video_path),
                    "lang_path": str(language_path),
                    "interception_path": str(interception_path),
                }
            )
        return episodes

    def _scan_all_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        task_counts: dict[str, int] = {}
        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.is_dir():
                raise FileNotFoundError(
                    f"DynamicWAM requires converted RoboTwin split: {split_dir}"
                )
            task_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            if not task_dirs:
                raise ValueError(f"RoboTwin split contains no tasks: {split_dir}")
            for task_dir in task_dirs:
                task_episodes = self._scan_task_folder(task_dir, split)
                episodes.extend(task_episodes)
                task_counts[task_dir.name] = task_counts.get(task_dir.name, 0) + len(
                    task_episodes
                )
        if not episodes:
            raise ValueError(
                f"no complete RoboTwin episodes found under {self.dataset_dir}"
            )
        for task_name, count in sorted(task_counts.items()):
            logger.info("  %s: %d episodes", task_name, count)
        return episodes

    def _get_video_frame_count_cached(self, video_path: str) -> int:
        cached = self._frame_count_cache.get(video_path)
        if cached is None:
            cached = get_video_frame_count(video_path)
            self._frame_count_cache[video_path] = cached
        return cached

    def _get_qpos_frame_count_cached(self, qpos_path: str) -> int:
        cached = self._qpos_frame_count_cache.get(qpos_path)
        if cached is None:
            value = torch.load(
                qpos_path,
                map_location="cpu",
                weights_only=True,
            )
            tensor = (
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            )
            cached = int(tensor.shape[0]) if tensor.ndim else 0
            self._qpos_frame_count_cache[qpos_path] = cached
        return cached

    def _effective_episode_frame_count(
        self,
        episode: dict[str, Any],
    ) -> int:
        return min(
            self._get_video_frame_count_cached(episode["video_path"]),
            self._get_qpos_frame_count_cached(episode["qpos_path"]),
        )

    def _condition_indices_for_episode(self, total_frames: int) -> range:
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        return range(max(0, total_frames - physical_chunk_size))

    def _build_sample_index(self) -> None:
        dropped_short = 0
        for episode_index, episode in enumerate(self.all_episodes):
            total_frames = self._effective_episode_frame_count(episode)
            indices = self._condition_indices_for_episode(total_frames)
            if not indices:
                dropped_short += 1
                continue
            self.sample_index.extend(
                (episode_index, condition_index) for condition_index in indices
            )
        if not self.sample_index:
            raise ValueError("DynamicWAM RoboTwin source has no valid samples")
        if dropped_short:
            logger.warning(
                "Dropped %d episodes shorter than the DynamicWAM action horizon",
                dropped_short,
            )

    def _calculate_sampling_indices(
        self,
        *,
        total_frames: int,
        condition_frame_idx: int,
    ) -> tuple[int, list[int], list[int]]:
        valid_indices = self._condition_indices_for_episode(total_frames)
        if condition_frame_idx not in valid_indices:
            raise ValueError(
                f"condition index {condition_frame_idx} is outside "
                f"[0, {max(0, valid_indices.stop - 1)}]"
            )
        action_indices = [
            condition_frame_idx + (action_step + 1) * self.global_downsample_rate
            for action_step in range(self.action_chunk_size)
        ]
        video_indices = [
            action_indices[(video_step + 1) * self.video_action_freq_ratio - 1]
            for video_step in range(self.num_video_frames)
        ]
        return condition_frame_idx, video_indices, action_indices

    def get_episode(self, episode_index: int) -> dict[str, Any]:
        try:
            return self.all_episodes[int(episode_index)]
        except IndexError as exc:
            raise IndexError(
                f"episode index {episode_index} is outside "
                f"[0, {len(self.all_episodes) - 1}]"
            ) from exc

    def __len__(self) -> int:
        return len(self.sample_index)
