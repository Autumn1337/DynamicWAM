from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch


def resize_with_padding(
    frame: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize an HWC image without distortion and center-pad to ``target_size``."""
    target_height, target_width = (int(value) for value in target_size)
    if frame.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {frame.shape}")
    original_height, original_width = frame.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError(f"image dimensions must be positive, got {frame.shape}")

    scale = min(
        target_height / original_height,
        target_width / original_width,
    )
    new_height = max(1, int(original_height * scale))
    new_width = max(1, int(original_width * scale))
    resized = cv2.resize(frame, (new_width, new_height))
    padded = np.zeros(
        (target_height, target_width, frame.shape[2]),
        dtype=frame.dtype,
    )
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    padded[top : top + new_height, left : left + new_width] = resized
    return padded


def load_video_frames(
    video_path: str,
    frame_indices: Sequence[int],
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Read RGB frames with decord and return ``[T,C,H,W]`` floats in [0, 1]."""
    from decord import VideoReader, cpu  # type: ignore[import-not-found]

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    indices = [int(index) for index in frame_indices]
    invalid = [index for index in indices if index < 0 or index >= len(reader)]
    if invalid:
        raise ValueError(
            f"frame indices {invalid} are outside [0, {len(reader) - 1}] "
            f"for {video_path}"
        )
    frames = reader.get_batch(indices).asnumpy()
    if target_size is not None and tuple(frames.shape[1:3]) != tuple(target_size):
        frames = np.stack(
            [resize_with_padding(frame, target_size) for frame in frames],
            axis=0,
        )
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)


def get_video_frame_count(video_path: str) -> int:
    from decord import VideoReader, cpu  # type: ignore[import-not-found]

    return len(VideoReader(video_path, ctx=cpu(0), num_threads=1))
