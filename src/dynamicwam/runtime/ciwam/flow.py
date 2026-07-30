"""Online exact-simulator-time head motion for DynamicWAM."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from dynamicwam.absolute_motion import (
    MotionObservationBatch,
    build_motion_features,
    compute_flow_observation,
    flow_compute_contract,
)


class HeadFlowBuffer:
    """Build chronological flow RGB and absolute motion at policy stride."""

    def __init__(self, config: dict[str, Any]) -> None:
        contract = flow_compute_contract(config)
        self.history_count = int(config["count"])
        self.policy_stride = int(contract["policy_stride"])
        self.compute_size = tuple(int(value) for value in contract["compute_size"])
        self.normalization_percentile = float(contract["normalization_percentile"])
        self.farneback = dict(contract["farneback"])
        self.quality = dict(contract["quality"])
        depth = (self.history_count + 1) * self.policy_stride + 1
        self._frames: deque[tuple[np.ndarray, float]] = deque(maxlen=depth)

    def reset(self) -> None:
        self._frames.clear()

    def push(
        self,
        frame_rgb_u8: np.ndarray,
        *,
        simulator_time_seconds: float,
    ) -> None:
        if (
            frame_rgb_u8.dtype != np.uint8
            or frame_rgb_u8.ndim != 3
            or frame_rgb_u8.shape[-1] != 3
        ):
            raise ValueError(
                f"expected HWC uint8 RGB, got {frame_rgb_u8.dtype} {frame_rgb_u8.shape}"
            )
        timestamp = float(simulator_time_seconds)
        if not np.isfinite(timestamp):
            raise ValueError("simulator_time_seconds must be finite")
        if self._frames and timestamp < self._frames[-1][1]:
            raise ValueError(
                "simulator time cannot move backwards between policy frames: "
                f"{self._frames[-1][1]:.9f} -> {timestamp:.9f}"
            )
        self._frames.append((np.ascontiguousarray(frame_rgb_u8).copy(), timestamp))

    def observation(self) -> MotionObservationBatch:
        if not self._frames:
            raise RuntimeError("motion observation requested before the first frame")
        samples = list(self._frames)
        last = len(samples) - 1

        def endpoint(index: int) -> tuple[np.ndarray, float, int]:
            resolved = max(0, index)
            frame, timestamp = samples[resolved]
            return frame, timestamp, resolved

        flow_rgb = []
        displacement = []
        starts = []
        ends = []
        valid = []
        for offset in range(self.history_count, -1, -1):
            previous_frame, previous_time, previous_index = endpoint(
                last - (offset + 1) * self.policy_stride
            )
            current_frame, current_time, current_index = endpoint(
                last - offset * self.policy_stride
            )
            temporal_valid = (
                current_index - previous_index == self.policy_stride
                and current_time > previous_time
            )
            if temporal_valid:
                rgb, statistics, _reliable_fraction, quality_valid = (
                    compute_flow_observation(
                        previous_frame,
                        current_frame,
                        compute_size=self.compute_size,
                        normalization_percentile=self.normalization_percentile,
                        farneback=self.farneback,
                        quality=self.quality,
                    )
                )
            else:
                rgb = np.zeros((*self.compute_size, 3), dtype=np.uint8)
                statistics = np.zeros(4, dtype=np.float32)
                quality_valid = False
            flow_rgb.append(rgb)
            displacement.append(statistics)
            starts.append(previous_time)
            ends.append(current_time)
            valid.append(temporal_valid and quality_valid)

        features, interval_valid, acceleration_valid = build_motion_features(
            np.stack(displacement),
            np.asarray(starts, dtype=np.float64),
            np.asarray(ends, dtype=np.float64),
            np.asarray(valid, dtype=np.bool_),
        )
        return MotionObservationBatch(
            flow_rgb=np.stack(flow_rgb[1:]),
            motion_features=features[1:],
            interval_valid_mask=interval_valid[1:],
            acceleration_valid_mask=acceleration_valid[1:],
        )
