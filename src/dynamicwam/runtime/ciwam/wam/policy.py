from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .bootstrap import add_dynamicwam_to_path

logger = logging.getLogger(__name__)


class _Runner(Protocol):
    runtime: Any

    def reset(self) -> None: ...

    def set_instruction(self, instruction: str) -> None: ...

    def sample_chunk(
        self,
        *,
        first_frame: Any,
        state: Any,
        flow_frames: Any,
        motion_features: Any,
        motion_interval_valid_mask: Any,
        motion_acceleration_valid_mask: Any,
    ) -> Any: ...


class DynamicWAMPolicy:
    """Synchronous DynamicWAM head-flow policy."""

    def __init__(
        self,
        deploy_config_path: str,
        *,
        project_root: str,
    ) -> None:
        config_path = Path(deploy_config_path).expanduser()
        if not config_path.is_absolute():
            config_path = Path(project_root).expanduser() / config_path
        self.deploy_config_path = str(config_path.resolve())
        self.project_root = project_root
        self.runner: _Runner | None = None

    def setup(self) -> None:
        add_dynamicwam_to_path(self.project_root)
        from dynamicwam.inference.model_loader import (
            build_runtime_from_config,
            load_deploy_config,
        )

        from .flow_runner import HeadFlowRunner

        config = load_deploy_config(self.deploy_config_path)
        runtime = build_runtime_from_config(config)
        self.runner = HeadFlowRunner(runtime=runtime)
        self._warmup()

    def _require_runner(self) -> _Runner:
        if self.runner is None:
            raise RuntimeError("policy setup() must run before policy execution")
        return self.runner

    @property
    def runtime(self):
        return self._require_runner().runtime

    def reset(self) -> None:
        if self.runner is not None:
            self.runner.reset()

    def set_instruction(self, instruction: str) -> None:
        self._require_runner().set_instruction(instruction)

    def sample(self, packet: dict[str, Any]) -> np.ndarray:
        actions = self._require_runner().sample_chunk(
            first_frame=self._frame_tensor(packet["frame"]),
            state=self._state_tensor(packet["state"]),
            flow_frames=packet["flow_frames"],
            motion_features=packet["motion_features"],
            motion_interval_valid_mask=packet["motion_interval_valid_mask"],
            motion_acceleration_valid_mask=packet["motion_acceleration_valid_mask"],
        )
        return actions.squeeze(0).float().cpu().numpy()

    def _frame_tensor(self, frame: np.ndarray):
        import torch

        from dynamicwam.image import resize_with_padding

        target = self.runtime.video_size
        if frame.shape[:2] != target:
            frame = resize_with_padding(frame, target)
        normalized = frame.astype(np.float32) / 255.0
        return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _state_tensor(state: np.ndarray):
        import torch

        return torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)

    def _warmup(self) -> None:
        runner = self._require_runner()
        state_dim = int(self.runtime.model.config.state_dim)
        flow = self.runtime.head_flow_config
        flow_height, flow_width = (int(value) for value in flow["compute_size"])
        runner.set_instruction("warmup")
        self.sample(
            {
                "frame": np.zeros(
                    (*self.runtime.composite_frame_size, 3),
                    dtype=np.uint8,
                ),
                "state": np.zeros(state_dim, dtype=np.float32),
                "flow_frames": np.zeros(
                    (
                        int(flow["count"]),
                        flow_height,
                        flow_width,
                        3,
                    ),
                    dtype=np.uint8,
                ),
                "motion_features": np.zeros(
                    (int(flow["count"]), 12),
                    dtype=np.float32,
                ),
                "motion_interval_valid_mask": np.zeros(
                    int(flow["count"]),
                    dtype=np.bool_,
                ),
                "motion_acceleration_valid_mask": np.zeros(
                    int(flow["count"]),
                    dtype=np.bool_,
                ),
            }
        )
        self.reset()
        logger.info("DynamicWAM absolute-motion warmup complete")
