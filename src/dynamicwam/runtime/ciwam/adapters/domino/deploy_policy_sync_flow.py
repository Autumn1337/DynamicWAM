"""Native synchronous DOMINO adapter with exact simulator-time motion."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from ...execution import NativeStepper
from ...flow import HeadFlowBuffer
from ...wam.policy import DynamicWAMPolicy
from .composite import composite_robotwin_frame, extract_state, head_camera_frame

logger = logging.getLogger(__name__)


class DynamicWAMDeployModel:
    def __init__(self, usr_args: Dict[str, Any]) -> None:
        self.policy = DynamicWAMPolicy(
            usr_args["dynamicwam_deploy_config"],
            project_root=usr_args["dynamicwam_root"],
        )
        self.policy.setup()
        self.flow = HeadFlowBuffer(self.policy.runtime.head_flow_config)
        self.stepper: Optional[NativeStepper] = None
        self.pending: deque[np.ndarray] = deque()

    def bind_env(self, task_env) -> None:
        if self.stepper is None or self.stepper.env is not task_env:
            self.stepper = NativeStepper(
                task_env,
                action_interval_seconds=(self.policy.runtime.action_interval_seconds),
            )
            self.policy.set_instruction(task_env.get_instruction())

    def finish_episode(self) -> None:
        if self.stepper is not None:
            summary = self.stepper.telemetry.summary()
            logger.info("sync-flow episode: %s", summary)
            print(f"SYNC-EPISODE-SUMMARY {summary}", flush=True)
        self.stepper = None
        self.pending.clear()
        self.flow.reset()
        self.policy.reset()


def get_model(usr_args: Dict[str, Any]) -> DynamicWAMDeployModel:
    return DynamicWAMDeployModel(usr_args)


def eval(TASK_ENV, model: DynamicWAMDeployModel, observation: Dict[str, Any]) -> None:
    model.bind_env(TASK_ENV)
    frame = composite_robotwin_frame(
        observation,
        model.policy.runtime.observation_config,
    )
    scene_clock = getattr(TASK_ENV, "_scene_step_clock", None)
    if scene_clock is None:
        raise RuntimeError(
            "absolute-motion deployment requires TASK_ENV._scene_step_clock"
        )
    simulator_time = float(scene_clock.snapshot().time_seconds)
    model.flow.push(
        head_camera_frame(observation),
        simulator_time_seconds=simulator_time,
    )
    if not model.pending:
        motion = model.flow.observation()
        packet = {
            "frame": frame,
            "state": extract_state(observation),
            "flow_frames": motion.flow_rgb,
            "motion_features": motion.motion_features,
            "motion_interval_valid_mask": motion.interval_valid_mask,
            "motion_acceleration_valid_mask": (motion.acceleration_valid_mask),
        }
        actions = model.policy.sample(packet)
        if actions.shape[0] != model.policy.runtime.chunk_size:
            raise RuntimeError(
                "absolute-motion policy returned the wrong action chunk: "
                f"expected {model.policy.runtime.chunk_size}, got {actions.shape}"
            )
        model.pending.extend(actions)
    if model.stepper is None:
        raise RuntimeError("DOMINO environment was not bound before execution")
    model.stepper.execute(model.pending.popleft())


def reset_model(model: DynamicWAMDeployModel) -> None:
    model.finish_episode()
