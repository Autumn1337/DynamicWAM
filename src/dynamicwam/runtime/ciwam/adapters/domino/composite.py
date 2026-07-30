from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def head_camera_frame(observation: Dict[str, Any]) -> np.ndarray:
    """Return the third-person head-camera frame used by DynamicWAM head flow.

    The WAM condition remains the upstream three-camera composite; only the
    optical-flow stream uses this exocentric view, avoiding wrist
    self-motion dominating the motion representation.
    """
    head = np.asarray(observation["observation"]["head_camera"]["rgb"])
    if head.dtype != np.uint8:
        head = np.clip(head, 0, 255).astype(np.uint8)
    return head


def composite_robotwin_frame(
    observation: Dict[str, Any],
    config: dict[str, Any],
) -> np.ndarray:
    """Build the configured DynamicWAM head-over-two-wrists camera composite."""
    obs = observation["observation"]
    head_size = tuple(int(value) for value in config["head_size"])
    wrist_size = tuple(int(value) for value in config["wrist_size"])
    head = np.asarray(obs["head_camera"]["rgb"])
    if head.shape[:2] != head_size:
        raise ValueError(
            f"DOMINO head camera shape differs from DynamicWAM: "
            f"expected {head_size}, got {head.shape[:2]}"
        )
    wrist_height, wrist_width = wrist_size
    left = cv2.resize(
        obs["left_camera"]["rgb"],
        (wrist_width, wrist_height),
    )
    right = cv2.resize(
        obs["right_camera"]["rgb"],
        (wrist_width, wrist_height),
    )
    bottom = np.concatenate([left, right], axis=1)
    frame = np.concatenate([head, bottom], axis=0)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def extract_state(observation: Dict[str, Any]) -> np.ndarray:
    return np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
