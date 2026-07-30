"""Pinned Aloha AgileX qpos ordering and joint-limit validation."""

from __future__ import annotations

from typing import Any

import numpy as np

ALOHA_QPOS_DIM = 14
# DOMINO interleaves each six-joint arm with one normalized gripper scalar.
ALOHA_QPOS_LOWER = np.asarray(
    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0] * 2,
    dtype=np.float32,
)
ALOHA_QPOS_UPPER = np.asarray(
    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0] * 2,
    dtype=np.float32,
)


def aloha_qpos_is_valid(values: Any) -> bool:
    """Return whether frame-major qpos follows the pinned Aloha limits."""

    qpos = np.asarray(values)
    return bool(
        qpos.ndim == 2
        and qpos.shape[0] > 0
        and qpos.shape[1] == ALOHA_QPOS_DIM
        and np.isfinite(qpos).all()
        and np.all(qpos >= ALOHA_QPOS_LOWER)
        and np.all(qpos <= ALOHA_QPOS_UPPER)
    )
