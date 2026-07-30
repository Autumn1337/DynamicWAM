from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def validate_action_normalization_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the mandatory DynamicWAM mean/std normalization contract."""
    if not isinstance(config, Mapping):
        raise ValueError("action_normalization must be a mapping")
    normalized = dict(config)
    if normalized.get("enabled") is not True:
        raise ValueError("DynamicWAM action normalization must be enabled")
    if normalized.get("type") != "mean_std":
        raise ValueError("DynamicWAM action normalization type must be mean_std")
    if "stats_path" in normalized:
        normalized["stats_path"] = str(normalized["stats_path"])
    return normalized


def _stats_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    stats = payload.get("robotwin_qpos", payload)
    if not isinstance(stats, Mapping):
        raise ValueError("qpos stats JSON must contain a mapping payload")
    return stats


class RoboTwinQposNormalizer:
    """Mean/std normalizer for RoboTwin qpos vectors."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        epsilon: float,
    ) -> None:
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError(f"normalization epsilon must be positive, got {epsilon}")
        self._mean = mean.detach().float().cpu()
        self._std = std.detach().float().cpu()
        if self._mean.dim() != 1 or self._std.dim() != 1:
            raise ValueError("qpos normalization mean/std must be 1D tensors")
        if self._mean.shape != self._std.shape:
            raise ValueError(
                "qpos normalization mean/std shape mismatch: "
                f"{tuple(self._mean.shape)} vs {tuple(self._std.shape)}"
            )
        if not bool(torch.isfinite(self._mean).all()) or not bool(
            torch.isfinite(self._std).all()
        ):
            raise ValueError("qpos normalization mean/std must be finite")
        if bool((self._std < 0.0).any()):
            raise ValueError("qpos normalization std must be non-negative")
        self._std.clamp_min_(self.epsilon)

    @classmethod
    def from_stats(
        cls,
        stats: Mapping[str, Any],
        *,
        epsilon: float,
    ) -> "RoboTwinQposNormalizer":
        payload = _stats_payload(stats)
        mean = torch.as_tensor(payload.get("mean"), dtype=torch.float32)
        std = torch.as_tensor(payload.get("std"), dtype=torch.float32)
        if mean.numel() == 0 or std.numel() == 0:
            raise ValueError("Invalid qpos mean/std stats")
        return cls(mean=mean, std=std, epsilon=epsilon)

    @classmethod
    def from_stats_path(
        cls,
        stats_path: str | Path,
        *,
        epsilon: float,
    ) -> "RoboTwinQposNormalizer":
        path = Path(stats_path).expanduser()
        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls.from_stats(stats, epsilon=epsilon)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor, inverse=False)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._apply(tensor, inverse=True)

    def _apply(self, tensor: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        if tensor.shape[-1] != self._mean.numel():
            raise ValueError(
                f"qpos dim mismatch: expected {self._mean.numel()}, got {tensor.shape[-1]}"
            )
        mean = self._mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self._std.to(device=tensor.device, dtype=tensor.dtype)
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / std
