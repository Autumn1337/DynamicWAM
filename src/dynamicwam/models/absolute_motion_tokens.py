"""Absolute image-plane motion tokens for the action expert."""

from __future__ import annotations

import torch
import torch.nn as nn

from dynamicwam.absolute_motion import MOTION_FEATURE_DIM


class AbsoluteMotionTokenModule(nn.Module):
    """Map four exact-time motion intervals into live joint-attention tokens."""

    def __init__(
        self,
        *,
        dim: int,
        history_count: int,
        feature_mean: tuple[float, ...],
        feature_scale: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.history_count = int(history_count)
        if self.dim <= 0 or self.history_count <= 0:
            raise ValueError("motion token dimensions must be positive")
        if (
            len(feature_mean) != MOTION_FEATURE_DIM
            or len(feature_scale) != MOTION_FEATURE_DIM
        ):
            raise ValueError("absolute motion normalization requires 12 values")
        mean = torch.tensor(feature_mean, dtype=torch.float32)
        scale = torch.tensor(feature_scale, dtype=torch.float32)
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise ValueError("motion normalization contains non-finite values")
        if not bool((scale > 0.0).all()):
            raise ValueError("motion normalization scales must be positive")
        self.register_buffer("feature_mean", mean, persistent=True)
        self.register_buffer("feature_scale", scale, persistent=True)

        self.encoder = nn.Sequential(
            nn.Linear(MOTION_FEATURE_DIM, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        init_scale = self.dim**-0.5
        self.positions = nn.Parameter(
            torch.randn(1, self.history_count, self.dim) * init_scale
        )
        self.motion_type = nn.Parameter(torch.randn(1, 1, self.dim) * init_scale)
        self.invalid_interval = nn.Parameter(torch.randn(1, 1, self.dim) * init_scale)
        self.invalid_acceleration = nn.Parameter(
            torch.randn(1, 1, self.dim) * init_scale
        )

    def forward(
        self,
        features: torch.Tensor,
        interval_valid: torch.Tensor,
        acceleration_valid: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        features = features.to(device=device, dtype=torch.float32)
        interval_valid = interval_valid.to(device=device, dtype=torch.bool)
        acceleration_valid = acceleration_valid.to(
            device=device,
            dtype=torch.bool,
        )
        expected_features = (
            features.shape[0],
            self.history_count,
            MOTION_FEATURE_DIM,
        )
        if tuple(features.shape) != expected_features:
            raise ValueError(
                "absolute_motion_features must be [B,"
                f"{self.history_count},{MOTION_FEATURE_DIM}], got "
                f"{tuple(features.shape)}"
            )
        expected_mask = (features.shape[0], self.history_count)
        if (
            tuple(interval_valid.shape) != expected_mask
            or tuple(acceleration_valid.shape) != expected_mask
        ):
            raise ValueError(
                f"absolute motion masks must be {expected_mask}, got "
                f"{tuple(interval_valid.shape)} and "
                f"{tuple(acceleration_valid.shape)}"
            )
        if bool((acceleration_valid & ~interval_valid).any()):
            raise ValueError("acceleration cannot be valid for an invalid interval")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("absolute_motion_features contains non-finite values")

        normalized = (
            features - self.feature_mean.to(device=device)
        ) / self.feature_scale.to(device=device)
        feature_valid = interval_valid[..., None].expand_as(normalized).clone()
        feature_valid[..., 9:12] &= acceleration_valid[..., None]
        normalized = torch.where(
            feature_valid,
            normalized,
            torch.zeros_like(normalized),
        )
        encoded = self.encoder(normalized.to(dtype=dtype))
        tokens = torch.where(
            interval_valid[..., None],
            encoded,
            self.invalid_interval.to(device=device, dtype=dtype).expand_as(encoded),
        )
        return (
            tokens
            + self.positions.to(device=device, dtype=dtype)
            + self.motion_type.to(device=device, dtype=dtype)
            + (interval_valid & ~acceleration_valid)[..., None].to(dtype=dtype)
            * self.invalid_acceleration.to(device=device, dtype=dtype)
        )
