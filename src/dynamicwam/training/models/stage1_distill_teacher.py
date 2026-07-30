from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .wan_model import WanVideoModel


@dataclass(frozen=True)
class Stage1DistillTeacherConfig:
    checkpoint_path: str
    config_path: str
    precision: str
    hidden_anchor_teacher_layers: List[int]
    motion_anchor_teacher_layers: List[int]


@dataclass(frozen=True)
class Stage1TeacherTargets:
    hidden: Dict[int, torch.Tensor]
    motion: Dict[int, torch.Tensor]


class FixedPCATeacherProjector:
    """Fixed teacher-side projector loaded from PCA prep artifacts."""

    def __init__(self, stats_path: str, device: str = "cpu"):
        payload = torch.load(
            stats_path,
            map_location=device,
            weights_only=True,
        )
        self.projection_dim = int(payload["projection_dim"])
        self.layer_stats = payload["layers"]
        self.device = device
        self._tensor_cache: Dict[
            tuple[str, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _layer_tensors(
        self,
        layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_key = str(layer)
        if layer_key not in self.layer_stats:
            available = ", ".join(sorted(self.layer_stats.keys(), key=int))
            raise KeyError(
                f"PCA stats missing teacher layer {layer}; available layers: {available}"
            )

        cache_key = (layer_key, str(device), dtype)
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        stats = self.layer_stats[layer_key]
        mean = stats["mean"].to(device=device, dtype=dtype, non_blocking=True)
        components = stats["components"].to(
            device=device, dtype=dtype, non_blocking=True
        )
        self._tensor_cache[cache_key] = (mean, components)
        return mean, components

    def project(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        mean, components = self._layer_tensors(layer, hidden.device, hidden.dtype)
        centered = F.layer_norm(hidden, (hidden.shape[-1],)) - mean
        return centered @ components


class Stage1DistillTeacher:
    """Teacher backend for Stage 1 distillation."""

    def __init__(
        self,
        config: Stage1DistillTeacherConfig,
        student_hidden_layers: List[int],
        student_motion_layers: List[int],
        pca_stats_path: str,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.student_hidden_layers = list(student_hidden_layers)
        self.student_motion_layers = list(student_motion_layers)
        self.pca_projector = FixedPCATeacherProjector(pca_stats_path, device="cpu")
        self.model = WanVideoModel.from_pretrained(
            checkpoint_path=config.checkpoint_path,
            config_path=config.config_path,
            device=device,
            precision=config.precision,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def targets(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        include_hidden: bool,
        include_motion: bool,
    ) -> Stage1TeacherTargets:
        x_t = batch["x_t"]
        timestep = batch["t"]
        text_embeddings = batch["text_embeddings"]
        hidden_teacher_layers = (
            self.config.hidden_anchor_teacher_layers if include_hidden else []
        )
        motion_teacher_layers = (
            self.config.motion_anchor_teacher_layers if include_motion else []
        )
        hidden_layers = sorted(set(hidden_teacher_layers + motion_teacher_layers))
        if not isinstance(x_t, dict):
            raise TypeError("DynamicWAM teacher input must contain multiscale latents")
        features = self.model.get_multiscale_layer_features(
            condition_latent=x_t["condition_latent"],
            future_latent=x_t["future_latent"],
            timestep=timestep,
            text_embeddings=text_embeddings,
            layer_indices=hidden_layers,
        )

        raw_by_layer = dict(zip(hidden_layers, features, strict=True))

        hidden_targets: Dict[int, torch.Tensor] = {}
        for student_layer, teacher_layer in zip(
            self.student_hidden_layers,
            hidden_teacher_layers,
            strict=True,
        ):
            hidden_targets[student_layer] = self.pca_projector.project(
                teacher_layer, raw_by_layer[teacher_layer]
            )

        motion_targets: Dict[int, torch.Tensor] = {}
        latent_frames = int(batch["num_motion_frames"])
        condition_tokens = int(batch["condition_tokens"])
        for student_layer, teacher_layer in zip(
            self.student_motion_layers,
            motion_teacher_layers,
            strict=True,
        ):
            projected = self.pca_projector.project(
                teacher_layer, raw_by_layer[teacher_layer]
            )
            projected = projected[:, condition_tokens:]
            batch_size, num_tokens, dim = projected.shape
            if num_tokens % latent_frames != 0:
                raise ValueError(
                    f"Teacher layer {teacher_layer} produced {num_tokens} tokens incompatible with {latent_frames} latent frames"
                )
            tokens_per_frame = num_tokens // latent_frames
            frames = projected.reshape(
                batch_size, latent_frames, tokens_per_frame, dim
            ).mean(dim=2)
            motion_targets[student_layer] = frames[:, 1:] - frames[:, :-1]
        return Stage1TeacherTargets(
            hidden=hidden_targets,
            motion=motion_targets,
        )
