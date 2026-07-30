from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillHeadConfig:
    hidden_dim: int
    projection_dim: int
    hidden_anchor_layers: Iterable[int]
    motion_anchor_layers: Iterable[int]
    eps: float = 1e-6


class _LayerProjector(nn.Module):
    """Per-layer LN + Linear projector used for distillation."""

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_weight = self.norm.weight
        x = x.to(device=norm_weight.device, dtype=norm_weight.dtype)
        normalized = self.norm(x)

        proj_weight = self.proj.weight
        normalized = normalized.to(device=proj_weight.device, dtype=proj_weight.dtype)
        return self.proj(normalized)


class Stage1DistillHeads(nn.Module):
    """Student-side distillation heads for hidden and motion supervision."""

    def __init__(self, config: DistillHeadConfig):
        super().__init__()
        self.config = config
        self.hidden_anchor_layers: List[int] = list(config.hidden_anchor_layers)
        self.motion_anchor_layers: List[int] = list(config.motion_anchor_layers)

        self.hidden_projectors = nn.ModuleDict(
            {
                str(layer): _LayerProjector(
                    config.hidden_dim, config.projection_dim, config.eps
                )
                for layer in self.hidden_anchor_layers
            }
        )
        self.motion_projectors = nn.ModuleDict(
            {
                str(layer): _LayerProjector(
                    config.hidden_dim, config.projection_dim, config.eps
                )
                for layer in self.motion_anchor_layers
            }
        )

    def project_hidden(
        self, hidden_features: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        projected: Dict[int, torch.Tensor] = {}
        for layer, feats in hidden_features.items():
            projected[layer] = self.hidden_projectors[str(layer)](feats)
        return projected

    def project_motion(
        self, hidden_features: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        projected: Dict[int, torch.Tensor] = {}
        for layer, feats in hidden_features.items():
            projected[layer] = self.motion_projectors[str(layer)](feats)
        return projected

    @staticmethod
    def build_motion_deltas(
        projected_tokens: Dict[int, torch.Tensor],
        num_frames: int,
        condition_tokens: int,
    ) -> Dict[int, torch.Tensor]:
        """Pool tokens to frame features and compute temporal differences.

        Expected token layout after WAN patchification: tokens are ordered by time first.
        This helper keeps the implementation simple for now and assumes the token count is
        divisible by num_frames.
        """

        deltas: Dict[int, torch.Tensor] = {}
        for layer, feats in projected_tokens.items():
            feats = feats[:, int(condition_tokens) :]
            batch, num_tokens, dim = feats.shape
            if num_tokens % num_frames != 0:
                raise ValueError(
                    f"Layer {layer} produced {num_tokens} tokens, not divisible by {num_frames} frames"
                )
            tokens_per_frame = num_tokens // num_frames
            frames = feats.reshape(batch, num_frames, tokens_per_frame, dim).mean(dim=2)
            deltas[layer] = frames[:, 1:] - frames[:, :-1]
        return deltas

    @staticmethod
    def hidden_cosine_loss_per_sample(
        student_hidden: Dict[int, torch.Tensor],
        teacher_hidden: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for layer, student in student_hidden.items():
            teacher = teacher_hidden[layer]
            student_f32 = student.float()
            teacher_f32 = teacher.float()
            per_token = 1.0 - F.cosine_similarity(student_f32, teacher_f32, dim=-1)
            losses.append(per_token.mean(dim=1))
        return torch.stack(losses, dim=0).mean(dim=0)

    @staticmethod
    def motion_cosine_loss_per_sample(
        student_motion: Dict[int, torch.Tensor],
        teacher_motion: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for layer, student in student_motion.items():
            teacher = teacher_motion[layer]
            student_f32 = student.float()
            teacher_f32 = teacher.float()
            per_token = 1.0 - F.cosine_similarity(student_f32, teacher_f32, dim=-1)
            losses.append(per_token.mean(dim=1))
        return torch.stack(losses, dim=0).mean(dim=0)
