from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .wan_model import WanVideoModel


@dataclass(frozen=True)
class CompactWANConfig:
    """Configuration for the DynamicWAM compact WAN backbone."""

    checkpoint_path: str
    config_path: str
    future_video_size: Tuple[int, int]
    precision: str
    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    head_dim: int
    hidden_anchor_layers: List[int]
    motion_anchor_layers: List[int]
    teacher_layer_mapping: List[int]

    def to_wan_model_config(self) -> Dict[str, int]:
        return {
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


class CompactWANModel(nn.Module):
    """Compact WAN backbone used by DynamicWAM training."""

    def __init__(self, config: CompactWANConfig, video_model: WanVideoModel):
        super().__init__()
        self.config = config
        self.video_model = video_model

    @classmethod
    def from_teacher_checkpoint(
        cls,
        config: CompactWANConfig,
        device: str = "cuda",
    ) -> "CompactWANModel":
        """Initialize DynamicWAM by structured slicing from the WAN teacher."""
        video_model = WanVideoModel.from_pretrained_compact(
            checkpoint_path=config.checkpoint_path,
            student_model_config=config.to_wan_model_config(),
            teacher_layer_mapping=config.teacher_layer_mapping,
            config_path=config.config_path,
            device=device,
            precision=config.precision,
        )
        return cls(config=config, video_model=video_model)

    @classmethod
    def from_config(
        cls,
        config: CompactWANConfig,
        device: str = "cuda",
    ) -> "CompactWANModel":
        """Build compact WAN architecture without loading teacher or compact weights."""
        video_model = WanVideoModel.from_compact_config(
            config_path=config.config_path,
            student_model_config=config.to_wan_model_config(),
            device=device,
            precision=config.precision,
        )
        return cls(config=config, video_model=video_model)

    def prepare_multiscale_video_tokens(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | int], torch.Tensor]:
        return self.video_model.prepare_multiscale_video_tokens(
            condition_latent, future_latent
        )

    def prepare_text_context(self, text_embeddings: List[torch.Tensor]) -> torch.Tensor:
        return self.video_model.prepare_text_context(text_embeddings)

    def prepare_time_embeddings(
        self,
        timestep: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_model.prepare_time_embeddings(timestep, seq_len)

    def apply_multiscale_video_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        return self.video_model.apply_multiscale_video_head(
            video_tokens, video_time_emb, layout
        )

    def apply_multiscale_rope(
        self,
        heads: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        return self.video_model.apply_multiscale_rope(heads, layout, freqs)

    def forward_multiscale_with_features(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: List[int],
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        video_pred, features = self.video_model.forward_multiscale_with_features(
            condition_latent=condition_latent,
            future_latent=future_latent,
            timestep=timestep,
            text_embeddings=text_embeddings,
            layer_indices=layer_indices,
        )
        hidden = dict(zip(layer_indices, features, strict=True))
        return video_pred, hidden

    def metadata(self) -> Dict[str, object]:
        return asdict(self.config)
