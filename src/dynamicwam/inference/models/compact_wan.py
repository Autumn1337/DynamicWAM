from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .wan_model import WanVideoModel


@dataclass(frozen=True)
class CompactWANConfig:
    checkpoint_path: str
    vae_path: str
    config_path: str
    precision: str
    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    head_dim: int
    future_video_size: tuple[int, int]

    def to_wan_model_config(self) -> dict[str, int]:
        return {
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }


class CompactWANModel(nn.Module):
    """Checkpoint-compatible compact WAN used by DynamicWAM deployment."""

    def __init__(
        self,
        config: CompactWANConfig,
        video_model: WanVideoModel,
    ) -> None:
        super().__init__()
        self.config = config
        self.video_model = video_model

    @classmethod
    def from_config(
        cls,
        config: CompactWANConfig,
        device: str = "cuda",
    ) -> "CompactWANModel":
        video_model = WanVideoModel.from_compact_config(
            config_path=config.config_path,
            vae_path=config.vae_path,
            student_model_config=config.to_wan_model_config(),
            device=device,
            precision=config.precision,
        )
        return cls(config=config, video_model=video_model)

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        return self.video_model.encode_video(video_pixels)

    def prepare_multiscale_video_tokens(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor | int],
        torch.Tensor,
    ]:
        return self.video_model.prepare_multiscale_video_tokens(
            condition_latent,
            future_latent,
        )

    def prepare_text_context(
        self,
        text_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
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
        layout: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        return self.video_model.apply_multiscale_video_head(
            video_tokens,
            video_time_emb,
            layout,
        )

    def apply_multiscale_rope(
        self,
        heads: torch.Tensor,
        layout: dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        return self.video_model.apply_multiscale_rope(
            heads,
            layout,
            freqs,
        )
