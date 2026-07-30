"""Minimal WAN video wrapper required by DynamicWAM deployment."""

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dynamicwam.vendor.wan.modules.model import (
    WanModel,
    rope_apply,
    sinusoidal_embedding_1d,
)
from dynamicwam.vendor.wan.modules.vae2_2 import Wan2_2_VAE

logger = logging.getLogger(__name__)


def _load_wan_arch_config(config_path: str) -> dict[str, Any]:
    config_json_path = Path(config_path) / "config.json"
    if not config_json_path.is_file():
        raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
    with config_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class WanVideoModel(nn.Module):
    """Compact WAN architecture plus the VAE needed for flow conditioning."""

    def __init__(
        self,
        model_config: dict[str, Any],
        vae_path: str,
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> None:
        super().__init__()
        if precision != "bfloat16":
            raise ValueError(
                f"DynamicWAM WAN precision must be bfloat16, got {precision!r}"
            )
        self.device = torch.device(device)
        self.precision = torch.bfloat16

        # Initialize WAN model
        self.wan_model = WanModel(**model_config)
        self.wan_model.to(device=self.device, dtype=self.precision)

        # Initialize VAE
        self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device)

        logger.info(
            f"WAN Video Model initialized with {sum(p.numel() for p in self.wan_model.parameters()):,} parameters"
        )

    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(video_pixels)

    @staticmethod
    def _validate_wan_latent(name: str, latent: torch.Tensor) -> None:
        if latent.ndim != 5:
            raise ValueError(
                f"Expected {name} latent [B, C, f, h, w], got {latent.ndim}D {latent.shape}"
            )
        if latent.shape[1] != 48:
            raise ValueError(
                f"Expected 48 channels for {name} WAN 2.2 latents, got {latent.shape[1]}"
            )

    def _patch_video_latent(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_wan_latent("video", latent)
        device = self.wan_model.patch_embedding.weight.device
        latent = latent.to(device=device, dtype=self.precision)
        with torch.autocast(
            "cuda", dtype=self.precision, enabled=device.type == "cuda"
        ):
            patched = self.wan_model.patch_embedding(latent)
            grid_sizes = torch.stack(
                [
                    torch.tensor(
                        patched[idx].shape[1:], dtype=torch.long, device=device
                    )
                    for idx in range(patched.shape[0])
                ]
            )
            tokens = patched.flatten(2).transpose(1, 2)
        return tokens, grid_sizes

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
        self._validate_wan_latent("condition", condition_latent)
        self._validate_wan_latent("future", future_latent)
        if condition_latent.shape[0] != future_latent.shape[0]:
            raise ValueError("Multiscale condition and future batch sizes differ")
        device = self.wan_model.patch_embedding.weight.device
        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)
        condition_tokens, condition_grid_sizes = self._patch_video_latent(
            condition_latent
        )
        future_tokens, future_grid_sizes = self._patch_video_latent(future_latent)
        tokens = torch.cat([condition_tokens, future_tokens], dim=1)
        seq_lens = torch.full(
            (tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=device
        )
        patch_f, patch_h, patch_w = self.wan_model.patch_size
        return (
            tokens,
            seq_lens,
            {
                "condition_grid_sizes": condition_grid_sizes,
                "future_grid_sizes": future_grid_sizes,
                "condition_seq_len": int(condition_tokens.shape[1]),
                "future_seq_len": int(future_tokens.shape[1]),
                "condition_grid_shape": (
                    int(condition_latent.shape[2] // patch_f),
                    int(condition_latent.shape[3] // patch_h),
                    int(condition_latent.shape[4] // patch_w),
                ),
                "future_grid_shape": (
                    int(future_latent.shape[2] // patch_f),
                    int(future_latent.shape[3] // patch_h),
                    int(future_latent.shape[4] // patch_w),
                ),
            },
            self.wan_model.freqs,
        )

    @staticmethod
    def multiscale_layout_grid_sizes(
        layout: dict[str, torch.Tensor | int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        condition_grid_sizes = layout["condition_grid_sizes"]
        future_grid_sizes = layout["future_grid_sizes"]
        if not isinstance(condition_grid_sizes, torch.Tensor) or not isinstance(
            future_grid_sizes, torch.Tensor
        ):
            raise TypeError("Multiscale video layout must carry tensor grid sizes")
        return condition_grid_sizes, future_grid_sizes, int(layout["condition_seq_len"])

    def apply_multiscale_rope(
        self,
        heads: torch.Tensor,
        layout: dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        condition_grid_sizes, future_grid_sizes, condition_seq_len = (
            self.multiscale_layout_grid_sizes(layout)
        )
        return torch.cat(
            [
                rope_apply(heads[:, :condition_seq_len], condition_grid_sizes, freqs),
                rope_apply(heads[:, condition_seq_len:], future_grid_sizes, freqs),
            ],
            dim=1,
        )

    def prepare_text_context(
        self,
        text_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Pad/truncate T5 embeddings and project them into WAN hidden space."""
        device = self.wan_model.patch_embedding.weight.device
        dtype = self.precision
        padded_embeddings = []
        for emb in text_embeddings:
            emb = emb.to(device=device, dtype=dtype)
            if emb.size(0) > self.wan_model.text_len:
                emb = emb[: self.wan_model.text_len]
            elif emb.size(0) < self.wan_model.text_len:
                pad = emb.new_zeros(self.wan_model.text_len - emb.size(0), emb.size(1))
                emb = torch.cat([emb, pad], dim=0)
            padded_embeddings.append(emb)

        autocast_enabled = device.type == "cuda"
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            return self.wan_model.text_embedding(torch.stack(padded_embeddings, dim=0))

    def prepare_time_embeddings(
        self,
        timestep: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build WAN head and AdaLN time embeddings for an already-tokenized sequence."""
        device = self.wan_model.patch_embedding.weight.device
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).expand(timestep.size(0), seq_len)
        timestep = timestep.to(device=device)
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=device.type == "cuda"
        ):
            batch = timestep.size(0)
            flat_t = timestep.flatten()
            head_time_emb = self.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.wan_model.freq_dim, flat_t)
                .unflatten(0, (batch, seq_len))
                .float()
                .to(device)
            )
            adaln = self.wan_model.time_projection(head_time_emb).unflatten(
                2, (6, self.wan_model.dim)
            )
        return head_time_emb, adaln

    def apply_multiscale_video_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        layout: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        _, future_grid_sizes, condition_seq_len = self.multiscale_layout_grid_sizes(
            layout
        )
        future_tokens = video_tokens[:, condition_seq_len:]
        future_time_emb = video_time_emb[:, condition_seq_len:]
        head_weight = self.wan_model.head.head.weight
        future_tokens = future_tokens.to(
            device=head_weight.device, dtype=head_weight.dtype
        )
        with torch.autocast(
            "cuda", dtype=self.precision, enabled=head_weight.device.type == "cuda"
        ):
            output = self.wan_model.head(future_tokens, future_time_emb)
            output = self.wan_model.unpatchify(output, future_grid_sizes)
        return torch.stack([item.float() for item in output], dim=0)

    @classmethod
    def from_compact_config(
        cls,
        config_path: str,
        vae_path: str,
        student_model_config: dict[str, Any],
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> "WanVideoModel":
        """Build the compact WAN architecture without loading teacher weights."""

        teacher_model_config = _load_wan_arch_config(config_path)
        merged_student_config = dict(teacher_model_config)
        merged_student_config.update(student_model_config)
        merged_student_config["num_layers"] = int(student_model_config["num_layers"])

        model = cls(
            model_config=merged_student_config,
            vae_path=vae_path,
            device=device,
            precision=precision,
        )
        logger.info(
            "Initialized compact WAN model from config only (no WAN weights loaded)"
        )
        return model
