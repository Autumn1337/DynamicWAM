"""DynamicWAM Action Expert shared by training and deployment."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from dynamicwam.vendor.wan.modules.model import WanLayerNorm, WanRMSNorm

logger = logging.getLogger(__name__)


def _sincos_positions(embed_dim: int, length: int) -> torch.Tensor:
    if embed_dim % 2:
        raise ValueError(f"Action Expert dimension must be even, got {embed_dim}")
    positions = np.arange(length, dtype=np.float64)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    phase = np.einsum("m,d->md", positions, omega)
    return torch.from_numpy(
        np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
    ).float()


def _three_layer_silu(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.SiLU(),
        nn.Linear(out_features, out_features),
        nn.SiLU(),
        nn.Linear(out_features, out_features),
    )


def _module_reference_tensor(module: nn.Module) -> torch.Tensor:
    return next(module.parameters())


@dataclass(frozen=True)
class ActionExpertConfig:
    dim: int
    ffn_dim: int
    num_layers: int
    state_dim: int
    action_dim: int
    chunk_size: int
    num_registers: int = 4
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.chunk_size < 2:
            raise ValueError(
                "chunk_size must include at least one state and one action"
            )
        if self.num_registers != 4:
            raise ValueError(
                f"DynamicWAM requires four register tokens, got {self.num_registers}"
            )


class StateActionEncoder(nn.Module):
    """Encode the current robot state followed by the noisy action chunk."""

    def __init__(self, config: ActionExpertConfig) -> None:
        super().__init__()
        self.state_encoder = _three_layer_silu(config.state_dim, config.dim)
        self.action_encoder = _three_layer_silu(config.action_dim, config.dim)
        max_seq_len = config.chunk_size + 1 + config.num_registers
        self.register_buffer(
            "pos_embedding",
            _sincos_positions(config.dim, max_seq_len).unsqueeze(0),
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        registers: torch.Tensor,
    ) -> torch.Tensor:
        state_ref = _module_reference_tensor(self.state_encoder)
        action_ref = _module_reference_tensor(self.action_encoder)
        state_encoded = self.state_encoder(
            state_tokens.to(device=state_ref.device, dtype=state_ref.dtype)
        )
        action_encoded = self.action_encoder(
            action_tokens.to(device=action_ref.device, dtype=action_ref.dtype)
        )
        encoded = torch.cat(
            [
                state_encoded,
                action_encoded,
                registers.to(device=state_encoded.device, dtype=state_encoded.dtype),
            ],
            dim=1,
        )
        return encoded + self.pos_embedding[:, : encoded.shape[1]].to(
            device=encoded.device,
            dtype=encoded.dtype,
        )


class ActionExpertBlock(nn.Module):
    """Action-side projections, feed-forward network, and AdaLN parameters."""

    def __init__(
        self,
        config: ActionExpertConfig,
        wan_config: dict[str, int],
    ) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)

        wan_num_heads = int(wan_config["num_heads"])
        wan_head_dim = int(wan_config["head_dim"])
        wan_dim = int(wan_config["dim"])
        if wan_num_heads * wan_head_dim != wan_dim:
            raise ValueError(
                "WAN attention dimensions are inconsistent: "
                f"{wan_num_heads} * {wan_head_dim} != {wan_dim}"
            )
        self.wan_action_qkv = nn.Parameter(
            torch.randn(3, wan_num_heads, config.dim, wan_head_dim)
            / (config.dim * wan_head_dim) ** 0.5
        )
        self.wan_action_o = nn.Linear(wan_dim, config.dim, bias=False)
        self.wan_action_norm_q = WanRMSNorm(wan_dim, eps=config.eps)
        self.wan_action_norm_k = WanRMSNorm(wan_dim, eps=config.eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, config.dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim**0.5)


class ActionDecoder(nn.Module):
    """Decode denoised action-token features."""

    def __init__(self, config: ActionExpertConfig) -> None:
        super().__init__()
        self.norm = WanLayerNorm(config.dim, eps=config.eps)
        self.action_head = nn.Sequential(nn.Linear(config.dim, config.action_dim))
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim**0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float32,
            enabled=tokens.is_cuda,
        ):
            shift, scale = (
                self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)
            ).chunk(2, dim=2)
        normalized = self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        head = self.action_head[0]
        return self.action_head(
            normalized.to(device=head.weight.device, dtype=head.weight.dtype)
        )


class ActionExpert(nn.Module):
    """DynamicWAM DiT-style action branch coupled one-to-one with compact WAN blocks."""

    def __init__(
        self,
        config: ActionExpertConfig,
        wan_config: dict[str, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.freq_dim = 256
        self.input_encoder = StateActionEncoder(config)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6),
        )
        self.blocks = nn.ModuleList(
            [ActionExpertBlock(config, wan_config) for _ in range(config.num_layers)]
        )
        self.registers = nn.Parameter(
            torch.empty(1, config.num_registers, config.dim).normal_(std=0.02)
        )
        self.decoder = ActionDecoder(config)
        self._initialize_weights()
        logger.info(
            "Action Expert initialized with %d parameters",
            sum(parameter.numel() for parameter in self.parameters()),
        )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.decoder.action_head[-1].weight)
        nn.init.zeros_(self.decoder.action_head[-1].bias)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
