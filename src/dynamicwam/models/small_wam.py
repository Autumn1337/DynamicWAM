from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import torch
import torch.nn as nn

from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
from dynamicwam.models.action_expert import ActionExpert, ActionExpertConfig
from dynamicwam.vendor.wan.modules.attention import flash_attention
from dynamicwam.vendor.wan.modules.model import sinusoidal_embedding_1d


class CompactWANArchitecture(Protocol):
    dim: int
    num_heads: int
    num_layers: int
    head_dim: int


@dataclass(frozen=True)
class SmallWAMActionConfig:
    compact_wan: CompactWANArchitecture
    action_dim: int
    state_dim: int
    chunk_size: int
    ae_dim: int
    ae_ffn_dim: int
    ae_num_layers: int
    wan_frozen: bool
    motion_history_count: int
    motion_feature_mean: tuple[float, ...]
    motion_feature_scale: tuple[float, ...]


class SmallWAMActionModel(nn.Module):
    """DynamicWAM mixture-of-tokens model for video-action flow matching."""

    def __init__(
        self,
        config: SmallWAMActionConfig,
        compact_wan: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.compact_wan = compact_wan
        if config.ae_num_layers != config.compact_wan.num_layers:
            raise ValueError(
                "DynamicWAM requires one action block per compact WAN block: "
                f"got ae_num_layers={config.ae_num_layers}, compact_wan.num_layers={config.compact_wan.num_layers}"
            )

        wan_cfg = {
            "dim": config.compact_wan.dim,
            "num_heads": config.compact_wan.num_heads,
            "head_dim": config.compact_wan.head_dim,
        }
        self.action_expert = ActionExpert(
            ActionExpertConfig(
                dim=config.ae_dim,
                ffn_dim=config.ae_ffn_dim,
                num_layers=config.ae_num_layers,
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                chunk_size=config.chunk_size,
            ),
            wan_cfg,
        )
        self.absolute_motion_tokens = AbsoluteMotionTokenModule(
            dim=config.ae_dim,
            history_count=config.motion_history_count,
            feature_mean=config.motion_feature_mean,
            feature_scale=config.motion_feature_scale,
        )
        if config.wan_frozen:
            for param in self.compact_wan.parameters():
                param.requires_grad_(False)

        self._rope_freq_grid_cache: Dict[tuple[object, ...], torch.Tensor] = {}

    def _cached_rope_freq_grid(
        self,
        freqs: torch.Tensor,
        grid_shape: tuple[int, int, int],
        complex_dim: int,
    ) -> torch.Tensor:
        f, h, w = (int(value) for value in grid_shape)
        key = (
            freqs.device.type,
            freqs.device.index,
            str(freqs.dtype),
            int(freqs.data_ptr()),
            int(complex_dim),
            f,
            h,
            w,
        )
        cached = self._rope_freq_grid_cache.get(key)
        if cached is not None:
            return cached

        c_f = complex_dim - 2 * (complex_dim // 3)
        c_h = complex_dim // 3
        c_w = complex_dim // 3
        fpart, hpart, wpart = freqs.split([c_f, c_h, c_w], dim=1)
        freq_grid = (
            torch.cat(
                [
                    fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, complex_dim)
            .contiguous()
        )
        self._rope_freq_grid_cache[key] = freq_grid
        return freq_grid

    def _rope_apply_exact_grid(
        self,
        heads: torch.Tensor,
        grid_shape: tuple[int, int, int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, num_heads, head_dim = heads.shape
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        expected_seq_len = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
        if seq_len != expected_seq_len:
            raise ValueError(
                "RoPE exact-grid fast path received mismatched sequence length: "
                f"{seq_len} != {expected_seq_len}"
            )
        complex_dim = head_dim // 2
        with torch.amp.autocast("cuda", enabled=False):
            freq_grid = self._cached_rope_freq_grid(freqs, grid_shape, complex_dim)
            heads_complex = torch.view_as_complex(
                heads.to(torch.float64).reshape(
                    batch, seq_len, num_heads, complex_dim, 2
                )
            ).contiguous()
            rotated = heads_complex * freq_grid
            return (
                torch.view_as_real(rotated)
                .reshape(batch, seq_len, num_heads, head_dim)
                .float()
            )

    def _multiscale_joint_attention_fast(
        self,
        attn: nn.Module,
        norm_video: torch.Tensor,
        action_q: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        joint_seq_lens: torch.Tensor,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
        attention_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = norm_video.shape[:2]
        action_len = action_q.shape[1]
        condition_seq_len = int(grid_sizes["condition_seq_len"])
        future_seq_len = int(grid_sizes["future_seq_len"])
        if condition_seq_len + future_seq_len != seq_len:
            raise ValueError(
                "Multiscale sequence layout does not match video tokens: "
                f"{condition_seq_len} + {future_seq_len} != {seq_len}"
            )
        condition_grid_shape = grid_sizes.get("condition_grid_shape")
        future_grid_shape = grid_sizes.get("future_grid_shape")
        if condition_grid_shape is None or future_grid_shape is None:
            video_q = attn.norm_q(attn.q(norm_video)).view(
                batch, seq_len, attn.num_heads, attn.head_dim
            )
            video_k = attn.norm_k(attn.k(norm_video)).view(
                batch, seq_len, attn.num_heads, attn.head_dim
            )
            video_v = attn.v(norm_video).view(
                batch, seq_len, attn.num_heads, attn.head_dim
            )
            video_q = self.compact_wan.apply_multiscale_rope(video_q, grid_sizes, freqs)
            video_k = self.compact_wan.apply_multiscale_rope(video_k, grid_sizes, freqs)
            attended = flash_attention(
                q=torch.cat([video_q, action_q.to(dtype=attention_dtype)], dim=1),
                k=torch.cat([video_k, action_k.to(dtype=attention_dtype)], dim=1),
                v=torch.cat([video_v, action_v.to(dtype=attention_dtype)], dim=1),
                k_lens=joint_seq_lens,
                window_size=attn.window_size,
            )
            return attn.o(attended[:, :seq_len].flatten(2)), attended[:, seq_len:]

        condition_grid_shape = tuple(int(value) for value in condition_grid_shape)
        future_grid_shape = tuple(int(value) for value in future_grid_shape)
        video_q = attn.norm_q(attn.q(norm_video)).view(
            batch, seq_len, attn.num_heads, attn.head_dim
        )
        video_k = attn.norm_k(attn.k(norm_video)).view(
            batch, seq_len, attn.num_heads, attn.head_dim
        )
        video_v = attn.v(norm_video).view(batch, seq_len, attn.num_heads, attn.head_dim)

        total_len = seq_len + action_len
        q_cat = video_q.new_empty(
            (batch, total_len, attn.num_heads, attn.head_dim), dtype=torch.float32
        )
        k_cat = video_k.new_empty(
            (batch, total_len, attn.num_heads, attn.head_dim), dtype=torch.float32
        )
        v_cat = video_v.new_empty(
            (batch, total_len, attn.num_heads, attn.head_dim), dtype=video_v.dtype
        )

        condition_slice = slice(0, condition_seq_len)
        future_slice = slice(condition_seq_len, seq_len)
        action_slice = slice(seq_len, total_len)
        q_cat[:, condition_slice] = self._rope_apply_exact_grid(
            video_q[:, condition_slice],
            condition_grid_shape,
            freqs,
        )
        q_cat[:, future_slice] = self._rope_apply_exact_grid(
            video_q[:, future_slice],
            future_grid_shape,
            freqs,
        )
        q_cat[:, action_slice] = action_q.to(dtype=q_cat.dtype)
        k_cat[:, condition_slice] = self._rope_apply_exact_grid(
            video_k[:, condition_slice],
            condition_grid_shape,
            freqs,
        )
        k_cat[:, future_slice] = self._rope_apply_exact_grid(
            video_k[:, future_slice],
            future_grid_shape,
            freqs,
        )
        k_cat[:, action_slice] = action_k.to(dtype=k_cat.dtype)
        v_cat[:, :seq_len] = video_v
        v_cat[:, action_slice] = action_v.to(dtype=v_cat.dtype)

        attended = flash_attention(
            q=q_cat,
            k=k_cat,
            v=v_cat,
            k_lens=joint_seq_lens,
            window_size=attn.window_size,
        )
        return attn.o(attended[:, :seq_len].flatten(2)), attended[:, seq_len:]

    @staticmethod
    def _expand_time_for_tokens(t: torch.Tensor, seq_len: int) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        return t

    def _build_action_time_embeddings(
        self,
        t: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._expand_time_for_tokens(t, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=t.is_cuda):
            batch = t.size(0)
            flat_t = t.flatten()
            emb = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, flat_t)
                .unflatten(0, (batch, seq_len))
                .float()
                .to(t.device)
            )
            adaln = self.action_expert.time_projection(emb).unflatten(
                2, (6, self.config.ae_dim)
            )
        return emb, adaln

    def _build_video_time_embeddings(
        self,
        t: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.compact_wan.prepare_time_embeddings(t, seq_len)

    @staticmethod
    def _block_modulation(block, adaln_params: torch.Tensor):
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=adaln_params.is_cuda
        ):
            return (block.modulation.unsqueeze(0) + adaln_params).chunk(6, dim=2)

    @staticmethod
    def _action_qkv_for_wan(
        block, norm_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = block.wan_action_qkv.to(
            device=norm_action.device, dtype=norm_action.dtype
        )
        action_qkv = (
            torch.einsum("btd,hdf->bthf", norm_action, qkv[0]),
            torch.einsum("btd,hdf->bthf", norm_action, qkv[1]),
            torch.einsum("btd,hdf->bthf", norm_action, qkv[2]),
        )
        action_q_h, action_k_h, action_v_h = action_qkv
        batch, seq_len, heads, head_dim = action_q_h.shape
        action_q = block.wan_action_norm_q(action_q_h.flatten(2)).view(
            batch, seq_len, heads, head_dim
        )
        action_k = block.wan_action_norm_k(action_k_h.flatten(2)).view(
            batch, seq_len, heads, head_dim
        )
        action_v = action_v_h.view(batch, seq_len, heads, head_dim)
        return action_q, action_k, action_v

    def _joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_modulation: tuple[torch.Tensor, ...],
        action_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        seq_lens: torch.Tensor,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        action_block = self.action_expert.blocks[layer_idx]
        video_mod = video_modulation
        action_mod = action_modulation

        norm_video = wan_layer.norm1(video_tokens).float() * (
            1 + video_mod[1].squeeze(2)
        ) + video_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens).float() * (
            1 + action_mod[1].squeeze(2)
        ) + action_mod[0].squeeze(2)
        action_q, action_k, action_v = self._action_qkv_for_wan(
            action_block, norm_action
        )

        joint_seq_lens = seq_lens + action_tokens.shape[1]
        attention_dtype = self.compact_wan.video_model.precision
        with torch.autocast("cuda", dtype=attention_dtype, enabled=norm_video.is_cuda):
            video_out, action_out_h = self._multiscale_joint_attention_fast(
                wan_layer.self_attn,
                norm_video,
                action_q,
                action_k,
                action_v,
                joint_seq_lens,
                grid_sizes,
                freqs,
                attention_dtype,
            )

        action_out_h = action_out_h.flatten(2).to(
            device=action_block.wan_action_o.weight.device,
            dtype=action_block.wan_action_o.weight.dtype,
        )
        action_out = action_block.wan_action_o(action_out_h)
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=video_tokens.is_cuda
        ):
            video_tokens = video_tokens + video_out * video_mod[2].squeeze(2)
            action_tokens = action_tokens + action_out * action_mod[2].squeeze(2)
        return video_tokens, action_tokens

    def _video_attention_kv_for_cache(
        self,
        video_tokens: torch.Tensor,
        video_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        attn = wan_layer.self_attn
        video_mod = video_modulation
        norm_video = wan_layer.norm1(video_tokens).float() * (
            1 + video_mod[1].squeeze(2)
        ) + video_mod[0].squeeze(2)
        batch, seq_len = norm_video.shape[:2]
        heads = attn.num_heads
        head_dim = attn.head_dim
        attention_dtype = self.compact_wan.video_model.precision
        with torch.autocast("cuda", dtype=attention_dtype, enabled=norm_video.is_cuda):
            video_k = attn.norm_k(attn.k(norm_video)).view(
                batch, seq_len, heads, head_dim
            )
            video_v = attn.v(norm_video).view(batch, seq_len, heads, head_dim)
            video_k = self.compact_wan.apply_multiscale_rope(
                video_k,
                grid_sizes,
                freqs,
            )
        return video_k.detach(), video_v.detach()

    def _empty_video_cache(
        self,
        seq_lens: torch.Tensor,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> Dict[str, object]:
        return {
            "seq_lens": seq_lens.detach(),
            "grid_sizes": {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in grid_sizes.items()
            },
            "freqs": freqs.detach(),
            "video_k": [],
            "video_v": [],
        }

    def _append_video_cache_layer(
        self,
        cache: Dict[str, object],
        video_tokens: torch.Tensor,
        video_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> None:
        video_k, video_v = self._video_attention_kv_for_cache(
            video_tokens,
            video_modulation,
            layer_idx,
            grid_sizes,
            freqs,
        )
        cache["video_k"].append(video_k)
        cache["video_v"].append(video_v)

    def _action_attention_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        video_cache: Dict[str, object],
    ) -> torch.Tensor:
        wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        action_block = self.action_expert.blocks[layer_idx]
        action_mod = action_modulation

        norm_action = action_block.norm1(action_tokens).float() * (
            1 + action_mod[1].squeeze(2)
        ) + action_mod[0].squeeze(2)
        action_q, action_k, action_v = self._action_qkv_for_wan(
            action_block, norm_action
        )

        video_k = video_cache["video_k"][layer_idx].to(
            device=action_q.device, dtype=action_q.dtype
        )
        video_v = video_cache["video_v"][layer_idx].to(
            device=action_v.device, dtype=action_v.dtype
        )
        joint_k = torch.cat([video_k, action_k], dim=1)
        joint_v = torch.cat([video_v, action_v], dim=1)
        video_seq_lens = video_cache["seq_lens"].to(device=action_q.device)
        joint_seq_lens = video_seq_lens + action_tokens.shape[1]

        attention_dtype = self.compact_wan.video_model.precision
        with torch.autocast("cuda", dtype=attention_dtype, enabled=action_q.is_cuda):
            action_out_h = flash_attention(
                q=action_q.to(dtype=attention_dtype),
                k=joint_k.to(dtype=attention_dtype),
                v=joint_v.to(dtype=attention_dtype),
                k_lens=joint_seq_lens,
                window_size=wan_layer.self_attn.window_size,
            )

        action_out_h = action_out_h.flatten(2).to(
            device=action_block.wan_action_o.weight.device,
            dtype=action_block.wan_action_o.weight.dtype,
        )
        action_out = action_block.wan_action_o(action_out_h)
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=action_tokens.is_cuda
        ):
            return action_tokens + action_out * action_mod[2].squeeze(2)

    def _build_action_tokens(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_actions: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        required = {
            "absolute_motion_features",
            "absolute_motion_interval_valid_mask",
            "absolute_motion_acceleration_valid_mask",
        }
        missing = sorted(required - batch.keys())
        if missing:
            raise KeyError(f"absolute motion inputs are missing: {missing}")
        registers = self.action_expert.registers.expand(
            noisy_actions.shape[0],
            -1,
            -1,
        )
        state_tokens = initial_state.unsqueeze(1)
        base_tokens = self.action_expert.input_encoder(
            state_tokens,
            noisy_actions,
            registers,
        )
        core_length = 1 + int(noisy_actions.shape[1])
        register_count = int(self.action_expert.config.num_registers)
        if int(base_tokens.shape[1]) != core_length + register_count:
            raise RuntimeError("unexpected state/action/register token layout")
        motion_tokens = self.absolute_motion_tokens(
            batch["absolute_motion_features"],
            batch["absolute_motion_interval_valid_mask"],
            batch["absolute_motion_acceleration_valid_mask"],
            dtype=base_tokens.dtype,
            device=base_tokens.device,
        )
        return (
            torch.cat(
                (
                    base_tokens[:, :core_length],
                    motion_tokens,
                    base_tokens[:, core_length:],
                ),
                dim=1,
            ),
            core_length,
        )

    def _decode_action_tokens(
        self,
        action_tokens: torch.Tensor,
        action_head_time_emb: torch.Tensor,
        core_length: int,
    ) -> torch.Tensor:
        if core_length != 1 + int(self.config.chunk_size):
            raise RuntimeError(
                f"action core length {core_length} differs from configured "
                f"chunk size {self.config.chunk_size}"
            )
        action_pred_full = self.action_expert.decoder(
            action_tokens[:, :core_length],
            action_head_time_emb[:, :core_length],
        )
        return action_pred_full[:, 1:, :]

    def forward_action_with_video_cache(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        video_cache = batch["video_cache"]
        noisy_actions = batch["noisy_actions"]
        action_t = batch["action_t"]
        initial_state = batch["initial_state"]

        action_tokens, action_core_length = self._build_action_tokens(
            batch,
            noisy_actions,
            initial_state,
        )
        action_head_time_emb, action_adaln_params = self._build_action_time_embeddings(
            action_t,
            action_tokens.shape[1],
        )
        for layer_idx in range(self.config.ae_num_layers):
            action_block = self.action_expert.blocks[layer_idx]
            action_modulation = self._block_modulation(
                action_block,
                action_adaln_params,
            )
            action_tokens = self._action_attention_with_video_cache(
                action_tokens,
                action_modulation,
                layer_idx,
                video_cache,
            )
            action_ffn_in = action_block.norm2(action_tokens).float() * (
                1 + action_modulation[4].squeeze(2)
            ) + action_modulation[3].squeeze(2)
            action_ffn_weight = action_block.ffn[0].weight
            action_ffn = action_block.ffn(
                action_ffn_in.to(
                    device=action_ffn_weight.device,
                    dtype=action_ffn_weight.dtype,
                )
            )
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float32,
                enabled=action_tokens.is_cuda,
            ):
                action_tokens = action_tokens + action_ffn * action_modulation[
                    5
                ].squeeze(2)

        action_pred = self._decode_action_tokens(
            action_tokens,
            action_head_time_emb,
            action_core_length,
        )
        return {"action_pred": action_pred}

    def _forward_video_action(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        include_video_prediction: bool,
        build_video_cache: bool,
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = batch["text_embeddings"]
        noisy_actions = batch["noisy_actions"]
        action_t = batch["action_t"]
        initial_state = batch["initial_state"]
        video_t = batch["video_t"]

        video_tokens, seq_lens, grid_sizes, freqs = (
            self.compact_wan.prepare_multiscale_video_tokens(
                batch["condition_latent"],
                batch["future_latent"],
            )
        )
        text_context = self.compact_wan.prepare_text_context(text_embeddings)
        video_head_time_emb, video_adaln_params = self._build_video_time_embeddings(
            video_t,
            video_tokens.shape[1],
        )

        action_tokens, action_core_length = self._build_action_tokens(
            batch,
            noisy_actions,
            initial_state,
        )
        action_head_time_emb, action_adaln_params = self._build_action_time_embeddings(
            action_t,
            action_tokens.shape[1],
        )

        video_cache = (
            self._empty_video_cache(seq_lens, grid_sizes, freqs)
            if build_video_cache
            else None
        )
        for layer_idx in range(self.config.ae_num_layers):
            wan_layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
            action_block = self.action_expert.blocks[layer_idx]
            video_modulation = self._block_modulation(
                wan_layer,
                video_adaln_params,
            )
            action_modulation = self._block_modulation(
                action_block,
                action_adaln_params,
            )
            if video_cache is not None:
                self._append_video_cache_layer(
                    video_cache,
                    video_tokens,
                    video_modulation,
                    layer_idx,
                    grid_sizes,
                    freqs,
                )
            video_tokens, action_tokens = self._joint_attention(
                video_tokens,
                action_tokens,
                video_modulation,
                action_modulation,
                layer_idx,
                seq_lens,
                grid_sizes,
                freqs,
            )

            cross_dtype = self.compact_wan.video_model.precision
            with torch.autocast(
                "cuda",
                dtype=cross_dtype,
                enabled=video_tokens.is_cuda,
            ):
                cross_in = wan_layer.norm3(video_tokens)
                cross_out = wan_layer.cross_attn(cross_in, text_context, None)
            video_tokens = video_tokens + cross_out
            video_ffn_in = wan_layer.norm2(video_tokens).float() * (
                1 + video_modulation[4].squeeze(2)
            ) + video_modulation[3].squeeze(2)
            video_ffn_weight = wan_layer.ffn[0].weight
            video_ffn = wan_layer.ffn(
                video_ffn_in.to(
                    device=video_ffn_weight.device,
                    dtype=video_ffn_weight.dtype,
                )
            )
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float32,
                enabled=video_tokens.is_cuda,
            ):
                video_tokens = video_tokens + video_ffn * video_modulation[5].squeeze(2)

            action_ffn_in = action_block.norm2(action_tokens).float() * (
                1 + action_modulation[4].squeeze(2)
            ) + action_modulation[3].squeeze(2)
            action_ffn_weight = action_block.ffn[0].weight
            action_ffn = action_block.ffn(
                action_ffn_in.to(
                    device=action_ffn_weight.device,
                    dtype=action_ffn_weight.dtype,
                )
            )
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float32,
                enabled=action_tokens.is_cuda,
            ):
                action_tokens = action_tokens + action_ffn * action_modulation[
                    5
                ].squeeze(2)

        action_pred = self._decode_action_tokens(
            action_tokens,
            action_head_time_emb,
            action_core_length,
        )
        outputs = {"action_pred": action_pred}
        if include_video_prediction:
            outputs["video_pred"] = self.compact_wan.apply_multiscale_video_head(
                video_tokens,
                video_head_time_emb,
                grid_sizes,
            )
        if video_cache is not None:
            outputs["video_cache"] = video_cache
        return outputs

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the frozen Stage 2 or joint Stage 3 training contract."""
        return self._forward_video_action(
            batch,
            include_video_prediction=not self.config.wan_frozen,
            build_video_cache=False,
        )

    def forward_with_video_cache(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run a slow inference refresh and return its reusable video cache."""
        return self._forward_video_action(
            batch,
            include_video_prediction=True,
            build_video_cache=True,
        )
