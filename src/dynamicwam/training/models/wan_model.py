"""WAN model support required by DynamicWAM distillation and training."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from dynamicwam.vendor.wan.modules.attention import flash_attention
from dynamicwam.vendor.wan.modules.model import (
    WanModel,
    rope_apply,
    sinusoidal_embedding_1d,
)

logger = logging.getLogger(__name__)


WAN_SHARD_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def _load_wan_arch_config(config_path: str) -> Dict[str, Any]:
    config_json_path = Path(config_path) / "config.json"
    if not config_json_path.is_file():
        raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
    with config_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_wan_sharded_state_dict(index_path: Path) -> Dict[str, torch.Tensor]:
    with index_path.open("r") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Shard index missing non-empty weight_map: {index_path}")

    state_dict: Dict[str, torch.Tensor] = {}
    shard_names = sorted({str(name) for name in weight_map.values()})
    if len(shard_names) != 3 or any(
        not name.startswith("diffusion_pytorch_model-")
        or not name.endswith(".safetensors")
        for name in shard_names
    ):
        raise ValueError(
            f"DynamicWAM WAN index must reference three safetensor shards: {shard_names}"
        )
    for shard_name in shard_names:
        shard_path = index_path.parent / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(
                f"WAN checkpoint shard listed in {index_path} not found: {shard_path}"
            )
        state_dict.update(safe_load_file(str(shard_path), device="cpu"))
    return state_dict


def _select_evenly_spaced_indices(total: int, keep: int) -> List[int]:
    if keep > total:
        raise ValueError(f"Cannot keep {keep} indices from total {total}")
    if keep == total:
        return list(range(total))
    steps = torch.linspace(0, total - 1, steps=keep)
    indices = torch.round(steps).to(torch.long).tolist()
    deduped: List[int] = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    cursor = 0
    while len(deduped) < keep:
        if cursor not in seen:
            deduped.append(cursor)
            seen.add(cursor)
        cursor += 1
    return sorted(deduped)


def _select_head_group_indices(
    total_heads: int, keep_heads: int, head_dim: int
) -> List[int]:
    head_indices = _select_evenly_spaced_indices(total_heads, keep_heads)
    hidden_indices: List[int] = []
    for head_idx in head_indices:
        start = head_idx * head_dim
        hidden_indices.extend(range(start, start + head_dim))
    return hidden_indices


def _select_ffn_indices(total_ffn_dim: int, keep_ffn_dim: int) -> List[int]:
    return _select_evenly_spaced_indices(total_ffn_dim, keep_ffn_dim)


def _slice_vector(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, idx).clone()


def _slice_linear(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    row_indices: Optional[Sequence[int]] = None,
    col_indices: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    out = weight
    if row_indices is not None:
        out = out.index_select(0, torch.as_tensor(row_indices, dtype=torch.long))
    if col_indices is not None:
        out = out.index_select(1, torch.as_tensor(col_indices, dtype=torch.long))
    result = {"weight": out.clone()}
    if bias is not None:
        if row_indices is not None:
            result["bias"] = bias.index_select(
                0, torch.as_tensor(row_indices, dtype=torch.long)
            ).clone()
        else:
            result["bias"] = bias.clone()
    return result


def _slice_conv3d_out_channels(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    idx = torch.as_tensor(out_indices, dtype=torch.long)
    result = {"weight": weight.index_select(0, idx).clone()}
    if bias is not None:
        result["bias"] = bias.index_select(0, idx).clone()
    return result


def _build_time_projection_row_indices(
    hidden_indices: Sequence[int], teacher_dim: int, groups: int = 6
) -> List[int]:
    row_indices: List[int] = []
    for group_idx in range(groups):
        offset = group_idx * teacher_dim
        row_indices.extend([offset + idx for idx in hidden_indices])
    return row_indices


def _copy_non_block_state(
    teacher_state: Dict[str, torch.Tensor],
    student_state: Dict[str, torch.Tensor],
    hidden_indices: Sequence[int],
    teacher_dim: int,
) -> None:
    patch = _slice_conv3d_out_channels(
        teacher_state["patch_embedding.weight"],
        teacher_state.get("patch_embedding.bias"),
        hidden_indices,
    )
    student_state["patch_embedding.weight"] = patch["weight"]
    if "bias" in patch:
        student_state["patch_embedding.bias"] = patch["bias"]

    text0 = _slice_linear(
        teacher_state["text_embedding.0.weight"],
        teacher_state.get("text_embedding.0.bias"),
        row_indices=hidden_indices,
    )
    student_state["text_embedding.0.weight"] = text0["weight"]
    student_state["text_embedding.0.bias"] = text0["bias"]
    text2 = _slice_linear(
        teacher_state["text_embedding.2.weight"],
        teacher_state.get("text_embedding.2.bias"),
        row_indices=hidden_indices,
        col_indices=hidden_indices,
    )
    student_state["text_embedding.2.weight"] = text2["weight"]
    student_state["text_embedding.2.bias"] = text2["bias"]

    time0 = _slice_linear(
        teacher_state["time_embedding.0.weight"],
        teacher_state.get("time_embedding.0.bias"),
        row_indices=hidden_indices,
    )
    student_state["time_embedding.0.weight"] = time0["weight"]
    student_state["time_embedding.0.bias"] = time0["bias"]
    time2 = _slice_linear(
        teacher_state["time_embedding.2.weight"],
        teacher_state.get("time_embedding.2.bias"),
        row_indices=hidden_indices,
        col_indices=hidden_indices,
    )
    student_state["time_embedding.2.weight"] = time2["weight"]
    student_state["time_embedding.2.bias"] = time2["bias"]

    row_indices = _build_time_projection_row_indices(
        hidden_indices, teacher_dim, groups=6
    )
    time_proj = _slice_linear(
        teacher_state["time_projection.1.weight"],
        teacher_state.get("time_projection.1.bias"),
        row_indices=row_indices,
        col_indices=hidden_indices,
    )
    student_state["time_projection.1.weight"] = time_proj["weight"]
    student_state["time_projection.1.bias"] = time_proj["bias"]

    head = _slice_linear(
        teacher_state["head.head.weight"],
        teacher_state.get("head.head.bias"),
        col_indices=hidden_indices,
    )
    student_state["head.head.weight"] = head["weight"]
    student_state["head.head.bias"] = head["bias"]
    student_state["head.modulation"] = teacher_state["head.modulation"][
        :, :, hidden_indices
    ].clone()


def _copy_block_state(
    teacher_state: Dict[str, torch.Tensor],
    student_state: Dict[str, torch.Tensor],
    teacher_idx: int,
    student_idx: int,
    hidden_indices: Sequence[int],
    ffn_indices: Sequence[int],
) -> None:
    teacher_prefix = f"blocks.{teacher_idx}."
    student_prefix = f"blocks.{student_idx}."

    def t(name: str) -> str:
        return teacher_prefix + name

    def s(name: str) -> str:
        return student_prefix + name

    # Norms
    for norm_name in [
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "cross_attn.norm_q.weight",
        "cross_attn.norm_k.weight",
        "norm3.weight",
        "norm3.bias",
    ]:
        if t(norm_name) in teacher_state:
            student_state[s(norm_name)] = _slice_vector(
                teacher_state[t(norm_name)], hidden_indices
            )

    # Attention projections
    for attn_name in [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
    ]:
        sliced = _slice_linear(
            teacher_state[t(f"{attn_name}.weight")],
            teacher_state.get(t(f"{attn_name}.bias")),
            row_indices=hidden_indices,
            col_indices=hidden_indices,
        )
        student_state[s(f"{attn_name}.weight")] = sliced["weight"]
        if "bias" in sliced:
            student_state[s(f"{attn_name}.bias")] = sliced["bias"]

    # FFN
    ffn0 = _slice_linear(
        teacher_state[t("ffn.0.weight")],
        teacher_state.get(t("ffn.0.bias")),
        row_indices=ffn_indices,
        col_indices=hidden_indices,
    )
    student_state[s("ffn.0.weight")] = ffn0["weight"]
    student_state[s("ffn.0.bias")] = ffn0["bias"]
    ffn2 = _slice_linear(
        teacher_state[t("ffn.2.weight")],
        teacher_state.get(t("ffn.2.bias")),
        row_indices=hidden_indices,
        col_indices=ffn_indices,
    )
    student_state[s("ffn.2.weight")] = ffn2["weight"]
    student_state[s("ffn.2.bias")] = ffn2["bias"]

    # Modulation
    student_state[s("modulation")] = teacher_state[t("modulation")][
        :, :, hidden_indices
    ].clone()


def _build_structured_sliced_wan_state_dict(
    teacher_state: Dict[str, torch.Tensor],
    teacher_model_config: Dict[str, Any],
    student_model_config: Dict[str, Any],
    teacher_layer_mapping: Sequence[int],
) -> Dict[str, torch.Tensor]:
    teacher_dim = int(teacher_model_config["dim"])
    teacher_heads = int(teacher_model_config["num_heads"])
    teacher_head_dim = teacher_dim // teacher_heads
    teacher_ffn_dim = int(teacher_model_config["ffn_dim"])

    student_heads = int(student_model_config["num_heads"])
    student_ffn_dim = int(student_model_config["ffn_dim"])
    student_layers = int(student_model_config["num_layers"])

    if len(teacher_layer_mapping) != student_layers:
        raise ValueError(
            f"Layer mapping length {len(teacher_layer_mapping)} must match student num_layers {student_layers}"
        )

    hidden_indices = _select_head_group_indices(
        teacher_heads, student_heads, teacher_head_dim
    )
    ffn_indices = _select_ffn_indices(teacher_ffn_dim, student_ffn_dim)

    student_state: Dict[str, torch.Tensor] = {}
    _copy_non_block_state(
        teacher_state=teacher_state,
        student_state=student_state,
        hidden_indices=hidden_indices,
        teacher_dim=teacher_dim,
    )

    zero_based_mapping = [layer - 1 for layer in teacher_layer_mapping]
    for student_idx, teacher_idx in enumerate(zero_based_mapping):
        _copy_block_state(
            teacher_state=teacher_state,
            student_state=student_state,
            teacher_idx=teacher_idx,
            student_idx=student_idx,
            hidden_indices=hidden_indices,
            ffn_indices=ffn_indices,
        )

    return student_state


def _load_wan_state_dict_from_checkpoint(
    checkpoint_path: str,
) -> Dict[str, torch.Tensor]:
    """Load the exact three-shard Wan2.2 checkpoint used by DynamicWAM."""
    path = Path(checkpoint_path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"DynamicWAM WAN checkpoint directory not found: {path}"
        )
    shard_index = path / WAN_SHARD_INDEX
    if not shard_index.is_file():
        raise FileNotFoundError(f"DynamicWAM WAN shard index not found: {shard_index}")
    logger.info("Loading DynamicWAM WAN weights from shard index %s", shard_index)
    return _load_wan_sharded_state_dict(shard_index)


class WanVideoModel(nn.Module):
    """Latent-only WAN wrapper for DynamicWAM teacher and compact student training."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        device: str = "cuda",
        precision: str = "bfloat16",
    ):
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

        logger.info(
            f"WAN Video Model initialized with {sum(p.numel() for p in self.wan_model.parameters()):,} parameters"
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor | int], torch.Tensor]:
        """Patchify high-resolution condition and low-resolution future latents."""
        self._validate_wan_latent("condition", condition_latent)
        self._validate_wan_latent("future", future_latent)
        if condition_latent.shape[0] != future_latent.shape[0]:
            raise ValueError(
                "Multiscale condition and future batch sizes differ: "
                f"{condition_latent.shape[0]} != {future_latent.shape[0]}"
            )
        device = self.wan_model.patch_embedding.weight.device
        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)
        condition_tokens, condition_grid_sizes = self._patch_video_latent(
            condition_latent
        )
        future_tokens, future_grid_sizes = self._patch_video_latent(future_latent)
        tokens = torch.cat([condition_tokens, future_tokens], dim=1)
        seq_lens = torch.full(
            (tokens.shape[0],),
            tokens.shape[1],
            dtype=torch.long,
            device=device,
        )
        layout: Dict[str, torch.Tensor | int] = {
            "condition_grid_sizes": condition_grid_sizes,
            "future_grid_sizes": future_grid_sizes,
            "condition_seq_len": int(condition_tokens.shape[1]),
            "future_seq_len": int(future_tokens.shape[1]),
        }
        patch_f, patch_h, patch_w = self.wan_model.patch_size
        layout["condition_grid_shape"] = (
            int(condition_latent.shape[2] // patch_f),
            int(condition_latent.shape[3] // patch_h),
            int(condition_latent.shape[4] // patch_w),
        )
        layout["future_grid_shape"] = (
            int(future_latent.shape[2] // patch_f),
            int(future_latent.shape[3] // patch_h),
            int(future_latent.shape[4] // patch_w),
        )
        return tokens, seq_lens, layout, self.wan_model.freqs

    @staticmethod
    def multiscale_layout_grid_sizes(
        layout: Dict[str, torch.Tensor | int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        condition_grid_sizes = layout["condition_grid_sizes"]
        future_grid_sizes = layout["future_grid_sizes"]
        condition_seq_len = int(layout["condition_seq_len"])
        if not isinstance(condition_grid_sizes, torch.Tensor) or not isinstance(
            future_grid_sizes, torch.Tensor
        ):
            raise TypeError("Multiscale video layout must carry tensor grid sizes")
        return condition_grid_sizes, future_grid_sizes, condition_seq_len

    def apply_multiscale_rope(
        self,
        heads: torch.Tensor,
        layout: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        condition_grid_sizes, future_grid_sizes, condition_seq_len = (
            self.multiscale_layout_grid_sizes(layout)
        )
        condition_heads = rope_apply(
            heads[:, :condition_seq_len], condition_grid_sizes, freqs
        )
        future_heads = rope_apply(
            heads[:, condition_seq_len:], future_grid_sizes, freqs
        )
        return torch.cat([condition_heads, future_heads], dim=1)

    def apply_video_self_attention(
        self,
        self_attn: nn.Module,
        video_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: Dict[str, torch.Tensor | int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = video_tokens.shape[:2]
        heads = int(self_attn.num_heads)
        head_dim = int(self_attn.head_dim)
        q = self_attn.norm_q(self_attn.q(video_tokens)).view(
            batch, seq_len, heads, head_dim
        )
        k = self_attn.norm_k(self_attn.k(video_tokens)).view(
            batch, seq_len, heads, head_dim
        )
        v = self_attn.v(video_tokens).view(batch, seq_len, heads, head_dim)
        attended = flash_attention(
            q=self.apply_multiscale_rope(q, grid_sizes, freqs),
            k=self.apply_multiscale_rope(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self_attn.window_size,
        )
        return self_attn.o(attended.flatten(2))

    def prepare_text_context(self, text_embeddings: List[torch.Tensor]) -> torch.Tensor:
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
        layout: Dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        """Unpatchify only future velocity tokens for a multiscale video."""
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

    def _run_multiscale_layers(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: List[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor | int],
        List[torch.Tensor],
    ]:
        requested_layer_indices = list(layer_indices)
        if requested_layer_indices and min(requested_layer_indices) >= 1:
            requested_layer_indices = [idx - 1 for idx in requested_layer_indices]
        x, seq_lens, layout, freqs = self.prepare_multiscale_video_tokens(
            condition_latent,
            future_latent,
        )
        e, e0 = self.prepare_time_embeddings(timestep, int(x.shape[1]))
        context = self.prepare_text_context(text_embeddings)
        device = self.wan_model.patch_embedding.weight.device
        layer_features = []
        with torch.autocast(
            "cuda", dtype=self.precision, enabled=device.type == "cuda"
        ):
            for i, block in enumerate(self.wan_model.blocks):
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.is_cuda):
                    modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                norm_video = block.norm1(x).float() * (
                    1 + modulation[1].squeeze(2)
                ) + modulation[0].squeeze(2)
                self_out = self.apply_video_self_attention(
                    block.self_attn,
                    norm_video,
                    seq_lens,
                    layout,
                    freqs,
                )
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.is_cuda):
                    x = x + self_out * modulation[2].squeeze(2)
                x = x + block.cross_attn(block.norm3(x), context, None)
                ffn_in = block.norm2(x).float() * (
                    1 + modulation[4].squeeze(2)
                ) + modulation[3].squeeze(2)
                ffn_weight = block.ffn[0].weight
                ffn_out = block.ffn(
                    ffn_in.to(device=ffn_weight.device, dtype=ffn_weight.dtype)
                )
                with torch.amp.autocast("cuda", dtype=torch.float32, enabled=x.is_cuda):
                    x = x + ffn_out * modulation[5].squeeze(2)
                if i in requested_layer_indices:
                    layer_features.append(x)
        return x, e, layout, layer_features

    def get_multiscale_layer_features(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: List[int],
    ) -> List[torch.Tensor]:
        """Extract selected WAN hidden features without running the video head."""
        _, _, _, layer_features = self._run_multiscale_layers(
            condition_latent,
            future_latent,
            timestep,
            text_embeddings,
            layer_indices,
        )
        return layer_features

    def forward_multiscale_with_features(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: List[int],
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Return the DynamicWAM video prediction and selected student features."""
        video_tokens, video_time_emb, layout, layer_features = (
            self._run_multiscale_layers(
                condition_latent,
                future_latent,
                timestep,
                text_embeddings,
                layer_indices,
            )
        )
        video_pred = self.apply_multiscale_video_head(
            video_tokens,
            video_time_emb,
            layout,
        )
        return video_pred, layer_features

    @classmethod
    def from_compact_config(
        cls,
        config_path: str,
        student_model_config: Dict[str, Any],
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> "WanVideoModel":
        """Initialize the compact WAN architecture without loading weights."""
        teacher_model_config = _load_wan_arch_config(config_path)
        merged_student_config = dict(teacher_model_config)
        merged_student_config.update(student_model_config)
        merged_student_config["num_layers"] = int(student_model_config["num_layers"])

        model = cls(
            model_config=merged_student_config,
            device=device,
            precision=precision,
        )
        logger.info(
            "Initialized compact WAN model from config only (no WAN weights loaded)"
        )
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> "WanVideoModel":
        """Load the full WAN teacher used by DynamicWAM distillation."""
        model_config = _load_wan_arch_config(config_path)
        model = cls(
            model_config=model_config,
            device=device,
            precision=precision,
        )
        try:
            logger.info(f"Loading WAN weights from {checkpoint_path}")
            wan_state_dict = _load_wan_state_dict_from_checkpoint(checkpoint_path)
            model.wan_model.load_state_dict(wan_state_dict, strict=True)
            logger.info("Successfully loaded WAN weights from checkpoint")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load WAN checkpoint from {checkpoint_path}"
            ) from e

        return model

    @classmethod
    def from_pretrained_compact(
        cls,
        checkpoint_path: str,
        student_model_config: Dict[str, Any],
        teacher_layer_mapping: Sequence[int],
        config_path: str,
        device: str = "cuda",
        precision: str = "bfloat16",
    ) -> "WanVideoModel":
        """Load a compact WAN initialized by structured slicing from the teacher checkpoint."""
        teacher_model_config = _load_wan_arch_config(config_path)
        model = cls.from_compact_config(
            config_path=config_path,
            student_model_config=student_model_config,
            device=device,
            precision=precision,
        )
        merged_student_config = dict(teacher_model_config)
        merged_student_config.update(student_model_config)
        merged_student_config["num_layers"] = int(student_model_config["num_layers"])

        try:
            logger.info(
                "Loading compact WAN from %s with structured slicing and teacher_layer_mapping=%s",
                checkpoint_path,
                list(teacher_layer_mapping),
            )
            teacher_state = _load_wan_state_dict_from_checkpoint(checkpoint_path)
            compact_state = _build_structured_sliced_wan_state_dict(
                teacher_state=teacher_state,
                teacher_model_config=teacher_model_config,
                student_model_config=merged_student_config,
                teacher_layer_mapping=teacher_layer_mapping,
            )
            model.wan_model.load_state_dict(compact_state, strict=True)
            logger.info("Successfully loaded compact WAN via structured slicing")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load compact WAN checkpoint from {checkpoint_path} "
                f"with teacher_layer_mapping={list(teacher_layer_mapping)}"
            ) from e

        return model
