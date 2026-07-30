"""Prepare PCA statistics for Stage 1 distillation."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

from dynamicwam.config import load_profile, write_config_snapshot
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.external_assets import verify_wan_assets
from dynamicwam.training.data.training_dataset import (
    build_packed_training_dataset,
    validate_dataset_identity,
)
from dynamicwam.training.models.wan_model import WanVideoModel
from dynamicwam.training.train.common import (
    build_arg_parser,
    build_training_flow_scheduler,
    setup_logging,
)
from dynamicwam.training.train.stage1_pca_sampling import build_packed_pca_sample_ids
from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler


def _build_dataset(cfg: Dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Stage 1 PCA dataset config must be a mapping")
    return build_packed_training_dataset(dataset_cfg)


def _build_teacher(cfg: Dict[str, Any], device: str) -> WanVideoModel:
    teacher_cfg = cfg["teacher"]
    return WanVideoModel.from_pretrained(
        checkpoint_path=teacher_cfg["checkpoint_path"],
        config_path=teacher_cfg["config_path"],
        device=device,
        precision=teacher_cfg["precision"],
    )


def _validate_config(config: Dict[str, Any]) -> None:
    require_exact_keys(
        config,
        {
            "name",
            "device",
            "log_level",
            "flow_matching",
            "dataset",
            "teacher",
            "student",
            "distill",
            "pca_prep",
            "artifacts",
        },
        "Stage 1 PCA config",
    )
    teacher = require_exact_keys(
        config["teacher"],
        {"checkpoint_path", "config_path", "precision"},
        "Stage 1 PCA teacher",
    )
    if teacher["precision"] != "bfloat16":
        raise ValueError("Stage 1 PCA requires bfloat16 teacher precision")
    require_exact_keys(
        config["student"],
        {"hidden_anchor_layers", "motion_anchor_layers"},
        "Stage 1 PCA student",
    )
    require_exact_keys(
        config["distill"],
        {
            "projection_dim",
            "hidden_teacher_layers",
            "motion_teacher_layers",
        },
        "Stage 1 PCA distill",
    )
    require_exact_keys(
        config["pca_prep"],
        {
            "episodes",
            "subclips_per_episode",
            "states_per_subclip",
            "max_tokens_per_layer",
            "seed",
        },
        "Stage 1 PCA sampling",
    )
    require_exact_keys(
        config["artifacts"],
        {"output_dir"},
        "Stage 1 PCA artifacts",
    )


def _prepare_packed_latents(
    sample: Dict[str, Any],
    teacher: WanVideoModel,
    scheduler: FlowMatchScheduler,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Noise an already-packed head-flow sample without re-encoding RGB frames."""
    if "condition_latent" not in sample or "future_latent" not in sample:
        raise KeyError("Packed PCA sample requires condition_latent and future_latent")
    condition_latent = (
        sample["condition_latent"]
        .unsqueeze(0)
        .to(
            device=teacher.device,
            dtype=teacher.precision,
        )
    )
    clean_future_latent = (
        sample["future_latent"]
        .unsqueeze(0)
        .to(
            device=teacher.device,
            dtype=teacher.precision,
        )
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    timestep_id = torch.randint(
        0, scheduler.num_train_timesteps, (1,), generator=generator
    )
    t_embed = scheduler.timesteps[timestep_id].to(
        dtype=teacher.precision, device=teacher.device
    )
    sigma = (
        scheduler.sigmas[timestep_id]
        .to(dtype=teacher.precision, device=teacher.device)
        .view(1, 1, 1, 1, 1)
    )
    noise = torch.randn(
        clean_future_latent.shape, generator=generator, dtype=torch.float32
    ).to(
        device=teacher.device,
        dtype=teacher.precision,
    )
    future_latent = clean_future_latent * (1 - sigma) + noise * sigma
    return {
        "x_t": {
            "condition_latent": condition_latent,
            "future_latent": future_latent,
        },
        "t": t_embed,
    }


def _distributed_info(device: str) -> tuple[int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    resolved_device = device

    if world_size > 1:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            resolved_device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            backend = "gloo"
        if not dist.is_initialized():
            if backend == "nccl":
                device_id = torch.device(resolved_device)
                try:
                    dist.init_process_group(backend=backend, device_id=device_id)
                except TypeError:
                    dist.init_process_group(backend=backend)
            else:
                dist.init_process_group(backend=backend)
    return rank, world_size, resolved_device


def _distributed_barrier(device: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _rank_item_count(total_items: int, rank: int, world_size: int) -> int:
    if rank >= total_items:
        return 0
    return (total_items - 1 - rank) // world_size + 1


def _rank_token_budget(max_tokens: int, rank: int, world_size: int) -> int:
    if world_size <= 1:
        return max_tokens
    base = max_tokens // world_size
    remainder = max_tokens % world_size
    return base + (1 if rank < remainder else 0)


def _write_progress_file(out_dir: Path, rank: int, completed_items: int) -> None:
    progress_path = out_dir / f".pca_progress_rank{rank:03d}.txt"
    tmp_path = out_dir / f".pca_progress_rank{rank:03d}.tmp"
    tmp_path.write_text(str(int(completed_items)))
    tmp_path.replace(progress_path)


def _read_total_progress(out_dir: Path, world_size: int) -> int:
    total = 0
    for rank in range(world_size):
        progress_path = out_dir / f".pca_progress_rank{rank:03d}.txt"
        if not progress_path.exists():
            continue
        try:
            total += int(progress_path.read_text().strip() or "0")
        except ValueError:
            continue
    return total


def _cleanup_progress_files(out_dir: Path, world_size: int) -> None:
    for rank in range(world_size):
        for suffix in ("txt", "tmp"):
            path = out_dir / f".pca_progress_rank{rank:03d}.{suffix}"
            if path.exists():
                path.unlink()


def _teacher_feature_dim(teacher: WanVideoModel) -> int:
    dim = getattr(teacher.wan_model, "dim", None)
    if dim is None:
        raise AttributeError(
            "WAN model does not expose hidden dimension via wan_model.dim"
        )
    return int(dim)


def _normalized_tokens(hidden: torch.Tensor) -> torch.Tensor:
    tokens = hidden.detach().reshape(-1, hidden.shape[-1]).float()
    return F.layer_norm(tokens, (tokens.shape[-1],))


def _subsample_tokens(tokens: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor:
    if max_tokens <= 0 or tokens.shape[0] == 0:
        return tokens[:0]
    if tokens.shape[0] <= max_tokens:
        return tokens
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(tokens.shape[0], generator=generator)[:max_tokens]
    return tokens[indices]


def run_pca_prep(
    config: Dict[str, Any],
    device: str,
    *,
    external_assets_manifest: Path,
) -> Path:
    _validate_config(config)
    rank, world_size, device = _distributed_info(device)
    if rank == 0:
        verify_wan_assets(
            root=Path(config["teacher"]["checkpoint_path"]),
            manifest_path=external_assets_manifest,
            purpose="training",
        )
    if world_size > 1:
        _distributed_barrier(device)
    dataset = _build_dataset(config)
    teacher = _build_teacher(config, device=device)
    teacher.eval()
    scheduler = build_training_flow_scheduler(config)

    teacher_layers = sorted(
        set(
            config["distill"]["hidden_teacher_layers"]
            + config["distill"]["motion_teacher_layers"]
        )
    )
    max_tokens = int(config["pca_prep"]["max_tokens_per_layer"])
    per_rank_token_budget = _rank_token_budget(max_tokens, rank, world_size)
    token_counts = dict.fromkeys(teacher_layers, 0)
    feature_dim = _teacher_feature_dim(teacher)
    stats_dtype = torch.float64
    layer_stats = {
        layer: {
            "count": torch.zeros(1, device=device, dtype=stats_dtype),
            "sum": torch.zeros(feature_dim, device=device, dtype=stats_dtype),
            "xtx": torch.zeros(
                (feature_dim, feature_dim), device=device, dtype=stats_dtype
            ),
        }
        for layer in teacher_layers
    }

    requested_episodes = int(config["pca_prep"]["episodes"])
    subclips_per_episode = int(config["pca_prep"]["subclips_per_episode"])
    states_per_subclip = int(config["pca_prep"]["states_per_subclip"])
    total_samples = requested_episodes * subclips_per_episode * states_per_subclip
    packed_sample_ids = build_packed_pca_sample_ids(
        dataset,
        requested_episodes=requested_episodes,
        samples_per_episode=subclips_per_episode * states_per_subclip,
        seed=int(config["pca_prep"]["seed"]),
    )
    if len(packed_sample_ids) != total_samples:
        raise RuntimeError(
            "Packed PCA selection size mismatch: "
            f"expected={total_samples}, got={len(packed_sample_ids)}"
        )
    rank_total_samples = _rank_item_count(total_samples, rank, world_size)
    per_rank_sample_token_budget = max(
        1, math.ceil(per_rank_token_budget / max(1, rank_total_samples))
    )
    out_dir = Path(config["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_progress_file(out_dir, rank, 0)
    if world_size > 1:
        _distributed_barrier(device)
    progress = tqdm(
        total=total_samples,
        desc="Stage1 PCA prep",
        dynamic_ncols=True,
        disable=rank != 0,
    )
    completed_items = 0
    last_progress_write = time.monotonic()
    last_progress_poll = time.monotonic()

    for global_sample_index in range(rank, total_samples, world_size):
        sample_id = int(packed_sample_ids[global_sample_index])
        raw = dataset[sample_id]
        prepared = _prepare_packed_latents(
            raw,
            teacher,
            scheduler,
            seed=global_sample_index,
        )
        episode_index = int(raw["episode_index"])
        subclip_id = global_sample_index % (subclips_per_episode * states_per_subclip)
        state_id = sample_id
        text_embedding = raw["language_embedding"]
        with torch.no_grad():
            hidden_features = teacher.get_multiscale_layer_features(
                condition_latent=prepared["x_t"]["condition_latent"],
                future_latent=prepared["x_t"]["future_latent"],
                timestep=prepared["t"],
                text_embeddings=[text_embedding.to(device)],
                layer_indices=teacher_layers,
            )
        for layer, hidden in zip(
            teacher_layers,
            hidden_features,
            strict=True,
        ):
            if token_counts[layer] >= per_rank_token_budget:
                continue
            tokens = _normalized_tokens(hidden)
            tokens = _subsample_tokens(
                tokens,
                max_tokens=per_rank_sample_token_budget,
                seed=(global_sample_index + 1) * 1000 + int(layer),
            )
            remaining = per_rank_token_budget - token_counts[layer]
            tokens = tokens[:remaining]
            if tokens.numel() == 0:
                continue
            tokens = tokens.to(device=device, dtype=stats_dtype)
            layer_stats[layer]["sum"] += tokens.sum(dim=0)
            layer_stats[layer]["xtx"] += tokens.transpose(0, 1) @ tokens
            layer_stats[layer]["count"] += tokens.shape[0]
            token_counts[layer] += tokens.shape[0]
        completed_items += 1
        if world_size == 1:
            progress.update(1)
            progress.set_postfix(
                {
                    "episode": episode_index,
                    "subclip": subclip_id,
                    "state": state_id,
                }
            )
        else:
            now = time.monotonic()
            if (
                now - last_progress_write >= 1.0
                or completed_items == rank_total_samples
            ):
                _write_progress_file(out_dir, rank, completed_items)
                last_progress_write = now
            if rank == 0 and now - last_progress_poll >= 1.0:
                progress.n = _read_total_progress(out_dir, world_size)
                progress.refresh()
                last_progress_poll = now

    if world_size > 1:
        _write_progress_file(out_dir, rank, completed_items)
        _distributed_barrier(device)
        if rank == 0:
            progress.n = _read_total_progress(out_dir, world_size)
            progress.refresh()
            _cleanup_progress_files(out_dir, world_size)
        _distributed_barrier(device)
    progress.close()

    if world_size > 1:
        for layer in teacher_layers:
            dist.all_reduce(layer_stats[layer]["count"], op=dist.ReduceOp.SUM)
            dist.all_reduce(layer_stats[layer]["sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(layer_stats[layer]["xtx"], op=dist.ReduceOp.SUM)

    payload: Dict[str, Any] = {
        "projection_dim": int(config["distill"]["projection_dim"]),
        "layers": {},
        "provenance": {
            "dataset_type": "packed_absolute_motion",
            "dataset_root": str(config["dataset"]["root"]),
            "dataset_identity": validate_dataset_identity(dataset.dataset_identity),
            "requested_episodes": requested_episodes,
            "samples_per_episode": subclips_per_episode * states_per_subclip,
            "sample_count": total_samples,
            "seed": int(config["pca_prep"]["seed"]),
        },
    }
    output_path = out_dir / "pca_stats.pt"

    if rank == 0:
        projection_dim = int(config["distill"]["projection_dim"])
        for layer in teacher_layers:
            count = int(layer_stats[layer]["count"].item())
            if count <= 0:
                raise RuntimeError(
                    f"No PCA samples collected for teacher layer {layer}"
                )

            sum_vec = layer_stats[layer]["sum"]
            xtx = layer_stats[layer]["xtx"]
            mean = sum_vec / count
            covariance = xtx / count - torch.outer(mean, mean)
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))

            eigvals, eigvecs = torch.linalg.eigh(covariance)
            keep = min(projection_dim, eigvecs.shape[1], count)
            order = torch.argsort(eigvals, descending=True)[:keep]
            components = (
                eigvecs[:, order].contiguous().to(dtype=torch.float32, device="cpu")
            )
            payload["layers"][str(layer)] = {
                "mean": mean.to(dtype=torch.float32, device="cpu"),
                "components": components,
            }

        torch.save(payload, output_path)

    if world_size > 1:
        _distributed_barrier(device)
    return output_path


def main() -> None:
    parser = build_arg_parser("Prepare DynamicWAM Stage 1 PCA statistics")
    args = parser.parse_args()
    profile = load_profile(args.config)
    config = profile.training_config("stage1_pca")
    setup_logging(str(config["log_level"]), rank=int(os.environ.get("RANK", "0")))
    if int(os.environ.get("RANK", "0")) == 0:
        write_config_snapshot(
            Path(config["artifacts"]["output_dir"]) / "config_audit",
            profile=profile,
            label="stage1_pca",
            resolved_config={
                "training": config,
            },
        )
    try:
        output_path = run_pca_prep(
            config,
            device=str(config["device"]),
            external_assets_manifest=Path(
                profile.raw["paths"]["external_assets_manifest"]
            ),
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Saved PCA stats to {output_path}")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
