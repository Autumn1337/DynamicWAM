"""Stage 1 DynamicWAM distillation training entry."""

from __future__ import annotations

import gc
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dynamicwam.config import load_profile, write_config_snapshot
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.training.data.packed_dataset import packed_collate_fn
from dynamicwam.training.data.training_dataset import (
    build_packed_training_dataset,
    make_packed_training_sampler,
    validate_dataset_identity,
)
from dynamicwam.training.models.compact_wan import CompactWANConfig, CompactWANModel
from dynamicwam.training.models.stage1_distill_heads import (
    DistillHeadConfig,
    Stage1DistillHeads,
)
from dynamicwam.training.models.stage1_distill_teacher import (
    Stage1DistillTeacher,
    Stage1DistillTeacherConfig,
)
from dynamicwam.training.train.common import (
    LogTracker,
    add_speed_metrics,
    build_accelerator,
    build_arg_parser,
    build_training_flow_scheduler,
    get_dataloader_config,
    get_effective_global_batch_size,
    get_epoch_progress_metrics,
    get_export_dir,
    get_learning_rate_metrics,
    get_per_device_batch_size,
    get_run_dir,
    init_experiment_trackers,
    logger,
    reduce_metrics_for_log,
    setup_logging,
)
from dynamicwam.training.utils.scheduler import create_scheduler


@dataclass
class Stage1System:
    bundle: nn.Module
    student: CompactWANModel
    distill_heads: Stage1DistillHeads
    teacher: Stage1DistillTeacher
    config: Dict[str, Any]


class Stage1TrainingBundle(nn.Module):
    def __init__(self, student: CompactWANModel, distill_heads: Stage1DistillHeads):
        super().__init__()
        self.student = student
        self.distill_heads = distill_heads

    def forward(
        self,
        x_t: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: List[int],
        hidden_anchor_layers: List[int],
        motion_anchor_layers: List[int],
        num_frames: int,
        condition_tokens: int,
    ) -> Dict[str, Any]:
        video_pred, hidden_features = self.student.forward_multiscale_with_features(
            condition_latent=x_t["condition_latent"],
            future_latent=x_t["future_latent"],
            timestep=timestep,
            text_embeddings=text_embeddings,
            layer_indices=layer_indices,
        )
        hidden_student = {
            layer: hidden_features[layer] for layer in hidden_anchor_layers
        }
        motion_student = {
            layer: hidden_features[layer] for layer in motion_anchor_layers
        }
        hidden_projected = self.distill_heads.project_hidden(hidden_student)
        motion_projected = self.distill_heads.project_motion(motion_student)
        motion_deltas = self.distill_heads.build_motion_deltas(
            motion_projected,
            num_frames=num_frames,
            condition_tokens=condition_tokens,
        )
        return {
            "video_pred": video_pred,
            "hidden_projected": hidden_projected,
            "motion_deltas": motion_deltas,
        }


class Stage1AccelerateTrainer:
    """Distributed Stage 1 trainer backed by Accelerate/DeepSpeed."""

    def __init__(self, config: Dict[str, Any], accelerator):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self._validate_config()
        self.train_loader = self._build_dataloader()
        self.system = self._build_system_staggered()
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
        self.fm_train_scheduler = build_training_flow_scheduler(self.config)
        self.timestep_sampling_weights = self._build_timestep_sampling_weights()
        self.global_step = 0

        self.system.bundle, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.system.bundle,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            )
        )

    def _unwrapped_bundle(self) -> Stage1TrainingBundle:
        return self.accelerator.unwrap_model(self.system.bundle)

    def _student(self) -> CompactWANModel:
        return self._unwrapped_bundle().student

    def _validate_config(self) -> None:
        require_exact_keys(
            self.config,
            {
                "name",
                "device",
                "log_level",
                "flow_matching",
                "allow_tf32",
                "cudnn_benchmark",
                "max_steps",
                "per_device_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "min_lr_ratio",
                "warmup_steps",
                "grad_clip_norm",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
                "checkpoint_dir",
                "log_interval",
                "checkpoint_interval",
                "checkpoint_total_limit",
                "resume_from",
                "dataset",
                "teacher",
                "student",
                "distill",
            },
            "Stage 1 config",
        )
        teacher = require_exact_keys(
            self.config["teacher"],
            {"checkpoint_path", "config_path", "precision", "pca_stats_path"},
            "Stage 1 teacher",
        )
        student = require_exact_keys(
            self.config["student"],
            {
                "checkpoint_path",
                "config_path",
                "precision",
                "dim",
                "ffn_dim",
                "num_heads",
                "num_layers",
                "head_dim",
                "future_video_size",
                "hidden_anchor_layers",
                "motion_anchor_layers",
                "teacher_layer_mapping",
            },
            "Stage 1 student",
        )
        distill = require_exact_keys(
            self.config["distill"],
            {
                "projection_dim",
                "hidden_teacher_layers",
                "motion_teacher_layers",
                "lambda_gt",
                "lambda_hidden_schedule",
                "lambda_motion_schedule",
                "schedule_boundaries",
                "timestep_sampler",
                "sigma_loss_weights",
            },
            "Stage 1 distill",
        )
        if teacher["precision"] != "bfloat16" or student["precision"] != "bfloat16":
            raise ValueError("DynamicWAM Stage 1 requires bfloat16 teacher and student")
        self._normalize_video_size(student.get("future_video_size"))
        boundaries = distill.get("schedule_boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError("DynamicWAM Stage 1 requires distill.schedule_boundaries")
        for name in ("hidden", "motion"):
            schedule = distill.get(f"lambda_{name}_schedule")
            if not isinstance(schedule, list) or len(schedule) != len(boundaries):
                raise ValueError(
                    f"distill.lambda_{name}_schedule must match schedule_boundaries"
                )
        if "lambda_gt" not in distill:
            raise ValueError("DynamicWAM Stage 1 requires distill.lambda_gt")
        require_exact_keys(
            distill["timestep_sampler"],
            {
                "uniform_weight",
                "low_center",
                "low_weight",
                "mid_center",
                "mid_weight",
                "high_center",
                "high_weight",
                "width",
            },
            "Stage 1 timestep_sampler",
        )
        sigma_terms = require_exact_keys(
            distill["sigma_loss_weights"],
            {"gt", "hidden", "motion"},
            "Stage 1 sigma_loss_weights",
        )
        require_exact_keys(
            sigma_terms["gt"],
            {"max_sigma", "softness", "floor"},
            "Stage 1 sigma_loss_weights.gt",
        )
        for name in ("hidden", "motion"):
            require_exact_keys(
                sigma_terms[name],
                {"center", "width", "floor"},
                f"Stage 1 sigma_loss_weights.{name}",
            )

    def _build_dataloader(self) -> DataLoader:
        dataset_cfg = self.config["dataset"]
        raw_dataset = build_packed_training_dataset(dataset_cfg)
        self.dataset_identity = validate_dataset_identity(raw_dataset.dataset_identity)
        sampler = make_packed_training_sampler(raw_dataset, dataset_cfg)
        dl_cfg = get_dataloader_config(self.config)
        return DataLoader(
            raw_dataset,
            batch_size=get_per_device_batch_size(self.config),
            shuffle=False,
            sampler=sampler,
            collate_fn=packed_collate_fn,
            drop_last=True,
            **dl_cfg,
        )

    def _student_config(self) -> CompactWANConfig:
        student_cfg = self.config["student"]
        future_video_size = self._normalize_video_size(
            student_cfg.get("future_video_size")
        )
        return CompactWANConfig(
            checkpoint_path=student_cfg["checkpoint_path"],
            config_path=student_cfg["config_path"],
            precision=student_cfg["precision"],
            dim=int(student_cfg["dim"]),
            ffn_dim=int(student_cfg["ffn_dim"]),
            num_heads=int(student_cfg["num_heads"]),
            num_layers=int(student_cfg["num_layers"]),
            head_dim=int(student_cfg["head_dim"]),
            future_video_size=future_video_size,
            hidden_anchor_layers=list(student_cfg["hidden_anchor_layers"]),
            motion_anchor_layers=list(student_cfg["motion_anchor_layers"]),
            teacher_layer_mapping=list(student_cfg["teacher_layer_mapping"]),
        )

    def _student_init_checkpoint_path(self) -> Path:
        return get_run_dir(self.config) / "init" / "stage1_init.pt"

    @staticmethod
    def _normalize_video_size(value: Any) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"future_video_size must contain two integers, got {value!r}"
            )
        return int(value[0]), int(value[1])

    def _validate_student_init_checkpoint(
        self,
        checkpoint_path: Path,
        payload: Dict[str, Any],
        compact_cfg: CompactWANConfig,
    ) -> None:
        exported_cfg = payload.get("compact_wan_config")
        if exported_cfg is None:
            raise RuntimeError(
                f"Student init checkpoint is missing compact_wan_config: {checkpoint_path}"
            )
        expected = {
            "dim": compact_cfg.dim,
            "ffn_dim": compact_cfg.ffn_dim,
            "num_heads": compact_cfg.num_heads,
            "num_layers": compact_cfg.num_layers,
            "head_dim": compact_cfg.head_dim,
            "hidden_anchor_layers": list(compact_cfg.hidden_anchor_layers),
            "motion_anchor_layers": list(compact_cfg.motion_anchor_layers),
            "teacher_layer_mapping": list(compact_cfg.teacher_layer_mapping),
        }
        mismatches = [
            f"{key}: expected {expected[key]!r}, got {exported_cfg.get(key)!r}"
            for key in (
                "dim",
                "ffn_dim",
                "num_heads",
                "num_layers",
                "head_dim",
                "hidden_anchor_layers",
                "motion_anchor_layers",
                "teacher_layer_mapping",
            )
            if exported_cfg.get(key) != expected[key]
        ]
        if mismatches:
            raise RuntimeError(
                f"Student init checkpoint metadata mismatch for {checkpoint_path}: "
                + "; ".join(mismatches)
            )
        exported_future_size = self._normalize_video_size(
            exported_cfg.get("future_video_size")
        )
        expected_future_size = self._normalize_video_size(compact_cfg.future_video_size)
        if exported_future_size != expected_future_size:
            raise RuntimeError(
                "Student init checkpoint future_video_size mismatch for "
                f"{checkpoint_path}: expected {expected_future_size!r}, "
                f"got {exported_future_size!r}"
            )

    def _save_student_init_checkpoint(
        self, student: CompactWANModel, checkpoint_path: Path
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        cpu_state = {
            key: value.detach().cpu() for key, value in student.state_dict().items()
        }
        try:
            torch.save(
                {
                    "model": cpu_state,
                    "compact_wan_config": student.metadata(),
                    "source": "stage1_distributed_student_init",
                },
                tmp_path,
            )
            tmp_path.replace(checkpoint_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        finally:
            del cpu_state
            gc.collect()
        logger.info(
            "Saved distributed Stage 1 student init checkpoint to %s", checkpoint_path
        )

    def _load_student_init_checkpoint(
        self,
        compact_cfg: CompactWANConfig,
        checkpoint_path: Path,
    ) -> CompactWANModel:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Student init checkpoint not found: {checkpoint_path}"
            )
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict) or "model" not in payload:
            raise TypeError(
                f"Unsupported student init checkpoint format at {checkpoint_path}"
            )
        self._validate_student_init_checkpoint(checkpoint_path, payload, compact_cfg)
        if not isinstance(payload["model"], dict):
            raise TypeError(
                f"Student init checkpoint model payload is not a state dict: {checkpoint_path}"
            )
        student = CompactWANModel.from_config(compact_cfg, device=str(self.device))
        student.load_state_dict(payload["model"], strict=True)
        del payload
        gc.collect()
        logger.info(
            "Loaded distributed Stage 1 student init checkpoint from %s",
            checkpoint_path,
        )
        return student

    def _cleanup_student_init_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            checkpoint_path.unlink(missing_ok=True)
            logger.info(
                "Removed distributed Stage 1 student init checkpoint %s",
                checkpoint_path,
            )
        except OSError as exc:
            logger.warning(
                "Failed to remove distributed Stage 1 student init checkpoint %s: %s",
                checkpoint_path,
                exc,
            )
            return
        with suppress(OSError):
            checkpoint_path.parent.rmdir()

    def _build_student(
        self,
        init_checkpoint_path: Path | None = None,
        save_init_checkpoint: bool = False,
    ) -> CompactWANModel:
        compact_cfg = self._student_config()
        if init_checkpoint_path is not None and not save_init_checkpoint:
            return self._load_student_init_checkpoint(compact_cfg, init_checkpoint_path)

        logger.info(
            "Initializing Stage 1 compact student by structured slicing from teacher checkpoint"
        )
        student = CompactWANModel.from_teacher_checkpoint(
            compact_cfg, device=str(self.device)
        )
        if init_checkpoint_path is not None and save_init_checkpoint:
            self._save_student_init_checkpoint(student, init_checkpoint_path)
        return student

    def _build_distill_heads(self, compact_cfg: CompactWANConfig) -> Stage1DistillHeads:
        distill_cfg = DistillHeadConfig(
            hidden_dim=compact_cfg.dim,
            projection_dim=int(self.config["distill"]["projection_dim"]),
            hidden_anchor_layers=compact_cfg.hidden_anchor_layers,
            motion_anchor_layers=compact_cfg.motion_anchor_layers,
        )
        return Stage1DistillHeads(distill_cfg)

    def _build_teacher(self) -> Stage1DistillTeacher:
        teacher_cfg = self.config["teacher"]
        pca_payload = torch.load(
            teacher_cfg["pca_stats_path"],
            map_location="cpu",
            weights_only=True,
        )
        pca_identity = (
            pca_payload.get("provenance", {}).get("dataset_identity")
            if isinstance(pca_payload, dict)
            else None
        )
        if validate_dataset_identity(pca_identity) != self.dataset_identity:
            raise RuntimeError(
                "Stage 1 PCA dataset identity differs from the training data"
            )
        distill_teacher_cfg = Stage1DistillTeacherConfig(
            checkpoint_path=teacher_cfg["checkpoint_path"],
            config_path=teacher_cfg["config_path"],
            precision=teacher_cfg["precision"],
            hidden_anchor_teacher_layers=list(
                self.config["distill"]["hidden_teacher_layers"]
            ),
            motion_anchor_teacher_layers=list(
                self.config["distill"]["motion_teacher_layers"]
            ),
        )
        return Stage1DistillTeacher(
            distill_teacher_cfg,
            student_hidden_layers=self.config["student"]["hidden_anchor_layers"],
            student_motion_layers=self.config["student"]["motion_anchor_layers"],
            pca_stats_path=teacher_cfg["pca_stats_path"],
            device=str(self.device),
        )

    def _build_system(
        self,
        student_init_checkpoint_path: Path | None = None,
        save_student_init_checkpoint: bool = False,
    ) -> Stage1System:
        student = self._build_student(
            init_checkpoint_path=student_init_checkpoint_path,
            save_init_checkpoint=save_student_init_checkpoint,
        )
        heads = self._build_distill_heads(student.config)
        teacher = self._build_teacher()
        bundle = Stage1TrainingBundle(student, heads).to(self.device)
        return Stage1System(
            bundle=bundle,
            student=student,
            distill_heads=heads,
            teacher=teacher,
            config=self.config,
        )

    def _build_system_staggered(self) -> Stage1System:
        if self.accelerator.num_processes <= 1:
            return self._build_system()

        student_init_checkpoint_path = self._student_init_checkpoint_path()

        student: CompactWANModel | None = None
        if self.accelerator.is_main_process:
            logger.info(
                "Building Stage 1 compact student on rank 0/%d",
                self.accelerator.num_processes,
            )
            student = self._build_student(
                init_checkpoint_path=student_init_checkpoint_path,
                save_init_checkpoint=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            logger.info(
                "Loading Stage 1 compact student init checkpoint on rank %d/%d",
                self.accelerator.process_index,
                self.accelerator.num_processes,
            )
            student = self._build_student(
                init_checkpoint_path=student_init_checkpoint_path,
                save_init_checkpoint=False,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.accelerator.wait_for_everyone()
        if student is None:
            raise RuntimeError(
                f"Rank {self.accelerator.process_index} did not build Stage 1 compact student"
            )

        heads = self._build_distill_heads(student.config)
        teacher = None
        for rank in range(self.accelerator.num_processes):
            if self.accelerator.process_index == rank:
                logger.info(
                    "Building Stage 1 distillation teacher on rank %d/%d",
                    rank,
                    self.accelerator.num_processes,
                )
                teacher = self._build_teacher()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
        if teacher is None:
            raise RuntimeError(
                f"Rank {self.accelerator.process_index} did not build Stage 1 distillation teacher"
            )

        bundle = Stage1TrainingBundle(student, heads).to(self.device)
        system = Stage1System(
            bundle=bundle,
            student=student,
            distill_heads=heads,
            teacher=teacher,
            config=self.config,
        )

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._cleanup_student_init_checkpoint(student_init_checkpoint_path)
        self.accelerator.wait_for_everyone()
        return system

    def _build_optimizer_and_scheduler(self):
        params = [p for p in self.system.bundle.parameters() if p.requires_grad]
        optimizer = AdamW(
            params,
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )

        scheduler = create_scheduler(
            optimizer,
            total_steps=int(self.config["max_steps"]),
            warmup_steps=int(self.config["warmup_steps"]),
            min_lr_ratio=float(self.config["min_lr_ratio"]),
        )
        return optimizer, scheduler

    def _build_timestep_sampling_weights(self) -> torch.Tensor:
        sampler_cfg = self.config["distill"]["timestep_sampler"]
        num_steps = int(self.fm_train_scheduler.num_train_timesteps)
        sigmas = self.fm_train_scheduler.sigmas[:num_steps].float()
        width = max(float(sampler_cfg["width"]), 1e-6)
        weights = torch.full_like(sigmas, float(sampler_cfg["uniform_weight"]))
        for name in ("low", "mid", "high"):
            center = float(sampler_cfg[f"{name}_center"])
            weight = float(sampler_cfg[f"{name}_weight"])
            weights = weights + weight * torch.exp(
                -0.5 * ((sigmas - center) / width).pow(2)
            )
        return weights.clamp_min(1e-8) / weights.sum().clamp_min(1e-8)

    def _sample_timestep_ids(self, batch_size: int) -> torch.Tensor:
        weights = self.timestep_sampling_weights.to(device=self.device)
        return torch.multinomial(weights, batch_size, replacement=True)

    @staticmethod
    def _masked_mse_per_sample(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _sigma_loss_weights(self, sigma: torch.Tensor, name: str) -> torch.Tensor:
        cfg = self.config["distill"]["sigma_loss_weights"][name]
        sigma = sigma.float().view(-1)
        floor = float(cfg["floor"])
        if name == "gt":
            max_sigma = float(cfg["max_sigma"])
            softness = max(float(cfg["softness"]), 1e-6)
            gate = torch.sigmoid((max_sigma - sigma) / softness)
            weights = floor + (1.0 - floor) * gate
        elif name in {"hidden", "motion"}:
            center = float(cfg["center"])
            width = max(float(cfg["width"]), 1e-6)
            gate = torch.exp(-0.5 * ((sigma - center) / width).pow(2))
            weights = floor + (1.0 - floor) * gate
        else:
            raise ValueError(f"Unknown DynamicWAM sigma loss term: {name}")
        return weights.clamp_min(1e-8)

    def _to_text_embedding_list(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        dtype = self._student().video_model.precision
        return [
            value.to(self.device, dtype=dtype, non_blocking=True)
            for value in batch["text_embeddings"]
        ]

    def _prepare_distill_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        student = self._student()
        dtype = student.video_model.precision
        required = {"condition_latent", "future_latent"}
        missing = sorted(required - batch.keys())
        if missing:
            raise KeyError(
                f"DynamicWAM Stage 1 batch is missing packed fields: {missing}"
            )
        condition_latent = batch["condition_latent"].to(
            self.device,
            dtype=dtype,
            non_blocking=True,
        )
        clean_future_latent = batch["future_latent"].to(
            self.device,
            dtype=dtype,
            non_blocking=True,
        )
        batch_size = clean_future_latent.shape[0]
        timestep_id = self._sample_timestep_ids(batch_size)
        timesteps = self.fm_train_scheduler.timesteps.to(
            dtype=dtype, device=self.device
        )
        sigmas = self.fm_train_scheduler.sigmas.to(dtype=dtype, device=self.device)
        t = timesteps[timestep_id]
        sigma = sigmas[timestep_id].view(batch_size, 1, 1, 1, 1)
        noise = torch.randn_like(clean_future_latent, dtype=dtype)
        future_latent = clean_future_latent * (1 - sigma) + noise * sigma
        future_target = noise - clean_future_latent
        condition_tokens = int(
            condition_latent.shape[2]
            * (condition_latent.shape[3] // 2)
            * (condition_latent.shape[4] // 2)
        )
        return {
            **batch,
            "x_t": {
                "condition_latent": condition_latent,
                "future_latent": future_latent,
            },
            "condition_latent": condition_latent,
            "clean_future_latent": clean_future_latent,
            "video_target": future_target,
            "t": t,
            "sigma": sigma,
            "condition_tokens": condition_tokens,
            "num_motion_frames": int(clean_future_latent.shape[2]),
            "text_embeddings": self._to_text_embedding_list(batch),
        }

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self._prepare_distill_batch(batch)

    @staticmethod
    def _scheduled_weight(
        progress: float,
        schedule: List[float],
        boundaries: List[float],
    ) -> float:
        for boundary, weight in zip(boundaries, schedule, strict=True):
            if progress <= float(boundary):
                return float(weight)
        return float(schedule[-1])

    @classmethod
    def distill_weights(
        cls, progress: float, config: Dict[str, Any]
    ) -> Dict[str, float]:
        distill_cfg = config["distill"]
        boundaries = distill_cfg["schedule_boundaries"]
        return {
            "gt": float(distill_cfg["lambda_gt"]),
            "hidden": cls._scheduled_weight(
                progress,
                distill_cfg["lambda_hidden_schedule"],
                boundaries,
            ),
            "motion": cls._scheduled_weight(
                progress,
                distill_cfg["lambda_motion_schedule"],
                boundaries,
            ),
        }

    def _forward_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prepared = self._prepare_batch(batch)
        compact_cfg = self._student().config
        progress = min(
            1.0, float(self.global_step + 1) / float(self.config["max_steps"])
        )
        weights = self.distill_weights(progress, self.config)
        enable_hidden_kd = weights["hidden"] > 0.0
        compute_motion_kd = weights["motion"] > 0.0
        teacher_active = enable_hidden_kd or compute_motion_kd
        hidden_anchor_layers = (
            compact_cfg.hidden_anchor_layers if enable_hidden_kd else []
        )
        motion_anchor_layers = (
            compact_cfg.motion_anchor_layers if compute_motion_kd else []
        )
        layer_indices = sorted(set(hidden_anchor_layers + motion_anchor_layers))
        num_motion_frames = int(prepared["num_motion_frames"])
        outputs = self.system.bundle(
            prepared["x_t"],
            prepared["t"],
            prepared["text_embeddings"],
            layer_indices,
            hidden_anchor_layers,
            motion_anchor_layers,
            num_motion_frames,
            int(prepared["condition_tokens"]),
        )
        video_pred_masked = outputs["video_pred"]
        condition_frames = int(prepared["condition_latent"].shape[2])
        future_frames = int(prepared["clean_future_latent"].shape[2])
        future_loss_scale = future_frames / max(
            1,
            condition_frames + future_frames,
        )
        batch_size = int(video_pred_masked.shape[0])
        sigma_values = prepared["sigma"].float().view(batch_size, -1)[:, 0]
        sigma_weight_gt = self._sigma_loss_weights(sigma_values, "gt")
        sigma_weight_hidden = self._sigma_loss_weights(sigma_values, "hidden")
        sigma_weight_motion = self._sigma_loss_weights(sigma_values, "motion")
        gt_loss_vec = (
            self._masked_mse_per_sample(video_pred_masked, prepared["video_target"])
            * future_loss_scale
        )
        gt_loss = self._weighted_mean(gt_loss_vec, sigma_weight_gt)
        zero_loss = gt_loss.detach().new_zeros(())

        teacher_targets = (
            self.system.teacher.targets(
                prepared,
                include_hidden=enable_hidden_kd,
                include_motion=compute_motion_kd,
            )
            if teacher_active
            else None
        )

        if enable_hidden_kd:
            if teacher_targets is None:
                raise RuntimeError(
                    "Hidden KD requested but teacher targets were not computed"
                )
            hidden_loss_vec = Stage1DistillHeads.hidden_cosine_loss_per_sample(
                outputs["hidden_projected"],
                teacher_targets.hidden,
            )
            hidden_loss = self._weighted_mean(hidden_loss_vec, sigma_weight_hidden)
        else:
            hidden_loss = zero_loss

        if compute_motion_kd:
            if teacher_targets is None:
                raise RuntimeError(
                    "Motion KD requested but teacher targets were not computed"
                )
            motion_loss_vec = Stage1DistillHeads.motion_cosine_loss_per_sample(
                outputs["motion_deltas"],
                teacher_targets.motion,
            )
            motion_loss = self._weighted_mean(motion_loss_vec, sigma_weight_motion)
        else:
            motion_loss = zero_loss

        total_loss = (
            float(weights["gt"]) * gt_loss.float()
            + float(weights["hidden"]) * hidden_loss.float()
            + float(weights["motion"]) * motion_loss.float()
        )
        return {
            "total_loss": total_loss,
            "gt_loss": gt_loss.detach(),
            "hidden_loss": hidden_loss.detach(),
            "motion_loss": motion_loss.detach(),
            "lambda_gt": gt_loss.detach().new_tensor(float(weights["gt"])),
            "lambda_hidden": gt_loss.detach().new_tensor(float(weights["hidden"])),
            "lambda_motion": gt_loss.detach().new_tensor(float(weights["motion"])),
            "sigma_mean": sigma_values.detach().mean(),
            "sigma_low_ratio": (sigma_values < 0.33).float().mean().detach(),
            "sigma_mid_ratio": ((sigma_values >= 0.33) & (sigma_values < 0.66))
            .float()
            .mean()
            .detach(),
            "sigma_high_ratio": (sigma_values >= 0.66).float().mean().detach(),
            "sigma_weight_gt_mean": sigma_weight_gt.detach().mean(),
            "sigma_weight_hidden_mean": sigma_weight_hidden.detach().mean(),
            "sigma_weight_motion_mean": sigma_weight_motion.detach().mean(),
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float] | None:
        self.system.bundle.train()
        grad_norm = None
        with self.accelerator.accumulate(self.system.bundle):
            losses = self._forward_losses(batch)
            self.accelerator.backward(losses["total_loss"])
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.system.bundle.parameters(),
                    max_norm=float(self.config["grad_clip_norm"]),
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if not self.accelerator.sync_gradients:
            return None
        metrics = {
            "loss/total": float(losses["total_loss"].item()),
            "loss/gt": float(losses["gt_loss"].item()),
            "loss/hidden": float(losses["hidden_loss"].item()),
            "loss/motion": float(losses["motion_loss"].item()),
            "loss_weight/gt": float(losses["lambda_gt"].item()),
            "loss_weight/hidden": float(losses["lambda_hidden"].item()),
            "loss_weight/motion": float(losses["lambda_motion"].item()),
            "sigma/mean": float(losses["sigma_mean"].item()),
            "sigma/low_ratio": float(losses["sigma_low_ratio"].item()),
            "sigma/mid_ratio": float(losses["sigma_mid_ratio"].item()),
            "sigma/high_ratio": float(losses["sigma_high_ratio"].item()),
            "sigma_loss_weight/gt_mean": float(losses["sigma_weight_gt_mean"].item()),
            "sigma_loss_weight/hidden_mean": float(
                losses["sigma_weight_hidden_mean"].item()
            ),
            "sigma_loss_weight/motion_mean": float(
                losses["sigma_weight_motion_mean"].item()
            ),
        }
        if grad_norm is not None:
            metrics["grad/norm"] = float(
                grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
            )
        metrics.update(get_learning_rate_metrics(self.optimizer))
        return metrics

    def _resume_if_needed(self) -> None:
        resume_from = self.config.get("resume_from")
        if not resume_from:
            return
        self.accelerator.load_state(str(resume_from))
        match = re.search(r"step_(\d+)", str(resume_from))
        if match:
            self.global_step = int(match.group(1))
        logger.info(
            "Resumed Stage 1 training from %s at step %d", resume_from, self.global_step
        )

    def save_checkpoint(self) -> None:
        ckpt_dir = get_run_dir(self.config) / f"step_{self.global_step}"
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            logger.info("Saved Stage 1 resume checkpoint to %s", ckpt_dir)
            self.export_compact_wan()

    def export_compact_wan(self) -> None:
        export_dir = get_export_dir(self.config)
        export_dir.mkdir(parents=True, exist_ok=True)
        bundle_state = self.accelerator.get_state_dict(self.system.bundle)
        compact_state = {
            key[len("student.") :]: value.cpu()
            for key, value in bundle_state.items()
            if key.startswith("student.")
        }
        output_path = export_dir / f"stage1_step_{self.global_step}.pt"
        checkpoint_config = dict(self.config)
        checkpoint_config["dataset_identity"] = dict(self.dataset_identity)
        torch.save(
            {
                "model": compact_state,
                "global_step": self.global_step,
                "config": checkpoint_config,
                "compact_wan_config": self._student().metadata(),
            },
            output_path,
        )
        logger.info("Exported Stage 1 compact WAN checkpoint to %s", output_path)

    def run(self) -> None:
        if self.accelerator.is_main_process:
            logger.info("Built Stage 1 DynamicWAM system")
            logger.info("Teacher: distillation")
            logger.info("Student initialization: structured teacher slicing")
            logger.info("Student WAN layers: %d", self._student().config.num_layers)
            logger.info(
                "Hidden anchors: %s", self._student().config.hidden_anchor_layers
            )
            logger.info(
                "Motion anchors: %s", self._student().config.motion_anchor_layers
            )
            logger.info("Run directory: %s", get_run_dir(self.config))

            weights = self.distill_weights(0.0, self.config)
            weight_str = " ".join([f"{k}={v}" for k, v in weights.items()])
            logger.info("")
            logger.info("Stage1 training config")
            logger.info("  max_steps      : %d", int(self.config["max_steps"]))
            logger.info("  log_interval   : %d", int(self.config["log_interval"]))
            logger.info("  loss_weights   : %s", weight_str)

        self.log_tracker = LogTracker("stage1", int(self.config["max_steps"]))
        self._resume_if_needed()
        self._last_log_step = self.global_step
        self._last_log_time = time.monotonic()
        samples_per_step = get_effective_global_batch_size(
            self.config, self.accelerator.num_processes
        )
        log_interval = int(self.config["log_interval"])
        checkpoint_interval = int(self.config["checkpoint_interval"])
        data_iter = iter(self.train_loader)

        while self.global_step < int(self.config["max_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)
            if metrics is None:
                continue
            self.global_step += 1
            metrics.update(
                get_epoch_progress_metrics(
                    self.global_step,
                    self.train_loader,
                    gradient_accumulation_steps=int(
                        self.config["gradient_accumulation_steps"]
                    ),
                    world_size=self.accelerator.num_processes,
                )
            )

            if self.global_step % log_interval == 0:
                self._last_log_step, self._last_log_time = add_speed_metrics(
                    metrics,
                    self.global_step,
                    self._last_log_step,
                    self._last_log_time,
                    samples_per_step=samples_per_step,
                )
                log_metrics = reduce_metrics_for_log(self.accelerator, metrics)
                self.accelerator.log(log_metrics, step=self.global_step)
                if self.accelerator.is_main_process:
                    lines_to_log = self.log_tracker.log_step(
                        self.global_step,
                        log_metrics,
                        loss_keys=[
                            ("total", "loss/total"),
                            ("gt", "loss/gt"),
                        ],
                        lr_keys=[("model", "lr/model")],
                        extra_keys=[
                            ("mean", "sigma/mean"),
                            ("low", "sigma/low_ratio"),
                            ("mid", "sigma/mid_ratio"),
                            ("high", "sigma/high_ratio"),
                        ],
                    )
                    for line in lines_to_log:
                        logger.info("%s", line)

            if self.global_step % checkpoint_interval == 0:
                self.save_checkpoint()

        if self.global_step == 0 or self.global_step % checkpoint_interval != 0:
            self.save_checkpoint()


def main() -> None:
    parser = build_arg_parser(description="Stage 1 DynamicWAM distillation training")
    args = parser.parse_args()
    profile = load_profile(args.config)
    config = profile.training_config("stage1")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(config["cudnn_benchmark"])
    accelerator = build_accelerator(config, profile.deepspeed_config())
    setup_logging(str(config["log_level"]), rank=accelerator.process_index)
    if accelerator.is_main_process:
        write_config_snapshot(
            get_run_dir(config) / "config_audit",
            profile=profile,
            label="stage1",
            resolved_config={
                "training": config,
                "deepspeed": profile.deepspeed_config(),
            },
        )
    accelerator.wait_for_everyone()
    init_experiment_trackers(accelerator, config)
    trainer = Stage1AccelerateTrainer(config=config, accelerator=accelerator)

    trainer.run()
    accelerator.end_training()


if __name__ == "__main__":
    main()
