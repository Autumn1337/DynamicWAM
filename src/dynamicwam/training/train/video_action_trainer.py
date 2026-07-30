"""Absolute-motion video-action training for Stage 2 and Stage 3."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dynamicwam.absolute_motion import (
    ABSOLUTE_MOTION_CHECKPOINT_VERSION,
    validate_checkpoint_motion_metadata,
)
from dynamicwam.config import load_profile, write_config_snapshot
from dynamicwam.config.schema import require_exact_keys
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from dynamicwam.training.checkpoint_merge import (
    compact_wan_training_contract,
)
from dynamicwam.training.data.packed_dataset import (
    TRAIN_DATASET_ACTION_STATS,
    TRAIN_DATASET_METADATA,
    packed_collate_fn,
)
from dynamicwam.training.data.training_dataset import (
    build_packed_training_dataset,
    make_packed_training_sampler,
    validate_dataset_identity,
)
from dynamicwam.training.models.compact_wan import CompactWANConfig, CompactWANModel
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
from dynamicwam.training.train.video_action_objective import VideoActionLossObjective
from dynamicwam.training.utils.scheduler import create_scheduler


@dataclass
class VideoActionSystem:
    model: SmallWAMActionModel
    config: Dict[str, Any]


class VideoActionTrainer:
    """Distributed video-action trainer backed by Accelerate/DeepSpeed."""

    def __init__(
        self,
        config: Dict[str, Any],
        accelerator,
        *,
        stage_name: str,
    ):
        self.config = config
        self.stage_name = str(stage_name).lower()
        if self.stage_name not in {"stage2", "stage3"}:
            raise ValueError(
                "absolute-motion stage must be stage2 or stage3, got "
                f"{self.stage_name!r}"
            )
        self.stage_title = "Stage 2" if self.stage_name == "stage2" else "Stage 3"
        self._validate_stage_contract()
        self.accelerator = accelerator
        self.device = accelerator.device
        self.loss_objective = VideoActionLossObjective.from_config(
            config,
            self.stage_name,
        )
        self.train_loader = self._build_dataloader()
        self.system = self._build_system()
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
        self._verify_trainable_setup_before_prepare()
        self.fm_train_scheduler_action = build_training_flow_scheduler(self.config)
        self.fm_train_scheduler_video = build_training_flow_scheduler(self.config)
        self._cache_scheduler_tensors()
        self.global_step = 0

        self.system.model, self.optimizer, self.train_loader, self.scheduler = (
            self.accelerator.prepare(
                self.system.model,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            )
        )

    def _validate_stage_contract(self) -> None:
        learning_rate_keys = (
            {"learning_rate"}
            if self.stage_name == "stage2"
            else {"action_learning_rate", "video_learning_rate"}
        )
        require_exact_keys(
            self.config,
            {
                "name",
                "device",
                "log_level",
                "flow_matching",
                "max_steps",
                "per_device_batch_size",
                "gradient_accumulation_steps",
                *learning_rate_keys,
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
                "model",
                "loss",
            },
            f"{self.stage_title} config",
        )
        model_keys = {
            "initial_checkpoint",
            "compact_wan",
            "action_expert",
            "absolute_motion",
        }
        model = require_exact_keys(
            self.config["model"],
            model_keys,
            f"{self.stage_title} model",
        )
        require_exact_keys(
            model["compact_wan"],
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
            f"{self.stage_title} compact_wan",
        )
        require_exact_keys(
            model["action_expert"],
            {
                "dim",
                "ffn_dim",
                "num_layers",
                "chunk_size",
                "state_dim",
                "action_dim",
            },
            f"{self.stage_title} action_expert",
        )
        require_exact_keys(
            model["absolute_motion"],
            {"history_count", "flow_contract"},
            f"{self.stage_title} absolute_motion",
        )
        expected_loss_keys = (
            {"action_weight"}
            if self.stage_name == "stage2"
            else {
                "action_weight",
                "video_weight_initial",
                "video_weight",
                "video_weight_anneal_steps",
            }
        )
        require_exact_keys(
            self.config["loss"],
            expected_loss_keys,
            f"{self.stage_title} loss",
        )
        if not model.get("initial_checkpoint"):
            raise ValueError(
                f"{self.stage_title} requires a strict full-model checkpoint"
            )

    def _model(self) -> SmallWAMActionModel:
        return self.accelerator.unwrap_model(self.system.model)

    def _cache_scheduler_tensors(self) -> None:
        dtype = self.system.model.compact_wan.video_model.precision
        self._action_timesteps = self.fm_train_scheduler_action.timesteps.to(
            device=self.device, dtype=dtype
        )
        self._action_sigmas = self.fm_train_scheduler_action.sigmas.to(
            device=self.device, dtype=dtype
        )
        self._video_timesteps = self.fm_train_scheduler_video.timesteps.to(
            device=self.device, dtype=dtype
        )
        self._video_sigmas = self.fm_train_scheduler_video.sigmas.to(
            device=self.device, dtype=dtype
        )

    def _build_dataloader(self) -> DataLoader:
        dataset_cfg = self.config["dataset"]
        raw_dataset = build_packed_training_dataset(dataset_cfg)
        self.motion_statistics = dict(raw_dataset.motion_statistics)
        self.motion_checkpoint_metadata = dict(raw_dataset.motion_checkpoint_metadata)
        self.motion_history_count = int(raw_dataset.motion_history_count)
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

    @staticmethod
    def _normalize_video_size(value: Any) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"future_video_size must contain two integers, got {value!r}"
            )
        return int(value[0]), int(value[1])

    def _build_compact_wan_architecture(self) -> CompactWANModel:
        wan_cfg = self.config["model"]["compact_wan"]
        future_video_size = self._normalize_video_size(wan_cfg.get("future_video_size"))
        compact_cfg = CompactWANConfig(
            checkpoint_path=wan_cfg["checkpoint_path"],
            config_path=wan_cfg["config_path"],
            precision=wan_cfg["precision"],
            dim=int(wan_cfg["dim"]),
            ffn_dim=int(wan_cfg["ffn_dim"]),
            num_heads=int(wan_cfg["num_heads"]),
            num_layers=int(wan_cfg["num_layers"]),
            head_dim=int(wan_cfg["head_dim"]),
            future_video_size=future_video_size,
            hidden_anchor_layers=list(wan_cfg["hidden_anchor_layers"]),
            motion_anchor_layers=list(wan_cfg["motion_anchor_layers"]),
            teacher_layer_mapping=list(wan_cfg["teacher_layer_mapping"]),
        )
        return CompactWANModel.from_config(
            compact_cfg,
            device=str(self.device),
        )

    def _validate_initial_checkpoint_config(
        self,
        checkpoint_path: str,
        checkpoint_config: Dict[str, Any],
    ) -> None:
        checkpoint_model = require_exact_keys(
            checkpoint_config.get("model"),
            {
                "initial_checkpoint",
                "compact_wan",
                "action_expert",
                "absolute_motion",
            },
            f"initial checkpoint model at {checkpoint_path}",
        )
        configured_model = self.config["model"]
        if compact_wan_training_contract(
            checkpoint_model["compact_wan"],
            label=f"initial checkpoint compact_wan at {checkpoint_path}",
        ) != compact_wan_training_contract(
            configured_model["compact_wan"],
            label="launch compact_wan",
        ):
            raise RuntimeError(
                "initial checkpoint compact_wan architecture differs from "
                f"the launch config: {checkpoint_path}"
            )
        if checkpoint_model["action_expert"] != configured_model["action_expert"]:
            raise RuntimeError(
                "initial checkpoint action_expert differs from the launch "
                f"config: {checkpoint_path}"
            )
        checkpoint_motion = validate_checkpoint_motion_metadata(
            checkpoint_model["absolute_motion"]
        )
        configured_motion = configured_model["absolute_motion"]
        if (
            int(configured_motion["history_count"])
            != int(checkpoint_motion["history_count"])
            or configured_motion["flow_contract"] != checkpoint_motion["flow_contract"]
        ):
            raise RuntimeError(
                "initial checkpoint motion computation differs from the "
                f"launch config: {checkpoint_path}"
            )
        if checkpoint_motion != self.motion_checkpoint_metadata:
            raise RuntimeError(
                "initial checkpoint motion statistics differ from the packed "
                f"dataset: {checkpoint_path}"
            )
        checkpoint_identity = validate_dataset_identity(
            checkpoint_config.get("dataset_identity")
        )
        if checkpoint_identity != self.dataset_identity:
            raise RuntimeError(
                "initial checkpoint dataset identity differs from the launch "
                f"data: {checkpoint_path}"
            )
        expected_normalization = self._action_normalization_from_train_dataset(
            self.config["dataset"]["root"],
            expected_epsilon=float(self.config["dataset"]["normalization_epsilon"]),
        )
        if checkpoint_config.get("action_normalization") != expected_normalization:
            raise RuntimeError(
                "initial checkpoint action normalization differs from the "
                f"packed dataset: {checkpoint_path}"
            )

    def _load_initial_checkpoint(self, model: SmallWAMActionModel) -> None:
        model_cfg = self.config["model"]
        checkpoint_path = model_cfg.get("initial_checkpoint")
        if not checkpoint_path:
            raise ValueError(
                f"{self.stage_title} has no full-model initialization checkpoint"
            )

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("model"), dict)
            or not isinstance(payload.get("config"), dict)
        ):
            raise TypeError(
                "Full-model initialization must contain model state and config "
                f"metadata: {checkpoint_path}"
            )
        if (
            payload.get("format") != "dynamicwam_absolute_motion_checkpoint"
            or payload.get("version") != ABSOLUTE_MOTION_CHECKPOINT_VERSION
        ):
            raise RuntimeError(
                "training accepts only absolute-motion checkpoint v2: "
                f"{checkpoint_path}"
            )
        expected_payload_keys = {
            "format",
            "version",
            "model",
            "global_step",
            "config",
        }
        if self.stage_name == "stage2":
            expected_payload_keys.add("initialization")
        if set(payload) != expected_payload_keys:
            raise RuntimeError(
                f"{self.stage_title} initialization checkpoint keys differ "
                f"from v2: {checkpoint_path}"
            )
        self._validate_initial_checkpoint_config(
            str(checkpoint_path),
            payload["config"],
        )
        model.load_state_dict(payload["model"], strict=True)
        logger.info(
            "Loaded strict full-model initialization from %s (%d tensors)",
            checkpoint_path,
            len(payload["model"]),
        )

    def _build_system(self) -> VideoActionSystem:
        compact_wan = self._build_compact_wan_architecture()
        model_cfg = self.config["model"]
        small_wam_cfg = SmallWAMActionConfig(
            compact_wan=compact_wan.config,
            action_dim=int(model_cfg["action_expert"]["action_dim"]),
            state_dim=int(model_cfg["action_expert"]["state_dim"]),
            chunk_size=int(model_cfg["action_expert"]["chunk_size"]),
            ae_dim=int(model_cfg["action_expert"]["dim"]),
            ae_ffn_dim=int(model_cfg["action_expert"]["ffn_dim"]),
            ae_num_layers=int(model_cfg["action_expert"]["num_layers"]),
            wan_frozen=self.stage_name == "stage2",
            motion_history_count=self.motion_history_count,
            motion_feature_mean=tuple(
                float(value) for value in self.motion_statistics["mean"]
            ),
            motion_feature_scale=tuple(
                float(value) for value in self.motion_statistics["scale"]
            ),
        )
        model = SmallWAMActionModel(config=small_wam_cfg, compact_wan=compact_wan).to(
            self.device
        )
        configured_history_count = int(model_cfg["absolute_motion"]["history_count"])
        if configured_history_count != self.motion_history_count:
            raise RuntimeError(
                "model motion history differs from the packed dataset: "
                f"{configured_history_count} != {self.motion_history_count}"
            )
        if (
            model_cfg["absolute_motion"]["flow_contract"]
            != self.motion_checkpoint_metadata["flow_contract"]
        ):
            raise RuntimeError(
                "model motion computation differs from the packed dataset"
            )
        self._load_initial_checkpoint(model)
        self._configure_trainable_parameters(model)
        return VideoActionSystem(model=model, config=self.config)

    @staticmethod
    def _count_trainable_parameters(parameters) -> int:
        return sum(p.numel() for p in parameters if p.requires_grad)

    @staticmethod
    def _optimizer_param_count(optimizer, parameters) -> int:
        param_ids = {id(p) for p in parameters}
        return sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
            if id(p) in param_ids
        )

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self._model().parameters() if p.requires_grad]

    def _configure_trainable_parameters(self, model: SmallWAMActionModel) -> None:
        for param in model.action_expert.parameters():
            param.requires_grad_(True)
        for param in model.absolute_motion_tokens.parameters():
            param.requires_grad_(True)
        for param in model.compact_wan.parameters():
            param.requires_grad_(False)
        if not model.config.wan_frozen:
            for param in model.compact_wan.video_model.wan_model.parameters():
                param.requires_grad_(True)

    def _build_optimizer_and_scheduler(self):
        model = self.system.model
        if self.stage_name == "stage2":
            action_learning_rate = float(self.config["learning_rate"])
            video_learning_rate = action_learning_rate
        else:
            action_learning_rate = float(self.config["action_learning_rate"])
            video_learning_rate = float(self.config["video_learning_rate"])
        weight_decay = float(self.config["weight_decay"])

        action_params = [
            p
            for module in (
                model.action_expert,
                model.absolute_motion_tokens,
            )
            for p in module.parameters()
            if p.requires_grad
        ]
        if not action_params:
            raise RuntimeError(
                "Video-action training expected trainable Action Expert parameters"
            )

        param_groups = [
            {
                "name": "action_expert",
                "params": action_params,
                "lr": action_learning_rate,
                "weight_decay": weight_decay,
            }
        ]
        if not model.config.wan_frozen:
            video_params = [
                p
                for p in model.compact_wan.video_model.wan_model.parameters()
                if p.requires_grad
            ]
            if not video_params:
                raise RuntimeError(
                    "Stage 3 joint contract expected trainable compact WAN "
                    "video expert parameters"
                )
            param_groups.append(
                {
                    "name": "video_expert",
                    "params": video_params,
                    "lr": video_learning_rate,
                    "weight_decay": weight_decay,
                }
            )

        optimizer = AdamW(
            param_groups,
            lr=action_learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = create_scheduler(
            optimizer,
            total_steps=int(self.config["max_steps"]),
            warmup_steps=int(self.config["warmup_steps"]),
            min_lr_ratio=float(self.config["min_lr_ratio"]),
        )
        return optimizer, scheduler

    def _verify_trainable_setup_before_prepare(self) -> None:
        model = self.system.model
        self._verify_trainable_setup(
            model, "before distributed wrapping", verify_optimizer=True
        )

    def _verify_trainable_setup_after_prepare(self) -> None:
        model = self._model()
        self._verify_trainable_setup(
            model, "after distributed wrapping", verify_optimizer=False
        )

    def _verify_trainable_setup(
        self,
        model: SmallWAMActionModel,
        context: str,
        verify_optimizer: bool,
    ) -> None:
        action_parameters = list(model.action_expert.parameters()) + list(
            model.absolute_motion_tokens.parameters()
        )
        action_trainable = self._count_trainable_parameters(action_parameters)
        video_trainable = self._count_trainable_parameters(
            model.compact_wan.video_model.wan_model.parameters()
        )
        wan_trainable = self._count_trainable_parameters(model.compact_wan.parameters())
        action_optimizer_params = 0
        video_optimizer_params = 0
        wan_optimizer_params = 0
        if verify_optimizer:
            action_optimizer_params = self._optimizer_param_count(
                self.optimizer,
                action_parameters,
            )
            video_optimizer_params = self._optimizer_param_count(
                self.optimizer,
                model.compact_wan.video_model.wan_model.parameters(),
            )
            wan_optimizer_params = self._optimizer_param_count(
                self.optimizer, model.compact_wan.parameters()
            )

        if action_trainable == 0:
            raise RuntimeError(
                f"Video-action training expected Action Expert trainable {context}, "
                f"found action_trainable={action_trainable}"
            )
        if verify_optimizer and action_optimizer_params != action_trainable:
            raise RuntimeError(
                f"Video-action training expected Action Expert fully included in optimizer {context}, "
                f"found action_trainable={action_trainable}, action_optimizer_params={action_optimizer_params}"
            )

        if model.config.wan_frozen:
            if wan_trainable != 0:
                raise RuntimeError(
                    f"Stage 2 requires compact WAN frozen {context}, "
                    f"found wan_trainable={wan_trainable}"
                )
            if verify_optimizer and wan_optimizer_params != 0:
                raise RuntimeError(
                    f"Stage 2 requires compact WAN excluded from optimizer {context}, "
                    f"found wan_optimizer_params={wan_optimizer_params}"
                )
        elif video_trainable == 0:
            raise RuntimeError(
                f"Stage 3 requires video expert trainable {context}, "
                f"found video_trainable={video_trainable}"
            )
        elif wan_trainable != video_trainable:
            raise RuntimeError(
                f"Stage 3 requires only compact WAN video expert trainable {context}, "
                f"found wan_trainable={wan_trainable}, video_trainable={video_trainable}"
            )
        elif verify_optimizer and video_optimizer_params != video_trainable:
            raise RuntimeError(
                f"Stage 3 requires video expert fully included in optimizer {context}, "
                f"found video_trainable={video_trainable}, video_optimizer_params={video_optimizer_params}"
            )
        elif verify_optimizer and wan_optimizer_params != video_optimizer_params:
            raise RuntimeError(
                f"Stage 3 requires only compact WAN video expert in optimizer {context}, "
                f"found wan_optimizer_params={wan_optimizer_params}, video_optimizer_params={video_optimizer_params}"
            )

        if self.accelerator.is_main_process:
            contract = (
                "frozen-video" if model.config.wan_frozen else "joint-video-action"
            )
            logger.info(
                "%s trainable setup verified %s: contract=%s action_trainable=%d action_optimizer=%d "
                "video_trainable=%d video_optimizer=%d compact_wan_trainable=%d compact_wan_optimizer=%d",
                self.stage_title,
                context,
                contract,
                action_trainable,
                action_optimizer_params,
                video_trainable,
                video_optimizer_params,
                wan_trainable,
                wan_optimizer_params,
            )

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        model = self._model()
        prepared = dict(batch)
        dtype = model.compact_wan.video_model.precision
        prepared["initial_state"] = prepared["initial_state"].to(
            self.device, dtype=dtype, non_blocking=True
        )
        actions = prepared["action_sequence"].to(
            self.device, dtype=dtype, non_blocking=True
        )
        batch_size = actions.shape[0]

        required = {
            "condition_latent",
            "future_latent",
            "absolute_motion_features",
            "absolute_motion_interval_valid_mask",
            "absolute_motion_acceleration_valid_mask",
        }
        missing = sorted(required - prepared.keys())
        if missing:
            raise KeyError(
                f"DynamicWAM {self.stage_title} batch is missing packed fields: "
                f"{missing}"
            )
        clean_future_latent = prepared["future_latent"].to(
            self.device,
            dtype=dtype,
            non_blocking=True,
        )
        condition_latent = prepared["condition_latent"].to(
            self.device,
            dtype=dtype,
            non_blocking=True,
        )

        video_timestep_id = torch.randint(
            0,
            self.fm_train_scheduler_video.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )
        video_t = self._video_timesteps[video_timestep_id]
        video_sigma = self._video_sigmas[video_timestep_id].view(batch_size, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_future_latent, dtype=dtype)
        future_latent = (
            clean_future_latent * (1 - video_sigma) + video_noise * video_sigma
        )
        if self.loss_objective.trains_video:
            video_target = video_noise - clean_future_latent
        prepared["condition_latent"] = condition_latent
        prepared["future_latent"] = future_latent
        prepared["absolute_motion_features"] = prepared["absolute_motion_features"].to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        prepared["absolute_motion_interval_valid_mask"] = prepared[
            "absolute_motion_interval_valid_mask"
        ].to(self.device, dtype=torch.bool, non_blocking=True)
        prepared["absolute_motion_acceleration_valid_mask"] = prepared[
            "absolute_motion_acceleration_valid_mask"
        ].to(self.device, dtype=torch.bool, non_blocking=True)

        timestep_id = torch.randint(
            0,
            self.fm_train_scheduler_action.num_train_timesteps,
            (batch_size,),
            device=self.device,
        )
        action_t = self._action_timesteps[timestep_id]
        sigma = self._action_sigmas[timestep_id].view(batch_size, 1, 1)
        noise = torch.randn_like(actions, dtype=dtype)
        prepared["noisy_actions"] = actions * (1 - sigma) + noise * sigma
        prepared["action_target"] = noise - actions
        prepared["action_t"] = action_t
        if self.loss_objective.trains_video:
            prepared["video_target"] = video_target
        prepared["video_t"] = video_t
        prepared["text_embeddings"] = [
            value.to(self.device, dtype=dtype, non_blocking=True)
            for value in batch["text_embeddings"]
        ]
        return prepared

    @staticmethod
    def _regression_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(prediction.float(), target.float(), reduction="mean")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float] | None:
        self.system.model.train()
        grad_norm = None
        with self.accelerator.accumulate(self.system.model):
            prepared = self._prepare_batch(batch)
            outputs = self.system.model(prepared)
            action_loss = self._regression_loss(
                outputs["action_pred"], prepared["action_target"]
            )
            if self.loss_objective.trains_video:
                video_loss = self._regression_loss(
                    outputs["video_pred"],
                    prepared["video_target"],
                )
            else:
                if "video_pred" in outputs:
                    raise RuntimeError(
                        "Action-only Stage 2 unexpectedly produced video_pred"
                    )
                video_loss = action_loss.detach().new_zeros(())
            action_weight = self.loss_objective.action_weight
            video_weight = self.loss_objective.video_weight_at(self.global_step + 1)
            total_loss = action_weight * action_loss + video_weight * video_loss
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self._trainable_parameters(),
                    max_norm=float(self.config["grad_clip_norm"]),
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if not self.accelerator.sync_gradients:
            return None
        grad_norm_value = 0.0
        if grad_norm is not None:
            grad_norm_value = float(
                grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
            )
        return {
            "loss/total": float(total_loss.detach().item()),
            "loss/action": float(action_loss.detach().item()),
            "loss/video": float(video_loss.detach().item()),
            "loss_weight/action": action_weight,
            "loss_weight/video": video_weight,
            "grad/norm": grad_norm_value,
            **get_learning_rate_metrics(self.optimizer),
        }

    def _resume_if_needed(self) -> None:
        resume_from = self.config.get("resume_from")
        if not resume_from:
            return
        self.accelerator.load_state(str(resume_from))
        match = re.search(r"step_(\d+)", str(resume_from))
        if match:
            self.global_step = int(match.group(1))
        logger.info(
            "Resumed %s training from %s at step %d",
            self.stage_title,
            resume_from,
            self.global_step,
        )

    def save_checkpoint(self) -> None:
        ckpt_dir = get_run_dir(self.config) / f"step_{self.global_step}"
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            logger.info("Saved %s resume checkpoint to %s", self.stage_title, ckpt_dir)
            self.export_dynamicwam()

    def export_dynamicwam(self) -> None:
        export_dir = get_export_dir(self.config)
        export_dir.mkdir(parents=True, exist_ok=True)
        model_state = {
            key: value.cpu()
            for key, value in self.accelerator.get_state_dict(self.system.model).items()
        }
        output_path = export_dir / f"{self.stage_name}_step_{self.global_step}.pt"
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        try:
            torch.save(
                {
                    "format": "dynamicwam_absolute_motion_checkpoint",
                    "version": ABSOLUTE_MOTION_CHECKPOINT_VERSION,
                    "model": model_state,
                    "global_step": self.global_step,
                    "config": self._checkpoint_config(),
                },
                temporary,
            )
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("Exported DynamicWAM checkpoint to %s", output_path)

    @staticmethod
    def _action_normalization_from_train_dataset(
        dataset_root: str | Path,
        *,
        expected_epsilon: float,
    ) -> Dict[str, Any]:
        root = Path(dataset_root).expanduser()
        metadata_path = root / TRAIN_DATASET_METADATA
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"DynamicWAM packed dataset metadata is missing: {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        data_cfg = metadata["data"]
        norm_cfg = data_cfg["action_normalization"]
        if not isinstance(norm_cfg, dict):
            raise ValueError(
                f"DynamicWAM dataset action_normalization is not a mapping: {metadata_path}"
            )
        if norm_cfg.get("enabled") is not True or norm_cfg.get("type") != "mean_std":
            raise ValueError(
                f"DynamicWAM dataset must use mean/std action normalization: {metadata_path}"
            )

        exported_norm = {"enabled": True, "type": "mean_std"}
        metadata_epsilon = norm_cfg.get("epsilon")
        if metadata_epsilon is not None and float(metadata_epsilon) != float(
            expected_epsilon
        ):
            raise ValueError(
                "Packed dataset normalization epsilon differs from the profile: "
                f"{metadata_epsilon!r} != {expected_epsilon!r}"
            )
        exported_norm["epsilon"] = float(expected_epsilon)
        stats_path = root / TRAIN_DATASET_ACTION_STATS
        if not stats_path.exists():
            raise FileNotFoundError(
                f"DynamicWAM action stats are missing: {stats_path}"
            )
        with stats_path.open("r", encoding="utf-8") as f:
            stats_payload = json.load(f)
        exported_norm["stats"] = stats_payload.get("robotwin_qpos", stats_payload)
        if norm_cfg.get("stats_file") != TRAIN_DATASET_ACTION_STATS:
            raise ValueError(
                f"DynamicWAM dataset stats_file must be {TRAIN_DATASET_ACTION_STATS!r}"
            )
        digest = hashlib.sha256()
        with stats_path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        stats_sha256 = digest.hexdigest()
        if norm_cfg.get("stats_sha256") != stats_sha256:
            raise ValueError(
                "Packed dataset action statistics do not match their pinned "
                f"SHA256: {stats_path}"
            )
        exported_norm["stats_file"] = TRAIN_DATASET_ACTION_STATS
        exported_norm["stats_sha256"] = stats_sha256
        return exported_norm

    def _checkpoint_config(self) -> Dict[str, Any]:
        checkpoint_config = dict(self.config)
        checkpoint_config["training_stage"] = self.stage_name
        dataset_cfg = checkpoint_config["dataset"]
        dataset_root = dataset_cfg["root"]
        checkpoint_config["action_normalization"] = (
            self._action_normalization_from_train_dataset(
                dataset_root,
                expected_epsilon=float(dataset_cfg["normalization_epsilon"]),
            )
        )
        checkpoint_config["model"] = dict(checkpoint_config["model"])
        checkpoint_config["model"]["absolute_motion"] = dict(
            self.motion_checkpoint_metadata
        )
        checkpoint_config["dataset_identity"] = dict(self.dataset_identity)
        return checkpoint_config

    def run(self) -> None:
        self._verify_trainable_setup_after_prepare()
        if self.accelerator.is_main_process:
            logger.info("Built %s DynamicWAM system", self.stage_title)
            logger.info(
                "%s contract: %s",
                self.stage_title,
                "frozen-video"
                if self._model().config.wan_frozen
                else "joint-video-action",
            )
            logger.info("WAN frozen: %s", self._model().config.wan_frozen)
            logger.info("AE layers: %d", self._model().config.ae_num_layers)
            logger.info("Run directory: %s", get_run_dir(self.config))

            action_w = self.loss_objective.action_weight
            video_w = self.loss_objective.video_weight_at(1)

            logger.info("")
            logger.info("Video-action training config")
            logger.info("  max_steps      : %d", int(self.config["max_steps"]))
            logger.info("  log_interval   : %d", int(self.config["log_interval"]))
            logger.info("  loss_weights   : action=%s video=%s", action_w, video_w)
            logger.info("  trains_video   : %s", self.loss_objective.trains_video)
            if self.loss_objective.video_weight_anneal_steps:
                logger.info(
                    "  video_anneal   : %s -> %s over %d steps (%s)",
                    self.loss_objective.video_weight_initial,
                    self.loss_objective.video_weight_final,
                    self.loss_objective.video_weight_anneal_steps,
                    "linear",
                )

        loss_precision = 3 if self._model().config.wan_frozen else 4
        self.log_tracker = LogTracker(
            self.stage_name,
            int(self.config["max_steps"]),
            loss_precision=loss_precision,
        )
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
                            ("action", "loss/action"),
                            ("video", "loss/video"),
                        ],
                        lr_keys=[
                            ("action", "lr/action_expert"),
                            ("video", "lr/video_expert"),
                        ],
                    )
                    for line in lines_to_log:
                        logger.info("%s", line)

            if self.global_step % checkpoint_interval == 0:
                self.save_checkpoint()

        if self.global_step == 0 or self.global_step % checkpoint_interval != 0:
            self.save_checkpoint()


def run_video_action_training(
    *,
    description: str,
    stage_name: str,
) -> None:
    parser = build_arg_parser(description=description)
    args = parser.parse_args()
    profile = load_profile(args.config)
    config = profile.training_config(stage_name)
    accelerator = build_accelerator(config, profile.deepspeed_config())
    setup_logging(str(config["log_level"]), rank=accelerator.process_index)
    if accelerator.is_main_process:
        write_config_snapshot(
            get_run_dir(config) / "config_audit",
            profile=profile,
            label=stage_name,
            resolved_config={
                "training": config,
                "deepspeed": profile.deepspeed_config(),
            },
        )
    accelerator.wait_for_everyone()
    init_experiment_trackers(accelerator, config)
    trainer = VideoActionTrainer(
        config=config,
        accelerator=accelerator,
        stage_name=stage_name,
    )
    trainer.run()
    accelerator.end_training()
