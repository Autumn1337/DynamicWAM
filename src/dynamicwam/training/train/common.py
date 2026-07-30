from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from dynamicwam.config.schema import require_exact_keys
from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler

logger = logging.getLogger(__name__)


_ANSI_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "gray": "90",
}


def color_logs_enabled() -> bool:
    flag = os.environ.get("DYNAMICWAM_COLOR_LOGS", "").lower()
    if flag in {"0", "false", "no", "off", "never"} or "NO_COLOR" in os.environ:
        return False
    if flag in {"1", "true", "yes", "on", "always"} or os.environ.get("FORCE_COLOR"):
        return True
    return os.environ.get("TERM", "") != "dumb"


def color_text(
    text: str, color: str | None = None, *, bold: bool = False, dim: bool = False
) -> str:
    if not color_logs_enabled():
        return text
    codes = []
    if bold:
        codes.append(_ANSI_CODES["bold"])
    if dim:
        codes.append(_ANSI_CODES["dim"])
    if color:
        codes.append(_ANSI_CODES[color])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def setup_logging(log_level: str = "INFO", rank: int = 0) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=("%(message)s" if int(rank) == 0 else f"[rank {int(rank)}] %(message)s"),
        force=True,
    )


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Accepted for launcher compatibility"
    )
    return parser


def serialize_tracker_config(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(
            {str(k): serialize_tracker_config(v) for k, v in value.items()},
            ensure_ascii=True,
            sort_keys=True,
        )
    if isinstance(value, (list, tuple)):
        return json.dumps(
            [serialize_tracker_config(item) for item in value], ensure_ascii=True
        )
    return str(value)


def flatten_tracker_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_tracker_config(value, name))
        else:
            flat[name] = serialize_tracker_config(value)
    return flat


def get_tracker_init_config(config: Dict[str, Any]) -> Dict[str, Any]:
    tracker_config = {
        str(key): serialize_tracker_config(value) for key, value in config.items()
    }
    tracker_config.update(flatten_tracker_config(config))
    return tracker_config


def build_accelerator(
    config: Dict[str, Any],
    deepspeed_config: Dict[str, Any],
):
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration

    run_dir = get_run_dir(config)
    log_dir = get_log_dir(config)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=deepspeed_config),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=str(run_dir),
        project_config=ProjectConfiguration(
            project_dir=str(run_dir),
            logging_dir=str(log_dir),
            total_limit=int(config["checkpoint_total_limit"]),
        ),
    )
    if config["device"] != "cuda" or accelerator.device.type != "cuda":
        raise RuntimeError(
            "DynamicWAM training requires a CUDA accelerator, got "
            f"config={config['device']!r}, runtime={accelerator.device}"
        )
    return accelerator


def get_run_dir(config: Dict[str, Any]) -> Path:
    return Path(config["checkpoint_dir"]) / str(config["name"])


def get_export_dir(config: Dict[str, Any]) -> Path:
    return get_run_dir(config) / "exports"


def get_log_dir(config: Dict[str, Any]) -> Path:
    return get_run_dir(config) / "logs"


def init_experiment_trackers(
    accelerator,
    config: Dict[str, Any],
) -> None:
    run_name = str(config["name"])
    tracker_config = get_tracker_init_config(
        {
            **config,
            "tracker_project_name": "dynamicwam",
            "tracker_run_name": run_name,
        }
    )
    accelerator.init_trackers("dynamicwam", config=tracker_config)


def get_per_device_batch_size(config: Dict[str, Any]) -> int:
    return max(1, int(config["per_device_batch_size"]))


def get_effective_global_batch_size(
    config: Dict[str, Any], world_size: int = 1
) -> float:
    return float(
        get_per_device_batch_size(config)
        * max(1, int(world_size))
        * max(1, int(config["gradient_accumulation_steps"]))
    )


def add_speed_metrics(
    metrics: Dict[str, float],
    global_step: int,
    last_log_step: int,
    last_log_time: float,
    samples_per_step: float | None = None,
) -> tuple[int, float]:
    now = time.monotonic()
    step_delta = max(1, int(global_step) - int(last_log_step))
    elapsed = max(now - float(last_log_time), 1e-9)
    steps_per_sec = float(step_delta) / elapsed
    metrics["speed/steps_per_sec"] = steps_per_sec
    metrics["speed/sec_per_step"] = elapsed / float(step_delta)
    if samples_per_step is not None:
        metrics["speed/samples_per_sec"] = steps_per_sec * float(samples_per_step)
    return int(global_step), now


def build_training_flow_scheduler(config: Dict[str, Any]) -> FlowMatchScheduler:
    flow_matching = require_exact_keys(
        config["flow_matching"],
        {
            "shift",
            "sigma_min",
            "extra_one_step",
            "training_timesteps",
        },
        "resolved training flow_matching",
    )
    scheduler = FlowMatchScheduler(
        shift=float(flow_matching["shift"]),
        sigma_min=float(flow_matching["sigma_min"]),
        extra_one_step=bool(flow_matching["extra_one_step"]),
    )
    scheduler.set_timesteps(
        num_inference_steps=int(flow_matching["training_timesteps"]),
        training=True,
    )
    return scheduler


def get_learning_rate_metrics(optimizer, prefix: str = "lr") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    seen_names: Dict[str, int] = {}
    single_group = len(optimizer.param_groups) == 1
    for idx, group in enumerate(optimizer.param_groups):
        lr = float(group.get("lr", 0.0))
        name = str(group.get("name") or ("model" if single_group else f"group_{idx}"))
        count = seen_names.get(name, 0)
        seen_names[name] = count + 1
        if count:
            name = f"{name}_{count}"
        metrics[f"{prefix}/{name}"] = lr
    return metrics


def get_epoch_progress_metrics(
    global_step: int,
    dataloader,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> Dict[str, float]:
    try:
        local_micro_batches_per_epoch = len(dataloader)
    except TypeError:
        return {}

    if local_micro_batches_per_epoch <= 0:
        return {}

    grad_accum = max(1, int(gradient_accumulation_steps))
    optimizer_steps_per_epoch = max(
        1, math.ceil(local_micro_batches_per_epoch / grad_accum)
    )
    current_epoch_step = (
        0 if global_step <= 0 else ((global_step - 1) % optimizer_steps_per_epoch) + 1
    )
    fractional_epoch = float(global_step) / float(optimizer_steps_per_epoch)

    return {
        "progress/epoch": fractional_epoch,
        "progress/epoch_index": float(math.floor(fractional_epoch)),
        "progress/epoch_step": float(current_epoch_step),
        "progress/steps_per_epoch": float(optimizer_steps_per_epoch),
        "progress/local_micro_batches_per_epoch": float(local_micro_batches_per_epoch),
        "progress/gradient_accumulation_steps": float(grad_accum),
        "progress/world_size": float(max(1, int(world_size))),
    }


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


class LogTracker:
    def __init__(self, stage: str, max_steps: int, loss_precision: int = 3):
        self.stage = stage
        self.max_steps = max_steps
        self.loss_precision = int(loss_precision)
        self.ema_loss: float | None = None
        self.ema_beta = 0.98
        self.ema_samp_sec: float | None = None

    def _format_loss(self, value: float) -> str:
        return f"{value:.{self.loss_precision}f}"

    def log_step(
        self,
        global_step: int,
        metrics: Dict[str, float],
        *,
        loss_keys: Iterable[tuple[str, str]],
        lr_keys: Iterable[tuple[str, str]] = (),
        grad_key: str = "grad/norm",
        extra_keys: Iterable[tuple[str, str]] = (),
    ) -> list[str]:
        lines_to_log = []

        current_loss = metrics.get("loss/total")
        if current_loss is not None:
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = (
                    self.ema_beta * self.ema_loss + (1 - self.ema_beta) * current_loss
                )

        samp_sec = metrics.get("speed/samples_per_sec")
        step_sec = metrics.get("speed/steps_per_sec")
        if samp_sec is not None:
            if self.ema_samp_sec is None:
                self.ema_samp_sec = samp_sec
            else:
                self.ema_samp_sec = 0.9 * self.ema_samp_sec + 0.1 * samp_sec

        remaining_steps = self.max_steps - global_step
        eta_seconds = remaining_steps / step_sec if step_sec and step_sec > 0 else 0
        eta_str = format_time(eta_seconds)

        grad_norm = metrics.get(grad_key)
        if current_loss is not None and (
            math.isnan(current_loss) or math.isinf(current_loss)
        ):
            lines_to_log.append(
                color_text(
                    f"WARNING step={global_step} loss={current_loss}, stop training or check input batch",
                    "red",
                    bold=True,
                )
            )
        if grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                lines_to_log.append(
                    color_text(
                        f"WARNING step={global_step} grad_norm={grad_norm}, possible instability",
                        "red",
                        bold=True,
                    )
                )
            elif grad_norm > 10:
                lines_to_log.append(
                    color_text(
                        f"WARNING step={global_step} grad_norm={grad_norm:.1f}, possible instability",
                        "red",
                        bold=True,
                    )
                )

        if (
            samp_sec is not None
            and self.ema_samp_sec is not None
            and samp_sec < 0.5 * self.ema_samp_sec
        ):
            lines_to_log.append(
                color_text(
                    f"WARNING step={global_step} throughput significantly lower than average",
                    "yellow",
                    bold=True,
                )
            )

        epoch_num = metrics.get("progress/epoch", 0.0)
        pct = 100.0 * global_step / max(1, self.max_steps)

        stage_lower = self.stage.lower()
        if stage_lower.startswith("stage") and stage_lower[5:].isdigit():
            short_stage = f"S{stage_lower[5:]}"
        else:
            short_stage = self.stage

        def dim_text(text: str) -> str:
            return color_text(text, dim=True)

        s_stage = dim_text(short_stage)
        s_step = dim_text(f"{global_step:06d}/{self.max_steps:06d}")
        s_pct = dim_text(f"{pct:.2f}%")
        s_epoch = f"{color_text('ep', 'cyan', bold=True)} {color_text(f'{epoch_num:.3f}', 'cyan')}"

        part_prefix = f"{s_stage} {s_step}  {s_pct}  {s_epoch}"

        loss_val = self._format_loss(current_loss) if current_loss is not None else "?"
        s_loss = (
            f"{color_text('loss', 'green', bold=True)} {color_text(loss_val, 'green')}"
        )
        s_ema = (
            dim_text(f"ema {self._format_loss(self.ema_loss)}")
            if self.ema_loss is not None
            else ""
        )
        part_loss = f"{s_loss}  {s_ema}".strip()

        loss_comps = []
        for label, key in loss_keys:
            if label != "total" and key in metrics:
                loss_comps.append(
                    dim_text(f"{label} {self._format_loss(metrics[key])}")
                )
        part_comps = "  ".join(loss_comps)

        lr_vals = []
        for _label, key in lr_keys:
            if key in metrics:
                lr_vals.append(color_text(f"{metrics[key]:.1e}", "magenta"))
        s_lr = (
            f"{color_text('lr', 'magenta', bold=True)} {','.join(lr_vals)}"
            if lr_vals
            else ""
        )
        s_grad = dim_text(f"grad {grad_norm:.2f}") if grad_norm is not None else ""
        if grad_norm is not None and grad_norm > 10:
            s_grad = color_text(f"grad {grad_norm:.2f}", "red")
        part_opt = f"{s_lr}  {s_grad}".strip()

        speed_vals = []
        if step_sec is not None:
            speed_vals.append(f"{step_sec:.2f} step/s")
        s_speed = dim_text("  ".join(speed_vals)) if speed_vals else ""
        s_eta = dim_text(f"ETA {eta_str}")
        part_speed = f"{s_speed}  {s_eta}".strip()

        delimiter = dim_text(" | ")
        parts = [
            p for p in [part_prefix, part_loss, part_comps, part_opt, part_speed] if p
        ]
        single_line = delimiter.join(parts)

        lines_to_log.append(single_line)

        if global_step % 500 == 0:
            detailed = []
            detailed.append("─" * 56)
            stage_num = short_stage[-1] if short_stage[-1].isdigit() else "X"
            detailed.append(
                f"Stage{stage_num} @ step {global_step} / {self.max_steps}   epoch {epoch_num:.3f}   ETA {eta_str}"
            )
            detailed.append("\nLoss")
            for label, key in loss_keys:
                if key in metrics:
                    detailed.append(f"  {label:<9} {self._format_loss(metrics[key])}")
            detailed.append("\nOptimization")
            for label, key in lr_keys:
                if key in metrics:
                    detailed.append(f"  lr_{label:<6} {metrics[key]:.2e}")
            if grad_norm is not None:
                detailed.append(f"  grad_norm {grad_norm:.2f}")
            if step_sec is not None:
                detailed.append(f"  step_rate {step_sec:.2f} steps/s")

            if extra_keys:
                has_extra = any(key in metrics for _, key in extra_keys)
                if has_extra:
                    detailed.append("\nSigma")
                    for label, key in extra_keys:
                        if key in metrics:
                            detailed.append(f"  {label:<9} {metrics[key]:.3f}")
            detailed.append("─" * 56)
            lines_to_log.append("\n".join(detailed))

        return lines_to_log


def reduce_metrics_for_log(
    accelerator, metrics: Dict[str, float], reduction: str = "mean"
) -> Dict[str, float]:
    """Average numeric metrics across ranks before console/tracker logging.

    Per-step losses are computed on each rank's local shard first; this reducer
    turns them into global means when each rank uses the same per-device batch
    size, which is how these distributed DataLoaders are configured.
    """
    if getattr(accelerator, "num_processes", 1) <= 1:
        return dict(metrics)

    numeric_items = [
        (key, float(value))
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    ]
    if not numeric_items:
        return dict(metrics)

    import torch

    keys = [key for key, _ in numeric_items]
    values = torch.tensor(
        [value for _, value in numeric_items],
        device=accelerator.device,
        dtype=torch.float32,
    )
    reduced_values = (
        accelerator.reduce(values, reduction=reduction).detach().cpu().tolist()
    )

    reduced_metrics = dict(metrics)
    for key, value in zip(keys, reduced_values, strict=True):
        reduced_metrics[key] = float(value)
    return reduced_metrics


def get_dataloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    num_workers = int(config["num_workers"])
    dataloader_config: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(config["pin_memory"]),
    }
    if num_workers > 0:
        dataloader_config["persistent_workers"] = bool(config["persistent_workers"])
        dataloader_config["prefetch_factor"] = max(1, int(config["prefetch_factor"]))
    return dataloader_config
