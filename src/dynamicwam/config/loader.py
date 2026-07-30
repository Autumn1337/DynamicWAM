"""Load the absolute-motion profile and derive every executable config."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamicwam.integrity import (
    domino_eval_policy_sha256,
    domino_source_sha256,
)
from dynamicwam.motion_contract import (
    SPATIAL_UNIT,
    TEMPORAL_CONTRACT,
    TIMESTAMP_SOURCE,
    flow_compute_contract,
)

from .schema import validate_absolute_motion_profile


def default_profile_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "absolute_motion_v2.yaml"


def _resolve_path(
    value: Any,
    *,
    base: Path,
    label: str,
    follow_symlinks: bool = True,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve() if follow_symlinks else path.absolute())


def _resolve_profile_paths(raw: Any, *, profile_path: Path) -> Any:
    """Resolve portable profile paths before applying the strict schema."""

    if not isinstance(raw, dict):
        return raw
    resolved = deepcopy(raw)
    paths = resolved.get("paths")
    if not isinstance(paths, dict):
        return resolved

    project_root = Path(
        _resolve_path(
            paths.get("project_root"),
            base=profile_path.parent,
            label="paths.project_root",
        )
    )
    paths["project_root"] = str(project_root)
    for key, value in tuple(paths.items()):
        if key != "project_root":
            paths[key] = _resolve_path(
                value,
                base=project_root,
                label=f"paths.{key}",
            )

    benchmark = resolved.get("benchmark")
    if isinstance(benchmark, dict):
        for key in ("python", "domino_root", "eval_policy", "curobo_root"):
            if key in benchmark:
                benchmark[key] = _resolve_path(
                    benchmark[key],
                    base=project_root,
                    label=f"benchmark.{key}",
                    follow_symlinks=key != "python",
                )
    return resolved


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _pretty_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise RuntimeError(
                f"Audit file differs from the resolved DynamicWAM launch: {path}"
            )
        return
    _atomic_write(path, content)


@dataclass(frozen=True)
class AbsoluteMotionProfile:
    """Immutable handle to the exact-time production profile."""

    path: Path
    sha256: str
    _raw_json: str

    @property
    def raw(self) -> dict[str, Any]:
        return json.loads(self._raw_json)

    def _compact_wan(self, *, include_distill_mapping: bool) -> dict[str, Any]:
        method = self.raw["method"]["compact_wan"]
        keys = {
            "precision",
            "dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "head_dim",
        }
        if include_distill_mapping:
            keys.update(
                {
                    "hidden_anchor_layers",
                    "motion_anchor_layers",
                    "teacher_layer_mapping",
                }
            )
        return {key: method[key] for key in method if key in keys}

    def _training_dataset(
        self,
        *,
        sampled: bool,
    ) -> dict[str, Any]:
        raw = self.raw
        dataset = {
            "root": raw["paths"]["packed_dataset"],
            "max_open_shards": raw["data"]["max_open_shards"],
        }
        if sampled:
            dataset.update(
                {
                    "sampler_seed": raw["data"]["sampler_seed"],
                    "normalization_epsilon": raw["method"]["normalization_epsilon"],
                    "samples_per_episode": raw["data"]["samples_per_episode"],
                }
            )
        return dataset

    def _training_common(self) -> dict[str, Any]:
        raw = self.raw
        return {
            **raw["training"]["common"],
            "checkpoint_dir": raw["paths"]["training_runs"],
        }

    def training_config(self, stage: str) -> dict[str, Any]:
        raw = self.raw
        paths = raw["paths"]
        method = raw["method"]
        training = raw["training"]
        if stage == "stage1_pca":
            stage1 = training["stage1"]
            return {
                "name": "absolute_motion_stage1_pca",
                "device": training["common"]["device"],
                "log_level": training["common"]["log_level"],
                "flow_matching": method["flow_matching"],
                "dataset": self._training_dataset(sampled=False),
                "teacher": {
                    "checkpoint_path": paths["wan_root"],
                    "config_path": paths["wan_root"],
                    "precision": method["compact_wan"]["precision"],
                },
                "student": {
                    "hidden_anchor_layers": method["compact_wan"][
                        "hidden_anchor_layers"
                    ],
                    "motion_anchor_layers": method["compact_wan"][
                        "motion_anchor_layers"
                    ],
                },
                "distill": {
                    "projection_dim": stage1["projection_dim"],
                    "hidden_teacher_layers": stage1["hidden_teacher_layers"],
                    "motion_teacher_layers": stage1["motion_teacher_layers"],
                },
                "pca_prep": training["stage1_pca"],
                "artifacts": {"output_dir": paths["pca_artifacts"]},
            }

        common = self._training_common()
        compact = {
            "checkpoint_path": paths["wan_root"],
            "config_path": paths["wan_root"],
            "future_video_size": method["video"]["future_size"],
            **self._compact_wan(include_distill_mapping=True),
        }
        if stage == "stage1":
            stage1 = training["stage1"]
            return {
                "name": "absolute_motion_stage1_video",
                "flow_matching": method["flow_matching"],
                **common,
                "allow_tf32": stage1["allow_tf32"],
                "cudnn_benchmark": stage1["cudnn_benchmark"],
                "max_steps": stage1["max_steps"],
                "learning_rate": stage1["learning_rate"],
                "min_lr_ratio": stage1["min_lr_ratio"],
                "warmup_steps": stage1["warmup_steps"],
                "dataset": self._training_dataset(sampled=True),
                "teacher": {
                    "checkpoint_path": paths["wan_root"],
                    "config_path": paths["wan_root"],
                    "precision": method["compact_wan"]["precision"],
                    "pca_stats_path": str(
                        Path(paths["pca_artifacts"]) / "pca_stats.pt"
                    ),
                },
                "student": compact,
                "distill": {
                    "projection_dim": stage1["projection_dim"],
                    "hidden_teacher_layers": stage1["hidden_teacher_layers"],
                    "motion_teacher_layers": stage1["motion_teacher_layers"],
                    "lambda_gt": stage1["lambda_gt"],
                    "lambda_hidden_schedule": stage1["lambda_hidden_schedule"],
                    "lambda_motion_schedule": stage1["lambda_motion_schedule"],
                    "schedule_boundaries": stage1["schedule_boundaries"],
                    "timestep_sampler": stage1["timestep_sampler"],
                    "sigma_loss_weights": stage1["sigma_loss_weights"],
                },
            }

        if stage not in {"stage2", "stage3"}:
            raise ValueError(f"Unknown absolute-motion training stage: {stage!r}")
        stage_config = training[stage]
        run_name = (
            "absolute_motion_stage2_action"
            if stage == "stage2"
            else "absolute_motion_stage3_joint"
        )
        config: dict[str, Any] = {
            "name": run_name,
            "flow_matching": method["flow_matching"],
            **common,
            "max_steps": stage_config["max_steps"],
            "min_lr_ratio": stage_config["min_lr_ratio"],
            "warmup_steps": stage_config["warmup_steps"],
            "dataset": self._training_dataset(sampled=True),
            "model": {
                "initial_checkpoint": (
                    paths["motion_init_checkpoint"]
                    if stage == "stage2"
                    else paths["stage2_checkpoint"]
                ),
                "compact_wan": compact,
                "action_expert": method["action_expert"],
                "absolute_motion": {
                    "history_count": method["head_flow"]["count"],
                    "flow_contract": flow_compute_contract(method["head_flow"]),
                },
            },
        }
        if stage == "stage2":
            config["learning_rate"] = stage_config["learning_rate"]
            config["loss"] = {"action_weight": stage_config["action_weight"]}
        else:
            config["action_learning_rate"] = stage_config["action_learning_rate"]
            config["video_learning_rate"] = stage_config["video_learning_rate"]
            config["loss"] = {
                "action_weight": stage_config["action_weight"],
                "video_weight_initial": stage_config["video_weight_initial"],
                "video_weight": stage_config["video_weight"],
                "video_weight_anneal_steps": stage_config["video_weight_anneal_steps"],
            }
        return config

    def deepspeed_config(self) -> dict[str, Any]:
        raw = self.raw
        distributed = raw["training"]["distributed"]
        common = raw["training"]["common"]
        return {
            "zero_optimization": {"stage": distributed["zero_stage"]},
            "bf16": {"enabled": distributed["bf16"]},
            "gradient_clipping": common["grad_clip_norm"],
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "steps_per_print": distributed["steps_per_print"],
        }

    def packing_config(self) -> dict[str, Any]:
        raw = self.raw
        paths = raw["paths"]
        method = raw["method"]
        video = method["video"]
        packing = raw["packing"]
        return {
            "output_root": paths["packed_dataset"],
            "external_assets_manifest": paths["external_assets_manifest"],
            "source": {
                "root": paths["source_dataset"],
                "splits": raw["data"]["splits"],
                "num_video_frames": video["num_frames"],
                "video_size": video["size"],
                "global_downsample_rate": video["global_downsample_rate"],
                "video_action_freq_ratio": video["video_action_frequency_ratio"],
                "normalization_epsilon": method["normalization_epsilon"],
            },
            "action_stats": {
                "source_path": paths["base_action_stats"],
                "sha256": raw["initialization"]["base_action_stats_sha256"],
                "qpos_dim": method["action_expert"]["action_dim"],
            },
            "head_flow": {
                "cache_root": paths["head_flow_cache"],
                "source_root": paths["source_dataset"],
                "motion_stats_path": str(
                    Path(paths["head_flow_cache"]) / "motion_stats.json"
                ),
                **method["head_flow"],
            },
            "latent": {
                "vae_path": str(Path(paths["wan_root"]) / "Wan2.2_VAE.pth"),
                "precision": method["compact_wan"]["precision"],
                "shard_size": packing["shard_size"],
                "future_video_size": video["future_size"],
            },
            "language": {
                "items_per_shard": packing["language_items_per_shard"],
            },
            "dataset": {
                "max_open_shards": packing["max_open_shards"],
                "samples_per_episode": raw["data"]["samples_per_episode"],
                "sampler_seed": raw["data"]["sampler_seed"],
            },
            "build": {
                "device": packing["device"],
                "batch_size": packing["batch_size"],
                "num_workers": packing["num_workers"],
                "pin_memory": packing["pin_memory"],
                "persistent_workers": packing["persistent_workers"],
                "prefetch_factor": packing["prefetch_factor"],
                "qpos_cache_size": packing["qpos_cache_size"],
                "progress_interval_seconds": packing["progress_interval_seconds"],
                "log_level": packing["log_level"],
                "dist_timeout_minutes": packing["distributed_timeout_minutes"],
            },
        }

    def inference_config(self) -> dict[str, Any]:
        raw = self.raw
        paths = raw["paths"]
        method = raw["method"]
        video = method["video"]
        return {
            "checkpoint_path": paths["stage3_checkpoint"],
            "checkpoint_artifact_id": raw["inference"]["checkpoint_artifact_id"],
            "checkpoint_manifest": paths["checkpoint_manifest"],
            "wan_root": paths["wan_root"],
            "external_assets_manifest": paths["external_assets_manifest"],
            "normalization_epsilon": method["normalization_epsilon"],
            "device": raw["inference"]["device"],
            "flow_matching": {
                key: method["flow_matching"][key]
                for key in ("shift", "sigma_min", "extra_one_step")
            },
            "observation": method["observation"],
            "common": {
                "num_video_frames": video["num_frames"],
                "video_height": video["size"][0],
                "video_width": video["size"][1],
            },
            "model": {
                "compact_wan": {
                    **self._compact_wan(include_distill_mapping=False),
                    "future_video_size": video["future_size"],
                },
                "action_expert": method["action_expert"],
                "absolute_motion": {
                    "history_count": method["head_flow"]["count"],
                    "temporal_contract": TEMPORAL_CONTRACT,
                    "timestamp_source": TIMESTAMP_SOURCE,
                    "spatial_unit": SPATIAL_UNIT,
                    "flow_contract": flow_compute_contract(method["head_flow"]),
                },
            },
            "head_flow": method["head_flow"],
            "inference": {
                key: raw["inference"][key]
                for key in (
                    "num_inference_steps",
                    "video_refresh_steps",
                    "action_interval_ms",
                    "scene_prefix",
                )
            },
        }

    def benchmark_config(self) -> dict[str, Any]:
        raw = self.raw
        paths = raw["paths"]
        benchmark = dict(raw["benchmark"])
        domino_commit = str(raw["collection"]["domino_commit"])
        project_root = Path(paths["project_root"])
        benchmark.update(
            {
                "baseline": raw["baseline"],
                "domino_commit": domino_commit,
                "domino_source_sha256": domino_source_sha256(domino_commit),
                "eval_policy_sha256": domino_eval_policy_sha256(domino_commit),
                "project_root": paths["project_root"],
                "deploy_config": str(self.path),
                "runtime_root": str(project_root / "src" / "dynamicwam" / "runtime"),
                "extra_pythonpath": [str(project_root / "src")],
                "run_root": str(
                    Path(paths["evaluation_runs"]) / "absolute_motion_domino_level1"
                ),
                "integrity_paths": [
                    str(project_root / "src" / "dynamicwam"),
                ],
            }
        )
        return benchmark

    def domino_base_config(self) -> dict[str, Any]:
        benchmark = self.benchmark_config()
        return {
            "policy_name": benchmark["policy_registration_name"],
            "task_name": benchmark["tasks"][0],
            "task_config": benchmark["task_config"],
            "ckpt_setting": benchmark["checkpoint_label"],
            "instruction_type": benchmark["instruction_type"],
            "seed": 0,
            "start_seed": benchmark["start_seed"],
            "episode_num": 1,
            "eval_output_root": benchmark["run_root"],
            "dynamicwam_root": benchmark["project_root"],
            "dynamicwam_deploy_config": benchmark["deploy_config"],
        }


def load_profile(path: str | Path | None = None) -> AbsoluteMotionProfile:
    resolved_path = Path(path or default_profile_path()).expanduser().resolve()
    if resolved_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"DynamicWAM profiles must be YAML files: {resolved_path}")
    source = resolved_path.read_bytes()
    text = source.decode("utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("Loading the DynamicWAM profile requires PyYAML") from exc
    raw = _resolve_profile_paths(
        yaml.safe_load(text),
        profile_path=resolved_path,
    )
    validated = validate_absolute_motion_profile(raw)
    return AbsoluteMotionProfile(
        path=resolved_path,
        sha256=_sha256_bytes(source),
        _raw_json=_canonical_json(validated),
    )


def write_config_snapshot(
    directory: str | Path,
    *,
    profile: AbsoluteMotionProfile,
    label: str,
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    """Write an immutable profile+resolved-config audit record."""

    if re.fullmatch(r"[a-z0-9_]+", label) is None:
        raise ValueError(f"Invalid config snapshot label: {label!r}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    resolved_text = _pretty_json(resolved_config)
    resolved_bytes = resolved_text.encode("utf-8")
    resolved_sha256 = _sha256_bytes(resolved_bytes)
    source_name = f"profile{profile.path.suffix.lower()}"
    resolved_name = f"resolved_{label}.json"
    audit = {
        "schema_version": 1,
        "baseline": profile.raw["baseline"],
        "label": label,
        "source_path": str(profile.path),
        "source_file": source_name,
        "source_sha256": profile.sha256,
        "resolved_file": resolved_name,
        "resolved_sha256": resolved_sha256,
    }
    audit_path = root / "audit.json"
    if audit_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing != audit:
            raise RuntimeError(f"Config audit mismatch; refusing to mix runs in {root}")

    _write_immutable(root / source_name, profile.path.read_bytes())
    _write_immutable(root / resolved_name, resolved_bytes)
    _write_immutable(audit_path, _pretty_json(audit).encode("utf-8"))
    return audit
