from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VideoActionLossObjective:
    """Validated output and loss-weight contract for Stage 2/3 training."""

    trains_video: bool
    action_weight: float
    video_weight_initial: float
    video_weight_final: float
    video_weight_anneal_steps: int

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        stage_name: str,
    ) -> "VideoActionLossObjective":
        loss_cfg = config["loss"]
        action_weight = float(loss_cfg["action_weight"])

        if action_weight <= 0.0:
            raise ValueError(
                f"loss.action_weight must be positive, got {action_weight}"
            )
        if stage_name == "stage2":
            return cls(
                trains_video=False,
                action_weight=action_weight,
                video_weight_initial=0.0,
                video_weight_final=0.0,
                video_weight_anneal_steps=0,
            )
        if stage_name != "stage3":
            raise ValueError(f"unknown DynamicWAM video-action stage: {stage_name}")

        video_weight_initial = float(loss_cfg["video_weight_initial"])
        video_weight_final = float(loss_cfg["video_weight"])
        anneal_steps = int(loss_cfg["video_weight_anneal_steps"])
        if video_weight_initial < 0.0 or video_weight_final < 0.0:
            raise ValueError(
                "Video loss weights must be non-negative, got "
                f"initial={video_weight_initial}, final={video_weight_final}"
            )
        if anneal_steps < 2:
            raise ValueError(
                "DynamicWAM Stage 3 requires video_weight_anneal_steps >= 2"
            )

        return cls(
            trains_video=True,
            action_weight=action_weight,
            video_weight_initial=video_weight_initial,
            video_weight_final=video_weight_final,
            video_weight_anneal_steps=anneal_steps,
        )

    def video_weight_at(self, optimizer_step: int) -> float:
        """Return the weight used by a one-indexed optimizer update."""
        if not self.trains_video:
            return 0.0

        step = max(1, int(optimizer_step))
        progress = min(
            1.0,
            float(step - 1) / float(self.video_weight_anneal_steps - 1),
        )
        return (
            self.video_weight_initial
            + (self.video_weight_final - self.video_weight_initial) * progress
        )
