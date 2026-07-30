from __future__ import annotations

import math
from typing import TypedDict


class SchedulerState(TypedDict):
    step_count: int
    base_lrs: list[float]
    total_steps: int
    warmup_steps: int
    min_lr_ratio: float


class ClampedCosineScheduler:
    """DynamicWAM cosine decay with optional warmup and a fixed learning-rate floor."""

    def __init__(
        self,
        optimizer,
        *,
        total_steps: int,
        min_lr_ratio: float,
        warmup_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_count = 0
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def _lr_at(self, step: int, base_lr: float) -> float:
        if self.warmup_steps and step <= self.warmup_steps:
            return base_lr * step / self.warmup_steps
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        decay_step = min(max(0, step - self.warmup_steps), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
        floor = base_lr * self.min_lr_ratio
        return floor + (base_lr - floor) * cosine

    def step(self) -> None:
        self.step_count += 1
        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = self._lr_at(self.step_count, base_lr)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> SchedulerState:
        return {
            "step_count": self.step_count,
            "base_lrs": self.base_lrs,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state: SchedulerState) -> None:
        self.step_count = int(state["step_count"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        if int(state["total_steps"]) != self.total_steps:
            raise ValueError("scheduler total_steps differs from the DynamicWAM config")
        if int(state["warmup_steps"]) != self.warmup_steps:
            raise ValueError(
                "scheduler warmup_steps differs from the DynamicWAM config"
            )
        if float(state["min_lr_ratio"]) != self.min_lr_ratio:
            raise ValueError(
                "scheduler min_lr_ratio differs from the DynamicWAM config"
            )


def create_scheduler(
    optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> ClampedCosineScheduler:
    return ClampedCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
