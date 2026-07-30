from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NativeStepRecord:
    index: int
    wall_ms: float
    overrun_ms: float


@dataclass
class NativeExecutionTelemetry:
    records: List[NativeStepRecord] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": len(self.records),
            "kinds": {"action": len(self.records)} if self.records else {},
            "overrun_steps": sum(record.overrun_ms > 1.0 for record in self.records),
        }


class NativeStepper:
    """Execute actions directly in DOMINO without pacing or synthetic holds."""

    def __init__(self, task_env, *, action_interval_seconds: float) -> None:
        self.env = task_env
        self.action_interval_s = float(action_interval_seconds)
        if self.action_interval_s <= 0.0:
            raise ValueError("action_interval_seconds must be positive")
        self.telemetry = NativeExecutionTelemetry()

    def execute(self, action) -> None:
        start = time.perf_counter()
        self.env.take_action(action, action_type="qpos")
        end = time.perf_counter()
        self.telemetry.records.append(
            NativeStepRecord(
                index=int(self.env.take_action_cnt),
                wall_ms=(end - start) * 1000.0,
                overrun_ms=max(
                    0.0,
                    (end - start) - self.action_interval_s,
                )
                * 1000.0,
            )
        )
