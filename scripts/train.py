#!/usr/bin/env python3
"""Stable dispatcher for the four DynamicWAM training stages."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
STAGES = {
    "stage1_pca": "dynamicwam.training.tools.prepare_stage1_pca",
    "stage1": "dynamicwam.training.train.stage1",
    "stage2": "dynamicwam.training.train.stage2",
    "stage3": "dynamicwam.training.train.stage3",
}


def usage() -> str:
    stages = "|".join(STAGES)
    return f"usage: {Path(sys.argv[0]).name} <{stages}> [stage arguments]"


def has_config(arguments: list[str]) -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in arguments)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        return 0 if len(sys.argv) >= 2 else 2
    stage = sys.argv[1]
    if stage not in STAGES:
        print(f"unknown training stage: {stage}\n{usage()}", file=sys.stderr)
        return 2

    module_name = STAGES[stage]
    downstream = sys.argv[2:]
    if not has_config(downstream):
        downstream = ["--config", str(DEFAULT_CONFIG), *downstream]

    sys.path.insert(0, str(SRC))
    sys.argv = [f"{Path(sys.argv[0]).name}:{stage}", *downstream]
    module = importlib.import_module(module_name)
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
