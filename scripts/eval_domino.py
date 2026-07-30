#!/usr/bin/env python3
"""Stable entrypoint for the official DynamicWAM DOMINO Level 1 suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
sys.path.insert(0, str(SRC))

from dynamicwam.evaluation.run_domino_level1_suite import main  # noqa: E402

if __name__ == "__main__":
    if not any(
        argument == "--config" or argument.startswith("--config=")
        for argument in sys.argv[1:]
    ):
        sys.argv[1:1] = ["--config", str(DEFAULT_CONFIG)]
    raise SystemExit(main())
