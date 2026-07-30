#!/usr/bin/env python3
"""Stable entrypoint for the DynamicWAM packed-dataset builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
sys.path.insert(0, str(SRC))

from dynamicwam.training.data.robotwin2.train_dataset_packer import main  # noqa: E402

if __name__ == "__main__":
    if not any(
        arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]
    ):
        sys.argv[1:1] = ["--config", str(DEFAULT_CONFIG)]
    main()
