#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
sys.path.insert(0, str(SRC))

from dynamicwam.training.collect_exact_domino import main  # noqa: E402

if __name__ == "__main__":
    if not any(
        argument == "--config" or argument.startswith("--config=")
        for argument in sys.argv[1:]
    ):
        sys.argv[1:1] = ["--config", str(DEFAULT_CONFIG)]
    main()
