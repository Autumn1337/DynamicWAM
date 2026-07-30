from __future__ import annotations

import sys
from pathlib import Path


def add_dynamicwam_to_path(project_root: str) -> Path:
    """Expose the source tree from the configured DynamicWAM project root."""
    source_root = (Path(project_root).expanduser() / "src").resolve()
    if not (source_root / "dynamicwam" / "inference" / "runner.py").is_file():
        raise FileNotFoundError(
            f"DynamicWAM source tree not found under project root: {source_root}"
        )
    text = str(source_root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return source_root
