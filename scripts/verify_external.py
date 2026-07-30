#!/usr/bin/env python3
"""Verify pinned external assets without downloading or modifying them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
sys.path.insert(0, str(SRC))

from dynamicwam.config import load_profile  # noqa: E402
from dynamicwam.external_assets import (  # noqa: E402
    verify_checkpoint_artifact,
    verify_robotwin_asset_trees,
    verify_wan_assets,
)
from dynamicwam.external_setup import (  # noqa: E402
    verify_curobo_runtime,
    verify_curobo_source,
    verify_domino_python_runtime,
    verify_domino_source,
    verify_robotwin_asset_links,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "scope",
        choices=(
            "checkpoints",
            "curobo-runtime",
            "curobo-source",
            "domino-runtime",
            "domino-source",
            "robotwin-assets",
            "wan-inference",
            "wan-architecture",
            "wan-language",
            "wan-packing",
            "wan-training",
        ),
    )
    arguments = parser.parse_args()

    profile = load_profile(arguments.config)
    raw = profile.raw
    paths = raw["paths"]
    manifest_path = Path(paths["external_assets_manifest"])
    scope = str(arguments.scope)
    if scope == "checkpoints":
        verify_checkpoint_artifact(
            root=Path(paths["stage3_checkpoint"]).parent,
            manifest_path=Path(paths["checkpoint_manifest"]),
            artifact_id=str(raw["inference"]["checkpoint_artifact_id"]),
        )
    elif scope == "domino-source":
        verify_domino_source(
            destination=Path(raw["benchmark"]["domino_root"]),
            manifest_path=manifest_path,
        )
    elif scope == "domino-runtime":
        verify_domino_python_runtime(
            python=Path(raw["benchmark"]["python"]),
            manifest_path=manifest_path,
        )
    elif scope == "robotwin-assets":
        asset_root = Path(paths["project_root"]) / "external" / "robotwin-assets"
        verify_robotwin_asset_trees(
            root=asset_root,
            manifest_path=manifest_path,
        )
        verify_robotwin_asset_links(
            asset_root=asset_root,
            domino_root=Path(raw["benchmark"]["domino_root"]),
        )
    elif scope == "curobo-source":
        verify_curobo_source(
            destination=Path(raw["benchmark"]["curobo_root"]).parent,
            manifest_path=manifest_path,
        )
    elif scope == "curobo-runtime":
        verify_curobo_runtime(
            destination=Path(raw["benchmark"]["curobo_root"]).parent,
            manifest_path=manifest_path,
        )
    else:
        verify_wan_assets(
            root=Path(paths["wan_root"]),
            manifest_path=manifest_path,
            purpose=scope.removeprefix("wan-"),
        )
    print(f"verified external scope: {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
