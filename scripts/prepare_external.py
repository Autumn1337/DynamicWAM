#!/usr/bin/env python3
"""Download or reconstruct pinned non-checkpoint dependencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
sys.path.insert(0, str(SRC))

from dynamicwam.config import load_profile  # noqa: E402
from dynamicwam.external_setup import (  # noqa: E402
    prepare_curobo_source,
    prepare_domino_python_runtime,
    prepare_domino_source,
    prepare_robotwin_assets,
    prepare_wan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="component", required=True)

    wan = subparsers.add_parser("wan")
    wan.add_argument(
        "--purpose",
        action="append",
        choices=("architecture", "inference", "language", "packing", "training"),
        default=None,
    )
    subparsers.add_parser("domino-source")
    subparsers.add_parser("domino-runtime")
    subparsers.add_parser("robotwin-assets")
    subparsers.add_parser("curobo-source")
    arguments = parser.parse_args()

    profile = load_profile(arguments.config)
    paths = profile.raw["paths"]
    manifest_path = Path(paths["external_assets_manifest"])
    component = str(arguments.component)
    if component == "wan":
        prepare_wan(
            destination=Path(paths["wan_root"]),
            manifest_path=manifest_path,
            purposes=tuple(arguments.purpose or ("inference",)),
        )
    elif component == "domino-source":
        prepare_domino_source(
            destination=Path(profile.raw["benchmark"]["domino_root"]),
            manifest_path=manifest_path,
        )
    elif component == "domino-runtime":
        prepare_domino_python_runtime(
            python=Path(profile.raw["benchmark"]["python"]),
            manifest_path=manifest_path,
        )
    elif component == "robotwin-assets":
        prepare_robotwin_assets(
            destination=Path(paths["project_root"]) / "external" / "robotwin-assets",
            domino_root=Path(profile.raw["benchmark"]["domino_root"]),
            manifest_path=manifest_path,
        )
    elif component == "curobo-source":
        prepare_curobo_source(
            destination=Path(profile.raw["benchmark"]["curobo_root"]).parent,
            manifest_path=manifest_path,
        )
    else:
        raise AssertionError(f"unhandled component: {component}")
    print(f"prepared and verified: {component}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
