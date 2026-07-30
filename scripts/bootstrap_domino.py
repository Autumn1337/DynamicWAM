#!/usr/bin/env python3
"""Reconstruct the exact DOMINO source used by the DynamicWAM mainline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_DESTINATION = ROOT / "external" / "DOMINO"
DEFAULT_MANIFEST = ROOT / "manifests" / "external_assets.json"
sys.path.insert(0, str(SRC))

from dynamicwam.external_setup import prepare_domino_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct the exact DOMINO dependency used by DynamicWAM",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"checkout destination (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    prepare_domino_source(
        destination=arguments.destination,
        manifest_path=arguments.manifest.expanduser().resolve(),
    )
    print(f"DOMINO reconstructed and verified: {arguments.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
