#!/usr/bin/env python3
"""Verify the released DynamicWAM checkpoint and matching config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_MANIFEST = ROOT / "manifests" / "checkpoints.json"
DEFAULT_CHECKPOINT_ROOT = ROOT / "external" / "checkpoints"
DEFAULT_CONFIG = ROOT / "configs" / "absolute_motion_v2.yaml"
sys.path.insert(0, str(SRC))

from dynamicwam.external_assets import (  # noqa: E402
    load_checkpoint_manifest,
    verify_checkpoint_artifact,
)
from dynamicwam.integrity import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the released DynamicWAM checkpoint and config",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.expanduser().resolve()
    root = arguments.root.expanduser().resolve()
    config_path = arguments.config.expanduser().resolve()
    manifest = load_checkpoint_manifest(manifest_path)
    config = manifest["config"]
    if (
        not config_path.is_file()
        or config_path.stat().st_size != int(config["size_bytes"])
        or sha256_file(config_path) != str(config["sha256"])
    ):
        raise RuntimeError(
            f"released config does not match the manifest: {config_path}"
        )
    artifact_id = str(manifest["artifacts"][0]["id"])
    verify_checkpoint_artifact(
        root=root,
        manifest_path=manifest_path,
        artifact_id=artifact_id,
    )
    print(f"verified {artifact_id} checkpoint and config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
