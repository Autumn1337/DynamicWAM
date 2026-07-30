"""Strict loaders and verifiers for release-time external artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dynamicwam.integrity import sha256_file, sha256_tree

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
WAN_PURPOSES = frozenset(
    {"architecture", "inference", "language", "packing", "training"}
)
CUROBO_EXTENSION_MODULES = frozenset(
    {
        "geom_cu",
        "kinematics_fused_cu",
        "lbfgs_step_cu",
        "line_search_cu",
        "tensor_step_cu",
    }
)


def _exact_object(
    value: Any,
    expected_keys: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != expected_keys:
        raise ValueError(
            f"{label} keys differ from the schema: "
            f"missing={sorted(expected_keys - set(value))}, "
            f"unknown={sorted(set(value) - expected_keys)}"
        )
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _git_revision(value: Any, *, label: str) -> str:
    revision = _nonempty_string(value, label=label)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"{label} must be a full lowercase Git revision")
    return revision


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must stay within its artifact root: {value!r}")
    return candidate


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _size(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def load_external_assets_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json_object(path, label="external asset manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"unsupported external asset manifest schema: {manifest.get('schema_version')!r}"
        )
    expected = {"schema_version", "wan", "domino", "robotwin_assets", "curobo"}
    if set(manifest) != expected:
        raise ValueError(
            "external asset manifest keys differ from the release schema: "
            f"missing={sorted(expected - set(manifest))}, "
            f"unknown={sorted(set(manifest) - expected)}"
        )
    _validate_wan_manifest(manifest["wan"])
    _validate_domino_manifest(manifest["domino"])
    _validate_robotwin_manifest(manifest["robotwin_assets"])
    _validate_curobo_manifest(manifest["curobo"])
    return manifest


def _validate_required_for(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
        or len(set(value)) != len(value)
        or not set(value).issubset(WAN_PURPOSES)
    ):
        raise ValueError(
            f"{label} must contain unique values from {sorted(WAN_PURPOSES)}"
        )
    return value


def _validate_wan_manifest(value: Any) -> None:
    wan = _exact_object(
        value,
        {"repo_id", "repo_type", "revision", "license", "root", "files", "trees"},
        label="wan",
    )
    _nonempty_string(wan["repo_id"], label="wan.repo_id")
    if wan["repo_type"] != "model":
        raise ValueError("wan.repo_type must equal model")
    _git_revision(wan["revision"], label="wan.revision")
    _nonempty_string(wan["license"], label="wan.license")
    _relative_path(wan["root"], label="wan.root")
    files = wan["files"]
    trees = wan["trees"]
    if not isinstance(files, list) or not files:
        raise ValueError("wan.files must be a non-empty list")
    if not isinstance(trees, list):
        raise ValueError("wan.trees must be a list")
    paths: set[str] = set()
    purposes: set[str] = set()
    for index, value in enumerate(files):
        entry = _exact_object(
            value,
            {"path", "sha256", "size_bytes", "required_for"},
            label=f"wan.files[{index}]",
        )
        relative = _relative_path(entry["path"], label=f"wan.files[{index}].path")
        if relative.as_posix() in paths:
            raise ValueError(f"duplicate WAN manifest path: {relative}")
        paths.add(relative.as_posix())
        _sha256(entry["sha256"], label=f"wan.files[{index}].sha256")
        _size(entry["size_bytes"], label=f"wan.files[{index}].size_bytes")
        purposes.update(
            _validate_required_for(
                entry["required_for"],
                label=f"wan.files[{index}].required_for",
            )
        )
    for index, value in enumerate(trees):
        entry = _exact_object(
            value,
            {"path", "sha256", "required_for"},
            label=f"wan.trees[{index}]",
        )
        _relative_path(entry["path"], label=f"wan.trees[{index}].path")
        _sha256(entry["sha256"], label=f"wan.trees[{index}].sha256")
        purposes.update(
            _validate_required_for(
                entry["required_for"],
                label=f"wan.trees[{index}].required_for",
            )
        )
    if purposes != set(WAN_PURPOSES):
        raise ValueError(f"WAN manifest purpose coverage differs: {sorted(purposes)}")


def _validate_domino_manifest(value: Any) -> None:
    domino = _exact_object(
        value,
        {
            "upstream_url",
            "upstream_base_revision",
            "evaluated_revision",
            "license",
            "patch",
            "patch_sha256",
            "source_root",
            "source_tree_sha256",
            "source_tree_excluded_prefixes",
            "eval_policy",
            "eval_policy_sha256",
            "evaluated_python_runtime",
        },
        label="domino",
    )
    upstream_url = _nonempty_string(
        domino["upstream_url"],
        label="domino.upstream_url",
    )
    if not upstream_url.startswith("https://"):
        raise ValueError("domino.upstream_url must use HTTPS")
    _git_revision(
        domino["upstream_base_revision"],
        label="domino.upstream_base_revision",
    )
    _git_revision(domino["evaluated_revision"], label="domino.evaluated_revision")
    _nonempty_string(domino["license"], label="domino.license")
    for key in ("patch", "source_root", "eval_policy"):
        _relative_path(domino[key], label=f"domino.{key}")
    for key in ("patch_sha256", "source_tree_sha256", "eval_policy_sha256"):
        _sha256(domino[key], label=f"domino.{key}")
    excluded = domino["source_tree_excluded_prefixes"]
    if (
        not isinstance(excluded, list)
        or not excluded
        or len(set(excluded)) != len(excluded)
    ):
        raise ValueError(
            "domino.source_tree_excluded_prefixes must be a unique non-empty list"
        )
    for index, prefix in enumerate(excluded):
        _relative_path(
            prefix,
            label=f"domino.source_tree_excluded_prefixes[{index}]",
        )
    runtime = domino["evaluated_python_runtime"]
    if not isinstance(runtime, list) or not runtime:
        raise ValueError("domino.evaluated_python_runtime must be non-empty")
    distributions: set[str] = set()
    relative_paths: set[str] = set()
    for index, value in enumerate(runtime):
        label = f"domino.evaluated_python_runtime[{index}]"
        entry = _exact_object(
            value,
            {
                "distribution",
                "version",
                "wheel_filename",
                "wheel_sha256",
                "relative_path",
                "pristine_sha256",
                "evaluated_sha256",
                "replacements",
            },
            label=label,
        )
        distribution = _nonempty_string(
            entry["distribution"],
            label=f"{label}.distribution",
        )
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", distribution) is None:
            raise ValueError(f"{label}.distribution must be canonical lowercase")
        if distribution in distributions:
            raise ValueError(f"duplicate DOMINO runtime distribution: {distribution}")
        distributions.add(distribution)
        _nonempty_string(entry["version"], label=f"{label}.version")
        wheel_filename = _relative_path(
            entry["wheel_filename"],
            label=f"{label}.wheel_filename",
        )
        if len(wheel_filename.parts) != 1 or wheel_filename.suffix != ".whl":
            raise ValueError(f"{label}.wheel_filename must be a wheel basename")
        _sha256(entry["wheel_sha256"], label=f"{label}.wheel_sha256")
        relative_path = _relative_path(
            entry["relative_path"],
            label=f"{label}.relative_path",
        ).as_posix()
        if relative_path in relative_paths:
            raise ValueError(f"duplicate DOMINO runtime module path: {relative_path}")
        relative_paths.add(relative_path)
        _sha256(entry["pristine_sha256"], label=f"{label}.pristine_sha256")
        _sha256(entry["evaluated_sha256"], label=f"{label}.evaluated_sha256")
        replacements = entry["replacements"]
        if not isinstance(replacements, list) or not replacements:
            raise ValueError(f"{label}.replacements must be non-empty")
        before_values: set[str] = set()
        for replacement_index, replacement_value in enumerate(replacements):
            replacement_label = f"{label}.replacements[{replacement_index}]"
            replacement = _exact_object(
                replacement_value,
                {"before", "after"},
                label=replacement_label,
            )
            before = _nonempty_string(
                replacement["before"],
                label=f"{replacement_label}.before",
            )
            after = _nonempty_string(
                replacement["after"],
                label=f"{replacement_label}.after",
            )
            if before == after or before in before_values:
                raise ValueError(f"{replacement_label} is duplicate or unchanged")
            before_values.add(before)


def _validate_robotwin_manifest(value: Any) -> None:
    section = _exact_object(
        value,
        {"repo_id", "repo_type", "revision", "license", "root", "archives"},
        label="robotwin_assets",
    )
    _nonempty_string(section["repo_id"], label="robotwin_assets.repo_id")
    if section["repo_type"] != "dataset":
        raise ValueError("robotwin_assets.repo_type must equal dataset")
    _git_revision(section["revision"], label="robotwin_assets.revision")
    _nonempty_string(section["license"], label="robotwin_assets.license")
    _relative_path(section["root"], label="robotwin_assets.root")
    archives = section["archives"]
    if not isinstance(archives, list) or not archives:
        raise ValueError("robotwin_assets.archives must be a non-empty list")
    paths: set[str] = set()
    directories: set[str] = set()
    for index, value in enumerate(archives):
        entry = _exact_object(
            value,
            {
                "path",
                "sha256",
                "size_bytes",
                "extracts_to",
                "tree_sha256",
                "file_count",
                "extracted_size_bytes",
            },
            label=f"robotwin_assets.archives[{index}]",
        )
        relative = _relative_path(
            entry["path"],
            label=f"robotwin_assets.archives[{index}].path",
        ).as_posix()
        directory = _relative_path(
            entry["extracts_to"],
            label=f"robotwin_assets.archives[{index}].extracts_to",
        ).as_posix()
        if relative in paths or directory in directories:
            raise ValueError("RoboTwin asset archive paths must be unique")
        paths.add(relative)
        directories.add(directory)
        _sha256(
            entry["sha256"],
            label=f"robotwin_assets.archives[{index}].sha256",
        )
        _sha256(
            entry["tree_sha256"],
            label=f"robotwin_assets.archives[{index}].tree_sha256",
        )
        for key in ("size_bytes", "file_count", "extracted_size_bytes"):
            _size(
                entry[key],
                label=f"robotwin_assets.archives[{index}].{key}",
            )


def _validate_curobo_manifest(value: Any) -> None:
    curobo = _exact_object(
        value,
        {
            "upstream_url",
            "tag",
            "revision",
            "license",
            "source_root",
            "source_tree_sha256",
            "evaluated_extensions",
            "evaluated_runtime",
        },
        label="curobo",
    )
    upstream_url = _nonempty_string(
        curobo["upstream_url"],
        label="curobo.upstream_url",
    )
    if not upstream_url.startswith("https://"):
        raise ValueError("curobo.upstream_url must use HTTPS")
    _nonempty_string(curobo["tag"], label="curobo.tag")
    _git_revision(curobo["revision"], label="curobo.revision")
    _nonempty_string(curobo["license"], label="curobo.license")
    _relative_path(curobo["source_root"], label="curobo.source_root")
    _sha256(curobo["source_tree_sha256"], label="curobo.source_tree_sha256")
    extensions = curobo["evaluated_extensions"]
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("curobo.evaluated_extensions must be non-empty")
    modules: set[str] = set()
    paths: set[str] = set()
    for index, value in enumerate(extensions):
        label = f"curobo.evaluated_extensions[{index}]"
        entry = _exact_object(
            value,
            {"module", "path", "sha256", "size_bytes"},
            label=label,
        )
        module = _nonempty_string(entry["module"], label=f"{label}.module")
        relative = _relative_path(entry["path"], label=f"{label}.path").as_posix()
        if module in modules or relative in paths:
            raise ValueError("CuRobo extension modules and paths must be unique")
        modules.add(module)
        paths.add(relative)
        _sha256(entry["sha256"], label=f"{label}.sha256")
        _size(entry["size_bytes"], label=f"{label}.size_bytes")
    if modules != set(CUROBO_EXTENSION_MODULES):
        raise ValueError(f"CuRobo extension module coverage differs: {sorted(modules)}")
    runtime = _exact_object(
        curobo["evaluated_runtime"],
        {
            "python",
            "torch",
            "torch_cuda",
            "cuda_extension_runtime",
            "gpu_compute_capability",
        },
        label="curobo.evaluated_runtime",
    )
    for key, item in runtime.items():
        _nonempty_string(item, label=f"curobo.evaluated_runtime.{key}")


def _verify_file(
    root: Path,
    entry: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    relative = _relative_path(entry.get("path"), label=f"{label}.path")
    expected_size = _size(entry.get("size_bytes"), label=f"{label}.size_bytes")
    expected_sha256 = _sha256(entry.get("sha256"), label=f"{label}.sha256")
    artifact = root / relative
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} is missing: {artifact}")
    actual_size = artifact.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}: "
            f"{artifact}"
        )
    actual_sha256 = sha256_file(artifact)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}: {artifact}"
        )
    return {
        "path": str(artifact),
        "kind": "file",
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
    }


def _verify_tree(
    root: Path,
    entry: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    relative = _relative_path(entry.get("path"), label=f"{label}.path")
    expected_sha256 = _sha256(entry.get("sha256"), label=f"{label}.sha256")
    artifact = root / relative
    if not artifact.is_dir():
        raise FileNotFoundError(f"{label} is missing: {artifact}")
    actual_sha256 = sha256_tree(artifact)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} tree SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}: {artifact}"
        )
    return {
        "path": str(artifact),
        "kind": "directory",
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
    }


def verify_wan_assets(
    *,
    root: Path,
    manifest_path: Path,
    purpose: str,
) -> dict[str, dict[str, Any]]:
    if purpose not in WAN_PURPOSES:
        raise ValueError(
            f"unsupported WAN verification purpose {purpose!r}; "
            f"expected one of {sorted(WAN_PURPOSES)}"
        )
    manifest = load_external_assets_manifest(manifest_path)
    wan = manifest["wan"]
    if not isinstance(wan, dict):
        raise ValueError("external asset manifest wan entry must be an object")
    files = wan.get("files")
    trees = wan.get("trees")
    if not isinstance(files, list) or not isinstance(trees, list):
        raise ValueError("external asset manifest wan files and trees must be lists")

    verified: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"wan.files[{index}] must be an object")
        required_for = raw_entry.get("required_for")
        if not isinstance(required_for, list) or not all(
            isinstance(value, str) for value in required_for
        ):
            raise ValueError(f"wan.files[{index}].required_for must be a string list")
        if purpose not in required_for:
            continue
        label = f"wan:{raw_entry.get('path')}"
        verified[label] = _verify_file(root, raw_entry, label=label)

    for index, raw_entry in enumerate(trees):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"wan.trees[{index}] must be an object")
        required_for = raw_entry.get("required_for")
        if not isinstance(required_for, list) or not all(
            isinstance(value, str) for value in required_for
        ):
            raise ValueError(f"wan.trees[{index}].required_for must be a string list")
        if purpose not in required_for:
            continue
        label = f"wan-tree:{raw_entry.get('path')}"
        verified[label] = _verify_tree(root, raw_entry, label=label)

    if not verified:
        raise ValueError(f"WAN manifest contains no artifacts for purpose {purpose!r}")
    return verified


def load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json_object(path, label="checkpoint manifest")
    _exact_object(
        manifest,
        {
            "schema_version",
            "repo_id",
            "revision",
            "checkpoint_root",
            "config",
            "artifacts",
        },
        label="checkpoint manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"unsupported checkpoint manifest schema: {manifest.get('schema_version')!r}"
        )
    repo_id = _nonempty_string(
        manifest["repo_id"],
        label="checkpoint manifest.repo_id",
    )
    if repo_id.count("/") != 1:
        raise ValueError("checkpoint manifest.repo_id must use owner/name")
    _nonempty_string(
        manifest["revision"],
        label="checkpoint manifest.revision",
    )
    _relative_path(
        manifest["checkpoint_root"],
        label="checkpoint manifest.checkpoint_root",
    )
    config = _exact_object(
        manifest["config"],
        {"filename", "hub_path", "sha256", "size_bytes"},
        label="checkpoint manifest.config",
    )
    config_filename = _relative_path(
        config["filename"],
        label="checkpoint manifest.config.filename",
    )
    if len(config_filename.parts) != 1:
        raise ValueError("checkpoint manifest.config.filename must be a filename")
    _relative_path(
        config["hub_path"],
        label="checkpoint manifest.config.hub_path",
    )
    _sha256(config["sha256"], label="checkpoint manifest.config.sha256")
    _size(config["size_bytes"], label="checkpoint manifest.config.size_bytes")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("checkpoint manifest must contain the one released checkpoint")
    artifact = _exact_object(
        artifacts[0],
        {
            "id",
            "filename",
            "hub_path",
            "sha256",
            "size_bytes",
            "global_step",
            "format",
            "version",
        },
        label="checkpoint artifacts[0]",
    )
    if artifact["id"] != "full":
        raise ValueError("checkpoint manifest must release only artifact id 'full'")
    filename = _relative_path(
        artifact["filename"],
        label="checkpoint full.filename",
    )
    if len(filename.parts) != 1:
        raise ValueError("checkpoint full.filename must be a filename")
    _relative_path(
        artifact["hub_path"],
        label="checkpoint full.hub_path",
    )
    _sha256(artifact["sha256"], label="checkpoint full.sha256")
    _size(artifact["size_bytes"], label="checkpoint full.size_bytes")
    if _size(artifact["global_step"], label="checkpoint full.global_step") == 0:
        raise ValueError("checkpoint full.global_step must be positive")
    if artifact["format"] != "dynamicwam_absolute_motion_checkpoint":
        raise ValueError("checkpoint full.format is unsupported")
    if artifact["version"] != 2:
        raise ValueError("checkpoint full.version must be 2")
    return manifest


def verify_checkpoint_artifact(
    *,
    root: Path,
    manifest_path: Path,
    artifact_id: str,
) -> dict[str, dict[str, Any]]:
    manifest = load_checkpoint_manifest(manifest_path)
    matches = [
        entry
        for entry in manifest["artifacts"]
        if isinstance(entry, dict) and entry.get("id") == artifact_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint artifact id must resolve exactly once: {artifact_id!r}"
        )
    artifact = matches[0]
    checkpoint_entry = {
        "path": artifact.get("filename"),
        "sha256": artifact.get("sha256"),
        "size_bytes": artifact.get("size_bytes"),
    }
    return {
        "checkpoint": _verify_file(
            root,
            checkpoint_entry,
            label=f"checkpoint:{artifact_id}",
        ),
    }


def verify_robotwin_asset_trees(
    *,
    root: Path,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    manifest = load_external_assets_manifest(manifest_path)
    section = manifest["robotwin_assets"]
    if not isinstance(section, dict) or not isinstance(section.get("archives"), list):
        raise ValueError("robotwin_assets manifest entry is invalid")
    verified: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(section["archives"]):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"robotwin_assets.archives[{index}] must be an object")
        directory = raw_entry.get("extracts_to")
        entry = {
            "path": directory,
            "sha256": raw_entry.get("tree_sha256"),
        }
        label = f"robotwin-assets:{directory}"
        record = _verify_tree(root, entry, label=label)
        expected_count = _size(
            raw_entry.get("file_count"),
            label=f"robotwin_assets.archives[{index}].file_count",
        )
        actual_count = sum(
            candidate.is_file() for candidate in (root / str(directory)).rglob("*")
        )
        if actual_count != expected_count:
            raise RuntimeError(
                f"{label} file-count mismatch: expected {expected_count}, "
                f"got {actual_count}"
            )
        record["file_count"] = actual_count
        expected_size = _size(
            raw_entry.get("extracted_size_bytes"),
            label=f"robotwin_assets.archives[{index}].extracted_size_bytes",
        )
        actual_size = sum(
            candidate.stat().st_size
            for candidate in (root / str(directory)).rglob("*")
            if candidate.is_file()
        )
        if actual_size != expected_size:
            raise RuntimeError(
                f"{label} extracted-size mismatch: expected {expected_size}, "
                f"got {actual_size}"
            )
        record["size_bytes"] = actual_size
        verified[label] = record
    return verified
