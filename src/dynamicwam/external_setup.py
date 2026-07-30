"""Reconstruct and verify external assets pinned by the release manifests."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from dynamicwam.external_assets import (
    WAN_PURPOSES,
    load_external_assets_manifest,
    verify_robotwin_asset_trees,
    verify_wan_assets,
)
from dynamicwam.integrity import (
    sha256_file,
    sha256_relative_files,
    sha256_tree,
)


def _run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def _require_string(mapping: dict[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _verify_file_record(path: Path, record: dict[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    expected_size = record.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError(f"{label}.size_bytes must be a non-negative integer")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    expected_sha256 = _require_string(record, "sha256", label=label)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def prepare_wan(
    *,
    destination: Path,
    manifest_path: Path,
    purposes: Iterable[str],
) -> None:
    manifest = load_external_assets_manifest(manifest_path)
    wan = manifest["wan"]
    if not isinstance(wan, dict) or not isinstance(wan.get("files"), list):
        raise ValueError("WAN manifest entry is invalid")
    requested = tuple(dict.fromkeys(purposes))
    if not requested or any(purpose not in WAN_PURPOSES for purpose in requested):
        raise ValueError(f"WAN purposes must be selected from {sorted(WAN_PURPOSES)}")
    allow_patterns = sorted(
        {
            str(record["path"])
            for record in wan["files"]
            if isinstance(record, dict)
            and isinstance(record.get("required_for"), list)
            and set(requested).intersection(record["required_for"])
        }
    )
    if not allow_patterns:
        raise ValueError(f"WAN manifest has no files for purposes {requested}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "preparing Hugging Face assets requires the project dependencies"
        ) from exc

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=_require_string(wan, "repo_id", label="wan"),
        repo_type=_require_string(wan, "repo_type", label="wan"),
        revision=_require_string(wan, "revision", label="wan"),
        local_dir=destination,
        allow_patterns=allow_patterns,
    )
    for purpose in requested:
        verify_wan_assets(
            root=destination,
            manifest_path=manifest_path,
            purpose=purpose,
        )


def verify_domino_source(*, destination: Path, manifest_path: Path) -> None:
    manifest = load_external_assets_manifest(manifest_path)
    domino = manifest["domino"]
    if not isinstance(domino, dict):
        raise ValueError("DOMINO manifest entry is invalid")
    excluded = domino.get("source_tree_excluded_prefixes")
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) for value in excluded
    ):
        raise ValueError("DOMINO excluded source prefixes must be a string list")
    actual_tree = sha256_tree(
        destination,
        excluded_relative_prefixes=excluded,
    )
    expected_tree = _require_string(domino, "source_tree_sha256", label="domino")
    if actual_tree != expected_tree:
        raise RuntimeError(
            f"DOMINO tree identity mismatch: expected {expected_tree}, got {actual_tree}"
        )
    policy = destination / _require_string(domino, "eval_policy", label="domino")
    actual_policy = sha256_file(policy)
    expected_policy = _require_string(
        domino,
        "eval_policy_sha256",
        label="domino",
    )
    if actual_policy != expected_policy:
        raise RuntimeError(
            "DOMINO evaluator identity mismatch: "
            f"expected {expected_policy}, got {actual_policy}"
        )


def prepare_domino_source(*, destination: Path, manifest_path: Path) -> None:
    manifest = load_external_assets_manifest(manifest_path)
    domino = manifest["domino"]
    if not isinstance(domino, dict):
        raise ValueError("DOMINO manifest entry is invalid")
    patch = manifest_path.parent.parent / _require_string(
        domino,
        "patch",
        label="domino",
    )
    expected_patch = _require_string(domino, "patch_sha256", label="domino")
    if sha256_file(patch) != expected_patch:
        raise RuntimeError(f"DOMINO patch identity mismatch: {patch}")
    if shutil.which("git") is None:
        raise RuntimeError("git is required to reconstruct DOMINO")

    destination = destination.expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"DOMINO destination is not a directory: {destination}"
            )
        verify_domino_source(
            destination=destination,
            manifest_path=manifest_path,
        )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=".domino-bootstrap-",
    ) as temporary_root:
        checkout = Path(temporary_root) / "DOMINO"
        _run("git", "init", "-q", str(checkout))
        _run(
            "git",
            "remote",
            "add",
            "origin",
            _require_string(domino, "upstream_url", label="domino"),
            cwd=checkout,
        )
        _run(
            "git",
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            _require_string(domino, "upstream_base_revision", label="domino"),
            cwd=checkout,
        )
        _run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=checkout)
        _run("git", "apply", "--check", str(patch), cwd=checkout)
        _run("git", "apply", str(patch), cwd=checkout)
        _run("git", "diff", "--check", cwd=checkout)
        verify_domino_source(
            destination=checkout,
            manifest_path=manifest_path,
        )
        checkout.replace(destination)


def _domino_python_runtime_targets(
    *,
    python: Path,
    manifest_path: Path,
) -> list[tuple[dict[str, Any], Path]]:
    manifest = load_external_assets_manifest(manifest_path)
    domino = manifest["domino"]
    if not isinstance(domino, dict) or not isinstance(
        domino.get("evaluated_python_runtime"),
        list,
    ):
        raise ValueError("DOMINO evaluated Python runtime manifest is invalid")
    records = domino["evaluated_python_runtime"]
    distributions = [str(record["distribution"]) for record in records]
    python = python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(
            f"evaluation Python is missing or not executable: {python}"
        )
    probe = "\n".join(
        (
            "import importlib.metadata",
            "import json",
            "import sys",
            "import sysconfig",
            "names = json.loads(sys.argv[1])",
            "print(json.dumps({",
            "    'purelib': sysconfig.get_path('purelib'),",
            "    'platlib': sysconfig.get_path('platlib'),",
            "    'versions': {name: importlib.metadata.version(name) for name in names},",
            "}, sort_keys=True))",
        )
    )
    probe_result = subprocess.run(
        (str(python), "-c", probe, json.dumps(distributions)),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe_result.returncode != 0:
        detail = probe_result.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"exit {probe_result.returncode}"
        raise RuntimeError(
            f"failed to inspect the evaluation Python runtime: {python}: {reason}"
        )
    raw_metadata = probe_result.stdout
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"evaluation Python returned invalid runtime metadata: {python}"
        ) from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("versions"), dict):
        raise RuntimeError(
            f"evaluation Python runtime metadata is incomplete: {python}"
        )
    library_roots = {
        Path(str(metadata[key])).expanduser().resolve()
        for key in ("purelib", "platlib")
        if isinstance(metadata.get(key), str) and metadata[key]
    }
    if not library_roots:
        raise RuntimeError(f"evaluation Python has no package-library roots: {python}")

    targets: list[tuple[dict[str, Any], Path]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("DOMINO evaluated Python runtime record is invalid")
        distribution = str(record["distribution"])
        expected_version = str(record["version"])
        actual_version = metadata["versions"].get(distribution)
        if actual_version != expected_version:
            raise RuntimeError(
                f"{distribution} version mismatch in {python}: "
                f"expected {expected_version}, got {actual_version}"
            )
        relative = Path(str(record["relative_path"]))
        candidates = {
            root / relative for root in library_roots if (root / relative).is_file()
        }
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one installed {distribution} module at {relative}, "
                f"found {sorted(str(path) for path in candidates)}"
            )
        target = candidates.pop()
        if target.is_symlink():
            raise RuntimeError(f"refusing a symlinked DOMINO runtime module: {target}")
        targets.append((record, target))
    return targets


def verify_domino_python_runtime(
    *,
    python: Path,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Verify the exact SAPIEN/MPLib source state used by the evaluator."""

    verified: dict[str, dict[str, Any]] = {}
    for record, target in _domino_python_runtime_targets(
        python=python,
        manifest_path=manifest_path,
    ):
        distribution = str(record["distribution"])
        expected = str(record["evaluated_sha256"])
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"{distribution} evaluated runtime mismatch: "
                f"expected {expected}, got {actual}: {target}"
            )
        verified[f"domino_runtime_{distribution}"] = {
            "path": str(target),
            "kind": "file",
            "version": str(record["version"]),
            "sha256": actual,
            "expected_sha256": expected,
        }
    return verified


def prepare_domino_python_runtime(*, python: Path, manifest_path: Path) -> None:
    """Apply only the exact dependency-source edits retained on the eval host."""

    targets = _domino_python_runtime_targets(
        python=python,
        manifest_path=manifest_path,
    )
    for record, target in targets:
        distribution = str(record["distribution"])
        actual = sha256_file(target)
        evaluated = str(record["evaluated_sha256"])
        if actual == evaluated:
            continue
        pristine = str(record["pristine_sha256"])
        if actual != pristine:
            raise RuntimeError(
                f"{distribution} runtime is neither pristine nor evaluated: "
                f"{actual}: {target}"
            )
        content = target.read_text(encoding="utf-8")
        replacements = record["replacements"]
        if not isinstance(replacements, list):
            raise ValueError(f"{distribution} runtime replacements are invalid")
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise ValueError(f"{distribution} runtime replacement is invalid")
            before = str(replacement["before"])
            after = str(replacement["after"])
            occurrences = content.count(before)
            if occurrences != 1:
                raise RuntimeError(
                    f"{distribution} runtime patch expected one occurrence, "
                    f"found {occurrences}: {before!r}"
                )
            content = content.replace(before, after, 1)
        encoded = content.encode("utf-8")
        temporary = target.with_name(f".{target.name}.dynamicwam.tmp")
        try:
            temporary.write_bytes(encoded)
            temporary.chmod(stat.S_IMODE(target.stat().st_mode))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        patched = sha256_file(target)
        if patched != evaluated:
            raise RuntimeError(
                f"{distribution} runtime patch produced {patched}, "
                f"expected {evaluated}: {target}"
            )
    verify_domino_python_runtime(
        python=python,
        manifest_path=manifest_path,
    )


def _safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    expected_file_count: int,
    expected_size_bytes: int,
) -> None:
    with zipfile.ZipFile(archive) as source:
        normalized_names: set[str] = set()
        file_count = 0
        size_bytes = 0
        for member in source.infolist():
            if "\x00" in member.filename or "\\" in member.filename:
                raise ValueError(f"unsafe ZIP member name: {member.filename!r}")
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"unsafe ZIP member path: {member.filename!r}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symlink is not allowed: {member.filename!r}")
            normalized = relative.as_posix().rstrip("/")
            if normalized in normalized_names:
                raise ValueError(f"duplicate ZIP member path: {member.filename!r}")
            normalized_names.add(normalized)
            if not member.is_dir():
                file_count += 1
                size_bytes += member.file_size
        if file_count != expected_file_count or size_bytes != expected_size_bytes:
            raise RuntimeError(
                f"ZIP contents differ before extraction: expected "
                f"{expected_file_count} files/{expected_size_bytes} bytes, got "
                f"{file_count} files/{size_bytes} bytes"
            )
        source.extractall(destination)


def _verify_extracted_asset(path: Path, record: dict[str, Any]) -> None:
    expected_tree = _require_string(record, "tree_sha256", label="asset")
    actual_tree = sha256_tree(path)
    if actual_tree != expected_tree:
        raise RuntimeError(
            f"extracted asset tree mismatch: expected {expected_tree}, got {actual_tree}"
        )
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    expected_count = record.get("file_count")
    if len(files) != expected_count:
        raise RuntimeError(
            f"extracted asset file count mismatch: expected {expected_count}, "
            f"got {len(files)}"
        )
    actual_size = sum(candidate.stat().st_size for candidate in files)
    expected_size = record.get("extracted_size_bytes")
    if actual_size != expected_size:
        raise RuntimeError(
            f"extracted asset size mismatch: expected {expected_size}, got {actual_size}"
        )


def _link_robotwin_assets(*, asset_root: Path, domino_root: Path) -> None:
    assets_directory = domino_root / "assets"
    assets_directory.mkdir(parents=True, exist_ok=True)
    for name in ("background_texture", "embodiments", "objects"):
        target = (asset_root / name).resolve()
        link = assets_directory / name
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError(f"DOMINO asset link points elsewhere: {link}")
            continue
        if link.exists():
            raise FileExistsError(
                f"refusing to replace an existing DOMINO asset path: {link}"
            )
        relative_target = os.path.relpath(target, start=link.parent.resolve())
        link.symlink_to(relative_target, target_is_directory=True)


def verify_robotwin_asset_links(*, asset_root: Path, domino_root: Path) -> None:
    """Verify that DOMINO resolves each separately versioned asset tree."""

    for name in ("background_texture", "embodiments", "objects"):
        expected = (asset_root / name).resolve()
        link = domino_root / "assets" / name
        if not link.is_symlink() or link.resolve() != expected:
            raise RuntimeError(
                f"DOMINO asset link is missing or points elsewhere: {link}"
            )


def prepare_robotwin_assets(
    *,
    destination: Path,
    domino_root: Path,
    manifest_path: Path,
) -> None:
    manifest = load_external_assets_manifest(manifest_path)
    section = manifest["robotwin_assets"]
    if not isinstance(section, dict) or not isinstance(section.get("archives"), list):
        raise ValueError("RoboTwin asset manifest entry is invalid")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "preparing Hugging Face assets requires the project dependencies"
        ) from exc

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for index, raw_record in enumerate(section["archives"]):
        if not isinstance(raw_record, dict):
            raise ValueError(f"robotwin_assets.archives[{index}] must be an object")
        filename = _require_string(raw_record, "path", label="asset")
        archive = Path(
            hf_hub_download(
                repo_id=_require_string(section, "repo_id", label="robotwin_assets"),
                repo_type=_require_string(
                    section,
                    "repo_type",
                    label="robotwin_assets",
                ),
                revision=_require_string(
                    section,
                    "revision",
                    label="robotwin_assets",
                ),
                filename=filename,
                local_dir=destination,
            )
        )
        _verify_file_record(archive, raw_record, label=f"asset archive {filename}")
        directory_name = _require_string(raw_record, "extracts_to", label="asset")
        target = destination / directory_name
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"asset target is not a directory: {target}")
            _verify_extracted_asset(target, raw_record)
            continue

        with tempfile.TemporaryDirectory(
            dir=destination,
            prefix=f".{directory_name}-extract-",
        ) as temporary_root:
            extraction_root = Path(temporary_root) / "payload"
            extraction_root.mkdir()
            _safe_extract_zip(
                archive,
                extraction_root,
                expected_file_count=int(raw_record["file_count"]),
                expected_size_bytes=int(raw_record["extracted_size_bytes"]),
            )
            top_level = extraction_root / directory_name
            candidate = top_level if top_level.is_dir() else extraction_root
            if candidate == top_level:
                extras = [
                    path.name
                    for path in extraction_root.iterdir()
                    if path.name != directory_name
                ]
                if extras:
                    raise RuntimeError(
                        f"asset archive {filename} has unexpected top-level entries: "
                        f"{sorted(extras)}"
                    )
            _verify_extracted_asset(candidate, raw_record)
            candidate.replace(target)

    verify_robotwin_asset_trees(
        root=destination,
        manifest_path=manifest_path,
    )
    verify_domino_source(
        destination=domino_root,
        manifest_path=manifest_path,
    )
    _link_robotwin_assets(
        asset_root=destination,
        domino_root=domino_root,
    )
    verify_robotwin_asset_links(
        asset_root=destination,
        domino_root=domino_root,
    )


def _git_output(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def verify_curobo_source(*, destination: Path, manifest_path: Path) -> None:
    manifest = load_external_assets_manifest(manifest_path)
    curobo = manifest["curobo"]
    if not isinstance(curobo, dict):
        raise ValueError("CuRobo manifest entry is invalid")
    if not (destination / ".git").is_dir():
        raise FileNotFoundError(f"CuRobo Git checkout is missing: {destination}")
    actual_revision = _git_output("git", "rev-parse", "HEAD", cwd=destination)
    expected_revision = _require_string(curobo, "revision", label="curobo")
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"CuRobo revision mismatch: expected {expected_revision}, "
            f"got {actual_revision}"
        )
    if _git_output(
        "git",
        "status",
        "--porcelain",
        "--untracked-files=no",
        cwd=destination,
    ):
        raise RuntimeError(
            f"CuRobo tracked source has local modifications: {destination}"
        )
    tracked = (
        subprocess.check_output(
            ("git", "ls-files", "-z"),
            cwd=destination,
        )
        .decode("utf-8")
        .split("\0")
    )
    tracked = [value for value in tracked if value]
    actual_tree = sha256_relative_files(destination, tracked)
    expected_tree = _require_string(curobo, "source_tree_sha256", label="curobo")
    if actual_tree != expected_tree:
        raise RuntimeError(
            f"CuRobo source tree mismatch: expected {expected_tree}, got {actual_tree}"
        )


def prepare_curobo_source(*, destination: Path, manifest_path: Path) -> None:
    manifest = load_external_assets_manifest(manifest_path)
    curobo = manifest["curobo"]
    if not isinstance(curobo, dict):
        raise ValueError("CuRobo manifest entry is invalid")
    destination = destination.expanduser().resolve()
    if destination.exists():
        verify_curobo_source(
            destination=destination,
            manifest_path=manifest_path,
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=".curobo-bootstrap-",
    ) as temporary_root:
        checkout = Path(temporary_root) / "curobo"
        _run("git", "init", "-q", str(checkout))
        _run(
            "git",
            "remote",
            "add",
            "origin",
            _require_string(curobo, "upstream_url", label="curobo"),
            cwd=checkout,
        )
        _run(
            "git",
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            _require_string(curobo, "revision", label="curobo"),
            cwd=checkout,
        )
        _run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=checkout)
        verify_curobo_source(
            destination=checkout,
            manifest_path=manifest_path,
        )
        checkout.replace(destination)


def inspect_curobo_runtime(
    *,
    destination: Path,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Record every CUDA extension required by the pinned CuRobo build."""

    manifest = load_external_assets_manifest(manifest_path)
    curobo = manifest["curobo"]
    if not isinstance(curobo, dict) or not isinstance(
        curobo.get("evaluated_extensions"),
        list,
    ):
        raise ValueError("CuRobo manifest entry is invalid")
    inspected: dict[str, dict[str, Any]] = {}
    for record in curobo["evaluated_extensions"]:
        if not isinstance(record, dict):
            raise ValueError("CuRobo extension record is invalid")
        module = _require_string(record, "module", label="curobo extension")
        relative = Path(
            _require_string(record, "path", label=f"curobo extension {module}")
        )
        extension = destination / relative
        if not extension.is_file():
            raise FileNotFoundError(f"CuRobo extension is missing: {extension}")
        inspected[f"curobo_extension_{module}"] = {
            "path": str(extension),
            "kind": "file",
            "module": module,
            "size_bytes": extension.stat().st_size,
            "sha256": sha256_file(extension),
        }
    return inspected


def verify_curobo_runtime(
    *,
    destination: Path,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    manifest = load_external_assets_manifest(manifest_path)
    curobo = manifest["curobo"]
    if not isinstance(curobo, dict) or not isinstance(
        curobo.get("evaluated_extensions"),
        list,
    ):
        raise ValueError("CuRobo manifest entry is invalid")
    inspected = inspect_curobo_runtime(
        destination=destination,
        manifest_path=manifest_path,
    )
    for record in curobo["evaluated_extensions"]:
        if not isinstance(record, dict):
            raise ValueError("CuRobo extension record is invalid")
        module = _require_string(record, "module", label="curobo extension")
        actual = inspected[f"curobo_extension_{module}"]
        expected_size = record.get("size_bytes")
        if actual["size_bytes"] != expected_size:
            raise RuntimeError(
                f"CuRobo {module} size mismatch: expected {expected_size}, "
                f"got {actual['size_bytes']}"
            )
        expected_sha256 = _require_string(
            record,
            "sha256",
            label=f"curobo extension {module}",
        )
        if actual["sha256"] != expected_sha256:
            raise RuntimeError(
                f"CuRobo {module} SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual['sha256']}"
            )
        actual["expected_sha256"] = expected_sha256
    return inspected
