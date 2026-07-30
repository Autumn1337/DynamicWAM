"""Deterministic integrity helpers for source and artifact identities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

DOMINO_RUNTIME_ASSET_PREFIXES = (
    "assets/background_texture",
    "assets/embodiments",
    "assets/objects",
)
DOMINO_SOURCE_SHA256_BY_COMMIT = {
    "9b1f34129323957baff35bcbc866fceca7e02779": (
        "54dd0280e44ea0539274365b7feaeba24e36b7629e8679c2999476addd9c314c"
    ),
}
DOMINO_EVAL_POLICY_SHA256_BY_COMMIT = {
    "9b1f34129323957baff35bcbc866fceca7e02779": (
        "fec0002fe1e30833b225d3559b09b583a9a4f7a22c29bc54a3817eb70d1aea38"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(
    path: Path,
    *,
    excluded_relative_paths: Iterable[str] = (),
    excluded_relative_prefixes: Iterable[str] = (),
) -> str:
    excluded = {
        _normalized_relative_path(value, label="excluded path")
        for value in excluded_relative_paths
    }
    excluded_prefixes = tuple(
        _normalized_relative_path(value, label="excluded prefix")
        for value in excluded_relative_prefixes
    )
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not _relative_path_is_excluded(
            candidate.relative_to(path).as_posix(),
            excluded_paths=excluded,
            excluded_prefixes=excluded_prefixes,
        )
        and ".git" not in candidate.parts
        and "__pycache__" not in candidate.parts
        and candidate.suffix not in {".pyc", ".pyo"}
    )
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def sha256_relative_files(path: Path, relative_paths: Iterable[str]) -> str:
    """Hash an explicit relative file set with the tree identity format."""

    normalized = sorted(
        {
            _normalized_relative_path(value, label="relative file")
            for value in relative_paths
        }
    )
    digest = hashlib.sha256()
    for relative_path in normalized:
        candidate = path / relative_path
        if not candidate.is_file():
            raise FileNotFoundError(f"tree identity file is missing: {candidate}")
        encoded = relative_path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def _normalized_relative_path(value: str, *, label: str) -> str:
    candidate = Path(value)
    normalized = candidate.as_posix().rstrip("/")
    if (
        not normalized
        or candidate.is_absolute()
        or normalized == "."
        or ".." in candidate.parts
    ):
        raise ValueError(f"{label} must be a non-empty relative path: {value!r}")
    return normalized


def _relative_path_is_excluded(
    relative_path: str,
    *,
    excluded_paths: set[str],
    excluded_prefixes: tuple[str, ...],
) -> bool:
    if relative_path in excluded_paths:
        return True
    return any(
        relative_path == prefix or relative_path.startswith(f"{prefix}/")
        for prefix in excluded_prefixes
    )


def domino_source_tree_sha256(path: Path) -> str:
    """Hash DOMINO code while excluding separately versioned runtime assets."""

    return sha256_tree(
        path,
        excluded_relative_prefixes=DOMINO_RUNTIME_ASSET_PREFIXES,
    )


def domino_source_sha256(commit: str) -> str:
    try:
        return DOMINO_SOURCE_SHA256_BY_COMMIT[commit]
    except KeyError as exc:
        raise ValueError(f"unsupported DOMINO source commit: {commit}") from exc


def domino_eval_policy_sha256(commit: str) -> str:
    try:
        return DOMINO_EVAL_POLICY_SHA256_BY_COMMIT[commit]
    except KeyError as exc:
        raise ValueError(f"unsupported DOMINO evaluator commit: {commit}") from exc
