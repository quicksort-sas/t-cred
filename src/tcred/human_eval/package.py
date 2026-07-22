from __future__ import annotations

import hashlib
from pathlib import Path


def artifact_hashes(*, root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def package_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
    return digest.hexdigest()


def verify_manifest_artifacts(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    require_hashes: bool,
) -> None:
    raw_hashes = manifest.get("artifact_sha256")
    if not isinstance(raw_hashes, dict):
        if require_hashes:
            raise ValueError(f"Frozen assignment manifest has no artifact hashes: {manifest_path}")
        return
    package_root = manifest_path.parent.resolve()
    for relative, expected in raw_hashes.items():
        path = (package_root / str(relative)).resolve()
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise ValueError(f"Manifest artifact escapes package root: {relative}") from exc
        if not path.exists():
            raise ValueError(f"Frozen human-evaluation artifact is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected):
            raise ValueError(f"Frozen human-evaluation artifact was modified: {path}")
