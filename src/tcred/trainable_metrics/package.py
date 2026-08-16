from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import orjson

from tcred.trainable_metrics.config import (
    DataBuildConfig,
    TrainingConfig,
    canonical_config_hash,
    load_yaml_model,
)
from tcred.trainable_metrics.source_io import file_sha256

BUNDLE_SCHEMA_VERSION = "tcred-sl-private-gpu-bundle-v1"
_CORPUS_METADATA = ("manifest.json", "license_ledger.json", "near_duplicate_audit.json")
_CONFIG_FILES = ("data.yaml", "training.a100-80gb.yaml")
_RUNTIME_FILES = (
    "pyproject.toml",
    "uv.lock",
    "src/tcred/__init__.py",
)
_DENIED_NAMES = {".env", "pytorch_model.bin"}
_DENIED_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: PurePosixPath


def create_private_gpu_bundle(
    *,
    workspace_root: Path,
    output_path: Path,
    corpus_dir: Path,
    tokenized_dir: Path,
    backbone_root: Path,
    readiness_path: Path,
    config_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a deterministic, allowlisted bundle for a private GPU workspace."""
    workspace_root = workspace_root.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"GPU bundle already exists: {output_path}")
    readiness = _read_object(readiness_path)
    if readiness.get("status") != "ready":
        raise RuntimeError("GPU bundle creation requires a passing readiness report")
    if readiness.get("schema_version") != "tcred-sl-gpu-readiness-v2":
        raise RuntimeError("GPU bundle creation requires the artifact-bound readiness schema")
    _verify_readiness_inputs(
        readiness,
        corpus_dir=corpus_dir,
        tokenized_dir=tokenized_dir,
        backbone_root=backbone_root,
        config_dir=config_dir,
    )

    files = _collect_files(
        workspace_root=workspace_root,
        corpus_dir=corpus_dir,
        tokenized_dir=tokenized_dir,
        backbone_root=backbone_root,
        readiness_path=readiness_path,
        config_dir=config_dir,
    )
    payload = [
        {
            "path": str(item.archive_path),
            "bytes": item.source.stat().st_size,
            "sha256": file_sha256(item.source),
        }
        for item in files
    ]
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "visibility": "private_training_workspace_only",
        "warning": (
            "Tokenized examples can be decoded back to source text. Do not publish this bundle. "
            "Use it only in an access-controlled research workspace under every upstream license."
        ),
        "readiness": {
            "data_config_hash": readiness.get("data_config_hash"),
            "training_config_hash": readiness.get("training_config_hash"),
            "status": readiness.get("status"),
        },
        "excluded_by_design": [
            "raw source corpora",
            "canonical record text",
            "unsafe pickle model weights",
            "environment files and credentials",
            "human annotations and personally identifying data",
        ],
        "files": payload,
    }
    manifest_bytes = orjson.dumps(
        manifest,
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        _write_archive(temporary, files=files, manifest_bytes=manifest_bytes)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    verification = verify_private_gpu_bundle(output_path)
    archive_hash = file_sha256(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {output_path.name}\n", encoding="ascii")
    return {
        **manifest,
        "archive": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": archive_hash,
            "checksum_path": str(checksum_path),
        },
        "verification": verification,
    }


def verify_private_gpu_bundle(path: Path) -> dict[str, Any]:
    """Verify archive paths, the embedded allowlist, sizes, and SHA-256 digests."""
    if not path.is_file():
        raise FileNotFoundError(f"GPU bundle does not exist: {path}")
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("GPU bundle contains duplicate archive paths")
        for member in members:
            _validate_archive_name(member.name)
            if not member.isfile():
                raise ValueError(f"GPU bundle contains a non-file member: {member.name}")
        manifest_member = archive.getmember("GPU_BUNDLE_MANIFEST.json")
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None:
            raise ValueError("GPU bundle manifest cannot be read")
        manifest = json.loads(manifest_handle.read())
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("GPU bundle has an unsupported manifest schema")
        expected = {str(row["path"]): row for row in manifest.get("files", [])}
        actual_names = set(names) - {"GPU_BUNDLE_MANIFEST.json"}
        if actual_names != set(expected):
            raise ValueError("GPU bundle members do not match the embedded allowlist")
        for name in sorted(actual_names):
            handle = archive.extractfile(name)
            if handle is None:
                raise ValueError(f"GPU bundle member cannot be read: {name}")
            digest = hashlib.sha256()
            size = 0
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            if size != int(expected[name]["bytes"]):
                raise ValueError(f"GPU bundle size mismatch: {name}")
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"GPU bundle checksum mismatch: {name}")
    return {
        "status": "passed",
        "payload_files": len(expected),
        "payload_bytes": sum(int(row["bytes"]) for row in expected.values()),
    }


def _collect_files(
    *,
    workspace_root: Path,
    corpus_dir: Path,
    tokenized_dir: Path,
    backbone_root: Path,
    readiness_path: Path,
    config_dir: Path,
) -> list[BundleFile]:
    candidates: list[Path] = []
    candidates.extend(corpus_dir / name for name in _CORPUS_METADATA)
    candidates.append(readiness_path)
    candidates.extend(config_dir / name for name in _CONFIG_FILES)
    candidates.extend(workspace_root / name for name in _RUNTIME_FILES)
    candidates.extend(sorted((workspace_root / "src/tcred/trainable_metrics").glob("*.py")))
    candidates.append(backbone_root / "manifest.json")
    candidates.extend(sorted((backbone_root / "safe_checkpoint").glob("*")))
    candidates.extend(sorted(tokenized_dir.rglob("*")))

    files: list[BundleFile] = []
    seen: set[PurePosixPath] = set()
    for candidate in candidates:
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise FileNotFoundError(f"Required GPU bundle artifact is missing: {candidate}")
        if candidate.is_symlink():
            raise ValueError(f"GPU bundles do not accept symbolic links: {candidate}")
        _validate_source_name(candidate)
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError(f"GPU bundle artifact is outside the workspace: {candidate}") from exc
        archive_path = PurePosixPath(relative.as_posix())
        _validate_archive_name(str(archive_path))
        if archive_path in seen:
            continue
        seen.add(archive_path)
        files.append(BundleFile(source=resolved, archive_path=archive_path))
    return sorted(files, key=lambda item: str(item.archive_path))


def _write_archive(
    path: Path,
    *,
    files: list[BundleFile],
    manifest_bytes: bytes,
) -> None:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        _add_bytes(archive, "GPU_BUNDLE_MANIFEST.json", manifest_bytes)
        for item in files:
            _add_file(archive, item)


def _add_file(archive: tarfile.TarFile, item: BundleFile) -> None:
    info = _tar_info(str(item.archive_path), size=item.source.stat().st_size)
    with item.source.open("rb") as handle:
        archive.addfile(info, handle)


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    archive.addfile(_tar_info(name, size=len(value)), io.BytesIO(value))


def _tar_info(name: str, *, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _validate_source_name(path: Path) -> None:
    lowered = path.name.lower()
    if lowered in _DENIED_NAMES or path.suffix.lower() in _DENIED_SUFFIXES:
        raise ValueError(f"Sensitive or unsafe artifact cannot enter GPU bundle: {path}")


def _validate_archive_name(name: str) -> None:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe GPU bundle archive path: {name}")
    if "\\" in name:
        raise ValueError(f"Non-portable GPU bundle archive path: {name}")


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root is not an object: {path}")
    return value


def _verify_readiness_inputs(
    readiness: dict[str, Any],
    *,
    corpus_dir: Path,
    tokenized_dir: Path,
    backbone_root: Path,
    config_dir: Path,
) -> None:
    expected_artifacts = readiness.get("input_artifacts")
    if not isinstance(expected_artifacts, dict):
        raise RuntimeError("Readiness report does not bind its input artifacts")
    actual_artifacts = {
        "corpus_manifest_sha256": file_sha256(corpus_dir / "manifest.json"),
        "tokenized_manifest_sha256": file_sha256(tokenized_dir / "manifest.json"),
        "near_duplicate_audit_sha256": file_sha256(
            corpus_dir / "near_duplicate_audit.json"
        ),
        "backbone_manifest_sha256": file_sha256(backbone_root / "manifest.json"),
    }
    mismatches = [
        name
        for name, digest in actual_artifacts.items()
        if expected_artifacts.get(name) != digest
    ]
    data_config = load_yaml_model(config_dir / "data.yaml", DataBuildConfig)
    training_config = load_yaml_model(
        config_dir / "training.a100-80gb.yaml", TrainingConfig
    )
    if canonical_config_hash(data_config) != readiness.get("data_config_hash"):
        mismatches.append("data_config_hash")
    if canonical_config_hash(training_config) != readiness.get("training_config_hash"):
        mismatches.append("training_config_hash")
    if mismatches:
        raise RuntimeError(
            "GPU bundle inputs changed after readiness validation: "
            + ", ".join(sorted(mismatches))
        )
