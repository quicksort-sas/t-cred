from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson


def validate_training_export(
    export_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a downloaded training export without changing remote provenance files.

    The first seed-42 export used basenames in ``run_manifest.json`` for files below the
    tokenizer directory. Validation recognizes that one known defect only when a basename
    resolves uniquely; bytes and hashes still have to match exactly.
    """

    root = export_root.resolve()
    final_dir = root / "final_model"
    manifest_path = root / "run_manifest.json"
    archive_path = root / "remote-export.tar.gz"
    sidecar_path = root / "remote-export.sha256"
    required = (final_dir, manifest_path, archive_path, sidecar_path)
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError(f"Training export is incomplete: {missing_required}")

    manifest = _read_object(manifest_path)
    described = manifest.get("final_model")
    if not isinstance(described, dict) or not described:
        raise ValueError("run_manifest.json has no final_model file inventory")

    actual_files = [path for path in sorted(final_dir.rglob("*")) if path.is_file()]
    by_relative = {path.relative_to(final_dir).as_posix(): path for path in actual_files}
    by_basename: dict[str, list[Path]] = {}
    for path in actual_files:
        by_basename.setdefault(path.name, []).append(path)

    errors: list[dict[str, Any]] = []
    resolutions: dict[str, str] = {}
    legacy_flattened: list[str] = []
    resolved_paths: set[str] = set()
    for manifest_name, raw_record in sorted(described.items()):
        if not isinstance(manifest_name, str) or not isinstance(raw_record, dict):
            errors.append({"kind": "invalid_manifest_entry", "entry": str(manifest_name)})
            continue
        path = by_relative.get(manifest_name)
        if path is None and "/" not in manifest_name:
            candidates = by_basename.get(manifest_name, [])
            if len(candidates) == 1:
                path = candidates[0]
                legacy_flattened.append(manifest_name)
            elif len(candidates) > 1:
                errors.append(
                    {
                        "kind": "ambiguous_legacy_basename",
                        "entry": manifest_name,
                        "candidates": [
                            candidate.relative_to(final_dir).as_posix()
                            for candidate in candidates
                        ],
                    }
                )
                continue
        if path is None:
            errors.append({"kind": "missing_file", "entry": manifest_name})
            continue
        relative = path.relative_to(final_dir).as_posix()
        resolutions[manifest_name] = relative
        resolved_paths.add(relative)
        actual_hash = file_sha256(path)
        actual_bytes = path.stat().st_size
        if raw_record.get("sha256") != actual_hash:
            errors.append(
                {
                    "kind": "hash_mismatch",
                    "entry": manifest_name,
                    "expected": raw_record.get("sha256"),
                    "actual": actual_hash,
                }
            )
        if raw_record.get("bytes") != actual_bytes:
            errors.append(
                {
                    "kind": "size_mismatch",
                    "entry": manifest_name,
                    "expected": raw_record.get("bytes"),
                    "actual": actual_bytes,
                }
            )

    unexpected = sorted(set(by_relative) - resolved_paths)
    if unexpected:
        errors.append({"kind": "unmanifested_final_model_files", "paths": unexpected})

    expected_archive_hash = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
    actual_archive_hash = file_sha256(archive_path)
    if expected_archive_hash != actual_archive_hash:
        errors.append(
            {
                "kind": "archive_hash_mismatch",
                "expected": expected_archive_hash,
                "actual": actual_archive_hash,
            }
        )

    model_config = _read_object(final_dir / "model_config.json")
    calibration = _read_object(final_dir / "calibration.json")
    identity_checks = {
        "steps_complete": manifest.get("completed_steps") == manifest.get("total_planned_steps"),
        "model_step_matches_manifest": (
            model_config.get("global_step") == manifest.get("completed_steps")
        ),
        "model_planned_steps_match_manifest": (
            model_config.get("total_planned_steps") == manifest.get("total_planned_steps")
        ),
        "training_config_hash_matches": (
            model_config.get("training_config_hash") == manifest.get("training_config_hash")
        ),
        "model_schema_supported": model_config.get("schema_version") == "tcred-sl-model-v1",
        "calibration_schema_supported": (
            calibration.get("schema_version") == "tcred-sl-temperature-calibration-v1"
        ),
    }
    for name, passed in identity_checks.items():
        if not passed:
            errors.append({"kind": "identity_check_failed", "check": name})

    warnings: list[dict[str, Any]] = []
    if legacy_flattened:
        warnings.append(
            {
                "kind": "legacy_flattened_manifest_paths",
                "detail": (
                    "The remote exporter recorded tokenizer basenames instead of relative paths. "
                    "Each basename resolved uniquely and its bytes/hash matched."
                ),
                "entries": sorted(legacy_flattened),
            }
        )
    report = {
        "schema_version": "tcred-sl-export-validation-v1",
        "status": "passed" if not errors else "failed",
        "export_root": root.as_posix(),
        "archive": {
            "path": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": actual_archive_hash,
            "sidecar_match": expected_archive_hash == actual_archive_hash,
        },
        "run": {
            "completed_steps": manifest.get("completed_steps"),
            "total_planned_steps": manifest.get("total_planned_steps"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "training_config_hash": manifest.get("training_config_hash"),
        },
        "model": {
            "weight_sha256": file_sha256(final_dir / "model.safetensors"),
            "weight_bytes": (final_dir / "model.safetensors").stat().st_size,
            "files": len(actual_files),
            "manifest_path_resolutions": resolutions,
        },
        "identity_checks": identity_checks,
        "warnings": warnings,
        "errors": errors,
    }
    if output_path is not None:
        _write_json(output_path, report)
    if errors:
        raise ValueError(f"Training export validation failed with {len(errors)} error(s)")
    return report


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
