from __future__ import annotations

from pathlib import Path

import orjson

from tcred.dataset.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from tcred.human_eval.assignments import (
    DEFAULT_ASSIGNMENT_SEED,
    assign_annotation_units,
    assignment_manifest_metadata,
)
from tcred.human_eval.package import (
    artifact_hashes,
    package_sha256,
    verify_manifest_artifacts,
)
from tcred.human_eval.protocol import PROTOCOL_VERSION, annotation_fields_manifest


def reassign_human_eval_package(
    *,
    package_dir: Path,
    annotators: int = 36,
    assignments_per_annotator: int = 20,
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED,
) -> dict[str, Path]:
    """Rebuild assignment files without resampling or rewriting evaluation units."""
    manifest_path = package_dir / "assignment_manifest.json"
    units_path = package_dir / "human_eval_units.jsonl"
    key_path = package_dir / "human_eval_key.jsonl"
    for path in (manifest_path, units_path, key_path):
        if not path.exists():
            raise FileNotFoundError(f"Human-evaluation package artifact is missing: {path}")

    manifest = orjson.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError(f"Assignment manifest must contain an object: {manifest_path}")
    verify_manifest_artifacts(
        manifest_path=manifest_path,
        manifest=manifest,
        require_hashes=True,
    )
    _require_no_production_labels(package_dir / "labels_raw")

    units_digest_before = artifact_hashes(root=package_dir, paths=[units_path])
    keys_digest_before = artifact_hashes(root=package_dir, paths=[key_path])
    unit_rows = read_jsonl(units_path)
    key_rows = read_jsonl(key_path)
    plan = assign_annotation_units(
        unit_rows=unit_rows,
        key_rows=key_rows,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        seed=assignment_seed,
    )

    assignments_dir = package_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {
        assignments_dir / f"{annotator_id}.jsonl" for annotator_id in plan.assignments
    }
    for annotator_id, rows in plan.assignments.items():
        write_jsonl_atomic(assignments_dir / f"{annotator_id}.jsonl", rows)
    for stale_path in sorted(assignments_dir.glob("annotator_*.jsonl")):
        if stale_path not in expected_paths:
            stale_path.unlink()

    assignment_paths = sorted(expected_paths)
    hashes = artifact_hashes(
        root=package_dir,
        paths=[units_path, key_path, *assignment_paths],
    )
    previous_package_sha256 = str(manifest.get("package_sha256", ""))
    manifest.update(
        {
            "annotators": annotators,
            "assignments_per_annotator": assignments_per_annotator,
            "total_assignments": annotators * assignments_per_annotator,
            "annotation_fields": annotation_fields_manifest(),
            "annotation_protocol_version": PROTOCOL_VERSION,
            **assignment_manifest_metadata(plan, key_rows=key_rows),
            "annotator_files": [str(path) for path in assignment_paths],
            "artifact_sha256": hashes,
            "package_sha256": package_sha256(hashes),
            "assignment_rebuild": {
                "unit_selection_preserved": True,
                "unit_and_key_files_rewritten": False,
                "previous_package_sha256": previous_package_sha256,
            },
            "blind_annotation_note": (
                "Assignment files contain a plain-text reference answer for the trusted server. "
                "The annotation API withholds it until evidence-stage judgments are locked and "
                "does not expose source, system, model, controlled labels, or computed verdicts."
            ),
        }
    )
    write_json_atomic(manifest_path, manifest)

    if units_digest_before != artifact_hashes(root=package_dir, paths=[units_path]):
        raise RuntimeError("Human-evaluation units changed during assignment-only rebuild")
    if keys_digest_before != artifact_hashes(root=package_dir, paths=[key_path]):
        raise RuntimeError("Human-evaluation keys changed during assignment-only rebuild")
    verify_manifest_artifacts(
        manifest_path=manifest_path,
        manifest=manifest,
        require_hashes=True,
    )
    return {
        "human_eval_units": units_path,
        "human_eval_key": key_path,
        "assignments": assignments_dir,
        "assignment_manifest": manifest_path,
    }


def _require_no_production_labels(labels_dir: Path) -> None:
    if not labels_dir.exists():
        return
    production_labels = [
        path for path in labels_dir.glob("annotator_*.jsonl") if path.stem != "annotator_00"
    ]
    if production_labels:
        raise RuntimeError(
            "Cannot rebuild assignments after production labels exist; preserve the frozen "
            f"assignment-to-annotator mapping. Found {len(production_labels)} label files."
        )
