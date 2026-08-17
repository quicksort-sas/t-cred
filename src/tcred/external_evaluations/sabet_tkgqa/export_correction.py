from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from tcred.external_evaluations.sabet_tkgqa.runner import (
    DATASETS,
    MODELS,
    _output_identity,
    _verify_instrumented_source,
    _verify_predictions,
    _write_json,
)

_TEST_BLOCK = re.compile(
    r"Split test\s+.*?Hits at 1:\s*([0-9.]+).*?Hits at 10:\s*([0-9.]+)",
    re.DOTALL,
)
_GOLD_FIELDS = {"schema_version", "gold_answer_ids", "gold_answer_labels"}


def promote_schema_1_1_export(
    *,
    run_dir: Path,
    corrected_path: Path,
    instrumented_source_root: Path,
) -> dict[str, object]:
    """Promote a v4 gold-encoding re-export while preserving the v3 file and audit trail."""

    run_dir = run_dir.resolve()
    corrected_path = corrected_path.resolve()
    if corrected_path.parent != run_dir:
        raise ValueError("Corrected export must be inside the run directory")
    canonical_path = run_dir / "predictions.jsonl"
    backup_path = run_dir / "predictions.schema-1.0.jsonl"
    if backup_path.exists():
        raise FileExistsError(f"Superseded export already exists: {backup_path}")
    if not canonical_path.is_file() or not corrected_path.is_file():
        raise FileNotFoundError("Both canonical and corrected prediction exports are required")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only a completed run can receive the audited export correction")
    dataset = str(manifest.get("dataset"))
    model = str(manifest.get("model"))
    run_id = str(manifest.get("run_id"))
    seed = manifest.get("seed")
    if dataset not in DATASETS or model not in MODELS:
        raise ValueError("Run manifest names an unknown dataset or model")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Run manifest has an invalid seed")

    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, dict):
        raise ValueError("Run manifest has no recorded outputs")
    recorded_prediction = recorded_outputs.get("predictions")
    if not isinstance(recorded_prediction, dict):
        raise ValueError("Run manifest has no recorded prediction identity")
    original_sha256 = _sha256(canonical_path)
    if recorded_prediction.get("sha256") != original_sha256:
        raise ValueError("Schema-1.0 prediction file changed after the run")

    comparison = compare_prediction_exports(canonical_path, corrected_path)
    if comparison["non_gold_difference_count"] != 0:
        raise ValueError("Corrected export changes predictions or non-gold metadata")
    if comparison["gold_rows_changed"] <= 0:
        raise ValueError("Corrected export does not contain the expected gold encoding change")
    source_identity = _verify_instrumented_source(instrumented_source_root.resolve())
    verification = _verify_predictions(
        corrected_path,
        expected_count=DATASETS[dataset].expected_test_examples,
        expected_run_id=run_id,
        expected_dataset=dataset,
        expected_model=MODELS[model].upstream_model,
        expected_variant=MODELS[model].variant,
        expected_seed=seed,
    )
    logged_hits = _final_logged_hits(Path(str(recorded_outputs["training_log"]["path"])))
    if round(float(verification["native_hits_at_1"]), 3) != logged_hits[0]:
        raise ValueError("Corrected identity H@1 does not match the upstream evaluator")
    if round(float(verification["native_hits_at_10"]), 3) != logged_hits[1]:
        raise ValueError("Corrected identity H@10 does not match the upstream evaluator")

    corrected_sha256 = _sha256(corrected_path)
    canonical_path.replace(backup_path)
    try:
        corrected_path.replace(canonical_path)
    except BaseException:
        backup_path.replace(canonical_path)
        raise

    manifest.setdefault("superseded_outputs", {})["predictions_schema_1_0"] = {
        **recorded_prediction,
        "path": str(backup_path),
        "superseded_reason": "row-level answer_type cannot encode mixed entity/time gold sets",
    }
    manifest["outputs"] = _output_identity(run_dir, dataset, run_id)
    manifest["prediction_verification"] = verification
    manifest["export_correction"] = {
        "corrected_utc": datetime.now(UTC).isoformat(),
        "from_schema": "1.0",
        "to_schema": "1.1",
        "original_sha256": original_sha256,
        "corrected_sha256": corrected_sha256,
        "gold_rows_changed": comparison["gold_rows_changed"],
        "non_gold_difference_count": comparison["non_gold_difference_count"],
        "predicted_rankings_and_scores_unchanged": True,
        "upstream_logged_hits_at_1": logged_hits[0],
        "upstream_logged_hits_at_10": logged_hits[1],
        "instrumented_source_identity": source_identity,
        "correction_utility_sha256": _sha256(Path(__file__)),
    }
    manifest["output_verification"] = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "verifier_file_sha256": _sha256(
            Path(__file__).with_name("runner.py")
        ),
        "mode": "audited_schema_1_1_reexport",
    }
    _write_json(manifest_path, manifest)
    return manifest


def compare_prediction_exports(original: Path, corrected: Path) -> dict[str, int]:
    line_count = 0
    gold_rows_changed = 0
    non_gold_difference_count = 0
    with original.open(encoding="utf-8") as left, corrected.open(encoding="utf-8") as right:
        while True:
            left_line = left.readline()
            right_line = right.readline()
            if not left_line and not right_line:
                break
            if not left_line or not right_line:
                raise ValueError("Original and corrected exports have different line counts")
            line_count += 1
            left_row = json.loads(left_line)
            right_row = json.loads(right_line)
            if left_row.get("schema_version") != "1.0":
                raise ValueError(f"Original line {line_count} is not schema 1.0")
            if right_row.get("schema_version") != "1.1":
                raise ValueError(f"Corrected line {line_count} is not schema 1.1")
            if any(
                left_row.get(field) != right_row.get(field)
                for field in ("gold_answer_ids", "gold_answer_labels")
            ):
                gold_rows_changed += 1
            left_non_gold = {
                key: value for key, value in left_row.items() if key not in _GOLD_FIELDS
            }
            right_non_gold = {
                key: value for key, value in right_row.items() if key not in _GOLD_FIELDS
            }
            if left_non_gold != right_non_gold:
                non_gold_difference_count += 1
    return {
        "line_count": line_count,
        "gold_rows_changed": gold_rows_changed,
        "non_gold_difference_count": non_gold_difference_count,
    }


def _final_logged_hits(path: Path) -> tuple[float, float]:
    matches = _TEST_BLOCK.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError("Training log has no complete test Hits block")
    hit_at_1, hit_at_10 = matches[-1]
    return float(hit_at_1), float(hit_at_10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an audited SABET schema-1.1 re-export")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corrected-path", type=Path, required=True)
    parser.add_argument("--instrumented-source-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = promote_schema_1_1_export(
        run_dir=args.run_dir,
        corrected_path=args.corrected_path,
        instrumented_source_root=args.instrumented_source_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
