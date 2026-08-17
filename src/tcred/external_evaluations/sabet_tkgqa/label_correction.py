from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tcred.external_evaluations.sabet_tkgqa.export_correction import _final_logged_hits
from tcred.external_evaluations.sabet_tkgqa.runner import (
    DATASETS,
    MODELS,
    _output_identity,
    _verify_predictions,
    _write_json,
)


def promote_entity_id_label_fallback(*, run_dir: Path) -> dict[str, object]:
    """Repair only blank exported labels using their unchanged namespaced answer IDs."""

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    canonical_path = run_dir / "predictions.jsonl"
    backup_path = run_dir / "predictions.empty-labels.jsonl"
    corrected_path = run_dir / "predictions.label-fallback-corrected.jsonl"
    if backup_path.exists() or corrected_path.exists():
        raise FileExistsError("A label-correction artifact already exists in this run")
    if not manifest_path.is_file() or not canonical_path.is_file():
        raise FileNotFoundError("Run manifest and canonical predictions are required")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "invalid_outputs" or manifest.get("return_code") != 0:
        raise ValueError("Only a successful process rejected for invalid outputs can be repaired")
    dataset = str(manifest.get("dataset"))
    model = str(manifest.get("model"))
    run_id = str(manifest.get("run_id"))
    seed = manifest.get("seed")
    if dataset not in DATASETS or model not in MODELS:
        raise ValueError("Run manifest names an unknown dataset or model")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Run manifest has an invalid seed")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("predictions"), dict):
        raise ValueError("Run manifest has no recorded prediction identity")
    recorded_prediction = outputs["predictions"]
    original_sha256 = _sha256(canonical_path)
    if recorded_prediction.get("sha256") != original_sha256:
        raise ValueError("Prediction file changed after the rejected run")

    correction = _write_corrected_export(canonical_path, corrected_path)
    verification = _verify_predictions(
        corrected_path,
        expected_count=DATASETS[dataset].expected_test_examples,
        expected_run_id=run_id,
        expected_dataset=dataset,
        expected_model=MODELS[model].upstream_model,
        expected_variant=MODELS[model].variant,
        expected_seed=seed,
    )
    training_log = outputs.get("training_log")
    if not isinstance(training_log, dict) or not isinstance(training_log.get("path"), str):
        raise ValueError("Run manifest has no training-log identity")
    logged_h1, logged_h10 = _final_logged_hits(Path(training_log["path"]))
    if round(float(verification["native_hits_at_1"]), 3) != logged_h1:
        raise ValueError("Label-only correction does not match upstream Hits@1")
    if round(float(verification["native_hits_at_10"]), 3) != logged_h10:
        raise ValueError("Label-only correction does not match upstream Hits@10")

    corrected_sha256 = _sha256(corrected_path)
    canonical_path.replace(backup_path)
    try:
        corrected_path.replace(canonical_path)
    except BaseException:
        backup_path.replace(canonical_path)
        raise

    manifest["status_before_label_correction"] = "invalid_outputs"
    manifest.setdefault("superseded_outputs", {})["predictions_empty_labels"] = {
        **recorded_prediction,
        "path": str(backup_path),
        "superseded_reason": "released entity label map contains blank English labels",
    }
    manifest["outputs"] = _output_identity(run_dir, dataset, run_id)
    manifest["prediction_verification"] = verification
    manifest["label_correction"] = {
        "corrected_utc": datetime.now(UTC).isoformat(),
        "policy": "blank label -> unchanged namespaced answer-ID payload",
        "original_sha256": original_sha256,
        "corrected_sha256": corrected_sha256,
        **correction,
        "answer_ids_unchanged": True,
        "rank_order_unchanged": True,
        "prediction_scores_unchanged": True,
        "nonblank_labels_unchanged": True,
        "upstream_logged_hits_at_1": logged_h1,
        "upstream_logged_hits_at_10": logged_h10,
        "correction_utility_sha256": _sha256(Path(__file__)),
    }
    manifest["output_verification"] = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "verifier_file_sha256": _sha256(Path(__file__).with_name("runner.py")),
        "mode": "audited_empty_label_fallback",
    }
    manifest["status"] = "completed"
    _write_json(manifest_path, manifest)
    return manifest


def _write_corrected_export(source: Path, target: Path) -> dict[str, object]:
    replacements_by_field: Counter[str] = Counter()
    replacements_by_answer_id: Counter[str] = Counter()
    replacements_by_rank: Counter[int] = Counter()
    changed_rows = 0
    line_count = 0
    try:
        with source.open(encoding="utf-8") as input_stream, target.open(
            "x", encoding="utf-8", newline="\n"
        ) as output_stream:
            for line_count, line in enumerate(input_stream, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Prediction line {line_count} is not an object")
                original = json.loads(line)
                changed = False
                for id_field, label_field in (
                    ("gold_answer_ids", "gold_answer_labels"),
                    ("predicted_answer_ids", "predicted_answer_labels"),
                ):
                    answer_ids = row.get(id_field)
                    labels = row.get(label_field)
                    if not isinstance(answer_ids, list) or not isinstance(labels, list):
                        raise ValueError(
                            f"Prediction line {line_count} has malformed answer arrays"
                        )
                    if len(answer_ids) != len(labels):
                        raise ValueError(
                            f"Prediction line {line_count} has misaligned answer arrays"
                        )
                    for rank, (answer_id, label) in enumerate(
                        zip(answer_ids, labels, strict=True)
                    ):
                        if not isinstance(answer_id, str):
                            raise ValueError(
                                f"Prediction line {line_count} has a non-string answer ID"
                            )
                        if not isinstance(label, str):
                            raise ValueError(
                                f"Prediction line {line_count} has a non-string label"
                            )
                        if label.strip():
                            continue
                        labels[rank] = _fallback_label(answer_id)
                        changed = True
                        replacements_by_field[label_field] += 1
                        replacements_by_answer_id[answer_id] += 1
                        if label_field == "predicted_answer_labels":
                            replacements_by_rank[rank] += 1
                if changed:
                    changed_rows += 1
                _validate_label_only_change(original, row, line_count)
                output_stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                )
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    replacement_count = sum(replacements_by_field.values())
    if replacement_count == 0:
        target.unlink(missing_ok=True)
        raise ValueError("Prediction export contains no blank labels to repair")
    return {
        "line_count": line_count,
        "changed_row_count": changed_rows,
        "replacement_count": replacement_count,
        "replacements_by_field": dict(sorted(replacements_by_field.items())),
        "replacements_by_answer_id": dict(sorted(replacements_by_answer_id.items())),
        "predicted_replacements_by_rank": {
            str(rank): count for rank, count in sorted(replacements_by_rank.items())
        },
    }


def _fallback_label(answer_id: str) -> str:
    namespace, separator, payload = answer_id.partition(":")
    if separator != ":" or namespace not in {"entity", "time"} or not payload.strip():
        raise ValueError(f"Cannot derive a label from invalid answer ID: {answer_id!r}")
    return payload


def _validate_label_only_change(
    original: dict[str, object], corrected: dict[str, object], line_number: int
) -> None:
    allowed = {"gold_answer_labels", "predicted_answer_labels"}
    if any(original.get(name) != corrected.get(name) for name in set(original) - allowed):
        raise ValueError(f"Label correction changed a protected field at line {line_number}")
    if set(original) != set(corrected):
        raise ValueError(f"Label correction changed the schema at line {line_number}")
    for id_field, label_field in (
        ("gold_answer_ids", "gold_answer_labels"),
        ("predicted_answer_ids", "predicted_answer_labels"),
    ):
        answer_ids = original[id_field]
        before = original[label_field]
        after = corrected[label_field]
        if not (
            isinstance(answer_ids, list)
            and isinstance(before, list)
            and isinstance(after, list)
        ):
            raise ValueError(f"Malformed corrected label arrays at line {line_number}")
        for answer_id, old_label, new_label in zip(answer_ids, before, after, strict=True):
            if old_label == new_label:
                continue
            if not isinstance(old_label, str) or old_label.strip():
                raise ValueError(f"A nonblank label changed at line {line_number}")
            if new_label != _fallback_label(str(answer_id)):
                raise ValueError(f"Unexpected fallback label at line {line_number}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote an audited empty-label correction")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = promote_entity_id_label_fallback(run_dir=args.run_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
