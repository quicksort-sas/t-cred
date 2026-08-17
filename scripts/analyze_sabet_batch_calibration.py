from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CANDIDATE_BATCHES = (8, 32, 64, 128, 256)
_SCORE_FIELDS = {
    "bertscore": ("bertscore_precision", "bertscore_recall", "bertscore_f1"),
    "sas": ("sas_cross_encoder",),
}


def analyze_calibration(
    *, results_root: Path, output_path: Path, tolerance: float, tie_fraction: float
) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_path = output_path.resolve()
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    if not 0 <= tie_fraction < 1:
        raise ValueError("tie_fraction must be in [0, 1)")

    metric_results: dict[str, Any] = {}
    for metric, expected_fields in _SCORE_FIELDS.items():
        candidates: dict[str, Any] = {}
        reference_rows: dict[str, dict[str, float]] | None = None
        reference_input: dict[str, Any] | None = None
        for batch_size in _CANDIDATE_BATCHES:
            candidate_dir = results_root / metric / f"batch-{batch_size}"
            exit_code = _read_exit_code(candidate_dir / "exit_code.txt")
            row: dict[str, Any] = {
                "batch_size": batch_size,
                "exit_code": exit_code,
                "eligible": False,
                "ineligibility_reasons": [],
            }
            if exit_code != 0:
                row["ineligibility_reasons"].append("worker_exit_nonzero")
                row["console"] = _optional_file_identity(candidate_dir / "console.log")
                candidates[str(batch_size)] = row
                continue

            manifest_path = candidate_dir / "manifest.json"
            score_path = candidate_dir / "scores.jsonl"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = _load_scores(score_path, expected_fields=expected_fields)
            row.update(
                {
                    "elapsed_seconds": float(manifest["elapsed_seconds"]),
                    "row_count": len(rows),
                    "manifest": _file_identity(manifest_path),
                    "scores": _file_identity(score_path),
                    "runtime_contract_sha256": manifest["runtime_contract_sha256"],
                    "input": manifest["input"],
                }
            )
            if manifest.get("requested_metrics") != [metric]:
                row["ineligibility_reasons"].append("requested_metric_mismatch")
            if manifest.get("row_count") != len(rows):
                row["ineligibility_reasons"].append("manifest_row_count_mismatch")

            if batch_size == _CANDIDATE_BATCHES[0]:
                reference_rows = rows
                reference_input = manifest["input"]
                row["maximum_absolute_difference_from_batch_8"] = 0.0
            elif reference_rows is None or reference_input is None:
                row["ineligibility_reasons"].append("batch_8_reference_unavailable")
            else:
                if manifest["input"] != reference_input:
                    row["ineligibility_reasons"].append("input_identity_mismatch")
                if set(rows) != set(reference_rows):
                    row["ineligibility_reasons"].append("metric_id_set_mismatch")
                else:
                    maximum_difference = max(
                        abs(rows[metric_id][field] - reference_rows[metric_id][field])
                        for metric_id in rows
                        for field in expected_fields
                    )
                    row["maximum_absolute_difference_from_batch_8"] = maximum_difference
                    if not math.isfinite(maximum_difference) or maximum_difference > tolerance:
                        row["ineligibility_reasons"].append("numerical_tolerance_exceeded")
            row["eligible"] = not row["ineligibility_reasons"]
            candidates[str(batch_size)] = row

        eligible = [row for row in candidates.values() if row["eligible"]]
        if not eligible:
            selection = None
            selection_reason = "no_eligible_candidate"
        else:
            fastest = min(float(row["elapsed_seconds"]) for row in eligible)
            tied = [
                row
                for row in eligible
                if float(row["elapsed_seconds"]) <= fastest * (1.0 + tie_fraction)
            ]
            selected = min(tied, key=lambda row: int(row["batch_size"]))
            selection = int(selected["batch_size"])
            selection_reason = (
                "smallest_batch_within_preregistered_elapsed_time_tie_of_fastest"
            )
        metric_results[metric] = {
            "reference_batch_size": _CANDIDATE_BATCHES[0],
            "score_fields": list(expected_fields),
            "candidates": candidates,
            "selected_batch_size": selection,
            "selection_reason": selection_reason,
        }

    report = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "results_root": str(results_root),
        "candidate_batch_sizes": list(_CANDIDATE_BATCHES),
        "maximum_absolute_score_tolerance": tolerance,
        "elapsed_time_tie_fraction": tie_fraction,
        "metrics": metric_results,
    }
    _write_json_atomic(output_path, report)
    return report


def _read_exit_code(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _load_scores(
    path: Path, *, expected_fields: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metric_id = row.get("metric_id")
            scores = row.get("scores")
            if not isinstance(metric_id, str) or not metric_id:
                raise ValueError(f"Missing metric_id at {path}:{line_number}")
            if metric_id in rows:
                raise ValueError(f"Duplicate metric_id in {path}: {metric_id}")
            if not isinstance(scores, dict) or set(scores) != set(expected_fields):
                raise ValueError(f"Unexpected score fields at {path}:{line_number}")
            numeric = {name: float(scores[name]) for name in expected_fields}
            if any(not math.isfinite(value) for value in numeric.values()):
                raise ValueError(f"Non-finite score at {path}:{line_number}")
            rows[metric_id] = numeric
    if not rows:
        raise ValueError(f"Score file is empty: {path}")
    return rows


def _optional_file_identity(path: Path) -> dict[str, Any] | None:
    return _file_identity(path) if path.is_file() else None


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": digest.hexdigest()}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite calibration analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SABET neural batch calibration")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--tie-fraction", type=float, default=0.03)
    args = parser.parse_args()
    report = analyze_calibration(
        results_root=args.results_root,
        output_path=args.output,
        tolerance=args.tolerance,
        tie_fraction=args.tie_fraction,
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
