from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict, Field

from tcred.dataset.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from tcred.human_eval.package import verify_manifest_artifacts
from tcred.human_eval.protocol import (
    CATEGORICAL_FIELDS,
    PROTOCOL_VERSION,
    AnnotationState,
    annotation_complete,
    label_distance,
    reason_allowed,
    reason_required,
)
from tcred.human_eval.quality_control import analyze_quality_control

SCORABLE_FIELDS = CATEGORICAL_FIELDS
ANNOTATOR_FILE = re.compile(r"^annotator_(\d{2})$")


class ImportedHumanLabel(BaseModel):
    unit_id: str
    annotator_id: str
    labels: dict[str, str]
    reasons: dict[str, str] = Field(default_factory=dict)
    issue_code: str = ""
    issue_detail: str = ""
    comment: str = ""


class FieldAgreement(BaseModel):
    comparable_pairs: int
    observed_agreement: float
    krippendorff_alpha: float | None
    krippendorff_alpha_ci95: tuple[float, float] | None
    units_with_two_or_more_labels: int
    mean_pairwise_cohen_kappa: float | None = None


class HumanLabelImportReport(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    imported_label_count: int
    unique_unit_count: int
    annotator_count: int
    field_agreement: dict[str, FieldAgreement]
    warnings: list[str]


def import_human_labels(
    *,
    assignment_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    allow_unfrozen: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    resolved_manifest = manifest_path or _discover_manifest(assignment_dir)
    if resolved_manifest is None and not allow_unfrozen:
        raise FileNotFoundError(
            "No assignment_manifest.json was found for the label directory. "
            "Pass the frozen manifest or set allow_unfrozen=True only for development data."
        )
    expected = (
        _expected_assignments(
            resolved_manifest,
            require_hashes=not allow_unfrozen,
        )
        if resolved_manifest
        else None
    )
    labels = _read_assignment_labels(
        assignment_dir,
        expected_assignments=expected,
        require_submitted=expected is not None,
    )
    majority_rows = _majority_rows(labels)
    report = _agreement_report(labels)
    quality_report = analyze_quality_control(labels, assignment_dir=assignment_dir)
    review_flagged = {row.annotator_id for row in quality_report.annotators if row.review_flagged}
    exclusion_review_candidates = {
        row.annotator_id for row in quality_report.annotators if row.exclusion_review_candidate
    }
    sensitivity_report = _quality_control_sensitivity(
        labels,
        review_flagged=review_flagged,
        exclusion_review_candidates=exclusion_review_candidates,
        policy_version=quality_report.policy_version,
    )

    labels_path = output_dir / "human_eval_labels_imported.jsonl"
    majority_path = output_dir / "human_eval_majority_labels.jsonl"
    report_path = output_dir / "human_eval_agreement_report.json"
    quality_path = output_dir / "human_eval_quality_control_report.json"
    sensitivity_path = output_dir / "human_eval_quality_control_sensitivity.json"
    for path in (labels_path, majority_path, report_path, quality_path, sensitivity_path):
        _guard(path, overwrite=overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(labels_path, [label.model_dump(mode="json") for label in labels])
    write_jsonl_atomic(majority_path, majority_rows)
    write_json_atomic(
        report_path,
        report.model_dump(mode="json"),
    )
    write_json_atomic(quality_path, quality_report.model_dump(mode="json"))
    write_json_atomic(sensitivity_path, sensitivity_report)
    return {
        "imported_labels": labels_path,
        "majority_labels": majority_path,
        "agreement_report": report_path,
        "quality_control_report": quality_path,
        "quality_control_sensitivity": sensitivity_path,
    }


def _read_assignment_labels(
    assignment_dir: Path,
    *,
    expected_assignments: set[tuple[str, str]] | None,
    require_submitted: bool,
) -> list[ImportedHumanLabel]:
    if not assignment_dir.exists():
        raise FileNotFoundError(f"Assignment directory does not exist: {assignment_dir}")
    labels: list[ImportedHumanLabel] = []
    seen: set[tuple[str, str]] = set()
    if require_submitted:
        _validate_submissions(assignment_dir, expected_assignments or set())
    for path in sorted(assignment_dir.glob("annotator_*.jsonl")):
        match = ANNOTATOR_FILE.fullmatch(path.stem)
        if match is None or match.group(1) == "00":
            continue
        for row in read_jsonl(path):
            annotator_id = str(row.get("annotator_id") or path.stem)
            if annotator_id != path.stem:
                raise ValueError(f"Annotator id {annotator_id!r} does not match file {path.name}")
            unit_id = str(row.get("unit_id") or "")
            raw_annotation = row.get("annotation")
            if not unit_id or not isinstance(raw_annotation, dict):
                raise ValueError(f"Malformed annotation row in {path}: missing unit or annotation")
            state = AnnotationState.model_validate(raw_annotation)
            identity = (annotator_id, unit_id)
            if identity in seen:
                raise ValueError(f"Duplicate annotation record: {identity}")
            if expected_assignments is not None and identity not in expected_assignments:
                raise ValueError(f"Annotation is not in the frozen assignment manifest: {identity}")
            cleaned: dict[str, str] = {}
            for field in SCORABLE_FIELDS:
                value = str(state.labels.get(field, "")).strip()
                if not value:
                    raise ValueError(f"Incomplete {field} label for {identity}")
                if value not in {"yes", "partial", "no", "not_applicable", "unjudgeable"}:
                    raise ValueError(f"Invalid {field} label {value!r} for {identity}")
                reason = str(state.reasons.get(field, "")).strip()
                if reason_required(value) and not reason_allowed(field, value, reason):
                    raise ValueError(f"Missing or invalid reason for {field}={value} in {identity}")
                cleaned[field] = value
            applicable = [field for field, value in cleaned.items() if value != "not_applicable"]
            if not annotation_complete({"applicable_fields": applicable}, state):
                raise ValueError(
                    f"Annotation is incomplete or logically contradictory for {identity}"
                )
            seen.add(identity)
            labels.append(
                ImportedHumanLabel(
                    unit_id=unit_id,
                    annotator_id=annotator_id,
                    labels=cleaned,
                    reasons=dict(state.reasons),
                    issue_code=state.issue_code,
                    issue_detail=state.issue_detail,
                    comment=state.comment,
                )
            )
    if expected_assignments is not None and seen != expected_assignments:
        missing = sorted(expected_assignments - seen)
        raise ValueError(
            "The annotation package is incomplete: "
            f"{len(missing)} frozen assignments have no submitted label row; "
            f"examples={missing[:5]}"
        )
    return labels


def _majority_rows(labels: list[ImportedHumanLabel]) -> list[dict[str, object]]:
    by_unit: dict[str, list[ImportedHumanLabel]] = defaultdict(list)
    for label in labels:
        by_unit[label.unit_id].append(label)

    rows: list[dict[str, object]] = []
    for unit_id, unit_labels in sorted(by_unit.items()):
        majority: dict[str, object] = {}
        for field in SCORABLE_FIELDS:
            values = [label.labels[field] for label in unit_labels if field in label.labels]
            if not values:
                majority[field] = None
                majority[f"{field}_tie"] = False
                majority[f"{field}_disagreement"] = False
                majority[f"{field}_votes"] = {}
                continue
            counts = Counter(values)
            top_count = max(counts.values())
            winners = sorted(value for value, count in counts.items() if count == top_count)
            tied = len(winners) > 1
            majority[field] = None if tied else winners[0]
            majority[f"{field}_tie"] = tied
            majority[f"{field}_disagreement"] = len(counts) > 1
            majority[f"{field}_votes"] = dict(sorted(counts.items()))
        rows.append(
            {
                "unit_id": unit_id,
                "annotator_ids": sorted(label.annotator_id for label in unit_labels),
                "label_count": len(unit_labels),
                "majority_labels": majority,
                "requires_resolution": any(
                    bool(majority.get(f"{field}_disagreement")) for field in SCORABLE_FIELDS
                ),
            }
        )
    return rows


def _agreement_report(labels: list[ImportedHumanLabel]) -> HumanLabelImportReport:
    by_unit: dict[str, list[ImportedHumanLabel]] = defaultdict(list)
    for label in labels:
        by_unit[label.unit_id].append(label)

    warnings: list[str] = []
    if not labels:
        warnings.append("No labels found in assignment files")
    field_agreement = {
        field: _field_agreement(field=field, by_unit=by_unit) for field in SCORABLE_FIELDS
    }
    return HumanLabelImportReport(
        imported_label_count=len(labels),
        unique_unit_count=len(by_unit),
        annotator_count=len({label.annotator_id for label in labels}),
        field_agreement=field_agreement,
        warnings=warnings,
    )


def _quality_control_sensitivity(
    labels: list[ImportedHumanLabel],
    *,
    review_flagged: set[str],
    exclusion_review_candidates: set[str],
    policy_version: str,
) -> dict[str, object]:
    def cohort(excluded: set[str]) -> dict[str, object]:
        retained = [label for label in labels if label.annotator_id not in excluded]
        return {
            "excluded_annotator_ids": sorted(excluded),
            "retained_annotator_count": len({label.annotator_id for label in retained}),
            "retained_label_count": len(retained),
            "agreement": _agreement_report(retained).model_dump(mode="json"),
        }

    return {
        "schema_version": "1.0",
        "quality_control_policy_version": policy_version,
        "primary_analysis": cohort(set()),
        "sensitivity_excluding_any_review_flag": cohort(review_flagged),
        "sensitivity_excluding_multi_domain_review_candidates": cohort(exclusion_review_candidates),
        "interpretation": (
            "All annotations remain in the primary analysis. Sensitivity cohorts are analytical "
            "comparisons, not automatic exclusions or modified raw data."
        ),
    }


def _field_agreement(
    *,
    field: str,
    by_unit: dict[str, list[ImportedHumanLabel]],
) -> FieldAgreement:
    values_by_unit = [
        [
            label.labels[field]
            for label in unit_labels
            if field in label.labels and label.labels[field] != "not_applicable"
        ]
        for unit_labels in by_unit.values()
    ]
    values_by_unit = [values for values in values_by_unit if len(values) >= 2]
    pairs = [pair for values in values_by_unit for pair in combinations(values, 2)]
    if not pairs:
        return FieldAgreement(
            comparable_pairs=0,
            observed_agreement=0.0,
            krippendorff_alpha=None,
            krippendorff_alpha_ci95=None,
            units_with_two_or_more_labels=0,
        )
    observed = sum(left == right for left, right in pairs) / len(pairs)
    alpha = _krippendorff_alpha(values_by_unit)
    interval = _bootstrap_alpha(values_by_unit) if len(values_by_unit) >= 2 else None
    return FieldAgreement(
        comparable_pairs=len(pairs),
        observed_agreement=round(observed, 4),
        krippendorff_alpha=round(alpha, 4) if alpha is not None else None,
        krippendorff_alpha_ci95=interval,
        units_with_two_or_more_labels=len(values_by_unit),
    )


def _krippendorff_alpha(values_by_unit: list[list[str]]) -> float | None:
    eligible = [values for values in values_by_unit if len(values) >= 2]
    if not eligible:
        return None
    observed_numerator = sum(
        2
        * sum(label_distance(left, right) for left, right in combinations(values, 2))
        / (len(values) - 1)
        for values in eligible
    )
    ratings = [value for values in eligible for value in values]
    observed_disagreement = observed_numerator / len(ratings)
    if len(ratings) < 2:
        return None
    counts = Counter(ratings)
    total = len(ratings)
    expected_disagreement = sum(
        left_count * (right_count - (1 if left == right else 0)) * label_distance(left, right)
        for left, left_count in counts.items()
        for right, right_count in counts.items()
    ) / (total * (total - 1))
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def _bootstrap_alpha(
    values_by_unit: list[list[str]],
    *,
    replicates: int = 500,
) -> tuple[float, float] | None:
    rng = random.Random(20260713)
    estimates: list[float] = []
    for _ in range(replicates):
        sample = [rng.choice(values_by_unit) for _ in values_by_unit]
        estimate = _krippendorff_alpha(sample)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return round(lower, 4), round(upper, 4)


def _validate_submissions(
    assignment_dir: Path,
    expected_assignments: set[tuple[str, str]],
) -> None:
    expected_annotators = sorted({annotator_id for annotator_id, _ in expected_assignments})
    candidate_dirs = [assignment_dir / "submissions", assignment_dir.parent / "submissions"]
    submission_dir = next((path for path in candidate_dirs if path.exists()), None)
    if submission_dir is None:
        raise ValueError(
            "Frozen annotations have no submission records; every annotator must use the final "
            "review and submit action"
        )
    for annotator_id in expected_annotators:
        submission_path = submission_dir / f"{annotator_id}.json"
        label_path = assignment_dir / f"{annotator_id}.jsonl"
        if not submission_path.exists():
            raise ValueError(f"Missing final submission record for {annotator_id}")
        if not label_path.exists():
            raise ValueError(f"Missing saved label file for {annotator_id}")
        payload = orjson.loads(submission_path.read_bytes())
        if not isinstance(payload, dict) or payload.get("annotator_id") != annotator_id:
            raise ValueError(f"Malformed final submission record for {annotator_id}")
        if payload.get("annotation_protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported annotation protocol for {annotator_id}")
        expected_count = sum(expected_id == annotator_id for expected_id, _ in expected_assignments)
        if payload.get("completed_units") != expected_count:
            raise ValueError(f"Incorrect submitted unit count for {annotator_id}")
        if not str(payload.get("submitted_at", "")).strip():
            raise ValueError(f"Missing submission timestamp for {annotator_id}")
        expected_revision = str(payload.get("label_file_revision", ""))
        actual_revision = hashlib.sha256(label_path.read_bytes()).hexdigest()
        if expected_revision != actual_revision:
            raise ValueError(f"Saved labels changed after final submission for {annotator_id}")


def _discover_manifest(assignment_dir: Path) -> Path | None:
    for candidate in (
        assignment_dir.parent / "assignment_manifest.json",
        assignment_dir.parent.parent / "assignment_manifest.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _expected_assignments(
    manifest_path: Path,
    *,
    require_hashes: bool,
) -> set[tuple[str, str]]:
    manifest = orjson.loads(manifest_path.read_bytes())
    verify_manifest_artifacts(
        manifest_path=manifest_path,
        manifest=manifest,
        require_hashes=require_hashes,
    )
    assignment_dir = manifest_path.parent / "assignments"
    expected: set[tuple[str, str]] = set()
    for path in sorted(assignment_dir.glob("annotator_*.jsonl")):
        if path.stem == "annotator_00":
            continue
        for row in read_jsonl(path):
            expected.add((path.stem, str(row.get("unit_id", ""))))
    if not expected:
        raise ValueError(f"Manifest has no readable frozen assignments: {manifest_path}")
    declared = int(manifest.get("total_assignments", len(expected)))
    if declared != len(expected):
        raise ValueError(f"Manifest declares {declared} assignments but {len(expected)} were found")
    return expected


def _guard(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
