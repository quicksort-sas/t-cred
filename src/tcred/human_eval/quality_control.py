from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Literal, Protocol

import orjson
from pydantic import BaseModel, ConfigDict, Field

from tcred.dataset.io import read_jsonl
from tcred.human_eval.protocol import CATEGORICAL_FIELDS, label_distance

QC_POLICY_VERSION = "1.0"
QC_POLICY_FROZEN_DATE = "2026-07-22"

MIN_RELATIVE_COHORT_SIZE = 12
FAST_COMPLETION_MAX_MINUTES = 30.0
FAST_COMPLETION_MEDIAN_FRACTION = 0.5
STRAIGHTLINE_MIN_JUDGMENTS = 40
STRAIGHTLINE_DOMINANT_RATE = 0.9
FIELD_STRAIGHTLINE_MIN_JUDGMENTS = 10
FIELD_STRAIGHTLINE_DOMINANT_RATE = 0.95
FIELD_STRAIGHTLINE_MIN_FIELDS = 3
TUTORIAL_FAILED_CHECKS_THRESHOLD = 3
CANNOT_JUDGE_MIN_JUDGMENTS = 40
CANNOT_JUDGE_MIN_RATE = 0.25
CANNOT_JUDGE_MEDIAN_EXCESS = 0.15
CANNOT_JUDGE_PEER_EXCESS = 0.15
LOO_MIN_COMPARISONS = 40
LOO_MAX_WEIGHTED_AGREEMENT = 0.6
LOO_MEDIAN_DEFICIT = 0.15
EXCLUSION_REVIEW_MIN_DOMAINS = 2

FlagDomain = Literal["process", "response_pattern", "peer_agreement"]


class HumanLabelLike(Protocol):
    unit_id: str
    annotator_id: str
    labels: dict[str, str]


class QualityFlag(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    domain: FlagDomain
    observed: dict[str, float | int | str | None]
    rule: str
    review_only: bool = True


class AnnotatorQualityRecord(BaseModel):
    annotator_id: str
    completed_units: int
    applicable_judgments: int
    response_distribution: dict[str, int]
    dominant_label: str | None
    dominant_label_rate: float | None
    field_straightline_count: int
    completion_minutes: float | None
    tutorial_attempts: int | None
    tutorial_failed_checks: int | None
    cannot_judge_rate: float | None
    peer_expected_cannot_judge_rate: float | None
    leave_one_out_comparisons: int
    leave_one_out_exact_agreement: float | None
    leave_one_out_weighted_agreement: float | None
    flags: list[QualityFlag] = Field(default_factory=list)
    flag_domains: list[FlagDomain] = Field(default_factory=list)
    review_flagged: bool = False
    exclusion_review_candidate: bool = False
    automatically_excluded: bool = False


class QualityControlReport(BaseModel):
    schema_version: str = "1.0"
    policy_version: str = QC_POLICY_VERSION
    policy_frozen_date: str = QC_POLICY_FROZEN_DATE
    preregistered_before_production_labels: bool = True
    annotator_count: int
    review_flagged_count: int
    exclusion_review_candidate_count: int
    automatically_excluded_count: int = 0
    thresholds: dict[str, object]
    annotators: list[AnnotatorQualityRecord]
    warnings: list[str] = Field(default_factory=list)


class _RawMetrics(BaseModel):
    annotator_id: str
    completed_units: int
    applicable_judgments: int
    response_distribution: dict[str, int]
    dominant_label: str | None
    dominant_label_rate: float | None
    field_straightline_count: int
    completion_minutes: float | None
    tutorial_attempts: int | None
    tutorial_failed_checks: int | None
    cannot_judge_rate: float | None
    peer_expected_cannot_judge_rate: float | None
    leave_one_out_comparisons: int
    leave_one_out_exact_agreement: float | None
    leave_one_out_weighted_agreement: float | None


def analyze_quality_control(
    labels: Sequence[HumanLabelLike],
    *,
    assignment_dir: Path,
) -> QualityControlReport:
    """Apply preregistered review flags without excluding or changing any label."""
    by_annotator: dict[str, list[HumanLabelLike]] = defaultdict(list)
    by_unit: dict[str, list[HumanLabelLike]] = defaultdict(list)
    for label in labels:
        by_annotator[label.annotator_id].append(label)
        by_unit[label.unit_id].append(label)

    activity, activity_warning = _read_activity(assignment_dir)
    raw = [
        _raw_metrics(
            annotator_id=annotator_id,
            labels=annotator_labels,
            by_unit=by_unit,
            assignment_dir=assignment_dir,
            activity=activity.get(annotator_id, {}),
        )
        for annotator_id, annotator_labels in sorted(by_annotator.items())
    ]
    completion_values = [
        row.completion_minutes for row in raw if row.completion_minutes is not None
    ]
    cannot_judge_values = [
        row.cannot_judge_rate for row in raw if row.cannot_judge_rate is not None
    ]
    loo_values = [
        row.leave_one_out_weighted_agreement
        for row in raw
        if row.leave_one_out_weighted_agreement is not None
    ]
    completion_median = _available_median(completion_values)
    cannot_judge_median = _available_median(cannot_judge_values)
    loo_median = _available_median(loo_values)
    completion_ready = len(completion_values) >= MIN_RELATIVE_COHORT_SIZE
    cannot_judge_ready = len(cannot_judge_values) >= MIN_RELATIVE_COHORT_SIZE
    loo_ready = len(loo_values) >= MIN_RELATIVE_COHORT_SIZE

    records = [
        _apply_flags(
            row,
            completion_ready=completion_ready,
            cannot_judge_ready=cannot_judge_ready,
            loo_ready=loo_ready,
            completion_median=completion_median,
            cannot_judge_median=cannot_judge_median,
            loo_median=loo_median,
        )
        for row in raw
    ]
    warnings: list[str] = []
    if activity_warning:
        warnings.append(activity_warning)
    unavailable_relative = [
        name
        for name, ready in (
            ("completion speed", completion_ready),
            ("Cannot-judge rate", cannot_judge_ready),
            ("leave-one-out agreement", loo_ready),
        )
        if not ready
    ]
    if unavailable_relative and raw:
        warnings.append(
            "Cohort-relative flags are suppressed until at least "
            f"{MIN_RELATIVE_COHORT_SIZE} valid annotator observations are available for: "
            + ", ".join(unavailable_relative)
            + "."
        )
    if any(row.completion_minutes is None for row in raw):
        warnings.append(
            "Completion-speed review is unavailable for annotators without both a research-save "
            "timestamp and a valid final-submission timestamp."
        )
    if any(row.tutorial_failed_checks is None for row in raw):
        warnings.append(
            "Tutorial-retry review is unavailable where the aggregate tutorial summary was not "
            "recorded; missing telemetry is not itself a quality flag."
        )

    return QualityControlReport(
        annotator_count=len(records),
        review_flagged_count=sum(row.review_flagged for row in records),
        exclusion_review_candidate_count=sum(row.exclusion_review_candidate for row in records),
        thresholds=quality_control_thresholds(),
        annotators=records,
        warnings=warnings,
    )


def quality_control_thresholds() -> dict[str, object]:
    return {
        "minimum_relative_cohort_size": MIN_RELATIVE_COHORT_SIZE,
        "fast_completion": {
            "maximum_minutes": FAST_COMPLETION_MAX_MINUTES,
            "maximum_fraction_of_cohort_median": FAST_COMPLETION_MEDIAN_FRACTION,
            "requires_both_conditions": True,
        },
        "straightlining": {
            "minimum_applicable_judgments": STRAIGHTLINE_MIN_JUDGMENTS,
            "global_dominant_label_rate": STRAIGHTLINE_DOMINANT_RATE,
            "field_minimum_judgments": FIELD_STRAIGHTLINE_MIN_JUDGMENTS,
            "field_dominant_label_rate": FIELD_STRAIGHTLINE_DOMINANT_RATE,
            "minimum_straightlined_fields": FIELD_STRAIGHTLINE_MIN_FIELDS,
        },
        "tutorial_retries": {
            "minimum_failed_independent_checks": TUTORIAL_FAILED_CHECKS_THRESHOLD,
        },
        "high_cannot_judge": {
            "minimum_applicable_judgments": CANNOT_JUDGE_MIN_JUDGMENTS,
            "minimum_rate": CANNOT_JUDGE_MIN_RATE,
            "minimum_excess_over_cohort_median": CANNOT_JUDGE_MEDIAN_EXCESS,
            "minimum_excess_over_same_item_peers": CANNOT_JUDGE_PEER_EXCESS,
            "requires_all_conditions": True,
        },
        "low_leave_one_out_agreement": {
            "minimum_pairwise_comparisons": LOO_MIN_COMPARISONS,
            "maximum_weighted_agreement": LOO_MAX_WEIGHTED_AGREEMENT,
            "minimum_deficit_from_cohort_median": LOO_MEDIAN_DEFICIT,
            "requires_all_conditions": True,
        },
        "exclusion_review": {
            "minimum_independent_flag_domains": EXCLUSION_REVIEW_MIN_DOMAINS,
            "domains": ["process", "response_pattern", "peer_agreement"],
            "automatic_exclusion": False,
        },
    }


def _raw_metrics(
    *,
    annotator_id: str,
    labels: Sequence[HumanLabelLike],
    by_unit: dict[str, list[HumanLabelLike]],
    assignment_dir: Path,
    activity: dict[str, object],
) -> _RawMetrics:
    values = [
        value
        for label in labels
        for field in CATEGORICAL_FIELDS
        if (value := label.labels.get(field, "")) != "not_applicable"
    ]
    distribution = Counter(values)
    dominant_label = None
    dominant_rate = None
    if values:
        dominant_label, dominant_count = sorted(
            distribution.items(), key=lambda item: (-item[1], item[0])
        )[0]
        dominant_rate = dominant_count / len(values)

    field_straightline_count = 0
    for field in CATEGORICAL_FIELDS:
        field_values = [
            label.labels[field]
            for label in labels
            if label.labels.get(field) not in {None, "not_applicable"}
        ]
        if (
            len(field_values) >= FIELD_STRAIGHTLINE_MIN_JUDGMENTS
            and max(Counter(field_values).values()) / len(field_values)
            >= FIELD_STRAIGHTLINE_DOMINANT_RATE
        ):
            field_straightline_count += 1

    exact_scores: list[float] = []
    weighted_scores: list[float] = []
    peer_cannot_judge: list[float] = []
    for label in labels:
        peers = [peer for peer in by_unit[label.unit_id] if peer.annotator_id != annotator_id]
        for field in CATEGORICAL_FIELDS:
            value = label.labels.get(field, "")
            if value == "not_applicable":
                continue
            peer_values = [
                peer.labels.get(field, "")
                for peer in peers
                if peer.labels.get(field, "") != "not_applicable"
            ]
            if not peer_values:
                continue
            exact_scores.append(sum(value == peer for peer in peer_values) / len(peer_values))
            weighted_scores.append(
                sum(1.0 - label_distance(value, peer) for peer in peer_values) / len(peer_values)
            )
            peer_cannot_judge.append(
                sum(peer == "unjudgeable" for peer in peer_values) / len(peer_values)
            )

    tutorial_attempts, tutorial_failed_checks = _tutorial_counts(activity)
    return _RawMetrics(
        annotator_id=annotator_id,
        completed_units=len(labels),
        applicable_judgments=len(values),
        response_distribution=dict(sorted(distribution.items())),
        dominant_label=dominant_label,
        dominant_label_rate=_rounded_rate(dominant_rate),
        field_straightline_count=field_straightline_count,
        completion_minutes=_completion_minutes(assignment_dir, annotator_id),
        tutorial_attempts=tutorial_attempts,
        tutorial_failed_checks=tutorial_failed_checks,
        cannot_judge_rate=_rounded_rate(
            distribution.get("unjudgeable", 0) / len(values) if values else None
        ),
        peer_expected_cannot_judge_rate=_rounded_rate(
            sum(peer_cannot_judge) / len(peer_cannot_judge) if peer_cannot_judge else None
        ),
        leave_one_out_comparisons=len(weighted_scores),
        leave_one_out_exact_agreement=_rounded_rate(
            sum(exact_scores) / len(exact_scores) if exact_scores else None
        ),
        leave_one_out_weighted_agreement=_rounded_rate(
            sum(weighted_scores) / len(weighted_scores) if weighted_scores else None
        ),
    )


def _apply_flags(
    row: _RawMetrics,
    *,
    completion_ready: bool,
    cannot_judge_ready: bool,
    loo_ready: bool,
    completion_median: float | None,
    cannot_judge_median: float | None,
    loo_median: float | None,
) -> AnnotatorQualityRecord:
    flags: list[QualityFlag] = []
    if (
        completion_ready
        and row.completion_minutes is not None
        and completion_median is not None
        and row.completion_minutes < FAST_COMPLETION_MAX_MINUTES
        and row.completion_minutes < completion_median * FAST_COMPLETION_MEDIAN_FRACTION
    ):
        flags.append(
            QualityFlag(
                code="fast_completion",
                domain="process",
                observed={
                    "completion_minutes": row.completion_minutes,
                    "cohort_median_minutes": completion_median,
                },
                rule="Below 30 minutes and below one half of the completed-cohort median.",
            )
        )

    global_straightline = (
        row.applicable_judgments >= STRAIGHTLINE_MIN_JUDGMENTS
        and row.dominant_label_rate is not None
        and row.dominant_label_rate >= STRAIGHTLINE_DOMINANT_RATE
    )
    field_straightline = row.field_straightline_count >= FIELD_STRAIGHTLINE_MIN_FIELDS
    if global_straightline or field_straightline:
        flags.append(
            QualityFlag(
                code="excessive_straightlining",
                domain="response_pattern",
                observed={
                    "dominant_label": row.dominant_label,
                    "dominant_label_rate": row.dominant_label_rate,
                    "field_straightline_count": row.field_straightline_count,
                },
                rule=(
                    "At least 90% one label over 40 applicable judgments, or at least three "
                    "fields with 95% one label over 10 judgments each."
                ),
            )
        )

    if (
        row.tutorial_failed_checks is not None
        and row.tutorial_failed_checks >= TUTORIAL_FAILED_CHECKS_THRESHOLD
    ):
        flags.append(
            QualityFlag(
                code="tutorial_retries",
                domain="process",
                observed={"failed_independent_checks": row.tutorial_failed_checks},
                rule="At least three failed checks across the independent tutorial items.",
            )
        )

    if (
        cannot_judge_ready
        and row.applicable_judgments >= CANNOT_JUDGE_MIN_JUDGMENTS
        and row.cannot_judge_rate is not None
        and row.peer_expected_cannot_judge_rate is not None
        and cannot_judge_median is not None
        and row.cannot_judge_rate >= CANNOT_JUDGE_MIN_RATE
        and row.cannot_judge_rate - cannot_judge_median >= CANNOT_JUDGE_MEDIAN_EXCESS
        and row.cannot_judge_rate - row.peer_expected_cannot_judge_rate >= CANNOT_JUDGE_PEER_EXCESS
    ):
        flags.append(
            QualityFlag(
                code="high_cannot_judge_rate",
                domain="response_pattern",
                observed={
                    "cannot_judge_rate": row.cannot_judge_rate,
                    "cohort_median_rate": cannot_judge_median,
                    "same_item_peer_rate": row.peer_expected_cannot_judge_rate,
                },
                rule=(
                    "At least 25%, at least 15 percentage points above the cohort median, and "
                    "at least 15 points above co-raters on the same unit-fields."
                ),
            )
        )

    if (
        loo_ready
        and row.leave_one_out_comparisons >= LOO_MIN_COMPARISONS
        and row.leave_one_out_weighted_agreement is not None
        and loo_median is not None
        and row.leave_one_out_weighted_agreement < LOO_MAX_WEIGHTED_AGREEMENT
        and loo_median - row.leave_one_out_weighted_agreement >= LOO_MEDIAN_DEFICIT
    ):
        flags.append(
            QualityFlag(
                code="low_leave_one_out_agreement",
                domain="peer_agreement",
                observed={
                    "weighted_agreement": row.leave_one_out_weighted_agreement,
                    "cohort_median_weighted_agreement": loo_median,
                    "comparisons": row.leave_one_out_comparisons,
                },
                rule=(
                    "Weighted agreement below 0.60 and at least 0.15 below the cohort median "
                    "over at least 40 pairwise comparisons."
                ),
            )
        )

    domains = sorted({flag.domain for flag in flags})
    return AnnotatorQualityRecord(
        **row.model_dump(mode="python"),
        flags=flags,
        flag_domains=domains,
        review_flagged=bool(flags),
        exclusion_review_candidate=len(domains) >= EXCLUSION_REVIEW_MIN_DOMAINS,
        automatically_excluded=False,
    )


def _completion_minutes(assignment_dir: Path, annotator_id: str) -> float | None:
    audit_path = _first_existing(
        assignment_dir / "audit" / f"{annotator_id}.events.jsonl",
        assignment_dir.parent / "save_audit" / f"{annotator_id}.events.jsonl",
    )
    submission_path = _first_existing(
        assignment_dir / "submissions" / f"{annotator_id}.json",
        assignment_dir.parent / "submissions" / f"{annotator_id}.json",
    )
    if audit_path is None or submission_path is None:
        return None
    save_times = [
        parsed
        for row in read_jsonl(audit_path)
        if row.get("event") == "annotation_save"
        and (parsed := _parse_timestamp(row.get("saved_at"))) is not None
    ]
    payload = orjson.loads(submission_path.read_bytes())
    submitted_at = (
        _parse_timestamp(payload.get("submitted_at")) if isinstance(payload, dict) else None
    )
    if not save_times or submitted_at is None:
        return None
    elapsed_seconds = (submitted_at - min(save_times)).total_seconds()
    if elapsed_seconds < 0:
        return None
    return round(elapsed_seconds / 60, 2)


def _read_activity(assignment_dir: Path) -> tuple[dict[str, dict[str, object]], str | None]:
    path = _first_existing(
        assignment_dir / ".admin" / "activity.json",
        assignment_dir.parent / ".admin" / "activity.json",
        assignment_dir / "quality_control" / "tutorial_activity.json",
        assignment_dir.parent / "quality_control" / "tutorial_activity.json",
    )
    if path is None:
        return {}, "No privacy-minimized tutorial activity file was found."
    payload = orjson.loads(path.read_bytes())
    raw_annotators = payload.get("annotators", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_annotators, dict):
        return {}, f"Tutorial activity file is malformed: {path.name}"
    return {
        str(annotator_id): dict(record)
        for annotator_id, record in raw_annotators.items()
        if isinstance(record, dict) and str(annotator_id) != "annotator_00"
    }, None


def _tutorial_counts(activity: dict[str, object]) -> tuple[int | None, int | None]:
    summary = activity.get("tutorial_summary")
    if not isinstance(summary, dict):
        return None, None
    items = summary.get("items")
    if not isinstance(items, dict):
        return None, None
    attempts = 0
    failed = 0
    observed = False
    for raw in items.values():
        if not isinstance(raw, dict):
            continue
        try:
            attempts += int(raw.get("attempts", 0))
            failed += int(raw.get("failed_attempts", 0))
        except (TypeError, ValueError):
            continue
        observed = True
    return (attempts, failed) if observed else (None, None)


def _available_median(values: Iterable[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return round(float(median(available)), 4) if available else None


def _rounded_rate(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)
