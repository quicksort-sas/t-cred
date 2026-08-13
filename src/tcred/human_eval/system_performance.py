from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict, Field

from tcred.dataset.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from tcred.human_eval.gold_schema import GOLD_POLICY_VERSION, GoldLabelRecord
from tcred.human_eval.protocol import CATEGORICAL_FIELDS, VISIBLE_LABEL_OPTIONS
from tcred.qa.models import ALL_QA_SYSTEMS

PERFORMANCE_SCHEMA_VERSION = "1.0"
LABEL_SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}
OUTPUT_FILENAMES = (
    "system_performance.json",
    "system_unit_scores.jsonl",
    "performance_manifest.json",
)


class RateEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float | None
    wilson_ci95: tuple[float, float] | None


class FieldPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicable_count: int = Field(ge=0)
    label_counts: dict[str, int]
    strict_yes: RateEstimate
    unjudgeable_count: int = Field(ge=0)
    determinate_count: int = Field(ge=0)
    determinate_strict_yes_rate: float | None
    determinate_partial_credit_score: float | None


class SystemPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_name: str
    unit_count: int = Field(ge=0)
    question_count: int = Field(ge=0)
    dataset_family_counts: dict[str, int]
    fields: dict[str, FieldPerformance]
    answer_by_dataset_family: dict[str, FieldPerformance]
    joint_answer_temporal_yes: RateEstimate
    all_applicable_fields_yes: RateEstimate
    micro_field_labels: FieldPerformance


class PairwiseAnswerComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_system: str
    right_system: str
    shared_question_count: int = Field(ge=0)
    comparable_question_count: int = Field(ge=0)
    left_strict_yes_count: int = Field(ge=0)
    right_strict_yes_count: int = Field(ge=0)
    left_partial_credit_score: float | None
    right_partial_credit_score: float | None
    left_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    right_wins: int = Field(ge=0)


class HumanGoldSystemPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PERFORMANCE_SCHEMA_VERSION
    generated_at: str
    gold_policy_version: str
    gold_manifest_sha256: str
    gold_snapshot: str
    gold_unit_count: int
    system_output_gold_unit_count: int
    system_names: tuple[str, ...]
    question_coverage_by_system_count: dict[str, int]
    complete_four_system_question_count: int
    scoring_policy: dict[str, str]
    systems: dict[str, SystemPerformance]
    descriptive_answer_ranking: tuple[str, ...]
    pairwise_answer_comparisons: tuple[PairwiseAnswerComparison, ...]
    limitations: tuple[str, ...]


def analyze_system_performance(
    *,
    gold_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Score the four QA systems against the human/adjudicated gold layer."""
    output_paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    _guard_outputs(output_paths.values(), overwrite=overwrite)

    manifest_path = gold_dir / "gold_manifest.json"
    labels_path = gold_dir / "gold_labels.jsonl"
    manifest = _load_object(manifest_path)
    _verify_gold_manifest(manifest, gold_dir=gold_dir)
    rows = tuple(GoldLabelRecord.model_validate(row) for row in read_jsonl(labels_path))
    expected_gold_count = int(_dict_value(manifest, "counts").get("gold_unit_count", -1))
    if len(rows) != expected_gold_count:
        raise ValueError("Gold label count differs from the gold manifest")

    system_rows = tuple(row for row in rows if row.metadata.get("source_kind") == "system_output")
    expected_systems = tuple(str(system) for system in ALL_QA_SYSTEMS)
    observed_systems = {row.metadata.get("system_name", "") for row in system_rows}
    if observed_systems != set(expected_systems):
        raise ValueError(
            "Gold system-output rows do not cover the exact configured systems: "
            f"expected={sorted(expected_systems)}, observed={sorted(observed_systems)}"
        )
    _validate_unique_system_questions(system_rows)

    grouped: dict[str, list[GoldLabelRecord]] = defaultdict(list)
    for row in system_rows:
        grouped[row.metadata["system_name"]].append(row)
    systems = {
        system_name: _system_performance(system_name, grouped[system_name])
        for system_name in expected_systems
    }

    by_qid: dict[str, set[str]] = defaultdict(set)
    for row in system_rows:
        by_qid[row.metadata["qid"]].add(row.metadata["system_name"])
    coverage = Counter(len(names) for names in by_qid.values())
    pairwise = _pairwise_answer_comparisons(system_rows, system_names=expected_systems)
    ranking = tuple(
        sorted(
            expected_systems,
            key=lambda name: (
                -(systems[name].fields["answer_correct"].strict_yes.rate or 0.0),
                -(systems[name].fields["answer_correct"].determinate_partial_credit_score or 0.0),
                name,
            ),
        )
    )
    report = HumanGoldSystemPerformance(
        generated_at=datetime.now(UTC).isoformat(),
        gold_policy_version=str(manifest.get("gold_policy_version", "")),
        gold_manifest_sha256=_sha256_file(manifest_path),
        gold_snapshot=str(manifest.get("annotation_snapshot", "")),
        gold_unit_count=len(rows),
        system_output_gold_unit_count=len(system_rows),
        system_names=expected_systems,
        question_coverage_by_system_count={
            str(count): qids for count, qids in sorted(coverage.items())
        },
        complete_four_system_question_count=coverage.get(4, 0),
        scoring_policy={
            "primary": "strict pass = human gold label yes",
            "partial_credit": "yes=1.0, partial=0.5, no=0.0",
            "unjudgeable": (
                "reported separately and excluded from determinate partial-credit scores"
            ),
            "confidence_interval": "Wilson 95% interval for strict yes / applicable count",
            "ranking": (
                "descriptive only; sorted by answer-correct strict pass rate, then partial credit"
            ),
        },
        systems=systems,
        descriptive_answer_ranking=ranking,
        pairwise_answer_comparisons=pairwise,
        limitations=(
            "Retained system samples are unequal and depend on interim annotation completion.",
            "No retained question has gold outputs for all four systems.",
            "Pairwise shared-question samples contain only 2 to 7 questions.",
            "Confidence intervals describe within-system binomial uncertainty but do not correct "
            "for non-random gold-set inclusion or question-composition differences.",
            "Adjudicated fields use one AI adjudicator rather than human-expert consensus.",
            "The descriptive ranking must not be reported as a statistically established system "
            "ordering.",
        ),
    )

    unit_scores = [_unit_score(row) for row in sorted(system_rows, key=_row_sort_key)]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_paths["system_performance.json"],
        report.model_dump(mode="json"),
        sort_keys=True,
    )
    write_jsonl_atomic(output_paths["system_unit_scores.jsonl"], unit_scores)
    write_json_atomic(
        output_paths["performance_manifest.json"],
        {
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "generated_at": report.generated_at,
            "gold_manifest": {
                "path": manifest_path.as_posix(),
                "bytes": manifest_path.stat().st_size,
                "sha256": report.gold_manifest_sha256,
            },
            "files": [
                _file_record(output_paths[name])
                for name in ("system_performance.json", "system_unit_scores.jsonl")
            ],
        },
        sort_keys=True,
    )
    return output_paths


def _system_performance(system_name: str, rows: list[GoldLabelRecord]) -> SystemPerformance:
    rows = sorted(rows, key=_row_sort_key)
    fields = {
        field: _field_performance(
            [row.gold_labels[field] for row in rows if field in row.gold_labels]
        )
        for field in CATEGORICAL_FIELDS
    }
    families: dict[str, list[GoldLabelRecord]] = defaultdict(list)
    for row in rows:
        families[row.metadata.get("dataset_family", "unknown")].append(row)
    answer_by_family = {
        family: _field_performance([row.gold_labels["answer_correct"] for row in family_rows])
        for family, family_rows in sorted(families.items())
    }

    temporal_rows = [row for row in rows if "temporal_correct" in row.gold_labels]
    joint_count = sum(
        row.gold_labels["answer_correct"] == "yes" and row.gold_labels["temporal_correct"] == "yes"
        for row in temporal_rows
    )
    all_yes = sum(all(label == "yes" for label in row.gold_labels.values()) for row in rows)
    all_field_labels = [label for row in rows for label in row.gold_labels.values()]
    return SystemPerformance(
        system_name=system_name,
        unit_count=len(rows),
        question_count=len({row.metadata["qid"] for row in rows}),
        dataset_family_counts=dict(
            sorted(Counter(row.metadata.get("dataset_family", "unknown") for row in rows).items())
        ),
        fields=fields,
        answer_by_dataset_family=answer_by_family,
        joint_answer_temporal_yes=_rate(joint_count, len(temporal_rows)),
        all_applicable_fields_yes=_rate(all_yes, len(rows)),
        micro_field_labels=_field_performance(all_field_labels),
    )


def _field_performance(labels: list[str]) -> FieldPerformance:
    counts = Counter(labels)
    for label in counts:
        if label not in VISIBLE_LABEL_OPTIONS:
            raise ValueError(f"Unsupported gold label in performance analysis: {label}")
    total = len(labels)
    unjudgeable = counts["unjudgeable"]
    determinate = total - unjudgeable
    determinate_score = (
        sum(LABEL_SCORES[label] for label in labels if label in LABEL_SCORES) / determinate
        if determinate
        else None
    )
    return FieldPerformance(
        applicable_count=total,
        label_counts={label: counts[label] for label in VISIBLE_LABEL_OPTIONS if counts[label] > 0},
        strict_yes=_rate(counts["yes"], total),
        unjudgeable_count=unjudgeable,
        determinate_count=determinate,
        determinate_strict_yes_rate=_rounded(counts["yes"] / determinate) if determinate else None,
        determinate_partial_credit_score=_rounded(determinate_score),
    )


def _pairwise_answer_comparisons(
    rows: tuple[GoldLabelRecord, ...], *, system_names: tuple[str, ...]
) -> tuple[PairwiseAnswerComparison, ...]:
    by_system = {
        system: {row.metadata["qid"]: row for row in rows if row.metadata["system_name"] == system}
        for system in system_names
    }
    comparisons: list[PairwiseAnswerComparison] = []
    for left_index, left in enumerate(system_names):
        for right in system_names[left_index + 1 :]:
            shared = sorted(set(by_system[left]) & set(by_system[right]))
            comparable: list[tuple[float, float]] = []
            for qid in shared:
                left_label = by_system[left][qid].gold_labels["answer_correct"]
                right_label = by_system[right][qid].gold_labels["answer_correct"]
                if left_label in LABEL_SCORES and right_label in LABEL_SCORES:
                    comparable.append((LABEL_SCORES[left_label], LABEL_SCORES[right_label]))
            comparisons.append(
                PairwiseAnswerComparison(
                    left_system=left,
                    right_system=right,
                    shared_question_count=len(shared),
                    comparable_question_count=len(comparable),
                    left_strict_yes_count=sum(score == 1.0 for score, _ in comparable),
                    right_strict_yes_count=sum(score == 1.0 for _, score in comparable),
                    left_partial_credit_score=_mean([score for score, _ in comparable]),
                    right_partial_credit_score=_mean([score for _, score in comparable]),
                    left_wins=sum(
                        left_score > right_score for left_score, right_score in comparable
                    ),
                    ties=sum(left_score == right_score for left_score, right_score in comparable),
                    right_wins=sum(
                        left_score < right_score for left_score, right_score in comparable
                    ),
                )
            )
    return tuple(comparisons)


def _unit_score(row: GoldLabelRecord) -> dict[str, object]:
    determinate = {
        field: LABEL_SCORES[label]
        for field, label in row.gold_labels.items()
        if label in LABEL_SCORES
    }
    return {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "unit_id": row.unit_id,
        "qid": row.metadata["qid"],
        "dataset_family": row.metadata.get("dataset_family", ""),
        "system_name": row.metadata["system_name"],
        "gold_labels": row.gold_labels,
        "numeric_scores": determinate,
        "all_applicable_fields_yes": all(label == "yes" for label in row.gold_labels.values()),
        "contains_unjudgeable": "unjudgeable" in row.gold_labels.values(),
    }


def _validate_unique_system_questions(rows: tuple[GoldLabelRecord, ...]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        system_name = row.metadata.get("system_name", "")
        qid = row.metadata.get("qid", "")
        if not system_name or not qid:
            raise ValueError(f"System gold row lacks system_name or qid: {row.unit_id}")
        identity = (system_name, qid)
        if identity in seen:
            raise ValueError(f"Duplicate gold system output for one question: {identity}")
        seen.add(identity)


def _verify_gold_manifest(manifest: dict[str, object], *, gold_dir: Path) -> None:
    if manifest.get("gold_policy_version") != GOLD_POLICY_VERSION:
        raise ValueError("Unsupported gold policy version")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Gold manifest has no file list")
    described: set[str] = set()
    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError("Malformed gold manifest file record")
        name = str(raw.get("path", ""))
        path = gold_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing gold payload: {path}")
        if path.stat().st_size != int(raw.get("bytes", -1)):
            raise ValueError(f"Gold payload size mismatch: {name}")
        if _sha256_file(path) != str(raw.get("sha256", "")):
            raise ValueError(f"Gold payload checksum mismatch: {name}")
        described.add(name)
    if "gold_labels.jsonl" not in described:
        raise ValueError("Gold manifest does not describe gold_labels.jsonl")


def _rate(numerator: int, denominator: int) -> RateEstimate:
    if denominator == 0:
        return RateEstimate(
            numerator=numerator,
            denominator=denominator,
            rate=None,
            wilson_ci95=None,
        )
    return RateEstimate(
        numerator=numerator,
        denominator=denominator,
        rate=_rounded(numerator / denominator),
        wilson_ci95=_wilson_interval(numerator, denominator),
    )


def _wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z_squared / (4 * total * total))
        / denominator
    )
    return _rounded(max(0.0, center - half_width)), _rounded(min(1.0, center + half_width))


def _mean(values: list[float]) -> float | None:
    return _rounded(sum(values) / len(values)) if values else None


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _row_sort_key(row: GoldLabelRecord) -> tuple[str, str, str]:
    return (
        row.metadata.get("system_name", ""),
        row.metadata.get("qid", ""),
        row.unit_id,
    )


def _dict_value(value: dict[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Gold manifest field {key} is not an object")
    return nested


def _load_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required performance input is missing: {path}")
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _guard_outputs(paths, *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"System-performance output already exists: {existing[0]}")
