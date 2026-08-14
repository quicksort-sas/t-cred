from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import orjson

from tcred.dataset.io import read_jsonl
from tcred.metrics.analysis import analyze_metric_scores
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.statistics import LABEL_SCORE, correlation_summary, mean_interval
from tcred.metrics.task_judge_models import (
    JUDGED_FIELDS,
    PromptSelection,
    PromptVariant,
    StageCacheRecord,
    TaskJudgeInput,
    TaskJudgeRecord,
)

FIELD_METRICS = {field: f"tcred_judge_{field}" for field in JUDGED_FIELDS}
CLASS_LABELS = ("yes", "partial", "no", "unjudgeable")
SELECTION_RULE = (
    "Highest calibration macro-average of per-field macro-F1 over gold-supported classes; "
    "then highest macro exact agreement; then fewer billed tokens; then lexical "
    "prompt-variant name."
)


def select_prompt_variant(
    rows: list[TaskJudgeInput],
    candidate_records: dict[PromptVariant, dict[str, TaskJudgeRecord]],
    *,
    calibration_ids: set[str],
    bootstrap_samples: int,
    seed: int,
    contract_version: str,
) -> tuple[PromptSelection, dict[str, object]]:
    selected_rows = [row for row in rows if row.metric_id in calibration_ids]
    candidates: dict[str, dict[str, object]] = {}
    detailed: dict[str, object] = {}
    for variant, records in candidate_records.items():
        summary = classification_analysis(
            selected_rows,
            records,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        detailed[variant] = summary
        field_summaries = [
            value
            for value in summary["fields"].values()
            if isinstance(value, dict) and int(value.get("n", 0)) >= 10
        ]
        macro_f1 = _mean(
            [
                float(value["macro_f1"])
                for value in field_summaries
                if value.get("macro_f1") is not None
            ]
        )
        macro_exact = _mean([float(value["exact_agreement"]) for value in field_summaries])
        usage = task_judge_usage(records)
        candidates[variant] = {
            "macro_field_f1": macro_f1,
            "macro_field_exact_agreement": macro_exact,
            "field_count": len(field_summaries),
            "stage_calls": usage["stage_calls"],
            "total_tokens": usage["total_tokens"],
        }
    ranking = sorted(
        candidates,
        key=lambda variant: (
            -_selection_value(candidates[variant]["macro_field_f1"]),
            -_selection_value(candidates[variant]["macro_field_exact_agreement"]),
            int(candidates[variant]["total_tokens"]),
            variant,
        ),
    )
    calibration_hash = hashlib.sha256(
        orjson.dumps(sorted(calibration_ids), option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
    selection = PromptSelection(
        contract_version=contract_version,
        selected_variant=ranking[0],  # type: ignore[arg-type]
        selection_rule=SELECTION_RULE,
        calibration_metric_ids_sha256=calibration_hash,
        candidates=candidates,
    )
    return selection, detailed


def classification_analysis(
    rows: list[TaskJudgeInput],
    records: dict[str, TaskJudgeRecord],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    fields: dict[str, object] = {}
    for offset, field in enumerate(JUDGED_FIELDS):
        selected = [row for row in rows if field in row.gold_labels and row.metric_id in records]
        if not selected:
            continue
        gold = [row.gold_labels[field] for row in selected]
        predicted = [records[row.metric_id].result.field(field).label for row in selected]
        confidence = [
            records[row.metric_id].result.field(field).confidence / 100 for row in selected
        ]
        if any(label == "not_applicable" for label in predicted):
            raise ValueError(f"Applicable gold field returned not_applicable: {field}")
        exact = [float(left == right) for left, right in zip(gold, predicted, strict=True)]
        clusters = [f"{row.dataset_family}:{row.qid}" for row in selected]
        confusion = {
            expected: {
                actual: sum(
                    left == expected and right == actual
                    for left, right in zip(gold, predicted, strict=True)
                )
                for actual in CLASS_LABELS
            }
            for expected in CLASS_LABELS
        }
        per_class = _per_class_metrics(gold, predicted)
        ordinal = [
            (LABEL_SCORE[right], LABEL_SCORE[left], cluster)
            for left, right, cluster in zip(gold, predicted, clusters, strict=True)
            if left in LABEL_SCORE and right in LABEL_SCORE
        ]
        correlation = correlation_summary(
            [item[0] for item in ordinal],
            [item[1] for item in ordinal],
            [item[2] for item in ordinal],
            samples=bootstrap_samples,
            seed=seed + offset,
        )
        supported_classes = [
            label for label, value in per_class.items() if int(value["support"]) > 0
        ]
        determinate_pairs = sum(
            left in LABEL_SCORE and right in LABEL_SCORE
            for left, right in zip(gold, predicted, strict=True)
        )
        fields[field] = {
            "n": len(selected),
            "unique_question_clusters": len(set(clusters)),
            "gold_distribution": dict(sorted(Counter(gold).items())),
            "predicted_distribution": dict(sorted(Counter(predicted).items())),
            "exact_agreement": sum(exact) / len(exact),
            "exact_agreement_ci95": mean_interval(
                exact,
                clusters,
                samples=bootstrap_samples,
                seed=seed + 100 + offset,
            )["ci95"],
            "macro_f1": _mean([per_class[label]["f1"] for label in supported_classes]),
            "macro_f1_classes": supported_classes,
            "per_class": per_class,
            "confusion_matrix": confusion,
            "cohen_kappa": _cohen_kappa(gold, predicted),
            "quadratic_weighted_kappa": _quadratic_weighted_kappa(gold, predicted),
            "quadratic_weighted_kappa_n": determinate_pairs,
            "ordinal_association": correlation,
            "confidence_calibration": _confidence_summary(confidence, exact),
        }
    return {
        "rows": len(rows),
        "question_clusters": len({(row.dataset_family, row.qid) for row in rows}),
        "fields": fields,
    }


def merge_with_baseline_scores(
    rows: list[TaskJudgeInput],
    judgments: dict[str, TaskJudgeRecord],
    *,
    non_llm_scores_path: Path,
    legacy_llm_scores_path: Path,
) -> list[MetricScoreRecord]:
    non_llm = _load_scores(non_llm_scores_path)
    legacy = _load_scores(legacy_llm_scores_path)
    expected = {row.metric_id for row in rows}
    if set(non_llm) != expected:
        raise ValueError("Non-LLM baseline score IDs do not match task-judge inputs")
    output: list[MetricScoreRecord] = []
    for row in rows:
        baseline = non_llm[row.metric_id]
        scores = dict(baseline.scores)
        legacy_record = legacy.get(row.metric_id)
        if legacy_record is not None:
            for name, value in legacy_record.scores.items():
                if name not in scores or scores[name] is None:
                    scores[name] = value
        judgment = judgments[row.metric_id]
        task_metadata: dict[str, object] = {}
        for field, metric_name in FIELD_METRICS.items():
            field_judgment = judgment.result.field(field)
            scores[metric_name] = LABEL_SCORE.get(field_judgment.label)
            task_metadata[field] = field_judgment.model_dump(mode="json")
        metadata = dict(baseline.metric_metadata)
        metadata["tcred_task_judge"] = {
            "provider": judgment.provider,
            "model": judgment.model,
            "prompt_variant": judgment.prompt_variant,
            "fields": task_metadata,
        }
        output.append(
            baseline.model_copy(
                update={
                    "scores": scores,
                    "metric_metadata": metadata,
                }
            )
        )
    return output


def complete_comparison_analysis(
    score_records: list[MetricScoreRecord],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    return analyze_metric_scores(
        score_records,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def stability_analysis(
    rows: list[TaskJudgeInput],
    primary: dict[str, TaskJudgeRecord],
    repeated: dict[str, TaskJudgeRecord],
) -> dict[str, object]:
    fields: dict[str, object] = {}
    for field in JUDGED_FIELDS:
        selected = [
            row
            for row in rows
            if row.metric_id in primary
            and row.metric_id in repeated
            and field in row.applicable_fields
        ]
        agreements = [
            primary[row.metric_id].result.field(field).label
            == repeated[row.metric_id].result.field(field).label
            for row in selected
        ]
        confidence_differences = [
            abs(
                primary[row.metric_id].result.field(field).confidence
                - repeated[row.metric_id].result.field(field).confidence
            )
            for row in selected
        ]
        fields[field] = {
            "n": len(selected),
            "label_exact_agreement": sum(agreements) / len(agreements) if agreements else None,
            "mean_absolute_confidence_difference": _mean(confidence_differences),
        }
    return {"rows": len(rows), "fields": fields}


def task_judge_usage(records: dict[str, TaskJudgeRecord]) -> dict[str, int]:
    unique: dict[str, object] = {}
    for record in records.values():
        unique[record.answer_stage.judgment_id] = record.answer_stage
        if record.evidence_stage is not None:
            unique[record.evidence_stage.judgment_id] = record.evidence_stage
    names = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    return {
        "stage_calls": len(unique),
        "request_attempts_for_accepted_records": sum(stage.attempts for stage in unique.values()),
        **{name: sum(stage.usage.get(name, 0) for stage in unique.values()) for name in names},
        "retried_stage_calls": sum(stage.attempts > 1 for stage in unique.values()),
    }


def support_pointer_analysis(records: dict[str, TaskJudgeRecord]) -> dict[str, object]:
    unique: dict[str, StageCacheRecord] = {}
    for record in records.values():
        unique[record.answer_stage.judgment_id] = record.answer_stage
        if record.evidence_stage is not None:
            unique[record.evidence_stage.judgment_id] = record.evidence_stage
    by_field: Counter[str] = Counter()
    unknown_evidence = 0
    unknown_paths = 0
    calls_with_warnings = 0
    for stage in unique.values():
        if stage.support_pointer_warnings:
            calls_with_warnings += 1
        for field, warning in stage.support_pointer_warnings.items():
            by_field[field] += 1
            unknown_evidence += len(warning.get("unknown_evidence_ids", []))
            unknown_paths += len(warning.get("unknown_path_ids", []))
    return {
        "stage_calls": len(unique),
        "calls_with_pointer_warnings": calls_with_warnings,
        "call_warning_rate": calls_with_warnings / len(unique) if unique else None,
        "fields_with_pointer_warnings": dict(sorted(by_field.items())),
        "unknown_evidence_id_occurrences": unknown_evidence,
        "unknown_path_id_occurrences": unknown_paths,
        "interpretation": (
            "Support pointers are auxiliary audit outputs. Unknown pointers do not alter the "
            "categorical metric label and remain explicit warnings."
        ),
    }


def full_judgment_distribution_analysis(
    rows: list[TaskJudgeInput],
    records: dict[str, TaskJudgeRecord],
) -> dict[str, object]:
    """Expose categorical coverage so unjudgeable outputs cannot disappear behind means."""

    full = [row for row in rows if row.population == "system_full"]
    return {
        "overall": _judgment_distribution(full, records),
        "by_system": {
            system: _judgment_distribution(
                [row for row in full if (row.system_name or "unknown") == system],
                records,
            )
            for system in sorted({row.system_name or "unknown" for row in full})
        },
        "by_dataset": {
            family: _judgment_distribution(
                [row for row in full if row.dataset_family == family],
                records,
            )
            for family in sorted({row.dataset_family for row in full})
        },
        "by_dataset_and_system": {
            group: _judgment_distribution(
                [
                    row
                    for row in full
                    if f"{row.dataset_family} / {row.system_name or 'unknown'}" == group
                ],
                records,
            )
            for group in sorted(
                {f"{row.dataset_family} / {row.system_name or 'unknown'}" for row in full}
            )
        },
    }


def _judgment_distribution(
    rows: list[TaskJudgeInput],
    records: dict[str, TaskJudgeRecord],
) -> dict[str, object]:
    fields: dict[str, object] = {}
    for field in JUDGED_FIELDS:
        applicable = [row for row in rows if field in row.applicable_fields]
        labels = [records[row.metric_id].result.field(field).label for row in applicable]
        invalid = sorted(set(labels) - set(CLASS_LABELS))
        if invalid:
            raise ValueError(f"Applicable field {field} has invalid labels: {invalid}")
        counts = Counter(labels)
        determinate = sum(counts[label] for label in LABEL_SCORE)
        numerator = counts["yes"] + 0.5 * counts["partial"]
        total = len(applicable)
        fields[field] = {
            "applicable": total,
            "label_counts": {label: counts[label] for label in CLASS_LABELS},
            "determinate": determinate,
            "determinate_coverage": determinate / total if total else None,
            "determinate_mean": numerator / determinate if determinate else None,
            "unjudgeable_as_no_lower_bound": numerator / total if total else None,
            "unjudgeable_as_yes_upper_bound": (
                (numerator + counts["unjudgeable"]) / total if total else None
            ),
        }
    return {"responses": len(rows), "fields": fields}


def combined_task_judge_usage(
    record_sets: list[dict[str, TaskJudgeRecord]],
) -> dict[str, int]:
    """Count unique accepted stage records across prompt variants and random seeds."""

    unique: dict[tuple[str, str, int, str, str], object] = {}
    for records in record_sets:
        for record in records.values():
            stages = [record.answer_stage]
            if record.evidence_stage is not None:
                stages.append(record.evidence_stage)
            for stage in stages:
                key = (
                    stage.provider,
                    stage.model,
                    stage.random_seed,
                    stage.prompt_variant,
                    stage.judgment_id,
                )
                unique[key] = stage
    names = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    return {
        "stage_calls": len(unique),
        "request_attempts_for_accepted_records": sum(stage.attempts for stage in unique.values()),
        **{name: sum(stage.usage.get(name, 0) for stage in unique.values()) for name in names},
        "retried_stage_calls": sum(stage.attempts > 1 for stage in unique.values()),
    }


def write_score_records(records: list[MetricScoreRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for record in records:
            stream.write(orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def write_disagreement_records(
    rows: list[TaskJudgeInput],
    records: dict[str, TaskJudgeRecord],
    path: Path,
) -> int:
    """Write one standalone audit record for every gold-versus-judge field mismatch."""

    output: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item.metric_id):
        record = records[row.metric_id]
        for field in JUDGED_FIELDS:
            gold = row.gold_labels.get(field)
            if gold is None:
                continue
            predicted = record.result.field(field)
            if predicted.label == gold:
                continue
            stage_record = (
                record.answer_stage
                if field in {"answer_correct", "response_decision_appropriate"}
                else record.evidence_stage
            )
            output.append(
                {
                    "metric_id": row.metric_id,
                    "unit_id": row.unit_id,
                    "dataset_family": row.dataset_family,
                    "source_kind": row.source_kind,
                    "system_name": row.system_name,
                    "qid": row.qid,
                    "scenario_id": row.scenario_id,
                    "field": field,
                    "gold_label": gold,
                    "gold_provenance": row.gold_provenance.get(field, {}),
                    "predicted": predicted.model_dump(mode="json"),
                    "support_pointer_warnings": (
                        stage_record.support_pointer_warnings.get(field, {})
                        if stage_record is not None
                        else {}
                    ),
                    "question": row.question,
                    "reference_answer": row.reference_answer,
                    "candidate_answer": row.candidate_answer,
                    "context_note": row.context_note,
                    "displayed_evidence": [
                        item.model_dump(mode="json") for item in row.displayed_evidence()
                    ],
                    "graph_paths": [item.model_dump(mode="json") for item in row.graph_paths],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for item in output:
            stream.write(orjson.dumps(item, option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)
    return len(output)


def _load_scores(path: Path) -> dict[str, MetricScoreRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline metric scores: {path}")
    records = [MetricScoreRecord.model_validate(raw) for raw in read_jsonl(path)]
    output = {record.metric_id: record for record in records}
    if len(output) != len(records):
        raise ValueError(f"Duplicate metric IDs in baseline score file: {path}")
    return output


def _per_class_metrics(gold: list[str], predicted: list[str]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for label in CLASS_LABELS:
        true_positive = sum(
            left == label and right == label for left, right in zip(gold, predicted, strict=True)
        )
        false_positive = sum(
            left != label and right == label for left, right in zip(gold, predicted, strict=True)
        )
        false_negative = sum(
            left == label and right != label for left, right in zip(gold, predicted, strict=True)
        )
        support = sum(left == label for left in gold)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return output


def _cohen_kappa(gold: list[str], predicted: list[str]) -> float | None:
    if not gold:
        return None
    observed = sum(left == right for left, right in zip(gold, predicted, strict=True)) / len(gold)
    gold_counts = Counter(gold)
    predicted_counts = Counter(predicted)
    expected = sum(gold_counts[label] * predicted_counts[label] for label in CLASS_LABELS) / (
        len(gold) ** 2
    )
    return (observed - expected) / (1 - expected) if expected < 1 else None


def _quadratic_weighted_kappa(gold: list[str], predicted: list[str]) -> float | None:
    """Quadratic kappa for determinate ordinal labels; unjudgeable is nominal and excluded."""

    order = {"no": 0, "partial": 1, "yes": 2}
    pairs = [
        (order[left], order[right])
        for left, right in zip(gold, predicted, strict=True)
        if left in order and right in order
    ]
    if not pairs:
        return None
    gold_counts = Counter(left for left, _right in pairs)
    predicted_counts = Counter(right for _left, right in pairs)
    denominator_scale = (len(order) - 1) ** 2
    observed_disagreement = sum(
        ((left - right) ** 2) / denominator_scale for left, right in pairs
    ) / len(pairs)
    expected_disagreement = sum(
        (gold_counts[left] / len(pairs))
        * (predicted_counts[right] / len(pairs))
        * (((left - right) ** 2) / denominator_scale)
        for left in order.values()
        for right in order.values()
    )
    return 1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else None


def _confidence_summary(confidence: list[float], exact: list[float]) -> dict[str, object]:
    brier = _mean(
        [
            (probability - outcome) ** 2
            for probability, outcome in zip(confidence, exact, strict=True)
        ]
    )
    thresholds = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    risk_coverage = []
    for threshold in thresholds:
        accepted = [
            outcome
            for probability, outcome in zip(confidence, exact, strict=True)
            if probability >= threshold
        ]
        risk_coverage.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": len(accepted) / len(exact) if exact else None,
                "error_rate": 1 - sum(accepted) / len(accepted) if accepted else None,
            }
        )
    return {
        "mean_confidence": _mean(confidence),
        "brier_score_for_exact_label": brier,
        "risk_coverage": risk_coverage,
    }


def _mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _selection_value(value: object) -> float:
    return float(value) if value is not None else -1.0
