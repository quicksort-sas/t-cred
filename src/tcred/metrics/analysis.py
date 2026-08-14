from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable

from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.statistics import (
    LABEL_SCORE,
    correlation_summary,
    mean_interval,
    paired_mean_difference,
    paired_spearman_difference,
)

ANSWER_METRICS = (
    "exact_match",
    "token_f1",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "pedants_match",
    "g_eval_answer_correctness",
    "ragchecker_precision",
    "ragchecker_recall",
    "ragchecker_f1",
    "tcred_judge_answer_correct",
    "tcred_answer_equivalence",
)
EVIDENCE_METRICS = (
    "ragchecker_faithfulness",
    "ragchecker_non_hallucination",
    "minicheck_retrieved_mean",
    "minicheck_retrieved_strict",
    "alignscore_retrieved",
    "tcred_judge_evidence_supports_answer",
    "tcred_semantic_attribution",
)
CITATION_TEXT_METRICS = (
    "required_citation_precision",
    "required_citation_recall",
    "citation_resolution_rate",
    "alce_citation_completeness",
    "alce_citation_precision",
    "minicheck_cited_mean",
    "minicheck_cited_strict",
    "alignscore_cited",
    "tcred_judge_citation_temporally_valid",
    "tcred_citation_precision",
    "tcred_citation_completeness",
    "tcred_citation_quality",
)

TARGET_METRICS = {
    "answer_correct": ANSWER_METRICS,
    "evidence_supports_answer": EVIDENCE_METRICS,
    "temporal_correct": ANSWER_METRICS
    + EVIDENCE_METRICS
    + (
        "tcred_judge_temporal_correct",
        "tcred_temporal_attribution",
        "tcred_temporal_correctness",
        "tcred_grounded_temporal_correctness",
    ),
    "citation_temporally_valid": CITATION_TEXT_METRICS,
    "graph_evidence_sufficient": (
        "tcred_judge_graph_evidence_sufficient",
        "tcred_graph_answer_coverage",
        "tcred_best_path_coherence",
    ),
    "response_decision_appropriate": (
        "tcred_judge_response_decision_appropriate",
        "tcred_response_decision",
    ),
}
UNSUPPORTED_HUMAN_TARGETS = (
    "graph_evidence_sufficient",
    "response_decision_appropriate",
)
DIAGNOSTIC_HUMAN_TARGETS = (
    "temporal_correct",
    "citation_temporally_valid",
)
PRIMARY_ANSWER_METRICS = (
    "token_f1",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "g_eval_answer_correctness",
    "ragchecker_f1",
    "tcred_judge_answer_correct",
    "tcred_answer_equivalence",
)
PRIMARY_EVIDENCE_METRICS = (
    "minicheck_retrieved_mean",
    "alignscore_retrieved",
    "ragchecker_faithfulness",
    "tcred_judge_evidence_supports_answer",
    "tcred_semantic_attribution",
)
PRIMARY_CITATION_METRICS = (
    "required_citation_precision",
    "required_citation_recall",
    "minicheck_cited_mean",
    "alignscore_cited",
    "alce_citation_completeness",
    "tcred_judge_citation_temporally_valid",
    "tcred_citation_quality",
)
PRIMARY_TEMPORAL_METRICS = (
    "pedants_probability",
    "g_eval_answer_correctness",
    "ragchecker_f1",
    "minicheck_retrieved_mean",
    "alignscore_retrieved",
    "tcred_judge_temporal_correct",
    "tcred_temporal_correctness",
    "tcred_grounded_temporal_correctness",
    "tcred_temporal_attribution",
)

DIRECT_TASK_METRICS = {
    field: f"tcred_judge_{field}"
    for field in (
        "answer_correct",
        "temporal_correct",
        "evidence_supports_answer",
        "citation_temporally_valid",
        "graph_evidence_sufficient",
        "response_decision_appropriate",
    )
}

# Independently documented before T-CRED suite development. The immutable primary gold remains
# unchanged; these exclusions are used only for a named field-level sensitivity analysis.
PREEXISTING_GOLD_ANOMALY_FIELDS = {
    "heu_766c85bb0a247b84dc64": {
        "evidence_supports_answer",
        "graph_evidence_sufficient",
    },
    "heu_98f2dd90833923f2512e": {"evidence_supports_answer"},
}


def analyze_metric_scores(
    records: list[MetricScoreRecord],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260813,
) -> dict[str, object]:
    human = [record for record in records if record.population == "human_gold"]
    human_system = [record for record in human if record.source_kind == "system_output"]
    full = [record for record in records if record.population == "system_full"]
    return {
        "schema_version": "1.5",
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "rng": "NumPy PCG64 via default_rng",
            "unit": (
                "dataset-family + qid cluster for human/pooled estimates; paired "
                "dataset-family + qid for full-system contrasts"
            ),
        },
        "coverage": _coverage(records),
        "human_gold_all": _population_analysis(
            human,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "human_gold_system_outputs": _population_analysis(
            human_system,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "human_metric_correlation_differences": _metric_correlation_differences(
            human,
            field="answer_correct",
            metric_names=PRIMARY_ANSWER_METRICS,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 4,
        ),
        "human_system_metric_correlation_differences": _metric_correlation_differences(
            human_system,
            field="answer_correct",
            metric_names=PRIMARY_ANSWER_METRICS,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 5,
        ),
        "human_evidence_metric_correlation_differences": _metric_correlation_differences(
            human,
            field="evidence_supports_answer",
            metric_names=PRIMARY_EVIDENCE_METRICS,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 6,
        ),
        "human_citation_metric_correlation_differences": _metric_correlation_differences(
            human,
            field="citation_temporally_valid",
            metric_names=PRIMARY_CITATION_METRICS,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 7,
        ),
        "human_temporal_metric_correlation_differences": _metric_correlation_differences(
            human,
            field="temporal_correct",
            metric_names=PRIMARY_TEMPORAL_METRICS,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 8,
        ),
        "human_gold_agreement_only": _population_analysis(
            _with_filtered_gold_labels(
                human,
                lambda provenance: provenance.get("resolution_method") == "annotator_agreement",
            ),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 2,
        ),
        "human_gold_without_medium_confidence_adjudication": _population_analysis(
            _with_filtered_gold_labels(
                human,
                lambda provenance: (
                    not (
                        provenance.get("resolution_method") == "adjudication"
                        and provenance.get("adjudication_confidence") == "medium"
                    )
                ),
            ),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 3,
        ),
        "human_gold_preexisting_anomaly_sensitivity": _population_analysis(
            _without_preexisting_anomaly_fields(human),
            bootstrap_samples=bootstrap_samples,
            seed=seed + 9,
        ),
        "human_by_system": _group_analysis(
            human_system,
            key=lambda record: record.system_name or "unknown",
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10,
        ),
        "human_by_source_kind": _group_analysis(
            human,
            key=lambda record: record.source_kind,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 15,
        ),
        "human_by_dataset": _group_analysis(
            human,
            key=lambda record: record.dataset_family,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 20,
        ),
        "human_system_outputs_by_dataset": _group_analysis(
            human_system,
            key=lambda record: record.dataset_family,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 30,
        ),
        "full_by_system": _group_analysis(
            full,
            key=lambda record: record.system_name or "unknown",
            bootstrap_samples=bootstrap_samples,
            seed=seed + 40,
        ),
        "full_by_dataset": _group_analysis(
            full,
            key=lambda record: record.dataset_family,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 50,
        ),
        "full_by_dataset_and_system": _group_analysis(
            full,
            key=lambda record: f"{record.dataset_family} / {record.system_name}",
            bootstrap_samples=bootstrap_samples,
            seed=seed + 60,
        ),
        "full_paired_system_differences": _paired_system_differences(
            full,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 70,
        ),
        "full_paired_system_differences_by_dataset": {
            family: _paired_system_differences(
                [record for record in full if record.dataset_family == family],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 80 + _stable_offset(family),
            )
            for family in sorted({record.dataset_family for record in full})
        },
        "human_target_coverage": _human_target_coverage(human),
    }


def _population_analysis(
    records: list[MetricScoreRecord],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    return {
        "units": len(records),
        "unique_qids": len({_cluster_id(record) for record in records}),
        "human_label_summary": _human_label_summary(
            records,
            samples=bootstrap_samples,
            seed=seed + 2,
        ),
        "metric_means": _metric_means(records, samples=bootstrap_samples, seed=seed),
        "human_correlations": _human_correlations(
            records,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
    }


def _human_label_summary(
    records: list[MetricScoreRecord],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    fields = sorted({field for record in records for field in record.gold_labels})
    output: dict[str, object] = {}
    for field in fields:
        selected = [record for record in records if record.gold_labels.get(field) in LABEL_SCORE]
        labels = [record.gold_labels[field] for record in selected]
        output[field] = {
            "n": len(selected),
            "counts": {
                label: sum(value == label for value in labels) for label in ("yes", "partial", "no")
            },
            "ordinal_mean": mean_interval(
                [LABEL_SCORE[label] for label in labels],
                [_cluster_id(record) for record in selected],
                samples=samples,
                seed=seed + _stable_offset(f"{field}:ordinal"),
            ),
            "strict_yes_rate": mean_interval(
                [float(label == "yes") for label in labels],
                [_cluster_id(record) for record in selected],
                samples=samples,
                seed=seed + _stable_offset(f"{field}:strict_yes"),
            ),
        }
    return output


def _with_filtered_gold_labels(
    records: list[MetricScoreRecord],
    predicate: Callable[[dict[str, object]], bool],
) -> list[MetricScoreRecord]:
    filtered: list[MetricScoreRecord] = []
    for record in records:
        labels = {
            field: label
            for field, label in record.gold_labels.items()
            if predicate(record.gold_provenance.get(field, {}))
        }
        filtered.append(record.model_copy(update={"gold_labels": labels}))
    return filtered


def _without_preexisting_anomaly_fields(
    records: list[MetricScoreRecord],
) -> list[MetricScoreRecord]:
    filtered: list[MetricScoreRecord] = []
    for record in records:
        excluded = PREEXISTING_GOLD_ANOMALY_FIELDS.get(record.unit_id or "", set())
        labels = {
            field: label
            for field, label in record.gold_labels.items()
            if field not in excluded
        }
        filtered.append(record.model_copy(update={"gold_labels": labels}))
    return filtered


def _human_target_coverage(records: list[MetricScoreRecord]) -> dict[str, object]:
    all_fields = sorted({field for record in records for field in record.gold_labels})
    output: dict[str, object] = {}
    for field in all_fields:
        available = [
            metric
            for metric in TARGET_METRICS.get(field, ())
            if any(record.scores.get(metric) is not None for record in records)
        ]
        direct_metric = DIRECT_TASK_METRICS.get(field)
        if direct_metric in available:
            status = "covered"
            reason = "The task-matched judge emits this field directly under the human rubric."
        elif field in DIAGNOSTIC_HUMAN_TARGETS and available:
            status = "diagnostic_only"
            reason = (
                "The selected metrics can be correlated with this label, but they do not directly "
                "validate evidence or citation valid-time."
            )
        elif available:
            status = "covered"
            reason = ""
        else:
            status = "not_covered"
            reason = (
                "No selected off-the-shelf metric measures this field without changing its "
                "construct. It is reserved for a future task-specific metric."
                if field in UNSUPPORTED_HUMAN_TARGETS
                else ""
            )
        output[field] = {
            "gold_label_count": sum(field in record.gold_labels for record in records),
            "automatic_metrics": available,
            "status": status,
            "reason": reason,
        }
    return output


def _group_analysis(
    records: list[MetricScoreRecord],
    *,
    key: Callable[[MetricScoreRecord], str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    groups: defaultdict[str, list[MetricScoreRecord]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return {
        name: _population_analysis(
            group,
            bootstrap_samples=bootstrap_samples,
            seed=seed + _stable_offset(name),
        )
        for name, group in sorted(groups.items())
    }


def _metric_means(
    records: list[MetricScoreRecord],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    names = sorted({name for record in records for name in record.scores})
    output: dict[str, object] = {}
    for name in names:
        selected = [record for record in records if record.scores.get(name) is not None]
        output[name] = mean_interval(
            [float(record.scores[name]) for record in selected],
            [_cluster_id(record) for record in selected],
            samples=samples,
            seed=seed + _stable_offset(name),
        )
    return output


def _human_correlations(
    records: list[MetricScoreRecord],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for field, metric_names in TARGET_METRICS.items():
        field_results: dict[str, object] = {}
        for metric_name in metric_names:
            selected = [
                record
                for record in records
                if record.gold_labels.get(field) in LABEL_SCORE
                and record.scores.get(metric_name) is not None
            ]
            if not selected:
                continue
            field_results[metric_name] = correlation_summary(
                [float(record.scores[metric_name]) for record in selected],
                [LABEL_SCORE[record.gold_labels[field]] for record in selected],
                [_cluster_id(record) for record in selected],
                samples=samples,
                seed=seed + _stable_offset(f"{field}:{metric_name}"),
            )
        output[field] = field_results
    return output


def _paired_system_differences(
    records: list[MetricScoreRecord],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    systems = sorted({record.system_name for record in records if record.system_name})
    names = sorted({name for record in records for name in record.scores})
    by_system = {
        system: {
            f"{record.dataset_family}:{record.qid}": record
            for record in records
            if record.system_name == system
        }
        for system in systems
    }
    output: dict[str, object] = {}
    for left_index, left in enumerate(systems):
        for right in systems[left_index + 1 :]:
            pair_name = f"{left} - {right}"
            metrics: dict[str, object] = {}
            for metric_name in names:
                left_scores = {
                    qid: float(record.scores[metric_name])
                    for qid, record in by_system[left].items()
                    if record.scores.get(metric_name) is not None
                }
                right_scores = {
                    qid: float(record.scores[metric_name])
                    for qid, record in by_system[right].items()
                    if record.scores.get(metric_name) is not None
                }
                metrics[metric_name] = paired_mean_difference(
                    left_scores,
                    right_scores,
                    samples=bootstrap_samples,
                    seed=seed + _stable_offset(f"{pair_name}:{metric_name}"),
                )
            output[pair_name] = metrics
    return output


def _metric_correlation_differences(
    records: list[MetricScoreRecord],
    *,
    field: str,
    metric_names: tuple[str, ...],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    available = tuple(
        metric
        for metric in metric_names
        if any(record.scores.get(metric) is not None for record in records)
    )
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            selected = [
                record
                for record in records
                if record.gold_labels.get(field) in LABEL_SCORE
                and record.scores.get(left) is not None
                and record.scores.get(right) is not None
            ]
            pair = f"{left} - {right}"
            output[pair] = paired_spearman_difference(
                [float(record.scores[left]) for record in selected],
                [float(record.scores[right]) for record in selected],
                [LABEL_SCORE[record.gold_labels[field]] for record in selected],
                [_cluster_id(record) for record in selected],
                samples=bootstrap_samples,
                seed=seed + _stable_offset(pair),
            )
    return output


def _coverage(records: list[MetricScoreRecord]) -> dict[str, object]:
    by_population: defaultdict[str, int] = defaultdict(int)
    by_population_system: defaultdict[str, int] = defaultdict(int)
    by_population_dataset: defaultdict[str, int] = defaultdict(int)
    for record in records:
        by_population[record.population] += 1
        by_population_system[f"{record.population} | {record.system_name or 'none'}"] += 1
        by_population_dataset[f"{record.population} | {record.dataset_family}"] += 1
    return {
        "total": len(records),
        "score_names": sorted({name for record in records for name in record.scores}),
        "by_population": dict(sorted(by_population.items())),
        "by_population_and_system": dict(sorted(by_population_system.items())),
        "by_population_and_dataset": dict(sorted(by_population_dataset.items())),
    }


def _stable_offset(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _cluster_id(record: MetricScoreRecord) -> str:
    # Keep qid first so a collision-free release retains the exact seeded bootstrap order.
    return f"{record.qid}\x1f{record.dataset_family}"
