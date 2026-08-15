from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean

from tcred.dataset.io import load_bundle
from tcred.metrics.analysis import TARGET_METRICS
from tcred.metrics.deterministic import reference_answer_scores
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.statistics import LABEL_SCORE

PRIMARY_FIELD_COMPONENT = {
    "answer_correct": "tcred_answer_equivalence",
    "temporal_correct": "tcred_temporal_correctness",
    "evidence_supports_answer": "tcred_semantic_attribution",
    "citation_temporally_valid": "tcred_citation_quality",
    "graph_evidence_sufficient": "tcred_graph_answer_coverage",
    "response_decision_appropriate": "tcred_response_decision",
}
_UPDATE_METRICS = (
    "exact_match",
    "token_f1",
    "rouge_l",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "g_eval_answer_correctness",
    "ragchecker_f1",
    "tcred_judge_answer_correct",
    "tcred_answer_equivalence",
    "tcred_temporal_correctness",
    "tcred_response_decision",
)
_SNAPSHOT_NUMBER = re.compile(r"(\d+)$")
_UPDATE_PASS_THRESHOLDS = (0.5, 0.75)


def analyze_human_calibration(records: Sequence[MetricScoreRecord]) -> dict[str, object]:
    """Measure bounded calibration and selective-risk behavior against available human gold."""

    human = [row for row in records if row.population == "human_gold"]
    fields: dict[str, object] = {}
    for field, metric_names in TARGET_METRICS.items():
        metrics: dict[str, object] = {}
        for metric_name in metric_names:
            selected = [
                row
                for row in human
                if row.gold_labels.get(field) in LABEL_SCORE
                and row.scores.get(metric_name) is not None
            ]
            if not selected:
                continue
            scores = [float(row.scores[metric_name]) for row in selected]
            labels = [LABEL_SCORE[row.gold_labels[field]] for row in selected]
            metrics[metric_name] = _calibration_summary(scores, labels)
        fields[field] = {
            "primary_tcred_component": PRIMARY_FIELD_COMPONENT.get(field),
            "metrics": metrics,
        }
    return {
        "interpretation": (
            "Calibration is exploratory because metric outputs are not trained probabilities. "
            "Partial labels are retained as ordinal targets at 0.5. Selective risk tests ranking, "
            "not causal confidence calibration."
        ),
        "human_units": len(human),
        "fields": fields,
    }


def analyze_update_stability(
    records: Sequence[MetricScoreRecord],
    *,
    dataset_root: Path,
    candidate_by_metric_id: dict[str, str],
) -> dict[str, object]:
    """Evaluate paired QA outputs over adjacent snapshots in each semantic series."""

    full = [row for row in records if row.population == "system_full" and row.system_name]
    by_output = {
        (row.dataset_family, str(row.system_name), row.qid): row
        for row in full
    }
    observations: list[dict[str, object]] = []
    for family in sorted({row.dataset_family for row in full}):
        bundle = load_bundle(dataset_root / family)
        by_series: defaultdict[str, list[object]] = defaultdict(list)
        for question in bundle.questions:
            if question.semantic_series_id:
                by_series[question.semantic_series_id].append(question)
        systems = sorted(
            {
                str(row.system_name)
                for row in full
                if row.dataset_family == family and row.system_name
            }
        )
        for series_id, questions in by_series.items():
            ordered = sorted(
                questions,
                key=lambda row: (_snapshot_order(row.program.snapshot_id), row.qid),
            )
            for before_question, after_question in zip(ordered, ordered[1:], strict=False):
                before_reference = _reference_signature(before_question)
                after_reference = _reference_signature(after_question)
                reference_changed = before_reference != after_reference
                before_entities = set(before_question.gold_answer_entity_ids)
                after_entities = set(after_question.gold_answer_entity_ids)
                disjoint_change = bool(
                    reference_changed
                    and before_entities
                    and after_entities
                    and before_entities.isdisjoint(after_entities)
                )
                for system in systems:
                    before = by_output.get((family, system, before_question.qid))
                    after = by_output.get((family, system, after_question.qid))
                    if before is None or after is None:
                        continue
                    surface_equivalence = reference_answer_scores(
                        candidate_by_metric_id[after.metric_id],
                        candidate_by_metric_id[before.metric_id],
                    )["token_f1"]
                    for metric_name in _UPDATE_METRICS:
                        before_score = before.scores.get(metric_name)
                        after_score = after.scores.get(metric_name)
                        if before_score is None or after_score is None:
                            continue
                        first = float(before_score)
                        second = float(after_score)
                        stability = max(0.0, 1.0 - abs(first - second))
                        continuous = (
                            min(first, second)
                            if reference_changed
                            else min(first, second, stability)
                        )
                        pass_by_threshold = {
                            _threshold_key(threshold): (
                                first >= threshold
                                and second >= threshold
                                and (reference_changed or abs(first - second) <= 0.05)
                            )
                            for threshold in _UPDATE_PASS_THRESHOLDS
                        }
                        observations.append(
                            {
                                "dataset_family": family,
                                "system_name": system,
                                "series_id": series_id,
                                "before_qid": before_question.qid,
                                "after_qid": after_question.qid,
                                "reference_changed": reference_changed,
                                "disjoint_change": disjoint_change,
                                "metric": metric_name,
                                "before_score": first,
                                "after_score": second,
                                "continuous_score": continuous,
                                "pass_by_threshold": pass_by_threshold,
                                "surface_equivalence": surface_equivalence,
                            }
                        )

    return {
        "definition": {
            "update_stable_score": (
                "The minimum of the two snapshot-local scores and one minus their absolute "
                "difference when the formal answer is unchanged."
            ),
            "update_adaptation_score": (
                "The minimum of the two snapshot-local scores when the formal answer changes."
            ),
            "pass_rates": (
                "Sensitivity analyses at score thresholds 0.50 and 0.75. A stable pair also "
                "requires an absolute score difference of at most 0.05. These are bounded-score "
                "operating points, not calibrated accuracy estimates."
            ),
            "update_flip_score": (
                "The adaptation score restricted to updates whose before/after gold entity sets "
                "are disjoint; this makes semantic answer change identifiable without a "
                "candidate-to-candidate paraphrase judge."
            ),
            "surface_equivalence": (
                "Normalized token F1 between outputs, reported only as a style-sensitive "
                "diagnostic and never used as the semantic update criterion."
            ),
        },
        "pair_observations": len(observations),
        "by_system": _group_update_observations(observations, keys=("system_name",)),
        "by_dataset_and_system": _group_update_observations(
            observations,
            keys=("dataset_family", "system_name"),
        ),
        "observations": observations,
    }


def _calibration_summary(scores: Sequence[float], labels: Sequence[float]) -> dict[str, object]:
    n = len(scores)
    errors = [score - label for score, label in zip(scores, labels, strict=True)]
    binary = [
        (score, label)
        for score, label in zip(scores, labels, strict=True)
        if label in {0.0, 1.0}
    ]
    aurc = _aurc(scores, labels)
    oracle_aurc = _aurc(labels, labels)
    random_risk = mean(1.0 - label for label in labels)
    random_aurc = random_risk
    denominator = random_aurc - oracle_aurc
    normalized_excess = (
        (aurc - oracle_aurc) / denominator if denominator > 1e-12 else None
    )
    return {
        "n": n,
        "brier": mean(error * error for error in errors),
        "mae": mean(abs(error) for error in errors),
        "ece_equal_count": _ece_equal_count(scores, labels),
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "normalized_excess_aurc": normalized_excess,
        "selective_risk_at_50pct": _selective_risk(scores, labels, 0.5),
        "selective_risk_at_80pct": _selective_risk(scores, labels, 0.8),
        "binary_n_excluding_partial": len(binary),
        "binary_accuracy_at_0_5": (
            mean(float((score >= 0.5) == bool(label)) for score, label in binary)
            if binary
            else None
        ),
    }


def _aurc(scores: Sequence[float], labels: Sequence[float]) -> float:
    by_score: defaultdict[float, list[float]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=True):
        by_score[score].append(1.0 - label)
    cumulative_error = 0.0
    cumulative_count = 0
    risks: list[float] = []
    for score in sorted(by_score, reverse=True):
        errors = by_score[score]
        block_error = sum(errors)
        block_size = len(errors)
        for offset in range(1, block_size + 1):
            expected_error = cumulative_error + offset * block_error / block_size
            risks.append(expected_error / (cumulative_count + offset))
        cumulative_error += block_error
        cumulative_count += block_size
    return mean(risks)


def _selective_risk(
    scores: Sequence[float],
    labels: Sequence[float],
    coverage: float,
) -> float:
    count = max(1, math.ceil(len(scores) * coverage))
    by_score: defaultdict[float, list[float]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=True):
        by_score[score].append(1.0 - label)
    selected_error = 0.0
    selected_count = 0
    for score in sorted(by_score, reverse=True):
        errors = by_score[score]
        needed = min(len(errors), count - selected_count)
        selected_error += needed * mean(errors)
        selected_count += needed
        if selected_count == count:
            break
    return selected_error / selected_count


def _ece_equal_count(scores: Sequence[float], labels: Sequence[float]) -> float:
    n = len(scores)
    bin_count = min(10, max(1, n // 10))
    ordered = sorted(zip(scores, labels, strict=True), key=lambda row: row[0])
    weighted = 0.0
    for index in range(bin_count):
        start = index * n // bin_count
        end = (index + 1) * n // bin_count
        rows = ordered[start:end]
        if not rows:
            continue
        weighted += len(rows) / n * abs(
            mean(score for score, _label in rows)
            - mean(label for _score, label in rows)
        )
    return weighted


def _group_update_observations(
    observations: Iterable[dict[str, object]],
    *,
    keys: tuple[str, ...],
) -> dict[str, object]:
    groups: defaultdict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output: dict[str, object] = {}
    for group, rows in sorted(groups.items()):
        metric_groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            metric_groups[str(row["metric"])].append(row)
        output[" / ".join(group)] = {
            metric: _update_metric_summary(metric_rows)
            for metric, metric_rows in sorted(metric_groups.items())
        }
    return output


def _update_metric_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    stable = [row for row in rows if not bool(row["reference_changed"])]
    changed = [row for row in rows if bool(row["reference_changed"])]
    disjoint = [row for row in changed if bool(row["disjoint_change"])]
    summary: dict[str, object] = {
        "stable_pairs": len(stable),
        "update_stable_score": _mean_float(stable, "continuous_score"),
        "affected_pairs": len(changed),
        "update_adaptation_score": _mean_float(changed, "continuous_score"),
        "disjoint_flip_pairs": len(disjoint),
        "update_flip_score": _mean_float(disjoint, "continuous_score"),
        "mean_surface_equivalence_stable": _mean_float(stable, "surface_equivalence"),
        "mean_surface_equivalence_changed": _mean_float(changed, "surface_equivalence"),
    }
    for threshold in _UPDATE_PASS_THRESHOLDS:
        suffix = _threshold_key(threshold)
        summary[f"update_stable_pass_at_{suffix}"] = _mean_threshold_pass(
            stable,
            suffix,
        )
        summary[f"update_adaptation_pass_at_{suffix}"] = _mean_threshold_pass(
            changed,
            suffix,
        )
        summary[f"update_flip_pass_at_{suffix}"] = _mean_threshold_pass(
            disjoint,
            suffix,
        )
    return summary


def _mean_threshold_pass(
    rows: Sequence[dict[str, object]],
    threshold_key: str,
) -> float | None:
    if not rows:
        return None
    return mean(
        float(bool(row["pass_by_threshold"][threshold_key]))  # type: ignore[index]
        for row in rows
    )


def _mean_float(rows: Sequence[dict[str, object]], key: str) -> float | None:
    return mean(float(row[key]) for row in rows) if rows else None


def _snapshot_order(snapshot_id: str) -> tuple[int, str]:
    match = _SNAPSHOT_NUMBER.search(snapshot_id)
    return (int(match.group(1)), snapshot_id) if match else (10**9, snapshot_id)


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def _reference_signature(question: object) -> tuple[bool, tuple[str, ...]]:
    return bool(question.should_abstain), tuple(sorted(question.gold_answer_entity_ids))
