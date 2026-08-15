from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median

import orjson

from tcred.dataset.io import load_bundle, read_jsonl
from tcred.human_eval.export import controlled_private_key
from tcred.metrics.diagnostic_models import (
    DiagnosticPair,
    DiagnosticSuite,
    diagnostic_inference_cluster_ids,
)
from tcred.metrics.models import MetricScoreRecord

_METRICS_BY_CONSTRUCT: dict[str, tuple[str, ...]] = {
    "answer_correctness": (
        "exact_match",
        "token_f1",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "bertscore_f1",
        "sas_cross_encoder",
        "pedants_probability",
        "g_eval_answer_correctness",
        "ragchecker_f1",
        "tcred_judge_answer_correct",
        "tcred_answer_equivalence",
    ),
    "temporal_correctness": (
        "token_f1",
        "bertscore_f1",
        "sas_cross_encoder",
        "pedants_probability",
        "g_eval_answer_correctness",
        "ragchecker_f1",
        "tcred_judge_answer_correct",
        "tcred_judge_temporal_correct",
        "tcred_temporal_correctness",
        "tcred_grounded_temporal_correctness",
    ),
    "temporal_attribution": (
        "minicheck_retrieved_mean",
        "minicheck_retrieved_strict",
        "alignscore_retrieved",
        "ragchecker_faithfulness",
        "tcred_judge_evidence_supports_answer",
        "tcred_judge_temporal_correct",
        "tcred_temporal_attribution",
        "tcred_ablation_temporal_attribution_no_time",
        "tcred_ablation_temporal_attribution_no_contradiction",
        "tcred_ablation_temporal_attribution_cross_evidence",
    ),
    "evidence_support": (
        "minicheck_retrieved_mean",
        "minicheck_retrieved_strict",
        "alignscore_retrieved",
        "ragchecker_faithfulness",
        "ragchecker_non_hallucination",
        "tcred_judge_evidence_supports_answer",
        "tcred_semantic_attribution",
    ),
    "citation_correctness": (
        "citation_presence",
        "citation_resolution_rate",
        "required_citation_precision",
        "required_citation_recall",
        "alce_citation_completeness",
        "alce_citation_precision",
        "minicheck_cited_mean",
        "minicheck_cited_strict",
        "alignscore_cited",
        "tcred_judge_citation_temporally_valid",
        "tcred_citation_precision",
        "tcred_citation_completeness",
        "tcred_citation_f1",
        "tcred_citation_quality",
        "tcred_ablation_citation_f1_no_time",
    ),
    "graph_sufficiency": (
        "tcred_judge_graph_evidence_sufficient",
        "tcred_graph_answer_coverage",
        "tcred_best_path_coherence",
        "tcred_mean_top3_path_coherence",
        "tcred_ablation_graph_answer_coverage_no_time",
        "tcred_ablation_graph_mean_top3_no_time",
    ),
    "response_decision": (
        "pedants_probability",
        "g_eval_answer_correctness",
        "ragchecker_f1",
        "tcred_judge_answer_correct",
        "tcred_judge_response_decision_appropriate",
        "tcred_response_decision",
    ),
    "retrieval_quality": (
        "retrieval_precision_at_10",
        "retrieval_recall_at_10",
        "retrieval_average_precision_at_10",
        "retrieval_mrr",
        "retrieval_r_precision",
        "retrieval_ndcg_at_10",
        "ragchecker_context_precision",
        "tcred_t_ndcg_at_10",
        "tcred_t_precision_at_10",
        "tcred_temporal_cleanliness_at_10",
        "tcred_ablation_retrieval_ndcg_no_time_at_10",
    ),
}


def analyze_diagnostic_suite(
    suite: DiagnosticSuite,
    score_records: list[MetricScoreRecord],
    *,
    dataset_root: Path,
    gold_dir: Path,
    bootstrap_samples: int = 2000,
    seed: int = 20260815,
) -> dict[str, object]:
    scores = {record.metric_id: record.scores for record in score_records}
    expected = {case.case_id for case in suite.cases}
    if set(scores) != expected:
        raise ValueError("Diagnostic score IDs do not match the frozen suite")
    available = sorted({name for values in scores.values() for name in values})
    inference_cluster_ids = diagnostic_inference_cluster_ids(suite.cases, suite.pairs)
    constructs: dict[str, object] = {}
    for construct in sorted(_METRICS_BY_CONSTRUCT):
        pairs = [pair for pair in suite.pairs if pair.target_construct == construct]
        metrics = [name for name in _METRICS_BY_CONSTRUCT[construct] if name in available]
        constructs[construct] = _analyze_construct(
            pairs,
            metrics=metrics,
            scores=scores,
            cluster_ids=inference_cluster_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    return {
        "schema_version": "1.0",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "suite_audit": suite.audit,
        "metric_count": len(available),
        "score_names": available,
        "constructs": constructs,
        "formal_oracle_human_audit": _formal_oracle_human_audit(
            dataset_root=dataset_root,
            gold_dir=gold_dir,
        ),
        "interpretation_contract": {
            "primary_directional_statistic": (
                "Macro-average across phenomena of pairwise utility: win=1, tie=0.5, reversal=0."
            ),
            "strict_consistency": (
                "BUMP-style success rate requiring score(good) > score(error); ties fail."
            ),
            "aces_tau_like": "2 * strict_consistency - 1; ties are discordant.",
            "discrimination": "Pair-instance ROC AUC with ties worth 0.5; no threshold is fit.",
            "invariance": (
                "Mean absolute change divided by the metric's theoretical score range; "
                "lower is better."
            ),
            "uncertainty": (
                "Percentile bootstrap over connected source-scenario components; pairwise "
                "metric tests use a paired component-level permutation test on the "
                "common-pair subset and Holm correction within each construct/test type."
            ),
        },
    }


def _analyze_construct(
    pairs: list[DiagnosticPair],
    *,
    metrics: list[str],
    scores: dict[str, dict[str, float | None]],
    cluster_ids: Mapping[str, str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    directional = [pair for pair in pairs if pair.test_type == "directional"]
    invariant = [pair for pair in pairs if pair.test_type == "invariance"]
    directional_metrics = {
        metric: _directional_summary(
            directional,
            metric=metric,
            scores=scores,
            cluster_ids=cluster_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for metric in metrics
    }
    invariance_metrics = {
        metric: _invariance_summary(
            invariant,
            metric=metric,
            scores=scores,
            cluster_ids=cluster_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for metric in metrics
    }
    directional_comparisons = _paired_metric_comparisons(
        directional,
        metrics=metrics,
        scores=scores,
        cluster_ids=cluster_ids,
        mode="directional",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    invariance_comparisons = _paired_metric_comparisons(
        invariant,
        metrics=metrics,
        scores=scores,
        cluster_ids=cluster_ids,
        mode="invariance",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "pair_count": len(pairs),
        "directional_pair_count": len(directional),
        "invariance_pair_count": len(invariant),
        "phenomena": dict(sorted(Counter(pair.phenomenon for pair in pairs).items())),
        "datasets": dict(sorted(Counter(pair.dataset_family for pair in pairs).items())),
        "directional_metrics": directional_metrics,
        "invariance_metrics": invariance_metrics,
        "directional_pairwise_comparisons": directional_comparisons,
        "invariance_pairwise_comparisons": invariance_comparisons,
        "evidence_tiers": _evidence_tiers(
            metrics,
            directional_metrics=directional_metrics,
            invariance_metrics=invariance_metrics,
            directional_comparisons=directional_comparisons,
            invariance_comparisons=invariance_comparisons,
        ),
    }


def _directional_summary(
    pairs: list[DiagnosticPair],
    *,
    metric: str,
    scores: dict[str, dict[str, float | None]],
    cluster_ids: Mapping[str, str] | None = None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    observations = _directional_observations(
        pairs,
        metric=metric,
        scores=scores,
        cluster_ids=cluster_ids,
    )
    total = len(pairs)
    if not observations:
        return {"n": 0, "coverage": 0.0 if total else None}
    utilities = [row[3] for row in observations]
    strict = [float(row[4] == "win") for row in observations]
    ties = [float(row[4] == "tie") for row in observations]
    margins = [row[5] for row in observations]
    macro = _macro_phenomenon(observations, value_index=3)
    ci = _cluster_bootstrap_ci(
        observations,
        statistic=lambda rows: _macro_phenomenon(rows, value_index=3),
        samples=bootstrap_samples,
        seed=seed + _stable_seed(metric, "directional"),
    )
    return {
        "n": len(observations),
        "coverage": len(observations) / total if total else None,
        "strict_consistency": mean(strict),
        "tie_adjusted_pairwise_accuracy": mean(utilities),
        "aces_tau_like": 2 * mean(strict) - 1,
        "tie_rate": mean(ties),
        "reversal_rate": 1 - mean(strict) - mean(ties),
        "mean_normalized_margin": mean(margins),
        "median_normalized_margin": median(margins),
        "macro_phenomenon_pairwise_accuracy": macro,
        "macro_phenomenon_ci95": ci,
        "roc_auc": _pair_instance_auc(observations),
        "per_phenomenon": _group_directional(observations, group_index=1),
        "per_dataset": _group_directional(observations, group_index=2),
    }


def _invariance_summary(
    pairs: list[DiagnosticPair],
    *,
    metric: str,
    scores: dict[str, dict[str, float | None]],
    cluster_ids: Mapping[str, str] | None = None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    observations = _invariance_observations(
        pairs,
        metric=metric,
        scores=scores,
        cluster_ids=cluster_ids,
    )
    total = len(pairs)
    if not observations:
        return {"n": 0, "coverage": 0.0 if total else None}
    changes = [row[3] for row in observations]
    ci = _cluster_bootstrap_ci(
        observations,
        statistic=lambda rows: _macro_phenomenon(rows, value_index=3),
        samples=bootstrap_samples,
        seed=seed + _stable_seed(metric, "invariance"),
    )
    return {
        "n": len(observations),
        "coverage": len(observations) / total if total else None,
        "mean_normalized_absolute_change": mean(changes),
        "median_normalized_absolute_change": median(changes),
        "exact_invariance_rate": mean(float(value == 0) for value in changes),
        "within_five_percent_range_rate": mean(float(value <= 0.05) for value in changes),
        "macro_phenomenon_absolute_change": _macro_phenomenon(observations, value_index=3),
        "macro_phenomenon_ci95": ci,
        "per_phenomenon": _group_invariance(observations, group_index=1),
        "per_dataset": _group_invariance(observations, group_index=2),
    }


def _directional_observations(
    pairs: list[DiagnosticPair],
    *,
    metric: str,
    scores: dict[str, dict[str, float | None]],
    cluster_ids: Mapping[str, str] | None = None,
) -> list[tuple[str, str, str, float, str, float, float, float]]:
    output = []
    for pair in pairs:
        left = _normalized_score(metric, scores[pair.left_case_id].get(metric))
        right = _normalized_score(metric, scores[pair.right_case_id].get(metric))
        if left is None or right is None:
            continue
        outcome = "win" if left > right else "tie" if left == right else "reversal"
        utility = 1.0 if outcome == "win" else 0.5 if outcome == "tie" else 0.0
        output.append(
            (
                _cluster_id(pair, cluster_ids),
                pair.phenomenon,
                pair.dataset_family,
                utility,
                outcome,
                left - right,
                left,
                right,
            )
        )
    return output


def _invariance_observations(
    pairs: list[DiagnosticPair],
    *,
    metric: str,
    scores: dict[str, dict[str, float | None]],
    cluster_ids: Mapping[str, str] | None = None,
) -> list[tuple[str, str, str, float]]:
    output = []
    for pair in pairs:
        left = _normalized_score(metric, scores[pair.left_case_id].get(metric))
        right = _normalized_score(metric, scores[pair.right_case_id].get(metric))
        if left is None or right is None:
            continue
        output.append(
            (
                _cluster_id(pair, cluster_ids),
                pair.phenomenon,
                pair.dataset_family,
                abs(left - right),
            )
        )
    return output


def _cluster_id(
    pair: DiagnosticPair,
    cluster_ids: Mapping[str, str] | None,
) -> str:
    if cluster_ids is not None:
        return cluster_ids[pair.pair_id]
    return f"{pair.dataset_family}:{pair.scenario_id}"


def _normalized_score(metric: str, value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    number = float(value)
    if metric.startswith("bertscore_"):
        return (number + 1.0) / 2.0
    return number


def _macro_phenomenon(rows: list[tuple], *, value_index: int) -> float:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[1])].append(float(row[value_index]))
    return mean(mean(values) for values in grouped.values())


def _group_directional(rows: list[tuple], *, group_index: int) -> dict[str, object]:
    grouped: defaultdict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_index])].append(row)
    return {
        key: {
            "n": len(values),
            "strict_consistency": mean(float(row[4] == "win") for row in values),
            "tie_adjusted_pairwise_accuracy": mean(row[3] for row in values),
            "tie_rate": mean(float(row[4] == "tie") for row in values),
            "roc_auc": _pair_instance_auc(values),
        }
        for key, values in sorted(grouped.items())
    }


def _group_invariance(rows: list[tuple], *, group_index: int) -> dict[str, object]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_index])].append(float(row[3]))
    return {
        key: {
            "n": len(values),
            "mean_normalized_absolute_change": mean(values),
            "exact_invariance_rate": mean(float(value == 0) for value in values),
        }
        for key, values in sorted(grouped.items())
    }


def _pair_instance_auc(rows: list[tuple]) -> float | None:
    if not rows:
        return None
    positives = [float(row[6]) for row in rows]
    negatives = [float(row[7]) for row in rows]
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _cluster_bootstrap_ci(
    rows: list[tuple],
    *,
    statistic,
    samples: int,
    seed: int,
) -> list[float] | None:
    clusters: defaultdict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        clusters[str(row[0])].append(row)
    keys = sorted(clusters)
    if len(keys) < 2 or samples <= 0:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[tuple] = []
        for key in rng.choices(keys, k=len(keys)):
            sampled.extend(clusters[key])
        estimates.append(float(statistic(sampled)))
    estimates.sort()
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _paired_metric_comparisons(
    pairs: list[DiagnosticPair],
    *,
    metrics: list[str],
    scores: dict[str, dict[str, float | None]],
    cluster_ids: Mapping[str, str],
    mode: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    values: dict[str, dict[str, tuple[str, str, str, float]]] = {}
    for metric in metrics:
        if mode == "directional":
            observations = _directional_observations(
                pairs,
                metric=metric,
                scores=scores,
                cluster_ids=cluster_ids,
            )
            values[metric] = {
                pair.pair_id: (row[0], row[1], row[2], row[3])
                for pair, row in _align_pair_observations(pairs, observations, metric, scores, mode)
            }
        else:
            observations = _invariance_observations(
                pairs,
                metric=metric,
                scores=scores,
                cluster_ids=cluster_ids,
            )
            values[metric] = {
                pair.pair_id: (row[0], row[1], row[2], 1.0 - row[3])
                for pair, row in _align_pair_observations(pairs, observations, metric, scores, mode)
            }
    comparisons: list[dict[str, object]] = []
    for left, right in itertools.combinations(metrics, 2):
        common = sorted(set(values[left]) & set(values[right]))
        if len(common) < 20:
            continue
        rows = [
            (
                values[left][pair_id][0],
                values[left][pair_id][1],
                values[left][pair_id][2],
                values[left][pair_id][3] - values[right][pair_id][3],
            )
            for pair_id in common
        ]
        estimate = _macro_phenomenon(rows, value_index=3)
        distribution = _cluster_bootstrap_distribution(
            rows,
            samples=bootstrap_samples,
            seed=seed + _stable_seed(left, right, mode),
        )
        ci = (
            [_percentile(distribution, 0.025), _percentile(distribution, 0.975)]
            if distribution
            else None
        )
        p_value = _cluster_permutation_p_value(
            rows,
            samples=bootstrap_samples,
            seed=seed + _stable_seed(left, right, mode, "permutation"),
        )
        comparisons.append(
            {
                "left": left,
                "right": right,
                "n_common_pairs": len(common),
                "macro_utility_difference": estimate,
                "ci95": ci,
                "permutation_p_value": p_value,
            }
        )
    _holm_adjust(comparisons)
    return comparisons


def _align_pair_observations(
    pairs: list[DiagnosticPair],
    observations: list[tuple],
    metric: str,
    scores: dict[str, dict[str, float | None]],
    mode: str,
) -> list[tuple[DiagnosticPair, tuple]]:
    output = []
    position = 0
    for pair in pairs:
        left = _normalized_score(metric, scores[pair.left_case_id].get(metric))
        right = _normalized_score(metric, scores[pair.right_case_id].get(metric))
        if left is None or right is None:
            continue
        output.append((pair, observations[position]))
        position += 1
    if position != len(observations):
        raise RuntimeError(f"Observation alignment failed for {metric}/{mode}")
    return output


def _cluster_bootstrap_distribution(
    rows: list[tuple],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    clusters: defaultdict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        clusters[str(row[0])].append(row)
    keys = sorted(clusters)
    if len(keys) < 2 or samples <= 0:
        return []
    rng = random.Random(seed)
    output = []
    for _ in range(samples):
        sampled = [row for key in rng.choices(keys, k=len(keys)) for row in clusters[key]]
        output.append(_macro_phenomenon(sampled, value_index=3))
    return sorted(output)


def _cluster_permutation_p_value(
    rows: list[tuple],
    *,
    samples: int,
    seed: int,
) -> float | None:
    """Two-sided paired randomization test with labels swapped by source-scenario cluster."""

    clusters: defaultdict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        clusters[str(row[0])].append(row)
    keys = sorted(clusters)
    if len(keys) < 2 or samples <= 0:
        return None
    observed = abs(_macro_phenomenon(rows, value_index=3))

    def statistic(signs: tuple[int, ...] | list[int]) -> float:
        permuted = [
            (*row[:3], float(row[3]) * sign)
            for key, sign in zip(keys, signs, strict=True)
            for row in clusters[key]
        ]
        return abs(_macro_phenomenon(permuted, value_index=3))

    if len(keys) <= 16:
        total = 2 ** len(keys)
        extreme = 0
        for mask in range(total):
            signs = tuple(1 if mask & (1 << index) else -1 for index in range(len(keys)))
            extreme += int(statistic(signs) >= observed - 1e-12)
        return extreme / total

    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        signs = [1 if rng.getrandbits(1) else -1 for _ in keys]
        extreme += int(statistic(signs) >= observed - 1e-12)
    return (extreme + 1) / (samples + 1)


def _holm_adjust(comparisons: list[dict[str, object]]) -> None:
    valid = [row for row in comparisons if row["permutation_p_value"] is not None]
    ordered = sorted(valid, key=lambda row: float(row["permutation_p_value"]))
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, float(row["permutation_p_value"]) * (total - index))
        running = max(running, adjusted)
        row["holm_adjusted_p_value"] = running
        row["significant_at_0_05"] = running < 0.05
    for row in comparisons:
        row.setdefault("holm_adjusted_p_value", None)
        row.setdefault("significant_at_0_05", False)


def _evidence_tiers(
    metrics: list[str],
    *,
    directional_metrics: dict[str, dict[str, object]],
    invariance_metrics: dict[str, dict[str, object]],
    directional_comparisons: list[dict[str, object]],
    invariance_comparisons: list[dict[str, object]],
) -> list[dict[str, object]]:
    wins = Counter()
    losses = Counter()
    for row in [*directional_comparisons, *invariance_comparisons]:
        if not row.get("significant_at_0_05"):
            continue
        difference = float(row["macro_utility_difference"])
        winner = str(row["left"] if difference > 0 else row["right"])
        loser = str(row["right"] if difference > 0 else row["left"])
        wins[winner] += 1
        losses[loser] += 1
    rows = []
    for metric in metrics:
        directional = directional_metrics.get(metric, {})
        invariance = invariance_metrics.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "significant_pairwise_wins": wins[metric],
                "significant_pairwise_losses": losses[metric],
                "directional_macro": directional.get("macro_phenomenon_pairwise_accuracy"),
                "directional_coverage": directional.get("coverage"),
                "invariance_macro_change": invariance.get("macro_phenomenon_absolute_change"),
                "invariance_coverage": invariance.get("coverage"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["significant_pairwise_wins"]),
            int(row["significant_pairwise_losses"]),
            -(float(row["directional_macro"]) if row["directional_macro"] is not None else -1),
            str(row["metric"]),
        ),
    )


def _formal_oracle_human_audit(
    *,
    dataset_root: Path,
    gold_dir: Path,
) -> dict[str, object]:
    bundles = {
        path.name: load_bundle(path) for path in sorted(dataset_root.iterdir()) if path.is_dir()
    }
    answers = {
        (family, answer.answer_id): (bundle, answer)
        for family, bundle in bundles.items()
        for answer in bundle.answer_variants
    }
    by_field: defaultdict[str, list[bool]] = defaultdict(list)
    by_resolution: defaultdict[str, list[bool]] = defaultdict(list)
    comparisons = 0
    controlled_units = 0
    for raw in read_jsonl(gold_dir / "gold_units.jsonl"):
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("source_kind") != "controlled_variant":
            continue
        key = (str(metadata["dataset_family"]), str(metadata["answer_id"]))
        if key not in answers:
            continue
        controlled_units += 1
        bundle, answer = answers[key]
        expected = controlled_private_key(answer, bundle=bundle)
        labels = raw.get("gold_labels")
        provenance = raw.get("field_provenance")
        if not isinstance(labels, dict):
            continue
        for field, human_label in labels.items():
            oracle_label = expected.get(field)
            if oracle_label is None:
                continue
            agreement = str(oracle_label) == str(human_label)
            by_field[str(field)].append(agreement)
            method = "unknown"
            if isinstance(provenance, dict) and isinstance(provenance.get(field), dict):
                method = str(provenance[field].get("resolution_method", "unknown"))
            by_resolution[method].append(agreement)
            comparisons += 1
    agreements = [value for values in by_field.values() for value in values]
    return {
        "controlled_units": controlled_units,
        "field_comparisons": comparisons,
        "exact_agreement": mean(agreements) if agreements else None,
        "wilson_ci95": _wilson_interval(sum(agreements), len(agreements)),
        "by_field": {
            field: {
                "n": len(values),
                "exact_agreement": mean(values),
                "wilson_ci95": _wilson_interval(sum(values), len(values)),
            }
            for field, values in sorted(by_field.items())
        },
        "by_resolution_method": {
            method: {"n": len(values), "exact_agreement": mean(values)}
            for method, values in sorted(by_resolution.items())
        },
        "scope_note": (
            "This audit validates the formal/presentation oracle against the available human-rated "
            "controlled variants. It does not turn synthetic challenge cases into real-world "
            "human gold."
        ),
    }


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total <= 0:
        return None
    probability = successes / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2)) / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _stable_seed(*parts: str) -> int:
    payload = ":".join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def write_analysis(analysis: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(analysis, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
