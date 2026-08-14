from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np

LABEL_SCORE = {"no": 0.0, "partial": 0.5, "yes": 1.0}


def mean_interval(
    values: list[float],
    clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | list[float] | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    estimate = float(np.mean(values))
    interval = _cluster_bootstrap_mean_interval(
        values,
        clusters,
        samples=samples,
        seed=seed,
    )
    return {"n": len(values), "mean": estimate, "ci95": interval}


def correlation_summary(
    scores: list[float],
    labels: list[float],
    clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    binary = [(score, label) for score, label in zip(scores, labels, strict=True) if label != 0.5]
    binary_scores = [score for score, _label in binary]
    binary_labels = [int(label) for _score, label in binary]
    binary_positive_count = sum(binary_labels)
    binary_summary = {
        "binary_n": len(binary),
        "binary_positive_count": binary_positive_count,
        "binary_positive_rate": binary_positive_count / len(binary) if binary else None,
        "auroc_yes_vs_no": _auroc(binary_scores, binary_labels),
        "auprc_yes_vs_no": _average_precision(binary_scores, binary_labels),
    }
    if len(scores) < 5 or len(set(scores)) < 2 or len(set(labels)) < 2:
        return {
            "n": len(scores),
            "spearman": None,
            "spearman_ci95": None,
            "kendall_tau_b": None,
            **binary_summary,
        }
    spearman = _spearman(scores, labels)
    interval = _cluster_bootstrap_interval(
        scores,
        labels,
        clusters,
        statistic=_spearman,
        samples=samples,
        seed=seed,
    )
    return {
        "n": len(scores),
        "spearman": spearman,
        "spearman_ci95": interval,
        "kendall_tau_b": _kendall_tau_b(scores, labels),
        **binary_summary,
    }


def paired_mean_difference(
    left: dict[str, float],
    right: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"n": 0, "mean_difference": None, "ci95": None}
    differences = [left[key] - right[key] for key in keys]
    values = np.asarray(differences, dtype=float)
    indices = _bootstrap_index_matrix(
        population_size=len(differences),
        draw_size=len(differences),
        samples=samples,
        seed=seed,
    )
    boot = values[indices].mean(axis=1).tolist()
    return {
        "n": len(keys),
        "mean_difference": float(np.mean(differences)),
        "ci95": _percentile_interval(boot),
    }


def paired_spearman_difference(
    left: list[float],
    right: list[float],
    labels: list[float],
    clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Compare two correlations on identical units with a joint cluster bootstrap."""
    if len(left) < 5 or len(set(labels)) < 2:
        return {"n": len(left), "mean_difference": None, "ci95": None}
    estimate = _correlation_difference(left, right, labels)
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[cluster].append(index)
    cluster_ids = sorted(grouped)
    draws = _bootstrap_index_matrix(
        population_size=len(cluster_ids),
        draw_size=len(cluster_ids),
        samples=samples,
        seed=seed,
    )
    boot: list[float] = []
    for draw in draws:
        indices = [
            index
            for sampled_cluster in draw
            for index in grouped[cluster_ids[int(sampled_cluster)]]
        ]
        value = _correlation_difference(
            [left[index] for index in indices],
            [right[index] for index in indices],
            [labels[index] for index in indices],
        )
        if value is not None and math.isfinite(value):
            boot.append(value)
    return {
        "n": len(left),
        "mean_difference": estimate,
        "ci95": _percentile_interval(boot),
    }


def _correlation_difference(
    left: list[float], right: list[float], labels: list[float]
) -> float | None:
    left_correlation = _spearman(left, labels)
    right_correlation = _spearman(right, labels)
    if left_correlation is None or right_correlation is None:
        return None
    return left_correlation - right_correlation


def _cluster_bootstrap_interval(
    left: list[float],
    right: list[float],
    clusters: list[str],
    *,
    statistic: Callable[[list[float], list[float]], float | None],
    samples: int,
    seed: int,
) -> list[float] | None:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[cluster].append(index)
    cluster_ids = sorted(grouped)
    if not cluster_ids:
        return None
    draws = _bootstrap_index_matrix(
        population_size=len(cluster_ids),
        draw_size=len(cluster_ids),
        samples=samples,
        seed=seed,
    )
    estimates: list[float] = []
    for draw in draws:
        indices = [
            index
            for sampled_cluster in draw
            for index in grouped[cluster_ids[int(sampled_cluster)]]
        ]
        value = statistic([left[index] for index in indices], [right[index] for index in indices])
        if value is not None and math.isfinite(value):
            estimates.append(value)
    return _percentile_interval(estimates)


def _cluster_bootstrap_mean_interval(
    values: list[float],
    clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    cluster_ids = sorted(grouped)
    if not cluster_ids:
        return None
    cluster_sums = np.asarray(
        [sum(grouped[cluster]) for cluster in cluster_ids],
        dtype=float,
    )
    cluster_counts = np.asarray(
        [len(grouped[cluster]) for cluster in cluster_ids],
        dtype=float,
    )
    indices = _bootstrap_index_matrix(
        population_size=len(cluster_ids),
        draw_size=len(cluster_ids),
        samples=samples,
        seed=seed,
    )
    estimates = cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(axis=1)
    return _percentile_interval(estimates.tolist())


def _bootstrap_index_matrix(
    *,
    population_size: int,
    draw_size: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        population_size,
        size=(samples, draw_size),
        dtype=np.intp,
    )


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def _kendall_tau_b(left: list[float], right: list[float]) -> float | None:
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_sign = _sign(left[first] - left[second])
            right_sign = _sign(right[first] - right[second])
            if left_sign == 0 and right_sign == 0:
                continue
            if left_sign == 0:
                left_ties += 1
            elif right_sign == 0:
                right_ties += 1
            elif left_sign == right_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties) * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = _rank(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hits = 0
    processed = 0
    area = 0.0
    position = 0
    while position < len(order):
        threshold = scores[order[position]]
        end = position
        group_hits = 0
        while end < len(order) and scores[order[end]] == threshold:
            group_hits += labels[order[end]]
            end += 1
        hits += group_hits
        processed += end - position
        area += (group_hits / positives) * (hits / processed)
        position = end
    return area


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return [float(low), float(high)]


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)
