from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def binary_metric_summary(
    scores: Sequence[float],
    labels: Sequence[int],
    clusters: Sequence[str],
    *,
    samples: int = 5_000,
    seed: int = 20260816,
    batch_size: int = 64,
) -> dict[str, object]:
    """Compute exact point statistics and an exact weighted cluster bootstrap."""

    score_array, label_array, cluster_index, cluster_count = _validated_arrays(
        scores, labels, clusters
    )
    point = _weighted_statistics(
        score_array,
        label_array,
        np.ones((1, len(score_array)), dtype=np.int64),
    )
    output: dict[str, object] = {
        "n": len(score_array),
        "cluster_count": cluster_count,
        "positive_count": int(label_array.sum()),
        "positive_rate": float(label_array.mean()),
        "unique_score_count": int(np.unique(score_array).size),
        **{name: _finite_or_none(values[0]) for name, values in point.items()},
    }
    if samples <= 0:
        output["bootstrap_samples"] = 0
        return output
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {
        name: []
        for name in (
            "spearman",
            "auroc",
            "average_precision",
            "mean_separation",
            "brier_score",
        )
    }
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        sampled_cluster_weights = rng.multinomial(
            cluster_count,
            probabilities,
            size=count,
        )
        row_weights = sampled_cluster_weights[:, cluster_index]
        values = _weighted_statistics(score_array, label_array, row_weights)
        for name in bootstrap:
            bootstrap[name].extend(
                float(value) for value in values[name] if math.isfinite(float(value))
            )
    output["bootstrap_samples"] = samples
    for name, values in bootstrap.items():
        output[f"{name}_ci95"] = _percentile_interval(values)
        output[f"{name}_valid_bootstrap_replicates"] = len(values)
    return output


def mcnemar_test(left: Sequence[int], right: Sequence[int]) -> dict[str, object]:
    """Return the paired binary contrast and an exact two-sided McNemar p-value."""

    if len(left) != len(right) or not left:
        raise ValueError("left and right must be non-empty and aligned")
    if not set(left) <= {0, 1} or not set(right) <= {0, 1}:
        raise ValueError("McNemar inputs must be binary")
    left_only = sum(
        left_value == 1 and right_value == 0
        for left_value, right_value in zip(left, right, strict=True)
    )
    right_only = sum(
        left_value == 0 and right_value == 1
        for left_value, right_value in zip(left, right, strict=True)
    )
    discordant = left_only + right_only
    return {
        "n": len(left),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_count": discordant,
        "accuracy_difference": (sum(left) - sum(right)) / len(left),
        "exact_two_sided_p": _two_sided_binomial_p(left_only, right_only),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust a named family of valid p-values using Holm's step-down method."""

    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values.values()):
        raise ValueError("p-values must be finite and within [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def clustered_mean_summary(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    samples: int = 5_000,
    seed: int = 20260816,
    batch_size: int = 128,
) -> dict[str, object]:
    if len(values) != len(clusters) or not values:
        raise ValueError("values and clusters must be non-empty and aligned")
    value_array = np.asarray(values, dtype=float)
    if not np.isfinite(value_array).all():
        raise ValueError("values must be finite")
    cluster_index, cluster_count = _cluster_indices(clusters)
    output: dict[str, object] = {
        "n": len(values),
        "cluster_count": cluster_count,
        "mean": float(value_array.mean()),
        "bootstrap_samples": samples,
    }
    if samples <= 0:
        return output
    rng = np.random.default_rng(seed)
    probabilities = np.full(cluster_count, 1.0 / cluster_count)
    estimates: list[float] = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        sampled = rng.multinomial(cluster_count, probabilities, size=count)
        weights = sampled[:, cluster_index]
        estimates.extend(
            ((weights @ value_array) / weights.sum(axis=1)).astype(float).tolist()
        )
    output["ci95"] = _percentile_interval(estimates)
    return output


def _weighted_statistics(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = weights[:, order].astype(float, copy=False)
    starts = np.flatnonzero(
        np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    )
    group_total = np.add.reduceat(sorted_weights, starts, axis=1)
    group_positive = np.add.reduceat(
        sorted_weights * sorted_labels[np.newaxis, :], starts, axis=1
    )
    total = group_total.sum(axis=1)
    positive = group_positive.sum(axis=1)
    negative = total - positive

    cumulative_before = np.cumsum(group_total, axis=1) - group_total
    midrank = cumulative_before + (group_total + 1.0) / 2.0
    sum_positive_ranks = (group_positive * midrank).sum(axis=1)
    sum_ranks = total * (total + 1.0) / 2.0
    sum_squared_ranks = (group_total * midrank**2).sum(axis=1)
    covariance = sum_positive_ranks - sum_ranks * positive / total
    score_variance = sum_squared_ranks - sum_ranks**2 / total
    label_variance = positive - positive**2 / total
    with np.errstate(divide="ignore", invalid="ignore"):
        spearman = covariance / np.sqrt(score_variance * label_variance)
        auroc = (
            sum_positive_ranks - positive * (positive + 1.0) / 2.0
        ) / (positive * negative)

    descending_total = group_total[:, ::-1]
    descending_positive = group_positive[:, ::-1]
    cumulative_total = np.cumsum(descending_total, axis=1)
    cumulative_positive = np.cumsum(descending_positive, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        average_precision = (
            (descending_positive / positive[:, np.newaxis])
            * (cumulative_positive / cumulative_total)
        ).sum(axis=1)

    sorted_score_values = sorted_scores[starts]
    weighted_score_sum = (group_total * sorted_score_values[np.newaxis, :]).sum(axis=1)
    positive_score_sum = (
        group_positive * sorted_score_values[np.newaxis, :]
    ).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive_mean = positive_score_sum / positive
        negative_mean = (weighted_score_sum - positive_score_sum) / negative
        brier_score = (
            sorted_weights
            * (sorted_scores[np.newaxis, :] - sorted_labels[np.newaxis, :]) ** 2
        ).sum(axis=1) / total
    return {
        "spearman": spearman,
        "auroc": auroc,
        "average_precision": average_precision,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "mean_separation": positive_mean - negative_mean,
        "brier_score": brier_score,
    }


def _validated_arrays(
    scores: Sequence[float],
    labels: Sequence[int],
    clusters: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not scores or len(scores) != len(labels) or len(scores) != len(clusters):
        raise ValueError("scores, labels, and clusters must be non-empty and aligned")
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=np.int8)
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    if not set(label_array.tolist()) <= {0, 1}:
        raise ValueError("labels must be binary")
    if len(set(label_array.tolist())) != 2:
        raise ValueError("both binary classes are required")
    cluster_index, cluster_count = _cluster_indices(clusters)
    return score_array, label_array, cluster_index, cluster_count


def _cluster_indices(clusters: Sequence[str]) -> tuple[np.ndarray, int]:
    if any(not isinstance(cluster, str) or not cluster for cluster in clusters):
        raise ValueError("cluster IDs must be non-empty strings")
    names = {name: index for index, name in enumerate(sorted(set(clusters)))}
    return np.asarray([names[name] for name in clusters], dtype=np.intp), len(names)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _percentile_interval(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return [float(low), float(high)]


def _two_sided_binomial_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(discordant - successes + 1)
        - discordant * math.log(2.0)
        for successes in range(lower + 1)
    ]
    maximum = max(log_probabilities)
    lower_tail = math.exp(maximum) * math.fsum(
        math.exp(value - maximum) for value in log_probabilities
    )
    return min(1.0, 2.0 * lower_tail)
