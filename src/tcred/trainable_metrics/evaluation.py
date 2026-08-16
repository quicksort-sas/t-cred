from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


class EvaluationAccumulator:
    def __init__(self) -> None:
        self.binary: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        self.classes: defaultdict[str, list[tuple[int, list[float]]]] = defaultdict(list)
        self.ratings: list[tuple[float, float]] = []
        self.pairs: defaultdict[str, dict[str, float]] = defaultdict(dict)
        self.losses: list[float] = []

    def add_batch(
        self,
        *,
        task: str,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        loss: float | None = None,
    ) -> None:
        if loss is not None:
            self.losses.append(float(loss))
        if task == "answer":
            self._binary("answer_u1", outputs["u1"], batch["answer_u1"])
            self._binary("answer_u2", outputs["u2"], batch["answer_u2"])
            self._binary("equivalence", outputs["equivalence"], batch["equivalence"])
            self._rating(outputs["score"], batch["scalar_rating"])
            score = outputs["score"]
        elif task in {"support", "temporal"}:
            self._class(task, outputs["class_probabilities"], batch["class_target"])
            self._binary(f"{task}_supported", outputs["supported"], batch["supported"])
            binary_mask = batch["supported"] >= 0
            score = outputs["supported"].clone()
            score[~binary_mask] = outputs["class_probabilities"][~binary_mask, 0]
        elif task == "relevance":
            self._binary("relevance", outputs["relevance"], batch["relevance"])
            score = outputs["relevance"]
        elif task == "answerability":
            self._binary("answerability", outputs["answerable"], batch["answerable"])
            score = outputs["answerable"]
        elif task == "citation":
            self._class("citation", outputs["class_probabilities"], batch["class_target"])
            score = outputs["class_probabilities"][:, 0]
        else:
            raise ValueError(f"Unknown evaluation task: {task}")
        for pair_id, role, value in zip(
            batch["pair_ids"],
            batch["pair_roles"],
            score.detach().float().cpu().tolist(),
            strict=True,
        ):
            if pair_id and role:
                self.pairs[pair_id][role] = float(value)

    def compute(self) -> dict[str, Any]:
        binary = {name: _binary_metrics(rows) for name, rows in sorted(self.binary.items())}
        classes = {name: _class_metrics(rows) for name, rows in sorted(self.classes.items())}
        rating = _rating_metrics(self.ratings) if self.ratings else None
        pair_metrics = _pair_metrics(self.pairs)
        task_scores = _task_headline_scores(binary=binary, classes=classes, rating=rating)
        normalized_scores = _task_normalized_scores(
            binary=binary,
            classes=classes,
            rating=rating,
        )
        return {
            "mean_loss": float(np.mean(self.losses)) if self.losses else None,
            "binary": binary,
            "classes": classes,
            "rating": rating,
            "controlled_pairs": pair_metrics,
            "task_headline_scores": task_scores,
            "task_normalized_scores": normalized_scores,
            "harmonic_normalized_objective": _harmonic_normalized(normalized_scores),
        }

    def _binary(self, name: str, probabilities: Any, targets: Any) -> None:
        for probability, target in zip(
            probabilities.detach().float().cpu().tolist(),
            targets.detach().float().cpu().tolist(),
            strict=True,
        ):
            if target >= 0:
                self.binary[name].append((float(target), float(probability)))

    def _class(self, name: str, probabilities: Any, targets: Any) -> None:
        for probability, target in zip(
            probabilities.detach().float().cpu().tolist(),
            targets.detach().float().cpu().tolist(),
            strict=True,
        ):
            if target[0] >= 0:
                self.classes[name].append(
                    (int(np.argmax(target)), [float(value) for value in probability])
                )

    def _rating(self, predictions: Any, targets: Any) -> None:
        for prediction, target in zip(
            predictions.detach().float().cpu().tolist(),
            targets.detach().float().cpu().tolist(),
            strict=True,
        ):
            if target >= 0:
                self.ratings.append((float(target), float(prediction)))


def _binary_metrics(rows: list[tuple[float, float]]) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    targets = np.asarray([row[0] for row in rows], dtype=np.float64)
    probabilities = np.asarray([row[1] for row in rows], dtype=np.float64)
    hard_targets = (targets >= 0.5).astype(np.int64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    result = {
        "n": len(rows),
        "classes_observed": int(len(np.unique(hard_targets))),
        "macro_f1": float(f1_score(hard_targets, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(hard_targets, predictions)),
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "ece_10": _ece(targets, probabilities, bins=10),
    }
    if len(np.unique(hard_targets)) > 1:
        result["auroc"] = float(roc_auc_score(hard_targets, probabilities))
        result["average_precision"] = float(
            average_precision_score(hard_targets, probabilities)
        )
    else:
        result["auroc"] = None
        result["average_precision"] = None
    return result


def _class_metrics(rows: list[tuple[int, list[float]]]) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    targets = np.asarray([row[0] for row in rows], dtype=np.int64)
    probabilities = np.asarray([row[1] for row in rows], dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[targets]
    return {
        "n": len(rows),
        "classes_observed": int(len(np.unique(targets))),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "accuracy": float(np.mean(predictions == targets)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10": _ece(targets, probabilities, bins=10),
    }


def _rating_metrics(rows: list[tuple[float, float]]) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr

    targets = np.asarray([row[0] for row in rows], dtype=np.float64)
    predictions = np.asarray([row[1] for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "target_standard_deviation": float(np.std(targets)),
        "mae": float(np.mean(np.abs(predictions - targets))),
        "pearson": _finite_statistic(pearsonr(targets, predictions).statistic),
        "spearman": _finite_statistic(spearmanr(targets, predictions).statistic),
    }


def _pair_metrics(pairs: dict[str, dict[str, float]]) -> dict[str, Any]:
    rankings = [
        values["positive"] > values["negative"]
        for values in pairs.values()
        if "positive" in values and "negative" in values
    ]
    invariance = [
        abs(values["invariant_a"] - values["invariant_b"])
        for values in pairs.values()
        if "invariant_a" in values and "invariant_b" in values
    ]
    return {
        "ranking_pairs": len(rankings),
        "ranking_accuracy": float(np.mean(rankings)) if rankings else None,
        "invariance_pairs": len(invariance),
        "mean_absolute_invariance_delta": float(np.mean(invariance)) if invariance else None,
    }


def _task_headline_scores(
    *,
    binary: dict[str, dict[str, Any]],
    classes: dict[str, dict[str, Any]],
    rating: dict[str, Any] | None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    answer_values = [
        binary[name]["macro_f1"]
        for name in ("answer_u2", "equivalence")
        if name in binary
    ]
    if rating and rating["spearman"] is not None:
        answer_values.append(max(0.0, float(rating["spearman"])))
    if answer_values:
        result["answer"] = float(np.mean(answer_values))
    for task in ("support", "temporal"):
        values = []
        if task in classes:
            values.append(classes[task]["macro_f1"])
        if f"{task}_supported" in binary:
            values.append(binary[f"{task}_supported"]["macro_f1"])
        if values:
            result[task] = float(np.mean(values))
    if "relevance" in binary:
        result["relevance"] = binary["relevance"]["macro_f1"]
    if "answerability" in binary:
        result["answerability"] = binary["answerability"]["macro_f1"]
    if "citation" in classes:
        result["citation"] = classes["citation"]["macro_f1"]
    return result


def _task_normalized_scores(
    *,
    binary: dict[str, dict[str, Any]],
    classes: dict[str, dict[str, Any]],
    rating: dict[str, Any] | None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    answer_values = [
        _above_chance(binary[name]["macro_f1"], chance=0.5)
        for name in ("answer_u2", "equivalence")
        if name in binary and binary[name]["classes_observed"] >= 2
    ]
    if (
        rating
        and rating["spearman"] is not None
        and rating["target_standard_deviation"] > 0
    ):
        answer_values.append(max(0.0, min(1.0, float(rating["spearman"]))))
    if answer_values:
        result["answer"] = float(np.mean(answer_values))
    for task in ("support", "temporal"):
        values = []
        if task in classes and classes[task]["classes_observed"] >= 2:
            values.append(_above_chance(classes[task]["macro_f1"], chance=1 / 3))
        binary_name = f"{task}_supported"
        if binary_name in binary and binary[binary_name]["classes_observed"] >= 2:
            values.append(_above_chance(binary[binary_name]["macro_f1"], chance=0.5))
        if values:
            result[task] = float(np.mean(values))
    for task in ("relevance", "answerability"):
        if task in binary and binary[task]["classes_observed"] >= 2:
            result[task] = _above_chance(binary[task]["macro_f1"], chance=0.5)
    if "citation" in classes and classes["citation"]["classes_observed"] >= 2:
        result["citation"] = _above_chance(classes["citation"]["macro_f1"], chance=1 / 3)
    return result


def _harmonic_normalized(normalized_scores: dict[str, float]) -> float | None:
    if not normalized_scores:
        return None
    bounded = [max(1e-6, min(1.0, value)) for value in normalized_scores.values()]
    return len(bounded) / sum(1.0 / value for value in bounded)


def summarize_source_macro(source_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    for report in source_reports.values():
        for task, value in report.get("task_normalized_scores", {}).items():
            by_task[task].append(float(value))
    task_scores = {
        task: float(np.mean(values))
        for task, values in sorted(by_task.items())
        if values
    }
    return {
        "source_families": len(source_reports),
        "contributing_source_families": {
            task: len(values) for task, values in sorted(by_task.items())
        },
        "task_normalized_scores": task_scores,
        "harmonic_normalized_objective": _harmonic_normalized(task_scores),
    }


def _above_chance(value: float, *, chance: float) -> float:
    return max(0.0, min(1.0, (float(value) - chance) / (1.0 - chance)))


def _ece(targets: np.ndarray, probabilities: np.ndarray, *, bins: int) -> float:
    if probabilities.ndim == 2:
        confidence = probabilities.max(axis=1)
        correctness = (probabilities.argmax(axis=1) == targets).astype(np.float64)
    else:
        confidence = np.maximum(probabilities, 1.0 - probabilities)
        correctness = ((probabilities >= 0.5) == (targets >= 0.5)).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def _finite_statistic(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None
