from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

from tcred.external_evaluations.sabet_tkgqa.external_statistics import (
    binary_metric_summary,
    clustered_mean_summary,
    holm_adjust,
    mcnemar_test,
)
from tcred.external_evaluations.sabet_tkgqa.schema import AnswerMetricRecord

PRIMARY_SEED = 1729
BINARY_BOOTSTRAP_BATCH_SIZE = 8
MEAN_BOOTSTRAP_BATCH_SIZE = 32
PRIMARY_METRICS = (
    "exact_match",
    "token_f1",
    "rouge_l",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "tcred_answer_equivalence",
    "tcred_temporal_correctness",
)
DECISION_THRESHOLDS = {
    "exact_match": (1.0, "definition: normalized strings must be identical"),
    "pedants_match": (0.5, "published PEDANTS binary decision"),
    "pedants_probability": (0.5, "published PEDANTS classifier decision boundary"),
    "tcred_answer_equivalence": (0.5, "frozen T-CRED v1.4 answer-equivalence boundary"),
    "tcred_temporal_correctness": (0.5, "frozen T-CRED v1.4 component boundary"),
}


@dataclass(frozen=True, slots=True)
class _SharedQuestion:
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class _CompactMetric:
    run_id: str
    dataset: str
    model: str
    variant: str
    seed: int
    source_index: int
    answer_type: str
    native_hit_at_1: float
    native_hit_at_10: float
    scores: dict[str, float]
    reference_text_available: bool
    candidate_text_available: bool


@dataclass(frozen=True, slots=True)
class _CompactContext(Mapping[str, object]):
    question: _SharedQuestion
    source_record_sha256: str
    predicted_answer_id: str
    predicted_answer_label: str
    resolved_predicted_answer_available: bool

    def __getitem__(self, key: str) -> object:
        if key == "source_record_sha256":
            return self.source_record_sha256
        if key == "predicted_answer_id":
            return self.predicted_answer_id
        if key == "predicted_answer_label":
            return self.predicted_answer_label
        if key == "resolved_predicted_answer":
            return True if self.resolved_predicted_answer_available else None
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "source_record_sha256",
                "predicted_answer_id",
                "predicted_answer_label",
                "resolved_predicted_answer",
            )
        )

    def __len__(self) -> int:
        return 4


@dataclass(frozen=True, slots=True)
class ScoredUnit:
    metric: AnswerMetricRecord | _CompactMetric
    context: Mapping[str, object]

    @property
    def system(self) -> str:
        return f"{self.metric.model}_{self.metric.variant}"

    @property
    def cluster(self) -> str:
        identity = (
            self.context.question.identity_sha256
            if isinstance(self.context, _CompactContext)
            else _question_identity_sha256(self.context)
        )
        return f"{self.metric.dataset}:{identity}"


def analyze_external_metrics(
    *,
    scores_path: Path,
    context_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 5_000,
    seed: int = 20260816,
    workers: int = 4,
) -> dict[str, Path]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    units = load_scored_units(scores_path=scores_path, context_path=context_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance = _performance_analysis(units)
    correlations = _correlation_analysis(
        units,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        workers=workers,
    )
    paired = _paired_analysis(
        units,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        workers=workers,
    )
    qualitative = _qualitative_audit(units, context_path=context_path, seed=seed)

    artifacts = {
        "performance": output_dir / "system_performance.json",
        "correlations": output_dir / "metric_oracle_alignment.json",
        "paired": output_dir / "paired_system_comparisons.json",
        "qualitative": output_dir / "qualitative_audit.json",
    }
    for name, path in artifacts.items():
        _write_json(path, locals()[name])
    manifest_path = output_dir / "external_analysis_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "analysis_implementation_sha256": _sha256(Path(__file__)),
        "inputs": {
            "unit_scores": _file_identity(scores_path.resolve()),
            "unit_context": _file_identity(context_path.resolve()),
        },
        "unit_count": len(units),
        "run_count": len({unit.metric.run_id for unit in units}),
        "datasets": sorted({unit.metric.dataset for unit in units}),
        "systems": sorted({unit.system for unit in units}),
        "bootstrap": {
            "replicates": bootstrap_samples,
            "seed": seed,
            "workers": workers,
            "binary_batch_size": BINARY_BOOTSTRAP_BATCH_SIZE,
            "mean_batch_size": MEAN_BOOTSTRAP_BATCH_SIZE,
            "cluster": "dataset plus immutable released-question identity SHA-256",
            "interval": "percentile 95%",
        },
        "primary_seed": PRIMARY_SEED,
        "primary_metrics": list(PRIMARY_METRICS),
        "memory_contract": {
            "loader": "lockstep streaming with compact retained records",
            "question_identity": "shared SHA-256 identity per dataset/source index",
            "qualitative_context": "second streaming pass limited to selected audit rows",
        },
        "artifacts": {name: _file_identity(path) for name, path in artifacts.items()},
    }
    _write_json(manifest_path, manifest)
    return {"manifest": manifest_path, **artifacts}


def load_scored_units(*, scores_path: Path, context_path: Path) -> list[ScoredUnit]:
    units: list[ScoredUnit] = []
    seen_units: set[tuple[str, int]] = set()
    shared_questions: dict[tuple[str, int], _SharedQuestion] = {}
    missing = object()
    pairs = zip_longest(
        _iter_jsonl(scores_path),
        _iter_jsonl(context_path),
        fillvalue=missing,
    )
    for metric_item, context_item in pairs:
        if metric_item is missing or context_item is missing:
            raise ValueError("Metric/context unit count mismatch")
        assert isinstance(metric_item, tuple)
        assert isinstance(context_item, tuple)
        metric_line_number, metric_payload = metric_item
        context_line_number, context_payload = context_item
        try:
            metric = AnswerMetricRecord.model_validate(metric_payload)
        except ValueError as error:
            raise ValueError(
                f"Invalid metric record at {scores_path}:{metric_line_number}"
            ) from error
        _required_string(
            context_payload,
            "source_record_sha256",
            context_path,
            context_line_number,
        )
        unit = ScoredUnit(metric=metric, context=context_payload)
        _validate_context_alignment(unit)
        _validate_scores(unit)
        key = (metric.run_id, metric.source_index)
        if key in seen_units:
            raise ValueError(f"Duplicate metric/context unit: {key}")
        seen_units.add(key)

        question_key = (metric.dataset, metric.source_index)
        identity_sha256 = _question_identity_sha256(context_payload)
        shared = shared_questions.get(question_key)
        if shared is None:
            shared = _SharedQuestion(
                identity_sha256=identity_sha256,
            )
            shared_questions[question_key] = shared
        elif shared.identity_sha256 != identity_sha256:
            raise ValueError(
                f"Released question identity changed across runs: {question_key}"
            )
        compact_metric = _CompactMetric(
            run_id=sys.intern(metric.run_id),
            dataset=sys.intern(metric.dataset),
            model=sys.intern(metric.model),
            variant=sys.intern(metric.variant),
            seed=metric.seed,
            source_index=metric.source_index,
            answer_type=sys.intern(metric.answer_type),
            native_hit_at_1=metric.native_hit_at_1,
            native_hit_at_10=metric.native_hit_at_10,
            scores={
                sys.intern(name): float(value)
                for name, value in metric.scores.items()
                if value is not None
            },
            reference_text_available=bool(
                metric.applicability["reference_text_available"]
            ),
            candidate_text_available=bool(
                metric.applicability["candidate_text_available"]
            ),
        )
        compact_context = _CompactContext(
            question=shared,
            source_record_sha256=sys.intern(
                str(context_payload["source_record_sha256"])
            ),
            predicted_answer_id=sys.intern(
                _required_string(
                    context_payload,
                    "predicted_answer_id",
                    context_path,
                    context_line_number,
                )
            ),
            predicted_answer_label=sys.intern(
                _required_string(
                    context_payload,
                    "predicted_answer_label",
                    context_path,
                    context_line_number,
                )
            ),
            resolved_predicted_answer_available=(
                context_payload.get("resolved_predicted_answer") is not None
            ),
        )
        units.append(ScoredUnit(metric=compact_metric, context=compact_context))
    _validate_repeated_question_identity(units)
    return units


def _performance_analysis(units: Sequence[ScoredUnit]) -> dict[str, object]:
    groups: dict[str, list[ScoredUnit]] = {
        f"run/{run_id}": [unit for unit in units if unit.metric.run_id == run_id]
        for run_id in sorted({unit.metric.run_id for unit in units})
    }
    for dataset in sorted({unit.metric.dataset for unit in units}):
        for system in sorted({unit.system for unit in units}):
            selected = [
                unit
                for unit in units
                if unit.metric.dataset == dataset and unit.system == system
            ]
            if selected:
                groups[f"dataset-system/{dataset}/{system}"] = selected
    primary = [unit for unit in units if unit.metric.seed == PRIMARY_SEED]
    for system in sorted({unit.system for unit in primary}):
        groups[f"primary-system/{system}"] = [
            unit for unit in primary if unit.system == system
        ]

    return {
        "schema_version": "1.0",
        "estimand_note": (
            "Per-run and dataset-system means are primary. Primary-system pooled means are "
            "question-micro averages and are dominated by larger datasets; dataset macro means "
            "are reported separately."
        ),
        "groups": {name: _mean_scores(rows) for name, rows in sorted(groups.items())},
        "primary_dataset_macro_means": _primary_dataset_macro_means(primary),
    }


def _correlation_analysis(
    units: Sequence[ScoredUnit],
    *,
    bootstrap_samples: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    primary = [unit for unit in units if unit.metric.seed == PRIMARY_SEED]
    groups: dict[str, list[ScoredUnit]] = {"primary-pooled": primary}
    for dataset in sorted({unit.metric.dataset for unit in primary}):
        groups[f"primary-dataset/{dataset}"] = [
            unit for unit in primary if unit.metric.dataset == dataset
        ]
    for system in sorted({unit.system for unit in primary}):
        groups[f"primary-system/{system}"] = [
            unit for unit in primary if unit.system == system
        ]
    for run_id in sorted({unit.metric.run_id for unit in units}):
        groups[f"run/{run_id}"] = [unit for unit in units if unit.metric.run_id == run_id]

    output: dict[str, dict[str, object]] = {}
    tasks: list[tuple[str, str, list[ScoredUnit], int]] = []
    for group_index, (name, rows) in enumerate(sorted(groups.items())):
        output[name] = {
            "unit_count": len(rows),
            "question_cluster_count": len({unit.cluster for unit in rows}),
            "metrics": {},
        }
        for metric_index, metric_name in enumerate(_available_metrics(rows)):
            summary_seed = seed + group_index * 10_000 + metric_index
            tasks.append((name, metric_name, list(rows), summary_seed))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _metric_alignment_summary,
                rows,
                metric_name,
                bootstrap_samples,
                summary_seed,
            ): (group_name, metric_name)
            for group_name, metric_name, rows, summary_seed in tasks
        }
        for future in as_completed(futures):
            group_name, metric_name = futures[future]
            metrics = output[group_name]["metrics"]
            assert isinstance(metrics, dict)
            metrics[metric_name] = future.result()
    return {
        "schema_version": "1.0",
        "oracle": "released closed-world top-1 answer identity",
        "score_direction": "higher means more correct for every included metric",
        "brier_note": (
            "No test-label calibration is fitted. Scores already defined on [0,1] use identity; "
            "BERTScore uses its theoretical monotone affine map (score + 1) / 2."
        ),
        "groups": output,
    }


def _paired_analysis(
    units: Sequence[ScoredUnit],
    *,
    bootstrap_samples: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    pair_groups = _system_pairs(units)
    comparisons: dict[str, object] = {}
    raw_p_values: dict[str, float] = {}
    grouped_by_dataset: defaultdict[str, list[tuple[ScoredUnit, ScoredUnit]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for (dataset, run_seed), pairs in sorted(pair_groups.items()):
            name = f"{dataset}/seed{run_seed}"
            futures[
                executor.submit(
                    _paired_group_summary,
                    pairs,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + run_seed,
                    include_mcnemar=True,
                )
            ] = (name, dataset, pairs)
        for future in as_completed(futures):
            name, dataset, pairs = futures[future]
            comparison = future.result()
            comparisons[name] = comparison
            raw_p_values[name] = float(
                comparison["native_hit_at_1"]["exact_two_sided_p"]
            )
            grouped_by_dataset[dataset].extend(pairs)
    adjusted = holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        native = comparisons[name]["native_hit_at_1"]
        assert isinstance(native, dict)
        native["holm_adjusted_p_across_dataset_seed_family"] = value

    repeated: dict[str, object] = {}
    for index, (dataset, pairs) in enumerate(sorted(grouped_by_dataset.items())):
        pairs.sort(
            key=lambda pair: (
                pair[0].metric.seed,
                pair[0].metric.source_index,
            )
        )
        repeated[dataset] = _paired_group_summary(
            pairs,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 50_000 + index,
            include_mcnemar=False,
        )
    return {
        "schema_version": "1.0",
        "left_system": "sabet_hard",
        "right_system": "tempo_qr_hard",
        "difference_direction": "left minus right",
        "per_dataset_seed": comparisons,
        "dataset_repeated_seed_descriptive": repeated,
        "repeated_seed_note": (
            "Repeated-seed rows are clustered by source question for bootstrap intervals. "
            "McNemar tests are only reported per dataset/seed because repeated seeds are not "
            "independent questions."
        ),
    }


def _paired_group_summary(
    pairs: Sequence[tuple[ScoredUnit, ScoredUnit]],
    *,
    bootstrap_samples: int,
    seed: int,
    include_mcnemar: bool,
) -> dict[str, object]:
    left_h1 = [int(left.metric.native_hit_at_1) for left, _ in pairs]
    right_h1 = [int(right.metric.native_hit_at_1) for _, right in pairs]
    clusters = [left.cluster for left, _ in pairs]
    h1_differences = [left - right for left, right in zip(left_h1, right_h1, strict=True)]
    native_h1 = clustered_mean_summary(
        h1_differences,
        clusters,
        samples=bootstrap_samples,
        seed=seed,
        batch_size=MEAN_BOOTSTRAP_BATCH_SIZE,
    )
    if include_mcnemar:
        native_h1.update(mcnemar_test(left_h1, right_h1))
    h10_differences = [
        left.metric.native_hit_at_10 - right.metric.native_hit_at_10
        for left, right in pairs
    ]
    native_h10 = clustered_mean_summary(
        h10_differences,
        clusters,
        samples=bootstrap_samples,
        seed=seed + 1,
        batch_size=MEAN_BOOTSTRAP_BATCH_SIZE,
    )
    metric_differences: dict[str, object] = {}
    names = sorted(
        {
            name
            for pair in pairs
            for unit in pair
            for name, value in unit.metric.scores.items()
            if value is not None
        }
    )
    for index, name in enumerate(names):
        available = [
            (left, right)
            for left, right in pairs
            if left.metric.scores.get(name) is not None
            and right.metric.scores.get(name) is not None
        ]
        differences = [
            float(left.metric.scores[name]) - float(right.metric.scores[name])
            for left, right in available
        ]
        if not differences:
            continue
        metric_differences[name] = clustered_mean_summary(
            differences,
            [left.cluster for left, _ in available],
            samples=bootstrap_samples,
            seed=seed + 100 + index,
            batch_size=MEAN_BOOTSTRAP_BATCH_SIZE,
        )
    return {
        "pair_count": len(pairs),
        "native_hit_at_1": native_h1,
        "native_hit_at_10": native_h10,
        "metric_mean_differences": metric_differences,
    }


def _qualitative_audit(
    units: Sequence[ScoredUnit],
    *,
    context_path: Path,
    seed: int,
) -> dict[str, object]:
    primary = [unit for unit in units if unit.metric.seed == PRIMARY_SEED]
    extreme_units: dict[str, tuple[list[ScoredUnit], list[ScoredUnit]]] = {}
    for name in PRIMARY_METRICS:
        available = [unit for unit in primary if unit.metric.scores.get(name) is not None]
        incorrect = sorted(
            (unit for unit in available if unit.metric.native_hit_at_1 == 0),
            key=lambda unit: (-float(unit.metric.scores[name]), _unit_key(unit)),
        )[:10]
        correct = sorted(
            (unit for unit in available if unit.metric.native_hit_at_1 == 1),
            key=lambda unit: (float(unit.metric.scores[name]), _unit_key(unit)),
        )[:10]
        extreme_units[name] = (incorrect, correct)

    collisions = [
        unit
        for unit in primary
        if unit.metric.native_hit_at_1 == 0 and unit.metric.scores.get("exact_match") == 1.0
    ]
    collision_sample = _deterministic_sample_units(collisions, count=100, seed=seed)
    unmapped = [unit for unit in units if _looks_unmapped(unit)]
    missing_reference = [
        unit for unit in units if not _reference_text_available(unit)
    ]
    discordant_units: dict[
        str, tuple[int, list[tuple[ScoredUnit, ScoredUnit]]]
    ] = {}
    for (dataset, run_seed), pairs in sorted(_system_pairs(primary).items()):
        candidates = [
            pair
            for pair in pairs
            if pair[0].metric.native_hit_at_1 != pair[1].metric.native_hit_at_1
        ]
        sampled = _deterministic_sample_pairs(candidates, count=25, seed=seed)
        discordant_units[f"{dataset}/seed{run_seed}"] = (len(candidates), sampled)
    decision_units: dict[
        str,
        tuple[float, str, int, int, int, list[ScoredUnit], list[ScoredUnit]],
    ] = {}
    for name, (threshold, source) in DECISION_THRESHOLDS.items():
        available = [unit for unit in primary if unit.metric.scores.get(name) is not None]
        false_positives = [
            unit
            for unit in available
            if float(unit.metric.scores[name]) >= threshold
            and unit.metric.native_hit_at_1 == 0
        ]
        false_negatives = [
            unit
            for unit in available
            if float(unit.metric.scores[name]) < threshold
            and unit.metric.native_hit_at_1 == 1
        ]
        decision_units[name] = (
            threshold,
            source,
            len(available),
            len(false_positives),
            len(false_negatives),
            _deterministic_sample_units(false_positives, count=25, seed=seed),
            _deterministic_sample_units(false_negatives, count=25, seed=seed),
        )

    selected_keys: set[tuple[str, int]] = set()
    for incorrect, correct in extreme_units.values():
        selected_keys.update(_unit_context_key(unit) for unit in (*incorrect, *correct))
    selected_keys.update(_unit_context_key(unit) for unit in collision_sample)
    selected_keys.update(_unit_context_key(unit) for unit in missing_reference)
    selected_keys.update(_unit_context_key(unit) for unit in unmapped)
    for _population_count, pairs in discordant_units.values():
        for left, right in pairs:
            selected_keys.add(_unit_context_key(left))
            selected_keys.add(_unit_context_key(right))
    for decision in decision_units.values():
        selected_keys.update(
            _unit_context_key(unit) for unit in (*decision[5], *decision[6])
        )
    contexts = _load_selected_contexts(context_path, selected_keys)

    extremes = {
        name: {
            "highest_scoring_identity_incorrect": [
                _audit_row(unit, contexts=contexts, metric_name=name)
                for unit in incorrect
            ],
            "lowest_scoring_identity_correct": [
                _audit_row(unit, contexts=contexts, metric_name=name)
                for unit in correct
            ],
        }
        for name, (incorrect, correct) in extreme_units.items()
    }
    discordant = {
        name: {
            "population_count": population_count,
            "sample": [
                _discordant_row(left, right, contexts=contexts)
                for left, right in pairs
            ],
        }
        for name, (population_count, pairs) in discordant_units.items()
    }
    decisions = {
        name: {
            "threshold": threshold,
            "threshold_source": source,
            "n": available_count,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "false_positive_sample": [
                _audit_row(unit, contexts=contexts, metric_name=name)
                for unit in false_positive_sample
            ],
            "false_negative_sample": [
                _audit_row(unit, contexts=contexts, metric_name=name)
                for unit in false_negative_sample
            ],
        }
        for name, (
            threshold,
            source,
            available_count,
            false_positive_count,
            false_negative_count,
            false_positive_sample,
            false_negative_sample,
        ) in decision_units.items()
    }
    return {
        "schema_version": "1.0",
        "selection_rules": {
            "metric_extremes": "deterministic top/bottom 10; ties use unit key",
            "identity_label_collisions": "SHA-256-ranked sample of at most 100",
            "system_discordance": "SHA-256-ranked sample of at most 25 per dataset/seed",
            "missing_or_unmapped": "complete population, no sampling",
            "decision_errors": (
                "only metrics with a definition-level or published/frozen decision boundary; "
                "no threshold is fitted on these test labels"
            ),
            "seed": seed,
        },
        "metric_extremes": extremes,
        "identity_incorrect_exact_label_collision": {
            "population_count": len(collisions),
            "sample": [
                _audit_row(unit, contexts=contexts, metric_name="exact_match")
                for unit in collision_sample
            ],
        },
        "model_discordant": discordant,
        "fixed_threshold_decision_errors": decisions,
        "missing_reference": [
            _audit_row(unit, contexts=contexts) for unit in missing_reference
        ],
        "apparently_unmapped_entity_label": [
            _audit_row(unit, contexts=contexts) for unit in unmapped
        ],
    }


def _mean_scores(rows: Sequence[ScoredUnit]) -> dict[str, object]:
    means: dict[str, object] = {
        "native_hit_at_1": _mean(unit.metric.native_hit_at_1 for unit in rows),
        "native_hit_at_10": _mean(unit.metric.native_hit_at_10 for unit in rows),
    }
    for name in _available_metrics(rows):
        values = [
            float(unit.metric.scores[name])
            for unit in rows
            if unit.metric.scores.get(name) is not None
        ]
        means[name] = {"n": len(values), "mean": _mean(values)}
    return {
        "unit_count": len(rows),
        "question_cluster_count": len({unit.cluster for unit in rows}),
        "means": means,
    }


def _primary_dataset_macro_means(units: Sequence[ScoredUnit]) -> dict[str, object]:
    output: dict[str, object] = {}
    for system in sorted({unit.system for unit in units}):
        system_rows = [unit for unit in units if unit.system == system]
        datasets = sorted({unit.metric.dataset for unit in system_rows})
        metric_names = ["native_hit_at_1", "native_hit_at_10", *_available_metrics(system_rows)]
        means: dict[str, object] = {}
        for name in metric_names:
            dataset_values: list[float] = []
            for dataset in datasets:
                selected = [unit for unit in system_rows if unit.metric.dataset == dataset]
                if name == "native_hit_at_1":
                    values = [unit.metric.native_hit_at_1 for unit in selected]
                elif name == "native_hit_at_10":
                    values = [unit.metric.native_hit_at_10 for unit in selected]
                else:
                    values = [
                        float(unit.metric.scores[name])
                        for unit in selected
                        if unit.metric.scores.get(name) is not None
                    ]
                if values:
                    dataset_values.append(float(sum(values) / len(values)))
            means[name] = {
                "dataset_count": len(dataset_values),
                "macro_mean": _mean(dataset_values),
            }
        output[system] = {"datasets": datasets, "means": means}
    return output


def _system_pairs(
    units: Sequence[ScoredUnit],
) -> dict[tuple[str, int], list[tuple[ScoredUnit, ScoredUnit]]]:
    grouped: defaultdict[tuple[str, int, int], dict[str, ScoredUnit]] = defaultdict(dict)
    for unit in units:
        key = (unit.metric.dataset, unit.metric.seed, unit.metric.source_index)
        if unit.system in grouped[key]:
            raise ValueError(f"Duplicate system output for paired key {key}: {unit.system}")
        grouped[key][unit.system] = unit
    output: defaultdict[tuple[str, int], list[tuple[ScoredUnit, ScoredUnit]]] = defaultdict(list)
    for (dataset, run_seed, _source_index), systems in grouped.items():
        if set(systems) != {"sabet_hard", "tempo_qr_hard"}:
            raise ValueError(
                "Incomplete paired system output for "
                f"{dataset}/seed{run_seed}/source{_source_index}: {sorted(systems)}"
            )
        left = systems["sabet_hard"]
        right = systems["tempo_qr_hard"]
        if left.cluster != right.cluster:
            raise ValueError("Paired outputs do not identify the same released source record")
        if (
            left.context["source_record_sha256"]
            != right.context["source_record_sha256"]
        ):
            raise ValueError(
                "Paired systems expose different source-record hashes for the same seed"
            )
        output[(dataset, run_seed)].append((left, right))
    for pairs in output.values():
        pairs.sort(key=lambda pair: pair[0].metric.source_index)
    return dict(output)


def _bounded_scores(name: str, values: Sequence[float]) -> tuple[list[float], str]:
    if name.startswith("bertscore_"):
        if any(value < -1.000001 or value > 1.000001 for value in values):
            raise ValueError(f"BERTScore value outside theoretical range for {name}")
        return [min(1.0, max(0.0, (value + 1.0) / 2.0)) for value in values], (
            "theoretical affine [-1,1] to [0,1]"
        )
    if any(value < -1e-6 or value > 1.000001 for value in values):
        raise ValueError(f"Metric value outside [0,1] for {name}")
    return [min(1.0, max(0.0, value)) for value in values], "identity [0,1]"


def _metric_alignment_summary(
    rows: Sequence[ScoredUnit],
    metric_name: str,
    bootstrap_samples: int,
    summary_seed: int,
) -> dict[str, object]:
    available = [
        unit for unit in rows if unit.metric.scores.get(metric_name) is not None
    ]
    raw_scores = [float(unit.metric.scores[metric_name]) for unit in available]
    bounded_scores, transform = _bounded_scores(metric_name, raw_scores)
    labels = [int(unit.metric.native_hit_at_1) for unit in available]
    if len(set(labels)) == 2:
        summary = binary_metric_summary(
            bounded_scores,
            labels,
            [unit.cluster for unit in available],
            samples=bootstrap_samples,
            seed=summary_seed,
            batch_size=BINARY_BOOTSTRAP_BATCH_SIZE,
        )
    else:
        summary = {
            "n": len(available),
            "cluster_count": len({unit.cluster for unit in available}),
            "positive_count": sum(labels),
            "positive_rate": sum(labels) / len(labels) if labels else None,
            "not_estimable": "both native correctness classes are required",
        }
    summary.update(
        {
            "score_transform_for_bounded_analysis": transform,
            "raw_score_mean": sum(raw_scores) / len(raw_scores) if raw_scores else None,
            "excluded_null_score_count": len(rows) - len(available),
        }
    )
    return summary


def _available_metrics(rows: Sequence[ScoredUnit]) -> list[str]:
    return sorted(
        {
            name
            for unit in rows
            for name, value in unit.metric.scores.items()
            if value is not None
        }
    )


def _validate_context_alignment(unit: ScoredUnit) -> None:
    expected = {
        "run_id": unit.metric.run_id,
        "dataset": unit.metric.dataset,
        "model": unit.metric.model,
        "variant": unit.metric.variant,
        "seed": unit.metric.seed,
        "source_index": unit.metric.source_index,
        "qid": unit.metric.qid,
        "question_type": unit.metric.question_type,
        "answer_type": unit.metric.answer_type,
    }
    for name, value in expected.items():
        if unit.context.get(name) != value:
            raise ValueError(
                f"Context mismatch for {unit.metric.run_id}/{unit.metric.source_index}: {name}"
            )
    for name in ("gold_answer_ids", "gold_answer_labels"):
        values = unit.context.get(name)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(
                f"Invalid {name} for {unit.metric.run_id}/{unit.metric.source_index}"
            )
    if len(unit.context["gold_answer_ids"]) != len(unit.context["gold_answer_labels"]):
        raise ValueError(
            f"Misaligned gold references for {unit.metric.run_id}/{unit.metric.source_index}"
        )
    source_hash = str(unit.context["source_record_sha256"])
    if len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise ValueError(
            f"Invalid source hash for {unit.metric.run_id}/{unit.metric.source_index}"
        )
    for field in ("reference_text_available", "candidate_text_available"):
        if not isinstance(unit.metric.applicability.get(field), bool):
            raise ValueError(
                f"Missing {field} applicability for "
                f"{unit.metric.run_id}/{unit.metric.source_index}"
            )


def _validate_scores(unit: ScoredUnit) -> None:
    if unit.metric.native_hit_at_1 not in {0.0, 1.0}:
        raise ValueError("Native Hits@1 must be binary per question")
    if unit.metric.native_hit_at_10 not in {0.0, 1.0}:
        raise ValueError("Native Hits@10 must be binary per question")
    for name, value in unit.metric.scores.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite unit score: {name}")


def _validate_repeated_question_identity(units: Sequence[ScoredUnit]) -> None:
    seen: dict[tuple[str, int], str] = {}
    for unit in units:
        key = (unit.metric.dataset, unit.metric.source_index)
        identity = (
            unit.context.question.identity_sha256
            if isinstance(unit.context, _CompactContext)
            else _question_identity_sha256(unit.context)
        )
        previous = seen.setdefault(key, identity)
        if previous != identity:
            raise ValueError(f"Released question identity changed across runs: {key}")


def _looks_unmapped(unit: ScoredUnit) -> bool:
    if unit.metric.answer_type != "entity" or unit.metric.dataset == "MultiTQ":
        return False
    if unit.context.get("resolved_predicted_answer") is not None:
        return False
    answer_id = str(unit.context["predicted_answer_id"]).removeprefix("entity:")
    label = str(unit.context["predicted_answer_label"]).strip()
    return label == answer_id or label == str(unit.context["predicted_answer_id"])


def _audit_row(
    unit: ScoredUnit,
    *,
    contexts: Mapping[tuple[str, int], dict[str, object]],
    metric_name: str | None = None,
) -> dict[str, object]:
    context = contexts[_unit_context_key(unit)]
    output = {
        "run_id": unit.metric.run_id,
        "dataset": unit.metric.dataset,
        "system": unit.system,
        "seed": unit.metric.seed,
        "source_index": unit.metric.source_index,
        "qid": context["qid"],
        "question": context["question"],
        "question_type": context["question_type"],
        "answer_type": context["answer_type"],
        "gold_answer_ids": context["gold_answer_ids"],
        "gold_answer_labels": context["gold_answer_labels"],
        "resolved_gold_answer_labels": context.get(
            "resolved_gold_answer_labels"
        ),
        "predicted_answer_id": context["predicted_answer_id"],
        "predicted_answer_label": context["predicted_answer_label"],
        "resolved_predicted_answer": context.get(
            "resolved_predicted_answer"
        ),
        "native_hit_at_1": unit.metric.native_hit_at_1,
    }
    if metric_name is not None:
        output["metric"] = metric_name
        output["metric_score"] = unit.metric.scores.get(metric_name)
    return output


def _discordant_row(
    left: ScoredUnit,
    right: ScoredUnit,
    *,
    contexts: Mapping[tuple[str, int], dict[str, object]],
) -> dict[str, object]:
    context = contexts[_unit_context_key(left)]
    return {
        "dataset": left.metric.dataset,
        "seed": left.metric.seed,
        "source_index": left.metric.source_index,
        "qid": context["qid"],
        "question": context["question"],
        "gold_answer_ids": context["gold_answer_ids"],
        "gold_answer_labels": context["gold_answer_labels"],
        "sabet": _audit_row(left, contexts=contexts),
        "tempo_qr": _audit_row(right, contexts=contexts),
    }


def _load_selected_contexts(
    context_path: Path,
    selected_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, object]]:
    if not selected_keys:
        return {}
    contexts: dict[tuple[str, int], dict[str, object]] = {}
    for line_number, row in _iter_jsonl(context_path):
        run_id = _required_string(row, "run_id", context_path, line_number)
        source_index = row.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"Invalid source_index at {context_path}:{line_number}")
        key = (run_id, source_index)
        if key not in selected_keys:
            continue
        if key in contexts:
            raise ValueError(f"Duplicate selected context unit: {key}")
        contexts[key] = row
        if len(contexts) == len(selected_keys):
            break
    missing = sorted(selected_keys - set(contexts))[:10]
    if missing:
        raise ValueError(f"Selected qualitative contexts are missing: {missing}")
    return contexts


def _question_identity_sha256(context: Mapping[str, object]) -> str:
    payload = {
        "qid": context.get("qid"),
        "question": context.get("question"),
        "gold_answer_ids": context.get("gold_answer_ids"),
        "gold_answer_labels": context.get("gold_answer_labels"),
        "question_type": context.get("question_type"),
        "answer_type": context.get("answer_type"),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _reference_text_available(unit: ScoredUnit) -> bool:
    if isinstance(unit.metric, _CompactMetric):
        return unit.metric.reference_text_available
    return bool(unit.metric.applicability["reference_text_available"])


def _unit_context_key(unit: ScoredUnit) -> tuple[str, int]:
    return unit.metric.run_id, unit.metric.source_index


def _deterministic_sample_units(
    units: Sequence[ScoredUnit], *, count: int, seed: int
) -> list[ScoredUnit]:
    return sorted(
        units,
        key=lambda unit: hashlib.sha256(
            f"{seed}:{_unit_key(unit)}".encode()
        ).hexdigest(),
    )[:count]


def _deterministic_sample_pairs(
    pairs: Sequence[tuple[ScoredUnit, ScoredUnit]], *, count: int, seed: int
) -> list[tuple[ScoredUnit, ScoredUnit]]:
    return sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{seed}:{_unit_key(pair[0])}".encode()
        ).hexdigest(),
    )[:count]


def _unit_key(unit: ScoredUnit) -> str:
    return f"{unit.metric.run_id}:{unit.metric.source_index}"


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(sum(materialized) / len(materialized)) if materialized else None


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield line_number, row


def _required_string(
    row: dict[str, object], name: str, path: Path, line_number: int
) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {name} at {path}:{line_number}")
    return value


def _file_identity(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        line_count = sum(block.count(b"\n") for block in iter(lambda: stream.read(1024**2), b""))
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "line_count": line_count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reproduced SABET metric outputs")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    outputs = analyze_external_metrics(
        scores_path=args.scores,
        context_path=args.context,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
