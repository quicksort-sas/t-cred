from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import orjson

from tcred.metrics.diagnostic_analysis import _analyze_construct
from tcred.metrics.diagnostic_models import diagnostic_inference_cluster_ids
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.source_disjoint_io import load_frozen_suite_for_retrospective_scoring
from tcred.metrics.statistics import (
    LABEL_SCORE,
    correlation_summary,
    mean_interval,
    paired_spearman_difference,
)
from tcred.metrics.task_judge_models import TaskJudgeInput
from tcred.metrics.tcred_models import TCredMetricResult
from tcred.trainable_metrics.artifacts import file_sha256, validate_training_export
from tcred.trainable_metrics.inference import SemanticInferenceInput, TCredSLInference
from tcred.trainable_metrics.schema import EvidencePassage, SemanticTask
from tcred.trainable_metrics.suite import score_metric_cases

HUMAN_COMPARISONS = (
    (
        "answer_correctness",
        "answer_correct",
        "tcred_sl_answer_equivalence_semantic",
        "tcred_answer_equivalence",
    ),
    (
        "evidence_support",
        "evidence_supports_answer",
        "tcred_sl_evidence_support",
        "tcred_semantic_attribution",
    ),
    (
        "temporal_correctness",
        "temporal_correct",
        "tcred_sl_temporal_correctness",
        "tcred_temporal_correctness",
    ),
    (
        "temporal_attribution",
        "temporal_correct",
        "tcred_sl_temporal_attribution",
        "tcred_temporal_attribution",
    ),
    (
        "citation_quality",
        "citation_temporally_valid",
        "tcred_sl_citation_quality",
        "tcred_citation_quality",
    ),
    (
        "graph_sufficiency",
        "graph_evidence_sufficient",
        "tcred_sl_graph_sufficiency",
        "tcred_graph_answer_coverage",
    ),
    (
        "response_decision",
        "response_decision_appropriate",
        "tcred_sl_response_decision",
        "tcred_response_decision",
    ),
)

DIAGNOSTIC_COMPARISONS = {
    "answer_correctness": (
        "tcred_sl_answer_equivalence_semantic",
        "tcred_answer_equivalence",
    ),
    "temporal_correctness": (
        "tcred_sl_temporal_correctness",
        "tcred_temporal_correctness",
    ),
    "temporal_attribution": (
        "tcred_sl_temporal_attribution",
        "tcred_temporal_attribution",
    ),
    "evidence_support": (
        "tcred_sl_evidence_support",
        "tcred_semantic_attribution",
    ),
    "citation_correctness": (
        "tcred_sl_citation_quality",
        "tcred_citation_quality",
    ),
    "graph_sufficiency": (
        "tcred_sl_graph_sufficiency",
        "tcred_graph_answer_coverage",
    ),
    "response_decision": (
        "tcred_sl_response_decision",
        "tcred_response_decision",
    ),
    "retrieval_quality": (
        "tcred_sl_retrieval_relevance",
        "tcred_t_ndcg_at_10",
    ),
}


def evaluate_checkpoint_against_tcred_v14(
    *,
    repository_root: Path,
    export_root: Path,
    backbone_dir: Path,
    population_dir: Path,
    source_disjoint_root: Path,
    output_dir: Path,
    batch_size: int = 64,
    bootstrap_samples: int = 2_000,
    seed: int = 20260816,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the frozen checkpoint and paired retrospective T-CRED v1.4 comparison."""

    repository_root = repository_root.resolve()
    export_root = _resolve(repository_root, export_root)
    backbone_dir = _resolve(repository_root, backbone_dir)
    population_dir = _resolve(repository_root, population_dir)
    source_disjoint_root = _resolve(repository_root, source_disjoint_root)
    output_dir = _resolve(repository_root, output_dir)
    _prepare_output(output_dir, overwrite=overwrite)

    artifact_validation = validate_training_export(
        export_root,
        output_path=output_dir / "artifact_validation.json",
    )
    engine = TCredSLInference(
        model_dir=export_root / "final_model",
        backbone_dir=backbone_dir,
    )

    population_input_validation = _validate_directory_manifest(
        population_dir,
        required_files=(
            "task_judge_inputs.jsonl",
            "tcred_component_results.jsonl",
            "metric_scores_with_tcred.jsonl",
        ),
    )
    population_rows = _read_models(population_dir / "task_judge_inputs.jsonl", TaskJudgeInput)
    population_exact = _read_models(
        population_dir / "tcred_component_results.jsonl", TCredMetricResult
    )
    population_baseline = _read_models(
        population_dir / "metric_scores_with_tcred.jsonl", MetricScoreRecord
    )
    population_learned, population_runtime = score_metric_cases(
        engine=engine,
        rows=population_rows,
        exact_results=population_exact,
        batch_size=batch_size,
    )
    population_merged = _merge_scores(population_baseline, population_learned)
    _write_models(output_dir / "population_scores_with_tcred_sl.jsonl", population_merged)

    suite, protocol, source_disjoint_integrity = load_frozen_suite_for_retrospective_scoring(
        repository_root=repository_root,
        study_root=source_disjoint_root,
    )
    source_score_input_validation = _validate_source_score_inputs(
        source_disjoint_root,
        source_disjoint_integrity=source_disjoint_integrity,
    )
    diagnostic_rows = [case.task_judge_input for case in suite.cases]
    diagnostic_exact = _read_models(
        source_disjoint_root / "tcred" / "tcred_component_results.jsonl",
        TCredMetricResult,
    )
    diagnostic_baseline = _read_models(
        source_disjoint_root / "tcred" / "metric_scores_with_tcred.jsonl",
        MetricScoreRecord,
    )
    diagnostic_learned, diagnostic_runtime = score_metric_cases(
        engine=engine,
        rows=diagnostic_rows,
        exact_results=diagnostic_exact,
        batch_size=batch_size,
    )
    diagnostic_merged = _merge_scores(diagnostic_baseline, diagnostic_learned)
    _write_models(output_dir / "source_disjoint_scores_with_tcred_sl.jsonl", diagnostic_merged)

    human = _human_comparison(
        population_merged,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    diagnostic = _diagnostic_comparison(
        suite=suite,
        records=diagnostic_merged,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    population = _population_description(
        population_merged,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    resource = _resource_benchmark(engine, population_rows[:32])
    training = {
        "run_manifest": _read_object(export_root / "run_manifest.json"),
        "development_stage_a": _read_object(export_root / "development_stage_a.json"),
        "development_stage_b": _read_object(export_root / "development_stage_b.json"),
        "calibration_evaluation": _read_object(export_root / "calibration_evaluation.json"),
    }
    result = {
        "schema_version": "tcred-sl-v14-retrospective-comparison-v1",
        "study_status": "retrospective_single_seed_not_confirmatory",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "artifact_validation": artifact_validation,
        "input_validation": {
            "population": population_input_validation,
            "source_disjoint_frozen_suite": source_disjoint_integrity,
            "source_disjoint_baseline_scores": source_score_input_validation,
        },
        "training": training,
        "human_gold": human,
        "source_disjoint_diagnostics": diagnostic,
        "population_description": population,
        "runtime": {
            "population": population_runtime,
            "source_disjoint": diagnostic_runtime,
            "semantic_microbenchmark": resource,
        },
        "protocol": {
            "source_disjoint_protocol_id": protocol.get("protocol_id"),
            "model_seed": 42,
            "comparison_boundary": (
                "Learned semantic components versus frozen non-trainable T-CRED v1.4 components; "
                "both use the same exact deterministic temporal/graph/citation layer."
            ),
            "no_global_scalar": True,
            "human_gold_role": (
                "Concurrent validity only; sparse fields and retrospective exposure prohibit a "
                "benchmark-level or confirmatory superiority claim."
            ),
            "unlabeled_population_role": (
                "Descriptive score means only; 2,344 unlabeled system outputs do not increase "
                "the effective human-validity sample size."
            ),
        },
    }
    _write_json(output_dir / "comparison.json", result)
    report = _render_report(result)
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    manifest = _output_manifest(output_dir)
    _write_json(output_dir / "manifest.json", manifest)
    return result


def _human_comparison(
    records: list[MetricScoreRecord],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    human = [record for record in records if record.population == "human_gold"]
    output: dict[str, Any] = {}
    for index, (name, field, learned_name, baseline_name) in enumerate(HUMAN_COMPARISONS):
        selected = [
            record
            for record in human
            if record.gold_labels.get(field) in LABEL_SCORE
            and record.scores.get(learned_name) is not None
            and record.scores.get(baseline_name) is not None
        ]
        labels = [LABEL_SCORE[record.gold_labels[field]] for record in selected]
        learned = [float(record.scores[learned_name]) for record in selected]
        baseline = [float(record.scores[baseline_name]) for record in selected]
        clusters = [f"{record.dataset_family}\x1f{record.qid}" for record in selected]
        local_seed = seed + index * 10_003
        output[name] = {
            "human_field": field,
            "learned_metric": learned_name,
            "baseline_metric": baseline_name,
            "learned": _human_metric_summary(
                learned,
                labels,
                clusters,
                samples=bootstrap_samples,
                seed=local_seed,
            ),
            "baseline": _human_metric_summary(
                baseline,
                labels,
                clusters,
                samples=bootstrap_samples,
                seed=local_seed + 1,
            ),
            "paired_spearman_difference_learned_minus_baseline": paired_spearman_difference(
                learned,
                baseline,
                labels,
                clusters,
                samples=bootstrap_samples,
                seed=local_seed + 2,
            ),
            "paired_absolute_error_difference_learned_minus_baseline": (
                _paired_error_difference(
                    learned,
                    baseline,
                    labels,
                    clusters,
                    samples=bootstrap_samples,
                    seed=local_seed + 3,
                )
            ),
            "label_distribution": {
                label: sum(record.gold_labels[field] == label for record in selected)
                for label in ("yes", "partial", "no")
            },
        }
    return {
        "units": len(human),
        "comparisons": output,
        "interpretation": (
            "Positive Spearman differences favor T-CRED-SL. Negative absolute-error differences "
            "favor T-CRED-SL. Confidence intervals crossing zero are inconclusive."
        ),
    }


def _human_metric_summary(
    scores: list[float],
    labels: list[float],
    clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    summary = correlation_summary(scores, labels, clusters, samples=samples, seed=seed)
    summary.update(
        {
            "mae": float(np.mean(np.abs(np.asarray(scores) - np.asarray(labels))))
            if scores
            else None,
            "brier": float(np.mean((np.asarray(scores) - np.asarray(labels)) ** 2))
            if scores
            else None,
            **_binary_threshold_summary(scores, labels),
        }
    )
    return summary


def _binary_threshold_summary(scores: list[float], labels: list[float]) -> dict[str, Any]:
    selected = [(score, label) for score, label in zip(scores, labels, strict=True) if label != 0.5]
    if not selected:
        return {"binary_macro_f1_at_0_5": None, "binary_accuracy_at_0_5": None}
    targets = [int(label) for _score, label in selected]
    predictions = [int(score >= 0.5) for score, _label in selected]
    class_f1 = []
    for positive in (0, 1):
        tp = sum(p == positive and y == positive for p, y in zip(predictions, targets, strict=True))
        fp = sum(p == positive and y != positive for p, y in zip(predictions, targets, strict=True))
        fn = sum(p != positive and y == positive for p, y in zip(predictions, targets, strict=True))
        denominator = 2 * tp + fp + fn
        class_f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "binary_macro_f1_at_0_5": mean(class_f1),
        "binary_accuracy_at_0_5": mean(
            prediction == target for prediction, target in zip(predictions, targets, strict=True)
        ),
    }


def _paired_error_difference(
    learned: list[float],
    baseline: list[float],
    labels: list[float],
    clusters: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not learned:
        return {"n": 0, "mean_difference": None, "ci95": None}
    differences = [
        abs(left - target) - abs(right - target)
        for left, right, target in zip(learned, baseline, labels, strict=True)
    ]
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(differences, clusters, strict=True):
        grouped[cluster].append(value)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in sampled_keys for value in grouped[str(key)]]
        estimates.append(float(np.mean(values)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "n": len(differences),
        "mean_difference": float(np.mean(differences)),
        "ci95": [float(low), float(high)],
    }


def _diagnostic_comparison(
    *,
    suite: Any,
    records: list[MetricScoreRecord],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    scores = {record.metric_id: record.scores for record in records}
    expected = {case.case_id for case in suite.cases}
    if set(scores) != expected:
        raise ValueError("Source-disjoint score IDs do not match the challenge")
    cluster_ids = diagnostic_inference_cluster_ids(suite.cases, suite.pairs)
    constructs = {}
    for index, (construct, metrics) in enumerate(DIAGNOSTIC_COMPARISONS.items()):
        pairs = [pair for pair in suite.pairs if pair.target_construct == construct]
        constructs[construct] = _analyze_construct(
            pairs,
            metrics=list(metrics),
            scores=scores,
            cluster_ids=cluster_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 10_003,
        )
    return {
        "cases": len(suite.cases),
        "pairs": len(suite.pairs),
        "constructs": constructs,
        "interpretation": (
            "Directional utility is higher-is-better; invariance absolute change is "
            "lower-is-better. "
            "Pairwise comparisons use connected-cluster bootstrap and paired randomization."
        ),
    }


def _population_description(
    records: list[MetricScoreRecord],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    system_rows = [record for record in records if record.population == "system_full"]
    metric_names = sorted(
        {
            metric
            for _name, _field, learned, baseline in HUMAN_COMPARISONS
            for metric in (learned, baseline)
        }
        | {
            "tcred_sl_retrieval_relevance",
            "tcred_t_ndcg_at_10",
        }
    )

    def grouped_summary(key) -> dict[str, Any]:
        groups: defaultdict[str, list[MetricScoreRecord]] = defaultdict(list)
        for record in system_rows:
            groups[str(key(record))].append(record)
        output = {}
        for group_name, group in sorted(groups.items()):
            output[group_name] = {}
            for metric_name in metric_names:
                selected = [
                    record for record in group if record.scores.get(metric_name) is not None
                ]
                output[group_name][metric_name] = mean_interval(
                    [float(record.scores[metric_name]) for record in selected],
                    [f"{record.dataset_family}\x1f{record.qid}" for record in selected],
                    samples=bootstrap_samples,
                    seed=seed + _stable_offset(f"{group_name}:{metric_name}"),
                )
        return output

    return {
        "rows": len(system_rows),
        "by_system": grouped_summary(lambda record: record.system_name or "none"),
        "by_dataset": grouped_summary(lambda record: record.dataset_family),
        "interpretation": "Unlabeled means are descriptive and are not accuracy estimates.",
    }


def _resource_benchmark(
    engine: TCredSLInference,
    rows: list[TaskJudgeInput],
) -> dict[str, Any]:
    inputs = []
    for index, row in enumerate(rows):
        evidence = row.displayed_evidence()[:1]
        inputs.append(
            SemanticInferenceInput(
                input_id=f"benchmark::{index}",
                task=SemanticTask.ANSWER,
                question=row.question,
                reference_answers=[row.reference_answer],
                candidate_or_claim=row.candidate_answer,
                evidence_passages=[
                    EvidencePassage(evidence_id=item.evidence_id, text=item.text, rank=1)
                    for item in evidence
                ],
            )
        )
    if not inputs:
        return {"examples": 0}
    engine.predict(inputs[: min(8, len(inputs))], batch_size=min(8, len(inputs)))
    results = {}
    for batch_size in (1, 8, 32):
        elapsed = []
        for _ in range(2):
            started = time.perf_counter()
            engine.predict(inputs, batch_size=batch_size)
            elapsed.append(time.perf_counter() - started)
        duration = min(elapsed)
        results[str(batch_size)] = {
            "examples": len(inputs),
            "best_of_two_seconds": duration,
            "semantic_inputs_per_second": len(inputs) / duration,
            "milliseconds_per_input": 1_000 * duration / len(inputs),
        }
    return {
        "examples": len(inputs),
        "scope": "End-to-end tokenizer plus one ANSWER-head forward pass per semantic input.",
        "batch_sizes": results,
        "model_parameters": engine.runtime["model_parameters"],
        "weight_bytes": (engine.model_dir / "model.safetensors").stat().st_size,
        "device": engine.runtime["device"],
        "runtime": engine.runtime,
    }


def _merge_scores(
    baseline: list[MetricScoreRecord],
    learned: list[MetricScoreRecord],
) -> list[MetricScoreRecord]:
    learned_by_id = {record.metric_id: record for record in learned}
    if len(learned_by_id) != len(learned):
        raise ValueError("Learned score rows contain duplicate metric IDs")
    if {record.metric_id for record in baseline} != set(learned_by_id):
        raise ValueError("Baseline and learned score IDs differ")
    output = []
    for record in baseline:
        extra = learned_by_id[record.metric_id]
        shared = set(record.scores) & set(extra.scores)
        if shared:
            raise ValueError(f"Score-name collision while merging {record.metric_id}: {shared}")
        identity = (
            "population",
            "dataset_family",
            "source_kind",
            "system_name",
            "unit_id",
            "qid",
            "scenario_id",
            "gold_labels",
        )
        if any(getattr(record, name) != getattr(extra, name) for name in identity):
            raise ValueError(f"Baseline/learned identity mismatch: {record.metric_id}")
        output.append(
            record.model_copy(
                update={
                    "scores": record.scores | extra.scores,
                    "metric_metadata": record.metric_metadata | extra.metric_metadata,
                }
            )
        )
    return output


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# T-CRED-SL Seed-42 Retrospective Comparison",
        "",
        "> This is a single-seed retrospective evaluation. It is not confirmatory evidence of "
        "universal or strict dominance.",
        "",
        "## Training and Artifact",
        "",
        f"- Export validation: **{result['artifact_validation']['status']}**",
        f"- Completed steps: **{result['training']['run_manifest']['completed_steps']} / "
        f"{result['training']['run_manifest']['total_planned_steps']}**",
        f"- Training time: **{result['training']['run_manifest']['elapsed_seconds']:.1f} s**",
        f"- Weight SHA-256: `{result['artifact_validation']['model']['weight_sha256']}`",
        "",
        "## Human Gold",
        "",
        "| Construct | n | T-CRED-SL rho | T-CRED v1.4 rho | Delta rho (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["human_gold"]["comparisons"].items():
        learned = row["learned"]
        baseline = row["baseline"]
        delta = row["paired_spearman_difference_learned_minus_baseline"]
        lines.append(
            f"| {name.replace('_', ' ')} | {learned['n']} | {_fmt(learned['spearman'])} | "
            f"{_fmt(baseline['spearman'])} | {_fmt(delta['mean_difference'])} "
            f"{_interval(delta['ci95'])} |"
        )
    lines.extend(
        [
            "",
            "## Source-Disjoint Diagnostics",
            "",
            "| Construct | Directional macro: SL / v1.4 | Invariance change: SL / v1.4 |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, row in result["source_disjoint_diagnostics"]["constructs"].items():
        learned_name, baseline_name = DIAGNOSTIC_COMPARISONS[name]
        directional = row["directional_metrics"]
        invariant = row["invariance_metrics"]
        learned_directional = directional.get(learned_name, {}).get(
            "macro_phenomenon_pairwise_accuracy"
        )
        baseline_directional = directional.get(baseline_name, {}).get(
            "macro_phenomenon_pairwise_accuracy"
        )
        learned_invariance = invariant.get(learned_name, {}).get("macro_phenomenon_absolute_change")
        baseline_invariance = invariant.get(baseline_name, {}).get(
            "macro_phenomenon_absolute_change"
        )
        lines.append(
            f"| {name.replace('_', ' ')} | "
            f"{_fmt(learned_directional)} / {_fmt(baseline_directional)} | "
            f"{_fmt(learned_invariance)} / {_fmt(baseline_invariance)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The human sample is sparse by field, the formal suite is controlled, and this is one "
            "training seed. The learned suite is better only where paired evidence and uncertainty "
            "support that local statement; no aggregate scalar or universal superiority claim "
            "is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [item for item in path.iterdir() if item.is_file()]
    if existing and not overwrite:
        raise FileExistsError(f"Comparison output is not empty: {path}")
    if overwrite:
        allowed = {
            "artifact_validation.json",
            "population_scores_with_tcred_sl.jsonl",
            "source_disjoint_scores_with_tcred_sl.jsonl",
            "comparison.json",
            "comparison_report.md",
            "manifest.json",
        }
        unexpected = sorted(item.name for item in existing if item.name not in allowed)
        if unexpected:
            raise ValueError(f"Refusing to overwrite unexpected comparison artifacts: {unexpected}")
        for item in existing:
            item.unlink()


def _read_models(path: Path, model: Any) -> list[Any]:
    return [
        model.model_validate(orjson.loads(line)) for line in path.read_bytes().splitlines() if line
    ]


def _read_object(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_directory_manifest(
    root: Path,
    *,
    required_files: tuple[str, ...],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Artifact manifest has no file inventory: {manifest_path}")
    inventory = {
        str(record.get("path")): record
        for record in artifacts
        if isinstance(record, dict) and record.get("path")
    }
    checked = []
    errors = []
    for relative in required_files:
        record = inventory.get(relative)
        path = root / relative
        if record is None:
            errors.append({"path": relative, "kind": "missing_manifest_entry"})
            continue
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = file_sha256(path) if exists else None
        item = {
            "path": relative,
            "expected_bytes": record.get("bytes"),
            "actual_bytes": actual_bytes,
            "expected_sha256": record.get("sha256"),
            "actual_sha256": actual_sha256,
        }
        if not exists:
            errors.append({**item, "kind": "missing_file"})
        elif actual_bytes != record.get("bytes"):
            errors.append({**item, "kind": "size_mismatch"})
        elif actual_sha256 != record.get("sha256"):
            errors.append({**item, "kind": "hash_mismatch"})
        checked.append(item)
    if errors:
        raise ValueError(f"Frozen metric inputs failed manifest validation: {errors}")
    return {
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "files_checked": checked,
    }


def _validate_source_score_inputs(
    source_disjoint_root: Path,
    *,
    source_disjoint_integrity: dict[str, Any],
) -> dict[str, Any]:
    score_root = source_disjoint_root / "tcred"
    report = _validate_directory_manifest(
        score_root,
        required_files=(
            "tcred_component_results.jsonl",
            "metric_scores_with_tcred.jsonl",
        ),
    )
    manifest = _read_object(score_root / "manifest.json")
    expected_lock = source_disjoint_integrity["implementation_lock_sha256"]
    if manifest.get("implementation_lock_sha256") != expected_lock:
        raise ValueError("Source-disjoint score manifest refers to a different implementation lock")
    declared_challenge = manifest.get("challenge_artifact_sha256")
    if not isinstance(declared_challenge, dict):
        raise ValueError("Source-disjoint score manifest has no challenge hash inventory")
    challenge_checks = {}
    for filename, expected_hash in sorted(declared_challenge.items()):
        path = source_disjoint_root / "challenge" / str(filename)
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Source-disjoint score manifest challenge hash mismatch: {filename}")
        challenge_checks[str(filename)] = actual_hash
    report.update(
        {
            "implementation_lock_sha256": expected_lock,
            "implementation_lock_link": "match",
            "challenge_artifact_sha256": challenge_checks,
        }
    )
    return report


def _write_models(path: Path, rows: list[Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def _output_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    return {"schema_version": "tcred-sl-comparison-manifest-v1", "files": files}


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _stable_offset(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _fmt(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return ""
    return f"[{float(value[0]):.3f}, {float(value[1]):.3f}]"
