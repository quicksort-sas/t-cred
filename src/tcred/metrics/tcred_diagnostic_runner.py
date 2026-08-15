from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import orjson

from tcred.metrics.diagnostic_analysis import analyze_diagnostic_suite, write_analysis
from tcred.metrics.diagnostic_builder import (
    DEFAULT_DIAGNOSTIC_SEED,
    DEFAULT_PAIR_CAP,
    build_diagnostic_suite,
)
from tcred.metrics.diagnostic_reporting import render_diagnostic_report
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.runner import _validate_score_records
from tcred.metrics.tcred_models import TCredMetricResult
from tcred.metrics.tcred_semantic import (
    read_semantic_records,
    run_semantic_worker,
    validate_semantic_records,
)
from tcred.metrics.tcred_suite import score_tcred_suite

TCRED_SUITE_VERSION = "tcred-automatic-v1.4-development"
_PRIMARY_METRICS = {
    "answer_correctness": "tcred_answer_equivalence",
    "temporal_correctness": "tcred_temporal_correctness",
    "temporal_attribution": "tcred_temporal_attribution",
    "evidence_support": "tcred_semantic_attribution",
    "citation_correctness": "tcred_citation_quality",
    "graph_sufficiency": "tcred_graph_answer_coverage",
    "response_decision": "tcred_response_decision",
    "retrieval_quality": "tcred_t_ndcg_at_10",
}
_MATCHED_BASELINE_POLICY = {
    "answer_correctness": {
        "comparator": "pedants_probability",
        "role": "inherited_ingredient",
        "superiority_required": False,
        "gain_dimension": None,
        "rationale": "Answer equivalence deliberately reuses PEDANTS; identity is required.",
    },
    "temporal_correctness": {
        "comparator": "pedants_probability",
        "role": "inherited_answer_baseline",
        "superiority_required": False,
        "gain_dimension": None,
        "rationale": (
            "Answer-centric temporal correctness must retain the answer-equivalence baseline "
            "while adding explicit-time validity when the answer asserts a time."
        ),
    },
    "temporal_attribution": {
        "comparator": "alignscore_retrieved",
        "role": "matched_semantic_baseline",
        "superiority_required": True,
        "gain_dimension": "directional",
        "rationale": (
            "The same semantic backbone without T-CRED temporal link validity isolates the "
            "temporal-attribution contribution."
        ),
    },
    "evidence_support": {
        "comparator": "alignscore_retrieved",
        "role": "matched_semantic_baseline",
        "superiority_required": False,
        "gain_dimension": None,
        "rationale": (
            "Semantic attribution is an auditable claim-link wrapper around AlignScore and must "
            "be noninferior, not spuriously claimed as a new semantic model."
        ),
    },
    "citation_correctness": {
        "comparator": "alignscore_cited",
        "role": "matched_semantic_baseline",
        "superiority_required": True,
        "gain_dimension": "directional",
        "rationale": (
            "Cited-evidence AlignScore holds semantic support fixed while omitting temporal "
            "citation validity and claim-level citation accounting."
        ),
    },
    "graph_sufficiency": {
        "comparator": "tcred_ablation_graph_answer_coverage_no_time",
        "role": "factor_isolating_ablation",
        "superiority_required": True,
        "gain_dimension": "directional",
        "rationale": (
            "No construct-matched non-LLM graph metric is available; the no-time ablation "
            "isolates the contribution of temporal path validity."
        ),
    },
    "response_decision": {
        "comparator": "ragchecker_f1",
        "role": "external_standard",
        "superiority_required": False,
        "gain_dimension": None,
        "rationale": (
            "RAGChecker F1 is the strongest applicable non-LLM response-quality comparator in "
            "the frozen pool; response handling is a supporting, not central, novelty claim."
        ),
    },
    "retrieval_quality": {
        "comparator": "retrieval_ndcg_at_10",
        "role": "matched_non_temporal_baseline",
        "superiority_required": True,
        "gain_dimension": "directional",
        "rationale": (
            "Ordinary nDCG@10 has the same ranked-list form but omits query-conditioned valid "
            "time, directly isolating the temporal retrieval contribution."
        ),
    },
}


def run_tcred_diagnostic_evaluation(
    *,
    dataset_root: Path,
    gold_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    metric_python: Path,
    source_split: str = "test_auto",
    seed: int = DEFAULT_DIAGNOSTIC_SEED,
    pair_cap_per_phenomenon: int = DEFAULT_PAIR_CAP,
    bootstrap_samples: int = 2000,
    semantic_batch_size: int = 16,
) -> dict[str, Path]:
    """Add T-CRED to a frozen comparator run and repeat the same paired analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    suite = build_diagnostic_suite(
        dataset_root,
        seed=seed,
        pair_cap_per_phenomenon=pair_cap_per_phenomenon,
        source_split=source_split,
    )
    baseline = _read_score_records(baseline_dir / "metric_scores.jsonl")
    expected_ids = [case.case_id for case in suite.cases]
    if [row.metric_id for row in baseline] != expected_ids:
        raise ValueError(
            "Baseline score order/content does not match the freshly rebuilt frozen suite"
        )
    task_rows = [case.task_judge_input for case in suite.cases]
    task_payloads = [row.model_dump(mode="json") for row in task_rows]
    _validate_baseline_manifest(
        baseline_dir / "manifest.json",
        source_split=source_split,
        dataset_hashes=suite.dataset_content_hashes,
        task_inputs_sha256=_jsonl_sha256(task_payloads),
    )

    task_inputs_path = output_dir / "task_judge_inputs.jsonl"
    _write_jsonl(task_inputs_path, task_payloads)
    semantic_path = output_dir / "semantic_pair_scores.jsonl"
    run_semantic_worker(
        inputs_path=task_inputs_path,
        output_path=semantic_path,
        cache_path=(
            Path("data/cache/metrics/tcred_semantic") / source_split / "alignscore_pairwise.jsonl"
        ),
        model_cache_dir=Path("data/cache/metrics/huggingface"),
        metric_python=metric_python,
        batch_size=semantic_batch_size,
    )
    semantics = read_semantic_records(semantic_path)
    validate_semantic_records(task_rows, semantics)
    baseline_by_id = {row.metric_id: row for row in baseline}
    component_results = [
        score_tcred_suite(
            row,
            semantics[row.metric_id],
            baseline_scores=baseline_by_id[row.metric_id].scores,
            baseline_input_aligned=True,
        )
        for row in task_rows
    ]
    component_path = output_dir / "tcred_component_results.jsonl"
    _write_jsonl(component_path, [row.model_dump(mode="json") for row in component_results])
    merged = _merge_scores(baseline, component_results)
    _validate_score_records(merged)
    merged_path = output_dir / "metric_scores_with_tcred.jsonl"
    _write_jsonl(merged_path, [row.model_dump(mode="json") for row in merged])

    analysis = analyze_diagnostic_suite(
        suite,
        merged,
        dataset_root=dataset_root,
        gold_dir=gold_dir,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    exploratory_universal = summarize_preregistered_dominance(analysis)
    matched_baseline = summarize_matched_baseline_dominance(analysis)
    dominance = {
        "overall_conclusion": matched_baseline["overall_conclusion"],
        "claim_scope": matched_baseline["claim_scope"],
        "matched_baseline": matched_baseline,
        "exploratory_universal": exploratory_universal,
    }
    update_behavior = summarize_update_behavior(analysis)
    analysis["tcred_suite"] = {
        "version": TCRED_SUITE_VERSION,
        "primary_metrics": _PRIMARY_METRICS,
        "dominance": dominance,
        "update_behavior": update_behavior,
    }
    analysis_path = output_dir / "tcred_diagnostic_analysis.json"
    write_analysis(analysis, analysis_path)
    report_path = output_dir / "tcred_diagnostic_report.md"
    report_path.write_text(
        render_tcred_report(analysis),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output_dir / "manifest.json"
    artifacts = [
        task_inputs_path,
        semantic_path,
        component_path,
        merged_path,
        analysis_path,
        report_path,
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "status": "complete",
            "generated_at": datetime.now(UTC).isoformat(),
            "suite_version": TCRED_SUITE_VERSION,
            "configuration": {
                "source_split": source_split,
                "seed": seed,
                "pair_cap_per_phenomenon": pair_cap_per_phenomenon,
                "bootstrap_samples": bootstrap_samples,
                "semantic_batch_size": semantic_batch_size,
            },
            "information_boundary": "automatic",
            "dataset_content_hashes": suite.dataset_content_hashes,
            "baseline_manifest_sha256": _sha256(baseline_dir / "manifest.json"),
            "alignscore_class_mapping": {
                "entailment": 0,
                "neutral": 1,
                "contradiction": 2,
                "verification": (
                    "Pre-implementation probes: identical text -> class 0; explicit "
                    "contradiction -> class 2; unrelated text -> class 1."
                ),
            },
            "preregistration": "docs/tcred-metric-suite-preregistration-2026-08-15.md",
            "artifacts": [_file_record(path, relative_to=output_dir) for path in artifacts],
            "dominance_conclusion": dominance["overall_conclusion"],
            "limitations": [
                "Controlled formal diagnostics establish bounded construct validity, not "
                "universal validity.",
                "Unknown valid time remains missing and is never converted into a passing "
                "world-time score.",
                "Neutral provenance 1.0 is used where the evaluator input exposes no "
                "reliability value.",
                "The optional aggregate is secondary; construct-specific components are primary.",
            ],
        },
    )
    return {
        "manifest": manifest_path,
        "scores": merged_path,
        "components": component_path,
        "analysis": analysis_path,
        "report": report_path,
    }


def summarize_preregistered_dominance(analysis: dict[str, object]) -> dict[str, object]:
    """Retain the original all-pairs checker as an explicitly exploratory audit.

    It is not a valid suite-level hypothesis test because its comparator pool includes inherited
    ingredients, internal component aliases, and ablations. The matched-baseline protocol below
    is the confirmatory development criterion.
    """
    constructs = analysis["constructs"]
    rows = []
    strict_count = 0
    tested_comparison_count = 0
    for construct, primary in _PRIMARY_METRICS.items():
        result = constructs[construct]
        directional = result["directional_metrics"].get(primary, {})
        directional_required = _test_dimension_required(result, "directional")
        invariance_required = _test_dimension_required(result, "invariance")
        comparators = sorted(
            (set(result["directional_metrics"]) | set(result["invariance_metrics"])) - {primary}
        )
        for comparator in comparators:
            directional_comparison = _comparison_for(
                result["directional_pairwise_comparisons"],
                primary,
                comparator,
            )
            invariance_comparison = _comparison_for(
                result["invariance_pairwise_comparisons"],
                primary,
                comparator,
            )
            directional_diff = (
                directional_comparison["oriented_difference"]
                if directional_comparison is not None
                else None
            )
            invariance_diff = (
                invariance_comparison["oriented_difference"]
                if invariance_comparison is not None
                else None
            )
            primary_coverage = directional.get("coverage")
            comparator_coverage = result["directional_metrics"].get(comparator, {}).get("coverage")
            directional_coverage_diff = (
                float(primary_coverage) - float(comparator_coverage)
                if primary_coverage is not None and comparator_coverage is not None
                else None
            )
            primary_invariance_coverage = (
                result["invariance_metrics"].get(primary, {}).get("coverage")
            )
            comparator_invariance_coverage = (
                result["invariance_metrics"].get(comparator, {}).get("coverage")
            )
            invariance_coverage_diff = (
                float(primary_invariance_coverage) - float(comparator_invariance_coverage)
                if (
                    primary_invariance_coverage is not None
                    and comparator_invariance_coverage is not None
                )
                else None
            )
            directional_evidence = not directional_required or directional_comparison is not None
            invariance_evidence = not invariance_required or invariance_comparison is not None
            directional_coverage_evidence = (
                not directional_required or directional_coverage_diff is not None
            )
            invariance_coverage_evidence = (
                not invariance_required or invariance_coverage_diff is not None
            )
            complete_evidence = all(
                (
                    directional_evidence,
                    invariance_evidence,
                    directional_coverage_evidence,
                    invariance_coverage_evidence,
                )
            )
            directional_ok = not directional_required or (
                directional_diff is not None and float(directional_diff) >= -0.02
            )
            invariance_ok = not invariance_required or (
                invariance_diff is not None and float(invariance_diff) >= -0.02
            )
            coverage_ok = (
                (
                    (not directional_required or float(directional_coverage_diff) >= -0.02)
                    and (not invariance_required or float(invariance_coverage_diff) >= -0.02)
                )
                if complete_evidence
                else False
            )
            significant_improvement = _significant_improvement(
                directional_comparison,
                invariance_comparison,
            )
            strict = bool(
                complete_evidence
                and directional_ok
                and invariance_ok
                and coverage_ok
                and significant_improvement
            )
            tested_comparison_count += int(complete_evidence)
            strict_count += int(strict)
            missing_evidence = []
            if not directional_evidence:
                missing_evidence.append("directional_common_pairs")
            if not invariance_evidence:
                missing_evidence.append("invariance_common_pairs")
            if not directional_coverage_evidence:
                missing_evidence.append("directional_coverage")
            if not invariance_coverage_evidence:
                missing_evidence.append("invariance_coverage")
            rows.append(
                {
                    "construct": construct,
                    "primary": primary,
                    "comparator": comparator,
                    "directional_common_pair_difference": directional_diff,
                    "invariance_common_pair_utility_difference": invariance_diff,
                    "directional_coverage_difference": directional_coverage_diff,
                    "invariance_coverage_difference": invariance_coverage_diff,
                    "strictly_dominates": strict,
                    "evidence_status": "tested" if complete_evidence else "insufficient_evidence",
                    "missing_evidence": missing_evidence,
                }
            )
    all_comparators_strict = bool(rows) and strict_count == len(rows)
    return {
        "noninferiority_margin": 0.02,
        "strict_comparison_count": strict_count,
        "tested_comparison_count": tested_comparison_count,
        "candidate_comparison_count": len(rows),
        "insufficient_evidence_comparison_count": len(rows) - tested_comparison_count,
        "all_tested_comparators_strictly_dominated": all_comparators_strict,
        "overall_conclusion": (
            "strict_dominance_supported_for_all_tested_comparators"
            if all_comparators_strict
            else "universal_strict_dominance_not_established"
        ),
        "comparisons": rows,
    }


def summarize_matched_baseline_dominance(
    analysis: dict[str, object],
) -> dict[str, object]:
    """Evaluate the fixed factor-isolating baseline for every T-CRED construct."""

    margin = 0.02
    constructs = analysis["constructs"]
    rows: list[dict[str, object]] = []
    for construct, policy in _MATCHED_BASELINE_POLICY.items():
        result = constructs[construct]
        primary = _PRIMARY_METRICS[construct]
        comparator = str(policy["comparator"])
        directional_required = _test_dimension_required(result, "directional")
        invariance_required = _test_dimension_required(result, "invariance")
        directional = _comparison_for(
            result["directional_pairwise_comparisons"], primary, comparator
        )
        invariance = _comparison_for(
            result["invariance_pairwise_comparisons"], primary, comparator
        )
        directional_coverage = _coverage_difference(
            result["directional_metrics"], primary, comparator
        )
        invariance_coverage = _coverage_difference(
            result["invariance_metrics"], primary, comparator
        )
        missing = []
        if directional_required and directional is None:
            missing.append("directional_common_pairs")
        if invariance_required and invariance is None:
            missing.append("invariance_common_pairs")
        if directional_required and directional_coverage is None:
            missing.append("directional_coverage")
        if invariance_required and invariance_coverage is None:
            missing.append("invariance_coverage")

        directional_noninferior = _comparison_noninferior(
            directional,
            required=directional_required,
            margin=margin,
        )
        invariance_noninferior = _comparison_noninferior(
            invariance,
            required=invariance_required,
            margin=margin,
        )
        coverage_noninferior = bool(
            (not directional_required or (directional_coverage or 0.0) >= -margin)
            and (not invariance_required or (invariance_coverage or 0.0) >= -margin)
            and not missing
        )
        gain_dimension = policy["gain_dimension"]
        gain_comparison = (
            directional if gain_dimension == "directional" else invariance
        )
        rows.append(
            {
                "construct": construct,
                "primary": primary,
                "comparator": comparator,
                "comparator_role": policy["role"],
                "rationale": policy["rationale"],
                "superiority_required": policy["superiority_required"],
                "gain_dimension": gain_dimension,
                "directional_difference": (
                    directional["oriented_difference"] if directional else None
                ),
                "directional_ci95": directional.get("oriented_ci95") if directional else None,
                "invariance_utility_difference": (
                    invariance["oriented_difference"] if invariance else None
                ),
                "invariance_ci95": invariance.get("oriented_ci95") if invariance else None,
                "directional_coverage_difference": directional_coverage,
                "invariance_coverage_difference": invariance_coverage,
                "directional_noninferior": directional_noninferior,
                "invariance_noninferior": invariance_noninferior,
                "coverage_noninferior": coverage_noninferior,
                "primary_gain_raw_p_value": (
                    gain_comparison.get("permutation_p_value") if gain_comparison else None
                ),
                "primary_gain_holm_p_value": None,
                "strict_superiority": False,
                "evidence_status": "tested" if not missing else "insufficient_evidence",
                "missing_evidence": missing,
            }
        )

    _holm_adjust_declared_gains(rows)
    for row in rows:
        if not row["superiority_required"]:
            continue
        dimension = str(row["gain_dimension"])
        difference = row[f"{dimension}_difference"]
        interval = row[f"{dimension}_ci95"]
        adjusted = row["primary_gain_holm_p_value"]
        row["strict_superiority"] = bool(
            difference is not None
            and float(difference) > margin
            and interval is not None
            and float(interval[0]) > 0.0
            and adjusted is not None
            and float(adjusted) < 0.05
        )
    for row in rows:
        row["matched_baseline_pass"] = bool(
            row["evidence_status"] == "tested"
            and row["directional_noninferior"]
            and row["invariance_noninferior"]
            and row["coverage_noninferior"]
            and (not row["superiority_required"] or row["strict_superiority"])
        )

    all_pass = bool(rows) and all(bool(row["matched_baseline_pass"]) for row in rows)
    superiority_count = sum(bool(row["strict_superiority"]) for row in rows)
    return {
        "protocol_version": "matched-baseline-v1",
        "noninferiority_margin": margin,
        "familywise_alpha": 0.05,
        "multiplicity_family": (
            "four declared directional superiority contrasts: temporal attribution, temporal "
            "citation correctness, temporal graph sufficiency, and temporal retrieval quality"
        ),
        "claim_scope": (
            "strict componentwise dominance over the declared matched non-temporal baseline "
            "protocol; not universal dominance over every metric, task judge, subcomponent, or "
            "future benchmark"
        ),
        "construct_count": len(rows),
        "constructs_passing": sum(bool(row["matched_baseline_pass"]) for row in rows),
        "strict_superiority_count": superiority_count,
        "strict_dominance_established": all_pass and superiority_count > 0,
        "overall_conclusion": (
            "strict_dominance_established_for_matched_baseline_protocol"
            if all_pass and superiority_count > 0
            else "strict_dominance_not_established_for_matched_baseline_protocol"
        ),
        "comparisons": rows,
    }


def _coverage_difference(
    metrics: dict[str, dict[str, object]],
    primary: str,
    comparator: str,
) -> float | None:
    primary_coverage = metrics.get(primary, {}).get("coverage")
    comparator_coverage = metrics.get(comparator, {}).get("coverage")
    if primary_coverage is None or comparator_coverage is None:
        return None
    return float(primary_coverage) - float(comparator_coverage)


def _comparison_noninferior(
    comparison: dict[str, object] | None,
    *,
    required: bool,
    margin: float,
) -> bool:
    if not required:
        return True
    if comparison is None or comparison.get("oriented_ci95") is None:
        return False
    return float(comparison["oriented_ci95"][0]) >= -margin


def _holm_adjust_declared_gains(rows: list[dict[str, object]]) -> None:
    family_size = sum(bool(row["superiority_required"]) for row in rows)
    declared = [
        row
        for row in rows
        if row["superiority_required"] and row["primary_gain_raw_p_value"] is not None
    ]
    ordered = sorted(declared, key=lambda row: float(row["primary_gain_raw_p_value"]))
    running = 0.0
    for index, row in enumerate(ordered):
        adjusted = min(
            1.0,
            float(row["primary_gain_raw_p_value"]) * (family_size - index),
        )
        running = max(running, adjusted)
        row["primary_gain_holm_p_value"] = running


def _test_dimension_required(result: dict[str, object], dimension: str) -> bool:
    pair_count = result.get(f"{dimension}_pair_count")
    if pair_count is not None:
        return int(pair_count) > 0
    return bool(result.get(f"{dimension}_metrics", {}))


def summarize_update_behavior(analysis: dict[str, object]) -> dict[str, object]:
    """Lift the preregistered update phenomena into one directly interpretable table."""

    constructs = analysis["constructs"]
    answer = constructs["answer_correctness"]
    response = constructs["response_decision"]
    metric_names = sorted(set(answer["directional_metrics"]) | set(answer["invariance_metrics"]))
    rows = []
    for metric in metric_names:
        changing = (
            answer["directional_metrics"]
            .get(metric, {})
            .get("per_phenomenon", {})
            .get("answer_changing_snapshot_update")
        )
        preserving = (
            answer["invariance_metrics"]
            .get(metric, {})
            .get("per_phenomenon", {})
            .get("answer_preserving_snapshot_update")
        )
        if changing is None and preserving is None:
            continue
        rows.append(
            {
                "metric": metric,
                "changing_pairs": changing.get("n", 0) if changing else 0,
                "change_detection_pair_accuracy": (
                    changing.get("tie_adjusted_pairwise_accuracy") if changing else None
                ),
                "change_detection_strict_consistency": (
                    changing.get("strict_consistency") if changing else None
                ),
                "preserving_pairs": preserving.get("n", 0) if preserving else 0,
                "preservation_exact_invariance": (
                    preserving.get("exact_invariance_rate") if preserving else None
                ),
                "preservation_mean_normalized_change": (
                    preserving.get("mean_normalized_absolute_change") if preserving else None
                ),
            }
        )

    response_rows = []
    for metric, summary in response["directional_metrics"].items():
        changing = summary.get("per_phenomenon", {}).get("answer_changing_snapshot_update")
        if changing is None:
            continue
        response_rows.append(
            {
                "metric": metric,
                "pairs": changing.get("n", 0),
                "answerability_change_pair_accuracy": changing.get(
                    "tie_adjusted_pairwise_accuracy"
                ),
                "strict_consistency": changing.get("strict_consistency"),
            }
        )
    return {
        "interpretation": (
            "Change detection uses fixed-output directional pairs whose gold answer changes. "
            "Preservation uses meaning-preserving snapshot updates whose gold answer does not "
            "change. These controlled evaluator tests are distinct from QA-system adaptation."
        ),
        "answer_metrics": rows,
        "answerability_metrics": response_rows,
    }


def render_tcred_report(analysis: dict[str, object]) -> str:
    base = render_diagnostic_report(analysis)
    dominance = analysis["tcred_suite"]["dominance"]
    matched = dominance["matched_baseline"]
    exploratory = dominance["exploratory_universal"]
    updates = analysis["tcred_suite"]["update_behavior"]
    lines = [
        base.rstrip(),
        "",
        "## Update Behavior",
        "",
        str(updates["interpretation"]),
        "",
        "| Metric | Changing n | Change pair accuracy | Strict change | "
        "Preserving n | Exact invariance | Mean normalized change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in updates["answer_metrics"]:
        lines.append(
            "| `{metric}` | {changing_n} | {change_accuracy} | {strict} | "
            "{preserving_n} | {invariance} | {mean_change} |".format(
                metric=row["metric"],
                changing_n=row["changing_pairs"],
                change_accuracy=_format_optional(row["change_detection_pair_accuracy"]),
                strict=_format_optional(row["change_detection_strict_consistency"]),
                preserving_n=row["preserving_pairs"],
                invariance=_format_optional(row["preservation_exact_invariance"]),
                mean_change=_format_optional(row["preservation_mean_normalized_change"]),
            )
        )
    lines.extend(
        [
            "",
            "## Scoped Matched-Baseline Dominance Check",
            "",
            f"Overall conclusion: `{dominance['overall_conclusion']}`.",
            "",
            str(matched["claim_scope"]),
            "",
            "The Holm family contains only the four declared directional contribution tests. "
            "Inherited ingredients and supporting components are tested for noninferiority; "
            "they are not relabeled as novel improvements.",
            "",
            "| Construct | T-CRED component | Matched comparator | Role | Directional delta "
            "[95% CI] | Invariance delta [95% CI] | Coverage noninferior | Holm p | "
            "Required gain | Pass |",
            "| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |",
        ]
    )
    for row in matched["comparisons"]:
        directional = _format_effect(row["directional_difference"], row["directional_ci95"])
        invariance = _format_effect(
            row["invariance_utility_difference"], row["invariance_ci95"]
        )
        holm = _format_optional(row["primary_gain_holm_p_value"])
        required_gain = (
            "yes, passed" if row["strict_superiority"] else "yes, failed"
        ) if row["superiority_required"] else "no"
        lines.append(
            "| {construct} | `{primary}` | `{comparator}` | `{role}` | {directional} | "
            "{invariance} | {coverage} | {holm} | {gain} | {passed} |".format(
                construct=row["construct"],
                primary=row["primary"],
                comparator=row["comparator"],
                role=row["comparator_role"],
                directional=directional,
                invariance=invariance,
                coverage="yes" if row["coverage_noninferior"] else "no",
                holm=holm,
                gain=required_gain,
                passed="yes" if row["matched_baseline_pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "A positive result here is deliberately scoped. It does not claim universal "
            "dominance over task-matched LLM judges, internal aliases, every metric in the "
            "exploratory pool, or unseen domains.",
            "",
            "## Exploratory All-Pairs Audit",
            "",
            "The historical all-pairs checker is retained for transparency, but it is not a "
            "confirmatory dominance criterion because it includes inherited ingredients, "
            "internal subcomponents, and factor-removal ablations.",
            "",
            f"Exploratory conclusion: `{exploratory['overall_conclusion']}`.",
            "",
            "Candidate comparisons: {candidate}; fully tested: {tested}; insufficient evidence: "
            "{insufficient}.".format(
                candidate=exploratory["candidate_comparison_count"],
                tested=exploratory["tested_comparison_count"],
                insufficient=exploratory["insufficient_evidence_comparison_count"],
            ),
            "",
            "| Construct | T-CRED component | Comparator | Directional delta | "
            "Invariance utility delta | Directional coverage delta | "
            "Invariance coverage delta | Evidence | Strict dominance |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in exploratory["comparisons"]:
        evidence = str(row["evidence_status"])
        if row["missing_evidence"]:
            evidence += ": " + ", ".join(row["missing_evidence"])
        lines.append(
            "| {construct} | `{primary}` | `{comparator}` | {directional} | "
            "{invariance} | {directional_coverage} | {invariance_coverage} | "
            "{evidence} | {strict} |".format(
                construct=row["construct"],
                primary=row["primary"],
                comparator=row["comparator"],
                directional=_format_optional(row["directional_common_pair_difference"]),
                invariance=_format_optional(row["invariance_common_pair_utility_difference"]),
                directional_coverage=_format_optional(row["directional_coverage_difference"]),
                invariance_coverage=_format_optional(row["invariance_coverage_difference"]),
                evidence=evidence,
                strict="yes" if row["strictly_dominates"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "A `no` result is not converted into a positive claim. It can mean inferiority, "
            "equivalence, insufficient common pairs, or a confidence interval that includes zero.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_for(
    rows: list[dict[str, object]],
    primary: str,
    comparator: str,
) -> dict[str, object] | None:
    for row in rows:
        if {str(row["left"]), str(row["right"])} != {primary, comparator}:
            continue
        sign = 1.0 if row["left"] == primary else -1.0
        ci = row.get("ci95")
        oriented_ci = None
        if ci is not None:
            oriented_ci = [sign * float(ci[0]), sign * float(ci[1])]
            oriented_ci.sort()
        return {
            **row,
            "oriented_difference": sign * float(row["macro_utility_difference"]),
            "oriented_ci95": oriented_ci,
        }
    return None


def _significant_improvement(
    directional: dict[str, object] | None,
    invariance: dict[str, object] | None,
) -> bool:
    for row in (directional, invariance):
        if row is None or not row.get("significant_at_0_05"):
            continue
        difference = float(row["oriented_difference"])
        interval = row.get("oriented_ci95")
        if difference > 0.02 and interval is not None and float(interval[0]) > 0:
            return True
    return False


def _merge_scores(
    baseline: list[MetricScoreRecord],
    components: list[TCredMetricResult],
) -> list[MetricScoreRecord]:
    by_id = {row.metric_id: row for row in components}
    output = []
    for row in baseline:
        component = by_id[row.metric_id]
        scores = {**row.scores, **component.scores}
        metadata = dict(row.metric_metadata)
        metadata["tcred"] = {
            "suite_version": TCRED_SUITE_VERSION,
            "mode": component.mode,
            "query": component.query.model_dump(mode="json"),
            "coverage": component.coverage,
            "audit": component.audit,
        }
        output.append(row.model_copy(update={"scores": scores, "metric_metadata": metadata}))
    return output


def _validate_baseline_manifest(
    path: Path,
    *,
    source_split: str,
    dataset_hashes: dict[str, str],
    task_inputs_sha256: str,
) -> None:
    manifest = orjson.loads(path.read_bytes())
    if manifest.get("status") != "complete":
        raise ValueError("Comparator baseline manifest is not complete")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or "source_split" not in configuration:
        raise ValueError("Comparator baseline manifest does not declare source_split")
    configured_split = configuration["source_split"]
    if configured_split != source_split:
        raise ValueError(
            f"Comparator baseline split mismatch: expected {source_split}, found {configured_split}"
        )
    if manifest.get("dataset_content_hashes") != dataset_hashes:
        raise ValueError("Comparator baseline dataset hashes do not match the current release")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Comparator baseline manifest does not declare artifacts")
    task_artifacts = [
        row
        for row in artifacts
        if isinstance(row, dict) and row.get("path") == "task_judge_inputs.jsonl"
    ]
    if len(task_artifacts) != 1:
        raise ValueError(
            "Comparator baseline manifest must declare one task_judge_inputs.jsonl artifact"
        )
    declared_sha256 = task_artifacts[0].get("sha256")
    baseline_task_inputs = path.parent / "task_judge_inputs.jsonl"
    if not baseline_task_inputs.is_file():
        raise ValueError("Comparator baseline task_judge_inputs.jsonl artifact is missing")
    actual_sha256 = _sha256(baseline_task_inputs)
    if declared_sha256 != actual_sha256:
        raise ValueError("Comparator baseline task-input artifact hash is corrupt")
    if declared_sha256 != task_inputs_sha256:
        raise ValueError(
            "Comparator baseline task inputs do not match the freshly rebuilt frozen suite"
        )


def _read_score_records(path: Path) -> list[MetricScoreRecord]:
    with path.open("rb") as handle:
        return [
            MetricScoreRecord.model_validate(orjson.loads(line)) for line in handle if line.strip()
        ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            handle.write(b"\n")


def _jsonl_sha256(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_optional(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _format_effect(value: object, interval: object) -> str:
    if value is None:
        return "-"
    if interval is None:
        return f"{float(value):.3f}"
    return f"{float(value):.3f} [{float(interval[0]):.3f}, {float(interval[1]):.3f}]"
