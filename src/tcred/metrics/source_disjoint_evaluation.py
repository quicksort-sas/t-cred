from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from tcred.dataset.extracted_source import load_extracted_sources
from tcred.dataset.io import load_bundle
from tcred.metrics.diagnostic_analysis import analyze_diagnostic_suite, write_analysis
from tcred.metrics.diagnostic_models import DiagnosticSuite, diagnostic_inference_cluster_ids
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.runner import _validate_score_records
from tcred.metrics.source_disjoint_io import (
    DEFAULT_STUDY_ROOT,
    challenge_hashes,
    file_record,
    load_prepared_suite,
    mapping,
    read_json,
    read_jsonl,
    resolve_path,
    sha256,
    write_json,
    write_jsonl,
)
from tcred.metrics.tcred_diagnostic_runner import (
    TCRED_SUITE_VERSION,
    render_tcred_report,
    summarize_matched_baseline_dominance,
    summarize_preregistered_dominance,
    summarize_update_behavior,
)
from tcred.metrics.tcred_models import TCredMetricResult
from tcred.metrics.tcred_semantic import (
    read_semantic_records,
    run_semantic_worker,
    validate_semantic_records,
)
from tcred.metrics.tcred_suite import score_tcred_suite

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


def run_source_disjoint_tcred_evaluation(
    *,
    repository_root: Path,
    metric_python: Path,
    gold_dir: Path,
    study_root: Path = DEFAULT_STUDY_ROOT,
    semantic_batch_size: int = 32,
) -> dict[str, Path]:
    """Run frozen T-CRED and the preregistered source-disjoint inference."""

    repository_root = repository_root.resolve()
    study_root = resolve_path(repository_root, study_root)
    metric_python = resolve_path(repository_root, metric_python)
    gold_dir = resolve_path(repository_root, gold_dir)
    suite, protocol = load_prepared_suite(
        repository_root=repository_root,
        study_root=study_root,
    )
    inference = mapping(protocol, "inference")
    challenge = mapping(protocol, "challenge_generation")
    baseline_dir = study_root / "comparators"
    baseline_manifest = _validate_comparator_manifest(
        baseline_dir / "manifest.json",
        suite=suite,
        study_root=study_root,
    )
    baseline = [
        MetricScoreRecord.model_validate(row)
        for row in read_jsonl(baseline_dir / "metric_scores.jsonl")
    ]
    expected_ids = [case.case_id for case in suite.cases]
    if [row.metric_id for row in baseline] != expected_ids:
        raise ValueError("Comparator score order does not match the locked challenge")

    output_dir = study_root / "tcred"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_inputs_path = study_root / "challenge" / "task_judge_inputs.jsonl"
    semantic_path = output_dir / "semantic_pair_scores.jsonl"
    run_semantic_worker(
        inputs_path=task_inputs_path,
        output_path=semantic_path,
        cache_path=(
            repository_root
            / "data/cache/metrics/source_disjoint_validation/tcred_semantic"
            / "alignscore_pairwise.jsonl"
        ),
        model_cache_dir=repository_root / "data/cache/metrics/huggingface",
        metric_python=metric_python,
        batch_size=semantic_batch_size,
    )
    task_rows = [case.task_judge_input for case in suite.cases]
    semantics = read_semantic_records(semantic_path)
    validate_semantic_records(task_rows, semantics)
    baseline_by_id = {row.metric_id: row for row in baseline}
    components = [
        score_tcred_suite(
            row,
            semantics[row.metric_id],
            baseline_scores=baseline_by_id[row.metric_id].scores,
            baseline_input_aligned=True,
        )
        for row in task_rows
    ]
    component_path = output_dir / "tcred_component_results.jsonl"
    write_jsonl(component_path, [row.model_dump(mode="json") for row in components])
    merged = _merge_scores(baseline, components)
    _validate_score_records(merged)
    merged_path = output_dir / "metric_scores_with_tcred.jsonl"
    write_jsonl(merged_path, [row.model_dump(mode="json") for row in merged])

    bootstrap_samples = int(inference["bootstrap_replicates"])
    inference_seed = int(inference["seed"])
    analysis_records, analysis_metric_names = _project_preregistered_scores(
        merged,
        protocol=protocol,
    )
    analysis = analyze_diagnostic_suite(
        suite,
        analysis_records,
        dataset_root=study_root / "dataset",
        gold_dir=gold_dir,
        bootstrap_samples=bootstrap_samples,
        seed=inference_seed,
    )
    exploratory = summarize_preregistered_dominance(analysis)
    matched = summarize_matched_baseline_dominance(analysis)
    confirmation = source_disjoint_confirmation(
        suite=suite,
        records=merged,
        matched=matched,
        protocol=protocol,
    )
    property_analysis = analyze_matched_results_by_property(
        suite=suite,
        records=merged,
        study_root=study_root,
        protocol=protocol,
    )
    analysis["tcred_suite"] = {
        "version": TCRED_SUITE_VERSION,
        "primary_metrics": _PRIMARY_METRICS,
        "analysis_score_projection": {
            "scope": "preregistered_matched_primary_and_comparator_scores_only",
            "score_names": analysis_metric_names,
            "reason": (
                "The confirmatory protocol contains eight matched comparisons. Excluding other "
                "stored score columns avoids running the generic analyzer over the complete "
                "80-column exploratory family. The confirmatory decision still reads only the "
                "eight protocol-matched rows, without changing any locked estimand."
            ),
        },
        "dominance": {
            "overall_conclusion": matched["overall_conclusion"],
            "claim_scope": matched["claim_scope"],
            "matched_baseline": matched,
            "exploratory_universal": exploratory,
        },
        "source_disjoint_confirmation": confirmation,
        "update_behavior": summarize_update_behavior(analysis),
        "by_source_property": property_analysis,
    }
    analysis_path = output_dir / "source_disjoint_analysis.json"
    write_analysis(analysis, analysis_path)
    full_report_path = output_dir / "full_diagnostic_report.md"
    full_report_path.write_text(
        render_tcred_report(analysis),
        encoding="utf-8",
        newline="\n",
    )
    report_path = output_dir / "source_disjoint_validation_report.md"
    report_path.write_text(
        render_source_disjoint_report(
            confirmation=confirmation,
            property_analysis=property_analysis,
            preflight=read_json(study_root / "preflight_audit.json"),
            baseline_manifest=baseline_manifest,
            protocol=protocol,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output_dir / "manifest.json"
    artifacts = [
        semantic_path,
        component_path,
        merged_path,
        analysis_path,
        full_report_path,
        report_path,
    ]
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "generated_at": datetime.now(UTC).isoformat(),
            "suite_version": TCRED_SUITE_VERSION,
            "configuration": {
                "source_split": suite.source_split,
                "diagnostic_seed": challenge["diagnostic_seed"],
                "inference_seed": inference_seed,
                "pair_cap_per_phenomenon": challenge["pair_cap_per_phenomenon"],
                "bootstrap_replicates": bootstrap_samples,
                "randomization_replicates": inference["randomization_replicates"],
                "semantic_batch_size": semantic_batch_size,
                "minimum_common_pairs": inference["minimum_common_pairs"],
                "minimum_common_source_clusters": inference[
                    "minimum_common_source_clusters"
                ],
                "analysis_score_names": analysis_metric_names,
            },
            "dataset_content_hashes": suite.dataset_content_hashes,
            "challenge_artifact_sha256": challenge_hashes(study_root),
            "implementation_lock_sha256": sha256(
                study_root / "implementation_lock.json"
            ),
            "comparator_manifest_sha256": sha256(baseline_dir / "manifest.json"),
            "confirmation": confirmation,
            "artifacts": [file_record(path, relative_to=output_dir) for path in artifacts],
            "claim_boundary": (
                "Source-disjoint real-world-backed formal construct validity. No new human "
                "concurrent-validity, ecological-validity, or production-validity claim."
            ),
        },
    )
    return {
        "manifest": manifest_path,
        "scores": merged_path,
        "components": component_path,
        "analysis": analysis_path,
        "report": report_path,
        "full_report": full_report_path,
    }


def _project_preregistered_scores(
    records: list[MetricScoreRecord],
    *,
    protocol: dict[str, Any],
) -> tuple[list[MetricScoreRecord], list[str]]:
    contracts = mapping(protocol, "comparators")
    names = sorted(
        {
            str(contract[field])
            for contract in contracts.values()
            if isinstance(contract, dict)
            for field in ("primary", "baseline")
        }
    )
    available = {name for record in records for name in record.scores}
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(f"Preregistered analysis scores are missing: {missing}")
    return (
        [
            record.model_copy(
                update={"scores": {name: record.scores.get(name) for name in names}}
            )
            for record in records
        ],
        names,
    )


def source_disjoint_confirmation(
    *,
    suite: DiagnosticSuite,
    records: list[MetricScoreRecord],
    matched: dict[str, object],
    protocol: dict[str, Any],
) -> dict[str, object]:
    inference = mapping(protocol, "inference")
    minimum_pairs = int(inference["minimum_common_pairs"])
    minimum_clusters = int(inference["minimum_common_source_clusters"])
    scores = {row.metric_id: row.scores for row in records}
    cluster_ids = diagnostic_inference_cluster_ids(suite.cases, suite.pairs)
    rows: list[dict[str, object]] = []
    for matched_row in matched["comparisons"]:
        construct = str(matched_row["construct"])
        primary = str(matched_row["primary"])
        comparator = str(matched_row["comparator"])
        dimensions: dict[str, object] = {}
        for test_type in ("directional", "invariance"):
            relevant = [
                pair
                for pair in suite.pairs
                if pair.target_construct == construct and pair.test_type == test_type
            ]
            if not relevant:
                continue
            common = [
                pair
                for pair in relevant
                if _pair_has_scores(pair, scores, primary, comparator)
            ]
            clusters = {cluster_ids[pair.pair_id] for pair in common}
            dimensions[test_type] = {
                "available_pairs": len(relevant),
                "common_pairs": len(common),
                "common_source_clusters": len(clusters),
                "minimum_pairs": minimum_pairs,
                "minimum_source_clusters": minimum_clusters,
                "floor_pass": len(common) >= minimum_pairs
                and len(clusters) >= minimum_clusters,
            }
        floor_pass = bool(dimensions) and all(
            bool(value["floor_pass"]) for value in dimensions.values()
        )
        matched_pass = bool(matched_row["matched_baseline_pass"])
        row = dict(matched_row)
        row.update(
            {
                "common_evidence": dimensions,
                "preregistered_floor_pass": floor_pass,
                "source_disjoint_requirement_pass": matched_pass and floor_pass,
            }
        )
        rows.append(row)
    all_pass = len(rows) == 8 and all(
        bool(row["source_disjoint_requirement_pass"]) for row in rows
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "confirmatory_constructs": len(rows),
        "constructs_passing": sum(
            bool(row["source_disjoint_requirement_pass"]) for row in rows
        ),
        "strict_gain_constructs_passing": sum(
            bool(row["strict_superiority"])
            and bool(row["source_disjoint_requirement_pass"])
            for row in rows
        ),
        "all_eight_requirements_pass": all_pass,
        "overall_conclusion": (
            "source_disjoint_formal_confirmation_established"
            if all_pass
            else "source_disjoint_formal_confirmation_not_established"
        ),
        "comparisons": rows,
        "interpretation": (
            "This decision is limited to the preregistered matched-baseline formal protocol. "
            "It does not imply universal metric dominance or replace human validation."
        ),
    }


def analyze_matched_results_by_property(
    *,
    suite: DiagnosticSuite,
    records: list[MetricScoreRecord],
    study_root: Path,
    protocol: dict[str, Any],
) -> dict[str, object]:
    sources = load_extracted_sources(study_root / "source_sample.jsonl")
    property_by_source = {source.source_id: source.property_id for source in sources}
    bundle = load_bundle(study_root / "dataset" / "tcred_synth")
    property_by_scenario = {
        scenario.scenario_id: property_by_source[str(scenario.split_group_id)]
        for scenario in bundle.scenarios
    }
    comparator_contract = mapping(protocol, "comparators")
    scores = {row.metric_id: row.scores for row in records}
    grouped: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for construct, contract in comparator_contract.items():
        primary = str(contract["primary"])
        comparator = str(contract["baseline"])
        for pair in suite.pairs:
            if pair.target_construct != construct:
                continue
            difference = _pair_utility_difference(
                pair,
                scores=scores,
                primary=primary,
                comparator=comparator,
            )
            if difference is None:
                continue
            key = (construct, property_by_scenario[pair.scenario_id], pair.test_type)
            grouped[key].append(difference)
            counts[key] += 1
    rows = [
        {
            "construct": construct,
            "property_id": property_id,
            "test_type": test_type,
            "common_pairs": counts[(construct, property_id, test_type)],
            "mean_pair_utility_difference": mean(values),
        }
        for (construct, property_id, test_type), values in sorted(grouped.items())
    ]
    return {
        "status": "descriptive_secondary_analysis",
        "warning": (
            "Property strata were not separately powered or included in the multiplicity family. "
            "Their means diagnose heterogeneity and cannot upgrade the confirmatory claim."
        ),
        "rows": rows,
    }


def render_source_disjoint_report(
    *,
    confirmation: dict[str, object],
    property_analysis: dict[str, object],
    preflight: dict[str, Any],
    baseline_manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    diagnostics = preflight["diagnostic_suite"]
    selection = preflight["source_selection"]
    lines = [
        "# T-CRED v1.4 Source-Disjoint Formal Validation",
        "",
        f"**Protocol:** `{protocol['protocol_id']}`  ",
        f"**Decision:** `{confirmation['overall_conclusion']}`",
        "",
        "## Design integrity",
        "",
        f"- New Wikidata source series: **{selection['selected_size']}** from "
        f"**{selection['eligible_size_after_all_exclusions']}** eligible cached series.",
        "- Released/new overlap: **0 source IDs, 0 context IDs, 0 statement IDs**.",
        f"- Formal challenge: **{diagnostics['cases']} cases** and "
        f"**{diagnostics['pairs']} pairs**.",
        "- Pair construction and the implementation lock were completed before any score was read.",
        "- RAGChecker-style claim judgments were limited to "
        f"**{baseline_manifest['configuration']['claim_judge_case_count']} response cases**.",
        "- Inference used 10,000 connected-source-cluster bootstrap and randomization replicates.",
        "",
        "## Preregistered matched comparisons",
        "",
        "| Construct | Comparator | Requirement | Effect (95% CI) | Common evidence | "
        "Holm p | Result |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in confirmation["comparisons"]:
        dimension = str(row.get("gain_dimension") or "directional")
        difference = row.get(f"{dimension}_difference")
        interval = row.get(f"{dimension}_ci95")
        effect = _format_effect(difference, interval)
        evidence = "; ".join(
            f"{name}: {value['common_pairs']} pairs/{value['common_source_clusters']} clusters"
            for name, value in row["common_evidence"].items()
        )
        requirement = "strict gain" if row["superiority_required"] else "non-inferiority"
        holm = row.get("primary_gain_holm_p_value")
        result = "PASS" if row["source_disjoint_requirement_pass"] else "FAIL"
        lines.append(
            f"| {row['construct']} | `{row['comparator']}` | {requirement} | {effect} | "
            f"{evidence} | {_format_number(holm)} | **{result}** |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{confirmation['constructs_passing']} of 8** construct requirements pass. "
            f"The machine decision is `{confirmation['overall_conclusion']}`.",
            "",
            str(confirmation["interpretation"]),
            "",
            "## Source-property heterogeneity",
            "",
            str(property_analysis["warning"]),
            "",
            "The complete property-level table is stored in `source_disjoint_analysis.json`.",
            "",
            "## Limitations",
            "",
            "- The source facts are real and source-disjoint, but conversion and intervention "
            "logic remain project-owned; this is not a wholly external benchmark.",
            "- Formal pair oracles test expected local sensitivity and invariance. They are not "
            "new human judgments and cannot estimate natural annotator preference.",
            "- The only hosted comparator component is the scoped RAGChecker-style claim judge; "
            "its cached outputs are auditable but not assumed bitwise reproducible.",
            "- Per-property analyses are descriptive and not separately powered.",
            "- The original opened reserve remains a separate result and is not pooled here.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_comparator_manifest(
    path: Path,
    *,
    suite: DiagnosticSuite,
    study_root: Path,
) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("status") != "complete":
        raise ValueError("Source-disjoint comparator run is not complete")
    if manifest.get("dataset_content_hashes") != suite.dataset_content_hashes:
        raise ValueError("Comparator dataset hashes differ from the locked challenge")
    if manifest.get("challenge_artifact_sha256") != challenge_hashes(study_root):
        raise ValueError("Comparator challenge hashes differ from the implementation lock")
    return manifest


def _merge_scores(
    baseline: list[MetricScoreRecord],
    components: list[TCredMetricResult],
) -> list[MetricScoreRecord]:
    by_id = {component.metric_id: component for component in components}
    output: list[MetricScoreRecord] = []
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


def _pair_has_scores(
    pair: object,
    scores: dict[str, dict[str, float | None]],
    primary: str,
    comparator: str,
) -> bool:
    return all(
        _finite(scores[case_id].get(metric))
        for case_id in (pair.left_case_id, pair.right_case_id)
        for metric in (primary, comparator)
    )


def _pair_utility_difference(
    pair: object,
    *,
    scores: dict[str, dict[str, float | None]],
    primary: str,
    comparator: str,
) -> float | None:
    if not _pair_has_scores(pair, scores, primary, comparator):
        return None
    primary_left = float(scores[pair.left_case_id][primary])
    primary_right = float(scores[pair.right_case_id][primary])
    comparator_left = float(scores[pair.left_case_id][comparator])
    comparator_right = float(scores[pair.right_case_id][comparator])
    if pair.test_type == "directional":
        return _directional_utility(primary_left, primary_right) - _directional_utility(
            comparator_left,
            comparator_right,
        )
    return (1.0 - abs(primary_left - primary_right)) - (
        1.0 - abs(comparator_left - comparator_right)
    )


def _directional_utility(left: float, right: float) -> float:
    return 1.0 if left > right else 0.5 if left == right else 0.0


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _format_effect(value: object, interval: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(interval, list) and len(interval) == 2:
        return f"{float(value):+.4f} [{float(interval[0]):+.4f}, {float(interval[1]):+.4f}]"
    return f"{float(value):+.4f}"


def _format_number(value: object) -> str:
    return "-" if value is None else f"{float(value):.4g}"
