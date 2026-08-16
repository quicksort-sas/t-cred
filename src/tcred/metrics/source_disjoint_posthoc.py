from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from tcred.dataset.io import load_bundle
from tcred.dataset.models import QuestionProgram, TemporalOperator
from tcred.metrics.diagnostic_analysis import analyze_diagnostic_suite, write_analysis
from tcred.metrics.diagnostic_models import DiagnosticCase, DiagnosticSuite
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.source_disjoint_evaluation import (
    _project_preregistered_scores,
    source_disjoint_confirmation,
)
from tcred.metrics.source_disjoint_io import (
    DEFAULT_STUDY_ROOT,
    file_record,
    load_prepared_suite,
    mapping,
    read_json,
    read_jsonl,
    resolve_path,
    sha256,
    write_json,
)
from tcred.metrics.task_judge_models import JudgeGraphPath
from tcred.metrics.tcred_diagnostic_runner import summarize_matched_baseline_dominance


@dataclass(frozen=True)
class _Observation:
    label: str
    start: date
    end: date


def run_source_disjoint_posthoc_audit(
    *,
    repository_root: Path,
    gold_dir: Path,
    study_root: Path = DEFAULT_STUDY_ROOT,
) -> dict[str, Path]:
    """Audit discovered label defects without changing the locked primary analysis."""

    repository_root = repository_root.resolve()
    study_root = resolve_path(repository_root, study_root)
    gold_dir = resolve_path(repository_root, gold_dir)
    suite, protocol = load_prepared_suite(
        repository_root=repository_root,
        study_root=study_root,
    )
    primary_manifest_path, scores_path = _verify_primary_scores(
        study_root,
        protocol_id=str(protocol["protocol_id"]),
    )
    records = [MetricScoreRecord.model_validate(row) for row in read_jsonl(scores_path)]
    bundle = load_bundle(study_root / "dataset" / "tcred_synth")
    audit = audit_graph_time_labels(
        suite=suite,
        questions={question.qid: question for question in bundle.questions},
    )
    invalid_pair_ids = {
        str(row["pair_id"])
        for row in audit["rows"]
        if row["status"] == "mutation_still_supports_gold"
    }

    output_dir = study_root / "posthoc_label_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "graph_time_label_audit.json"
    write_json(audit_path, audit)

    filtered_suite = DiagnosticSuite(
        seed=suite.seed,
        source_split=suite.source_split,
        pair_cap_per_phenomenon=suite.pair_cap_per_phenomenon,
        dataset_content_hashes=suite.dataset_content_hashes,
        cases=suite.cases,
        pairs=[pair for pair in suite.pairs if pair.pair_id not in invalid_pair_ids],
        audit={
            **suite.audit,
            "posthoc_sensitivity_excluded_pair_count": len(invalid_pair_ids),
            "posthoc_sensitivity_exclusion_reason": (
                "The graph-time mutation still selects the formal gold answer."
            ),
        },
    )
    projected, score_names = _project_preregistered_scores(records, protocol=protocol)
    inference = mapping(protocol, "inference")
    sensitivity = analyze_diagnostic_suite(
        filtered_suite,
        projected,
        dataset_root=study_root / "dataset",
        gold_dir=gold_dir,
        bootstrap_samples=int(inference["bootstrap_replicates"]),
        seed=int(inference["seed"]),
    )
    matched = summarize_matched_baseline_dominance(sensitivity)
    confirmation = source_disjoint_confirmation(
        suite=filtered_suite,
        records=records,
        matched=matched,
        protocol=protocol,
    )
    sensitivity["posthoc_label_sensitivity"] = {
        "status": "posthoc_sensitivity_not_a_replacement_confirmatory_test",
        "excluded_pair_ids": sorted(invalid_pair_ids),
        "excluded_pair_count": len(invalid_pair_ids),
        "analysis_score_names": score_names,
        "confirmation_under_exclusion": confirmation,
    }
    sensitivity_path = output_dir / "sensitivity_analysis.json"
    write_analysis(sensitivity, sensitivity_path)
    report_path = output_dir / "posthoc_label_audit_report.md"
    report_path.write_text(
        _render_report(audit=audit, confirmation=confirmation),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output_dir / "manifest.json"
    artifacts = [audit_path, sensitivity_path, report_path]
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "status": "complete_posthoc_integrity_audit",
            "generated_at": datetime.now(UTC).isoformat(),
            "protocol_id": protocol["protocol_id"],
            "primary_result_modified": False,
            "invalid_pair_count": len(invalid_pair_ids),
            "sensitivity_conclusion": confirmation["overall_conclusion"],
            "source_artifacts": {
                "primary_manifest": file_record(
                    primary_manifest_path,
                    relative_to=study_root,
                ),
                "primary_scores": file_record(scores_path, relative_to=study_root),
                "challenge_pairs": file_record(
                    study_root / "challenge" / "diagnostic_pairs.jsonl",
                    relative_to=study_root,
                ),
            },
            "implementation": file_record(
                repository_root / "src/tcred/metrics/source_disjoint_posthoc.py",
                relative_to=repository_root,
            ),
            "artifacts": [file_record(path, relative_to=output_dir) for path in artifacts],
            "interpretation": (
                "The locked primary result remains immutable. This audit identifies an "
                "operator-insensitive challenge mutation and measures result sensitivity to "
                "excluding objectively mislabeled pairs."
            ),
        },
    )
    return {
        "manifest": manifest_path,
        "audit": audit_path,
        "sensitivity": sensitivity_path,
        "report": report_path,
    }


def _verify_primary_scores(
    study_root: Path,
    *,
    protocol_id: str,
) -> tuple[Path, Path]:
    manifest_path = study_root / "tcred" / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Primary source-disjoint T-CRED manifest is not complete")
    if manifest.get("protocol_id") != protocol_id:
        raise ValueError("Primary source-disjoint T-CRED protocol does not match the audit")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Primary source-disjoint T-CRED manifest has no artifact ledger")
    expected_name = "metric_scores_with_tcred.jsonl"
    entries = [
        row
        for row in artifacts
        if isinstance(row, dict) and row.get("path") == expected_name
    ]
    if len(entries) != 1:
        raise ValueError("Primary manifest must contain exactly one score-table entry")
    scores_path = manifest_path.parent / expected_name
    if not scores_path.is_file() or sha256(scores_path) != str(entries[0].get("sha256", "")):
        raise ValueError("Primary source-disjoint score table differs from its manifest")
    return manifest_path, scores_path


def audit_graph_time_labels(
    *,
    suite: DiagnosticSuite,
    questions: dict[str, Any],
) -> dict[str, Any]:
    cases = {case.case_id: case for case in suite.cases}
    rows = []
    for pair in suite.pairs:
        if pair.phenomenon != "temporally_invalid_graph_path_only":
            continue
        question = questions[pair.qid]
        expected = _normalize_labels(question.gold_answer_text)
        left = cases[pair.left_case_id]
        right = cases[pair.right_case_id]
        canonical = _select_graph_answers(left, question.program)
        mutated = _select_graph_answers(right, question.program)
        if canonical != expected:
            status = "canonical_projection_mismatch"
        elif mutated == expected:
            status = "mutation_still_supports_gold"
        else:
            status = "mutation_invalidates_gold_as_intended"
        rows.append(
            {
                "pair_id": pair.pair_id,
                "qid": pair.qid,
                "scenario_id": pair.scenario_id,
                "operator": str(question.program.operator),
                "property_id": str(question.template_family_id or "").split(":", 1)[0],
                "expected_gold_labels": sorted(expected),
                "canonical_selected_labels": sorted(canonical),
                "mutated_selected_labels": sorted(mutated),
                "status": status,
            }
        )
    status_counts: dict[str, int] = {}
    operator_defects: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "mutation_still_supports_gold":
            operator = str(row["operator"])
            operator_defects[operator] = operator_defects.get(operator, 0) + 1
    return {
        "schema_version": "1.0",
        "status": "posthoc_after_primary_scores",
        "method": (
            "Replay one-edge displayed graph observations with the independent dataset "
            "QuestionProgram operator; compare canonical and mutated selected labels with the "
            "frozen gold answer text."
        ),
        "pair_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "defects_by_operator": dict(sorted(operator_defects.items())),
        "rows": rows,
    }


def _select_graph_answers(
    case: DiagnosticCase,
    program: QuestionProgram,
) -> set[str]:
    observations = _path_observations(case.task_judge_input.graph_paths)
    selected = _select_observations(observations, program)
    return _normalize_labels(observation.label for observation in selected)


def _path_observations(paths: list[JudgeGraphPath]) -> list[_Observation]:
    output = []
    for path in paths:
        if len(path.edges) != 1:
            continue
        edge = path.edges[0]
        if edge.valid_time.type == "unknown":
            continue
        endpoint = edge.target if edge.traversal_direction == "forward" else edge.source
        output.append(
            _Observation(
                label=endpoint.label,
                start=(
                    date.fromisoformat(edge.valid_time.start)
                    if edge.valid_time.start is not None
                    else date.min
                ),
                end=(
                    date.fromisoformat(edge.valid_time.end)
                    if edge.valid_time.end is not None
                    else date.max
                ),
            )
        )
    return output


def _select_observations(
    observations: list[_Observation],
    program: QuestionProgram,
) -> list[_Observation]:
    point = program.query_time.start
    if point is None:
        return []
    operator = program.operator
    if operator in {TemporalOperator.CURRENT, TemporalOperator.AS_OF, TemporalOperator.EFFECTIVE}:
        return [row for row in observations if row.start <= point <= row.end]
    if operator == TemporalOperator.DURING:
        end = program.query_time.end
        if end is None:
            return []
        return [row for row in observations if row.start <= point and row.end >= end]
    if operator == TemporalOperator.BETWEEN:
        end = program.query_time.end or point
        return [row for row in observations if max(row.start, point) <= min(row.end, end)]
    if operator == TemporalOperator.PREVIOUS:
        current = [row for row in observations if row.start <= point <= row.end]
        if len({row.label.casefold() for row in current}) != 1:
            return []
        return _latest_ended(
            [row for row in observations if row.end < min(item.start for item in current)]
        )
    if operator == TemporalOperator.NEXT:
        current = [row for row in observations if row.start <= point <= row.end]
        boundary = max((row.end for row in current), default=point)
        return _earliest_started([row for row in observations if row.start > boundary])
    if operator in {TemporalOperator.LATEST, TemporalOperator.LAST}:
        return _latest_started([row for row in observations if row.start <= point])
    if operator == TemporalOperator.FIRST:
        return _earliest_started(observations)
    if operator in {TemporalOperator.BEFORE, TemporalOperator.EXPIRED}:
        return _latest_ended([row for row in observations if row.end < point])
    if operator == TemporalOperator.AFTER:
        return _earliest_started([row for row in observations if row.start > point])
    return []


def _latest_started(rows: list[_Observation]) -> list[_Observation]:
    if not rows:
        return []
    latest = max(row.start for row in rows)
    return [row for row in rows if row.start == latest]


def _earliest_started(rows: list[_Observation]) -> list[_Observation]:
    if not rows:
        return []
    earliest = min(row.start for row in rows)
    return [row for row in rows if row.start == earliest]


def _latest_ended(rows: list[_Observation]) -> list[_Observation]:
    if not rows:
        return []
    latest = max(row.end for row in rows)
    return [row for row in rows if row.end == latest]


def _normalize_labels(values: Any) -> set[str]:
    return {" ".join(str(value).split()).casefold() for value in values if str(value).strip()}


def _render_report(
    *,
    audit: dict[str, Any],
    confirmation: dict[str, object],
) -> str:
    defects = int(audit["status_counts"].get("mutation_still_supports_gold", 0))
    canonical_mismatches = int(
        audit["status_counts"].get("canonical_projection_mismatch", 0)
    )
    graph = next(
        row
        for row in confirmation["comparisons"]
        if row["construct"] == "graph_sufficiency"
    )
    return "\n".join(
        [
            "# Source-Disjoint Post-Hoc Label Audit",
            "",
            "> **Status: post-hoc integrity and sensitivity analysis.** This document does not",
            "> overwrite the locked primary result or convert this audit into a new untouched",
            "> test.",
            "",
            "## Defect",
            "",
            "The graph-time diagnostic builder replaced every displayed edge interval with year",
            "1000. That fixed date is not guaranteed to invalidate every temporal operator. The",
            "independent dataset `QuestionProgram` replay checks whether the mutated paths still",
            "select exactly the frozen gold answer.",
            "",
            f"- graph-time pairs audited: **{audit['pair_count']}**",
            f"- mutation still supports gold: **{defects}**",
            f"- canonical projection mismatches: **{canonical_mismatches}**",
            f"- defects by operator: `{audit['defects_by_operator']}`",
            "",
            "## Sensitivity result",
            "",
            f"After excluding only those {defects} objectively mislabeled pairs, the graph matched",
            f"effect is **{float(graph['directional_difference']):+.4f}** with 95% CI",
            f"**[{float(graph['directional_ci95'][0]):+.4f}, "
            f"{float(graph['directional_ci95'][1]):+.4f}]** and Holm-adjusted",
            f"`p={float(graph['primary_gain_holm_p_value']):.4g}`.",
            "",
            f"The eight-row sensitivity decision is `{confirmation['overall_conclusion']}`.",
            "The primary result remains the preregistered calculation; this sensitivity analysis",
            "shows whether its conclusion depends on the discovered label noise.",
            "",
            "## Scientific interpretation",
            "",
            "A passing sensitivity result supports robustness of the numerical conclusion but does",
            "not erase the failed preflight guarantee. The study must be described as a locked",
            "formal result with a disclosed, downward-biasing graph-label defect and a post-hoc",
            "robustness check, not as a flawless external confirmation.",
            "",
        ]
    )
