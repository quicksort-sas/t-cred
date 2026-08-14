from __future__ import annotations

from collections.abc import Mapping

from tcred.metrics.task_judge_analysis import FIELD_METRICS
from tcred.metrics.task_judge_models import JUDGED_FIELDS, PromptSelection

_FIELD_LABELS = {
    "answer_correct": "Answer correctness",
    "temporal_correct": "Temporal correctness",
    "evidence_supports_answer": "Evidence supports answer",
    "citation_temporally_valid": "Citation temporal validity",
    "graph_evidence_sufficient": "Graph-evidence sufficiency",
    "response_decision_appropriate": "Response-decision appropriateness",
}

_COMPARISON_METRICS = {
    "answer_correct": (
        "tcred_judge_answer_correct",
        "pedants_probability",
        "g_eval_answer_correctness",
        "ragchecker_f1",
        "sas_cross_encoder",
        "bertscore_f1",
        "token_f1",
    ),
    "temporal_correct": (
        "tcred_judge_temporal_correct",
        "g_eval_answer_correctness",
        "ragchecker_f1",
        "minicheck_retrieved_mean",
        "alignscore_retrieved",
    ),
    "evidence_supports_answer": (
        "tcred_judge_evidence_supports_answer",
        "minicheck_retrieved_mean",
        "alignscore_retrieved",
        "ragchecker_faithfulness",
    ),
    "citation_temporally_valid": (
        "tcred_judge_citation_temporally_valid",
        "minicheck_cited_mean",
        "alignscore_cited",
        "required_citation_precision",
        "required_citation_recall",
    ),
    "graph_evidence_sufficient": ("tcred_judge_graph_evidence_sufficient",),
    "response_decision_appropriate": ("tcred_judge_response_decision_appropriate",),
}


def render_task_judge_report(
    *,
    split: Mapping[str, object],
    selection: PromptSelection,
    calibration_classification: Mapping[str, object],
    held_out_classification: Mapping[str, object],
    all_gold_classification: Mapping[str, object],
    human_system_classification: Mapping[str, object],
    held_out_metric_analysis: Mapping[str, object],
    complete_metric_analysis: Mapping[str, object],
    stability: Mapping[str, object],
    pointer_audit: Mapping[str, object],
    judgment_distributions: Mapping[str, object],
    usage: Mapping[str, object],
    provider: str,
    model: str,
    requests_per_second: float,
) -> str:
    lines = [
        "# T-CRED Task-Matched LLM Judge Evaluation",
        "",
        "## Experimental Contract",
        "",
        f"- Provider/model: `{provider}` / `{model}`.",
        f"- Prompt selected on calibration only: `{selection.selected_variant}`.",
        "- Judge design: blinded, reference-hidden evidence stage followed by a reference-aware "
        "answer stage.",
        f"- Direct request throttle: `{requests_per_second}` requests/second.",
        f"- Calibration responses: {len(split.get('calibration_metric_ids', []))}.",
        f"- Held-out responses: {len(split.get('held_out_metric_ids', []))}.",
        "- Split unit: dataset family plus question ID; no question crosses partitions.",
        "- Human gold is the validity population. Full-system results are descriptive automatic "
        "scores, not additional human-validation evidence.",
        "",
        "## Prompt Selection",
        "",
        f"Selection rule: {selection.selection_rule}",
        "",
        "| Prompt | Macro field F1 | Macro exact | Stage calls | Tokens | Selected |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, value in sorted(selection.candidates.items()):
        lines.append(
            f"| `{name}` | {_number(value.get('macro_field_f1'))} | "
            f"{_number(value.get('macro_field_exact_agreement'))} | "
            f"{value.get('stage_calls', 0)} | {value.get('total_tokens', 0)} | "
            f"{'yes' if name == selection.selected_variant else 'no'} |"
        )

    lines.extend(_classification_section("Calibration Classification", calibration_classification))
    lines.extend(_classification_section("Held-Out Classification", held_out_classification))
    lines.extend(_classification_section("All Human Gold", all_gold_classification))
    lines.extend(
        _classification_section("Human-Judged QA-System Outputs", human_system_classification)
    )
    lines.extend(
        _metric_comparison_section(
            "Held-Out Metric-to-Human Comparison",
            held_out_metric_analysis,
        )
    )
    lines.extend(
        _metric_comparison_section(
            "All-Gold Metric-to-Human Comparison",
            complete_metric_analysis,
        )
    )
    lines.extend(_full_system_section(complete_metric_analysis))
    lines.extend(_judgment_distribution_section(judgment_distributions))
    lines.extend(_dataset_section(complete_metric_analysis))
    lines.extend(_stability_section(stability))
    lines.extend(_pointer_audit_section(pointer_audit))
    final_usage = _mapping(usage.get("final_selected_logical", {}))
    experiment_usage = _mapping(usage.get("unique_accepted_stage_records", {}))
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "The final logical count includes every selected-prompt score, including values reused "
            "from calibration caches. The accepted-record count deduplicates those reused records "
            "and includes the losing calibration prompt and stability repeat. It is not a billing "
            "count: abandoned contract trials and failed requests without an accepted response are "
            "not represented in these caches.",
            "",
            f"- Final selected logical stage calls: {final_usage.get('stage_calls', 0)}.",
            f"- Unique accepted stage records: {experiment_usage.get('stage_calls', 0)}.",
            "- Request attempts associated with accepted records: "
            f"{experiment_usage.get('request_attempts_for_accepted_records', 0)}.",
            f"- Retried accepted stage records: {experiment_usage.get('retried_stage_calls', 0)}.",
            f"- Experiment input tokens: {experiment_usage.get('input_tokens', 0)}.",
            f"- Experiment output tokens: {experiment_usage.get('output_tokens', 0)}.",
            f"- Experiment total tokens: {experiment_usage.get('total_tokens', 0)}.",
            "",
            "## Interpretation Limits",
            "",
            "1. Prompt selection and final evaluation use disjoint question clusters, but the "
            "held-out set is still small for rare graph and response-decision labels.",
            "2. Only one model/provider family is evaluated in this run. It is not an independent "
            "judge panel and cannot establish model-family robustness.",
            "3. Confidence is self-reported and must be evaluated empirically; it is not a "
            "probability guarantee.",
            "4. The 2,344 complete-run system answers have no additional human labels. Their "
            "means support system comparison only under the judge's measured validity limits.",
            "5. This judge is an automatic metric. Human consensus and adjudication remain the "
            "gold authority.",
            "",
        ]
    )
    return "\n".join(lines)


def _classification_section(title: str, analysis: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        f"Responses: {analysis.get('rows', 0)}; question clusters: "
        f"{analysis.get('question_clusters', 0)}.",
        "",
        "| Field | n | Exact agreement [95% CI] | Macro F1 | Cohen kappa | "
        "Quadratic kappa (n) | Spearman rho [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fields = _mapping(analysis.get("fields", {}))
    for field in JUDGED_FIELDS:
        value = _mapping(fields.get(field, {}))
        if not value:
            continue
        ordinal = _mapping(value.get("ordinal_association", {}))
        exact_interval = _estimate_interval(
            value.get("exact_agreement"), value.get("exact_agreement_ci95")
        )
        lines.append(
            f"| {_FIELD_LABELS[field]} | {value.get('n', 0)} | "
            f"{exact_interval} | "
            f"{_number(value.get('macro_f1'))} | "
            f"{_number(value.get('cohen_kappa'))} | "
            f"{_number(value.get('quadratic_weighted_kappa'))} "
            f"({value.get('quadratic_weighted_kappa_n', 0)}) | "
            f"{_estimate_interval(ordinal.get('spearman'), ordinal.get('spearman_ci95'))} |"
        )
    return lines


def _metric_comparison_section(title: str, analysis: Mapping[str, object]) -> list[str]:
    human = _mapping(analysis.get("human_gold_all", {}))
    correlations = _mapping(human.get("human_correlations", {}))
    lines = ["", f"## {title}", ""]
    for field in JUDGED_FIELDS:
        field_metrics = _mapping(correlations.get(field, {}))
        selected = [name for name in _COMPARISON_METRICS[field] if name in field_metrics]
        if not selected:
            continue
        lines.extend(
            [
                f"### {_FIELD_LABELS[field]}",
                "",
                "| Metric | n | Spearman rho [95% CI] | Kendall tau-b | AUROC yes/no |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in selected:
            value = _mapping(field_metrics[metric])
            lines.append(
                f"| `{metric}` | {value.get('n', 0)} | "
                f"{_estimate_interval(value.get('spearman'), value.get('spearman_ci95'))} | "
                f"{_number(value.get('kendall_tau_b'))} | "
                f"{_number(value.get('auroc_yes_vs_no'))} |"
            )
        lines.append("")
    return lines


def _full_system_section(analysis: Mapping[str, object]) -> list[str]:
    groups = _mapping(analysis.get("full_by_system", {}))
    lines = [
        "",
        "## Complete-Run System Scores",
        "",
        "Each cell is mean (n). It uses only responses for which the corresponding field is "
        "applicable and the judge did not return unjudgeable.",
        "",
        "| System | Answer | Temporal | Evidence | Citation time | Graph | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system, raw in sorted(groups.items()):
        means = _mapping(_mapping(raw).get("metric_means", {}))
        values = []
        for field in JUDGED_FIELDS:
            metric = FIELD_METRICS[field]
            estimate = _mapping(means.get(metric, {}))
            values.append(f"{_number(estimate.get('mean'))} ({int(estimate.get('n', 0) or 0)})")
        lines.append(f"| `{system}` | " + " | ".join(values) + " |")
    return lines


def _dataset_section(analysis: Mapping[str, object]) -> list[str]:
    groups = _mapping(analysis.get("full_by_dataset_and_system", {}))
    lines = [
        "",
        "## Complete-Run Scores by Dataset",
        "",
        "Each cell is mean (n), with the same applicability and unjudgeable handling as above.",
        "",
        "| Dataset / system | Answer | Temporal | Evidence | Citation time | Graph | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, raw in sorted(groups.items()):
        means = _mapping(_mapping(raw).get("metric_means", {}))
        values = []
        for field in JUDGED_FIELDS:
            estimate = _mapping(means.get(FIELD_METRICS[field], {}))
            values.append(f"{_number(estimate.get('mean'))} ({int(estimate.get('n', 0) or 0)})")
        lines.append(f"| `{group}` | " + " | ".join(values) + " |")
    return lines


def _judgment_distribution_section(analysis: Mapping[str, object]) -> list[str]:
    groups = _mapping(analysis.get("by_system", {}))
    lines = [
        "",
        "## Complete-Run Categorical Coverage",
        "",
        "Determinate means use yes=1, partial=0.5, and no=0. Coverage reports how often the "
        "judge did not return unjudgeable among applicable responses. Bounds map every "
        "unjudgeable result to no or yes; they are transparency diagnostics, not primary scores.",
        "",
        "| System / field | Applicable | yes | partial | no | unjudgeable | Coverage | "
        "Determinate mean | Bounds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, raw in sorted(groups.items()):
        fields = _mapping(_mapping(raw).get("fields", {}))
        for field in JUDGED_FIELDS:
            value = _mapping(fields.get(field, {}))
            if not value or not value.get("applicable"):
                continue
            counts = _mapping(value.get("label_counts", {}))
            lower = _number(value.get("unjudgeable_as_no_lower_bound"))
            upper = _number(value.get("unjudgeable_as_yes_upper_bound"))
            lines.append(
                f"| `{system}` / {_FIELD_LABELS[field]} | {value.get('applicable', 0)} | "
                f"{counts.get('yes', 0)} | {counts.get('partial', 0)} | "
                f"{counts.get('no', 0)} | {counts.get('unjudgeable', 0)} | "
                f"{_number(value.get('determinate_coverage'))} | "
                f"{_number(value.get('determinate_mean'))} | {lower}-{upper} |"
            )
    return lines


def _stability_section(stability: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## Seed Stability",
        "",
        "| Field | n | Exact label agreement | Mean absolute confidence difference |",
        "|---|---:|---:|---:|",
    ]
    for field, raw in _mapping(stability.get("fields", {})).items():
        value = _mapping(raw)
        lines.append(
            f"| {_FIELD_LABELS.get(field, field)} | {value.get('n', 0)} | "
            f"{_number(value.get('label_exact_agreement'))} | "
            f"{_number(value.get('mean_absolute_confidence_difference'))} |"
        )
    return lines


def _pointer_audit_section(audit: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## Support-Pointer Audit",
        "",
        "Unknown evidence/path handles are explanation-provenance errors. They are reported here "
        "but never change, suppress, or repair the categorical metric label.",
        "",
        "| Scope | Stage calls | Calls with warnings | Warning rate | Unknown evidence IDs | "
        "Unknown path IDs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope, raw in sorted(_mapping(audit).items()):
        value = _mapping(raw)
        lines.append(
            f"| `{scope}` | {value.get('stage_calls', 0)} | "
            f"{value.get('calls_with_pointer_warnings', 0)} | "
            f"{_number(value.get('call_warning_rate'))} | "
            f"{value.get('unknown_evidence_id_occurrences', 0)} | "
            f"{value.get('unknown_path_id_occurrences', 0)} |"
        )
    return lines


def _estimate_interval(estimate: object, interval: object) -> str:
    if estimate is None:
        return "NA"
    if not isinstance(interval, list) or len(interval) != 2:
        return _number(estimate)
    return f"{_number(estimate)} [{_number(interval[0])}, {_number(interval[1])}]"


def _number(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
