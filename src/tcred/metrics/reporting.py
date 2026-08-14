from __future__ import annotations

from collections.abc import Iterable, Mapping

_HUMAN_SCORE_COLUMNS = (
    "token_f1",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "pedants_match",
    "g_eval_answer_correctness",
    "ragchecker_f1",
    "ragchecker_faithfulness",
    "minicheck_retrieved_mean",
    "alignscore_retrieved",
    "tcred_answer_equivalence",
    "tcred_semantic_attribution",
    "tcred_temporal_correctness",
    "tcred_citation_quality",
    "tcred_graph_answer_coverage",
    "tcred_response_decision",
)
_FULL_SCORE_COLUMNS = (
    "exact_match",
    "token_f1",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "pedants_match",
    "alignscore_retrieved",
    "alignscore_cited",
    "citation_presence",
    "required_citation_precision",
    "required_citation_recall",
    "citation_resolution_rate",
    "retrieval_precision_at_10",
    "retrieval_recall_at_10",
    "retrieval_average_precision_at_10",
    "retrieval_mrr",
    "retrieval_r_precision",
    "retrieval_ndcg_at_10",
    "tcred_answer_equivalence",
    "tcred_semantic_attribution",
    "tcred_temporal_attribution",
    "tcred_temporal_correctness",
    "tcred_citation_quality",
    "tcred_t_ndcg_at_10",
    "tcred_graph_answer_coverage",
    "tcred_response_decision",
)
_PAIRED_SCORE_COLUMNS = (
    "token_f1",
    "rouge_2",
    "rouge_l",
    "bertscore_f1",
    "sas_cross_encoder",
    "pedants_probability",
    "alignscore_retrieved",
    "alignscore_cited",
    "required_citation_recall",
    "retrieval_recall_at_10",
    "retrieval_average_precision_at_10",
    "retrieval_ndcg_at_10",
    "tcred_answer_equivalence",
    "tcred_temporal_correctness",
    "tcred_citation_quality",
    "tcred_t_ndcg_at_10",
    "tcred_graph_answer_coverage",
    "tcred_response_decision",
)
_TARGET_LABELS = {
    "answer_correct": "Answer correctness",
    "evidence_supports_answer": "Evidence support",
    "temporal_correct": "Temporal correctness (diagnostic)",
    "citation_temporally_valid": "Citation temporal validity (diagnostic)",
    "graph_evidence_sufficient": "Graph evidence sufficiency",
    "response_decision_appropriate": "Response decision",
}


def render_metric_report(analysis: Mapping[str, object]) -> str:
    """Render a self-contained Markdown view of the machine-readable analysis."""
    human_metrics = set(_mapping(_mapping(analysis["human_gold_all"])["metric_means"]))
    coverage = _mapping(analysis["coverage"])
    score_names = coverage.get("score_names")
    computed_metrics = (
        {str(value) for value in score_names} if isinstance(score_names, list) else human_metrics
    )
    human_score_columns = tuple(
        metric for metric in _HUMAN_SCORE_COLUMNS if metric in computed_metrics
    )
    full_score_columns = tuple(
        metric for metric in _FULL_SCORE_COLUMNS if metric in computed_metrics
    )
    judge_metrics_present = bool({"g_eval_answer_correctness", "ragchecker_f1"} & human_metrics)
    complete_scope_note = (
        "MiniCheck is intentionally limited to human-gold units; the other selected local and "
        "deterministic metrics cover the complete run."
    )
    if judge_metrics_present:
        complete_scope_note = (
            "LLM-judge and MiniCheck scores are intentionally limited to human-gold units; the "
            "other selected local and deterministic metrics cover the complete run."
        )
    lines = [
        "# Current Automatic Metric Evaluation",
        "",
        "## Scope",
        "",
        "This report deliberately separates two populations. The human-gold population is used "
        "for metric-to-human meta-evaluation. The complete QA run is used for higher-precision "
        "descriptive system scoring, but it cannot add human-correlation evidence.",
        "",
    ]
    lines.extend(_coverage_table(_mapping(analysis["coverage"])))
    lines.extend(_answer_overlap_section(_mapping(analysis.get("human_full_answer_overlap", {}))))
    lines.extend(_target_coverage_table(_mapping(analysis["human_target_coverage"])))
    lines.extend(
        [
            "",
            "## Metric Suite",
            "",
            "| Component | Metrics | Role |",
            "|---|---|---|",
            *_metric_suite_rows(computed_metrics),
        ]
    )
    lines.extend(
        [
            "",
            "No ordinary reference, grounding, or citation metric is reinterpreted as a temporal "
            "metric. Correlation with temporal labels is reported diagnostically to measure how "
            "much temporal validity these non-temporal metrics happen to capture.",
            "",
            "## Selection Boundary",
            "",
            "This artifact reports only metrics that were actually computed. The peer-reviewed "
            "candidate audit, exact inclusion criteria, implementation variants, and reasons for "
            "excluding or deferring other methods are documented in "
            "`docs/metric-suite-selection-and-non-llm-expansion-2026-08-14.md`. This run makes no "
            "claim to contain every method described as state of the art in a paper.",
        ]
    )

    lines.extend(
        _correlation_section(
            "Human-Gold Correlations: All Candidate Types",
            _mapping(analysis["human_gold_all"]),
        )
    )
    lines.extend(
        _metric_difference_section(
            "Paired Differences Between Answer-Metric Correlations: All Gold",
            _mapping(analysis["human_metric_correlation_differences"]),
        )
    )
    lines.extend(
        _metric_difference_section(
            "Paired Differences Between Answer-Metric Correlations: QA Outputs",
            _mapping(analysis["human_system_metric_correlation_differences"]),
        )
    )
    lines.extend(
        _metric_difference_section(
            "Paired Differences Between Evidence-Metric Correlations: All Gold",
            _mapping(analysis.get("human_evidence_metric_correlation_differences", {})),
        )
    )
    lines.extend(
        _metric_difference_section(
            "Paired Differences Between Citation-Text Metrics on Temporal Labels (Diagnostic)",
            _mapping(analysis.get("human_citation_metric_correlation_differences", {})),
        )
    )
    lines.extend(
        _correlation_section(
            "Human-Gold Correlations: Actual QA-System Answers Only",
            _mapping(analysis["human_gold_system_outputs"]),
        )
    )
    lines.extend(
        _correlation_group_section(
            "Human Correlations Within Each QA System",
            _mapping(analysis["human_by_system"]),
            note=(
                "These correlations use only the human-presented output from the named system. "
                "They are exploratory: each system has 12-19 units, some targets have fewer "
                "applicable labels, and several labels are constant within a system."
            ),
        )
    )
    lines.extend(
        _correlation_group_section(
            "Human Correlations by Candidate Source",
            _mapping(analysis["human_by_source_kind"]),
            targets=("answer_correct", "temporal_correct"),
        )
    )
    lines.extend(
        _correlation_group_section(
            "Human Correlations by Dataset",
            _mapping(analysis["human_by_dataset"]),
            targets=("answer_correct", "temporal_correct", "evidence_supports_answer"),
        )
    )
    lines.extend(
        _correlation_group_section(
            "Human-Judged QA-System Correlations by Dataset",
            _mapping(analysis["human_system_outputs_by_dataset"]),
            targets=("answer_correct", "temporal_correct", "evidence_supports_answer"),
        )
    )
    lines.extend(
        _group_mean_section(
            "QA Systems on the Human-Judged Subset",
            _mapping(analysis["human_by_system"]),
            human_score_columns,
            human_target="answer_correct",
            note=(
                "These are the scores directly comparable with human judgments. Samples are "
                "small and unequal across systems, so do not use unpaired differences as a "
                "definitive system ranking."
            ),
        )
    )
    lines.extend(
        _group_mean_section(
            "QA Systems on the Complete Generated Run",
            _mapping(analysis["full_by_system"]),
            full_score_columns,
            note=(
                "These estimates use every generated answer. Their confidence intervals quantify "
                "sampling variation in this benchmark, not validity against human judgment. "
                f"{complete_scope_note}"
            ),
        )
    )
    lines.extend(
        _group_mean_section(
            "Human-Gold Results by Dataset",
            _mapping(analysis["human_by_dataset"]),
            human_score_columns,
            human_target="answer_correct",
        )
    )
    lines.extend(
        _group_mean_section(
            "Human-Judged QA-System Results by Dataset",
            _mapping(analysis["human_system_outputs_by_dataset"]),
            human_score_columns,
            human_target="answer_correct",
        )
    )
    lines.extend(
        _group_mean_section(
            "Complete-Run Results by Dataset",
            _mapping(analysis["full_by_dataset"]),
            full_score_columns,
        )
    )
    lines.extend(
        _group_mean_section(
            "Complete-Run Results by Dataset and QA System",
            _mapping(analysis["full_by_dataset_and_system"]),
            full_score_columns,
        )
    )
    lines.extend(_paired_section(_mapping(analysis["full_paired_system_differences"])))
    lines.extend(
        _dataset_paired_sections(_mapping(analysis["full_paired_system_differences_by_dataset"]))
    )
    lines.extend(_sensitivity_section(analysis))
    if judge_metrics_present:
        lines.extend(_stability_section(_mapping(analysis.get("judge_stability", {}))))
    limitations = [
        "- MiniCheck and AlignScore assess textual entailment and do not independently reason "
        "about evidence valid-time or graph-path validity.",
        "- BERTScore, SAS, PEDANTS, and lexical overlap compare against a single reference and "
        "can penalize valid alternatives or reward a temporally wrong near-paraphrase.",
        "- PEDANTS' released probability is not assumed to be calibrated on T-CRED; no local "
        "threshold or recalibration is fitted.",
        "- Dataset-level results for PAT and HoH have much smaller sample sizes than synthetic "
        "results and should be read with their intervals, not as equally precise estimates.",
        "- The frozen gold set is selected from the co-rated portion of an incomplete collection; "
        "the 113 units are not a random sample of all 320 planned units.",
        "- Hard adjudications were produced by one AI adjudicator. Primary results therefore need "
        "the included agreement-only and medium-confidence sensitivity analyses.",
        "- Only the exact human-presented response inherits a human label. A later response for "
        "the same dataset/question/system tuple is not treated as human-scored.",
    ]
    references = [
        "- Tang, Laban, and Durrett (2024). *MiniCheck: Efficient Fact-Checking of LLMs on "
        "Grounding Documents*. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.499/",
        "- Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR 2020. "
        "https://openreview.net/forum?id=SkeHuCVFDr",
        "- Risch et al. (2021). *Semantic Answer Similarity for Evaluating Question Answering "
        "Models*. MRQA 2021. https://aclanthology.org/2021.mrqa-1.15/",
        "- Li et al. (2024). *PEDANTS: Cheap but Effective and Interpretable Answer "
        "Equivalence*. Findings of EMNLP 2024. "
        "https://aclanthology.org/2024.findings-emnlp.548/",
        "- Zha et al. (2023). *AlignScore: Evaluating Factual Consistency with a Unified "
        "Alignment Function*. ACL 2023. https://aclanthology.org/2023.acl-long.634/",
        "- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. Text "
        "Summarization Branches Out. https://aclanthology.org/W04-1013/",
    ]
    if judge_metrics_present:
        limitations[0:0] = [
            "- The RAGChecker family follows the published claim-level definitions, but uses a "
            "pinned structured LLM judge rather than the paper's original Llama-3-70B backend. "
            "Results are therefore explicitly named RAGChecker-style.",
            "- G-Eval-style scores can inherit model, prompt, and self-preference biases. A "
            "repeat sample is reported to quantify run-to-run instability.",
        ]
        references[0:0] = [
            "- Ru et al. (2024). *RAGChecker: A Fine-grained Framework for Diagnosing "
            "Retrieval-Augmented Generation*. NeurIPS 2024 Datasets and Benchmarks Track. "
            "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
            "27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html",
            "- Liu et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human "
            "Alignment*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.153/",
            "- Gao et al. (2023). *Enabling Large Language Models to Generate Text with "
            "Citations*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.398/",
        ]
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "1. The unit-level human analyses use the final frozen gold labels. `yes`, `partial`, "
            "and `no` map to 1, 0.5, and 0. `unjudgeable` and non-applicable fields are excluded "
            "only from the affected target analysis.",
            "2. Confidence intervals use a qid-cluster bootstrap, preserving dependence when "
            "multiple answer variants share a question. Cluster identity is the pair of dataset "
            "family and qid.",
            "3. Spearman rho and Kendall tau-b measure ordinal association. AUROC and average "
            "precision compare only unambiguous `yes` versus `no` cases.",
            "4. System-specific human correlations are exploratory because each system has only "
            "a small number of human-scored answers. The pooled human analysis is the primary "
            "metric meta-evaluation.",
            "5. The human-presented system answer is immutable. It is not silently replaced by a "
            "later run for the same question, so the human subset and complete run remain "
            "distinct.",
            "6. Correlation does not establish calibration, causal validity, or sensitivity to "
            "all temporal error types. Confidence intervals and failure analysis remain necessary.",
            "7. Graph-evidence sufficiency and response-decision appropriateness have human gold "
            "but no construct-matched off-the-shelf metric in this run; they are not proxied by "
            "answer or retrieval scores.",
            "",
            "## Methodological Limitations",
            "",
            *limitations,
            "",
            "## References",
            "",
            *references,
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _metric_suite_rows(metrics: set[str]) -> list[str]:
    rows: list[str] = []
    if "exact_match" in metrics:
        rouge_names = [
            name
            for name in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
            if name.lower().replace("-", "_") in metrics
        ]
        lexical = ", ".join(["Exact match", "token F1", *rouge_names])
        rows.append(f"| Lexical reference baselines | {lexical} | Transparent overlap baselines |")
    semantic: list[str] = []
    if "bertscore_f1" in metrics:
        semantic.append("BERTScore")
    if "sas_cross_encoder" in metrics:
        semantic.append("SAS cross-encoder")
    if "pedants_probability" in metrics:
        semantic.append("PEDANTS")
    if semantic:
        rows.append(
            "| Semantic reference metrics | "
            + ", ".join(semantic)
            + " | Contextual similarity and QA answer equivalence |"
        )
    grounding: list[str] = []
    if "minicheck_retrieved_mean" in metrics:
        grounding.append("MiniCheck-DeBERTa-v3-Large")
    if "alignscore_retrieved" in metrics:
        grounding.append("AlignScore-base NLI-SP")
    if grounding:
        rows.append(
            "| Grounded fact checking | "
            + ", ".join(grounding)
            + " | Evidence support at sentence level |"
        )
    if "required_citation_precision" in metrics:
        citation = "Required-evidence precision/recall and citation resolution"
        if "alignscore_cited" in metrics:
            citation += ", AlignScore over cited evidence"
        rows.append(
            f"| Citation evaluation | {citation} | Citation selection and textual support |"
        )
    if "retrieval_precision_at_10" in metrics:
        retrieval = ["P@10", "Recall@10", "Hit@10"]
        if "retrieval_average_precision_at_10" in metrics:
            retrieval.append("AP@10")
        retrieval.append("MRR")
        if "retrieval_r_precision" in metrics:
            retrieval.append("R-precision")
        retrieval.append("nDCG@10")
        rows.append(
            "| Retrieval evaluation | "
            + ", ".join(retrieval)
            + " | Gold-evidence retrieval quality |"
        )
    if "ragchecker_f1" in metrics:
        rows.append(
            "| Claim-level LLM diagnosis | RAGChecker-style answer and grounding metrics | "
            "Claim completeness, correctness, and grounding |"
        )
    if "g_eval_answer_correctness" in metrics:
        rows.append(
            "| Rubric-based LLM judge | G-Eval-style correctness and relevance | "
            "Holistic answer quality |"
        )
    if "alce_citation_completeness" in metrics:
        rows.append(
            "| Claim-level citation judge | ALCE-style completeness and precision | "
            "Whether citations support response claims |"
        )
    return rows


def _coverage_table(coverage: Mapping[str, object]) -> list[str]:
    populations = _mapping(coverage.get("by_population", {}))
    rows = [
        "| Population | Units | Human labels | Valid use |",
        "|---|---:|---|---|",
    ]
    descriptions = {
        "human_gold": (
            "Yes",
            "Metric-to-human association and directly comparable descriptive scores",
        ),
        "system_full": (
            "No",
            "Aggregate automatic scores, paired system contrasts, and dataset diagnostics",
        ),
    }
    for population, count in populations.items():
        labels, purpose = descriptions.get(str(population), ("Unknown", "Descriptive analysis"))
        rows.append(f"| `{population}` | {count} | {labels} | {purpose} |")
    return rows


def _answer_overlap_section(overlap: Mapping[str, object]) -> list[str]:
    if not overlap:
        return []
    totals = _mapping(overlap.get("all", {}))
    lines = [
        "",
        "## Human-Presented vs Complete-Run Answer Identity",
        "",
        "Human labels attach to the immutable answer shown during annotation. The complete QA "
        "run is a later scored population; matching a dataset, question, and system name does "
        "not transfer the label unless the answer text is identical.",
        "",
        "| Group | Human system answers | Exact text in complete run | Different text | Missing |",
        "|---|---:|---:|---:|---:|",
        _overlap_row("All systems", totals),
    ]
    for system, counts in sorted(_mapping(overlap.get("by_system", {})).items()):
        lines.append(_overlap_row(system, _mapping(counts)))
    return lines


def _overlap_row(name: str, counts: Mapping[str, object]) -> str:
    return (
        f"| {name} | {int(counts.get('total', 0))} | {int(counts.get('exact', 0))} | "
        f"{int(counts.get('different', 0))} | {int(counts.get('missing', 0))} |"
    )


def _target_coverage_table(targets: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## Human-Target Coverage",
        "",
        "| Human field | Gold labels | Automatic metrics | Status |",
        "|---|---:|---|---|",
    ]
    for field, value in targets.items():
        target = _mapping(value)
        metrics = target.get("automatic_metrics", [])
        rendered = ", ".join(f"`{metric}`" for metric in metrics) if metrics else "None"
        lines.append(
            f"| `{field}` | {target.get('gold_label_count', 0)} | {rendered} | "
            f"{target.get('status', 'unknown')} |"
        )
    lines.extend(
        [
            "",
            "Uncovered fields are retained in the human gold but excluded from automatic "
            "meta-evaluation. Substituting a semantically different proxy would overstate "
            "coverage.",
        ]
    )
    return lines


def _correlation_section(title: str, population: Mapping[str, object]) -> list[str]:
    lines = ["", f"## {title}", ""]
    lines.append(
        f"Units: **{population.get('units', 0)}**; unique questions: "
        f"**{population.get('unique_qids', 0)}**."
    )
    correlations = _mapping(population.get("human_correlations", {}))
    for target, metrics_value in correlations.items():
        lines.extend(["", f"### {_TARGET_LABELS.get(target, target)}", ""])
        lines.extend(
            [
                "| Metric | n | Spearman rho [95% CI] | Kendall tau-b | "
                "Binary n (yes rate) | AUROC yes/no | AUPRC yes/no |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, summary_value in _mapping(metrics_value).items():
            summary = _mapping(summary_value)
            lines.append(
                f"| `{metric}` | {summary.get('n', 0)} | "
                f"{_estimate_interval(summary.get('spearman'), summary.get('spearman_ci95'))} | "
                f"{_number(summary.get('kendall_tau_b'))} | "
                f"{_binary_coverage(summary)} | "
                f"{_number(summary.get('auroc_yes_vs_no'))} | "
                f"{_number(summary.get('auprc_yes_vs_no'))} |"
            )
    return lines


def _correlation_group_section(
    title: str,
    groups: Mapping[str, object],
    *,
    targets: tuple[str, ...] = ("answer_correct", "temporal_correct"),
    metrics: tuple[str, ...] = (
        "token_f1",
        "bertscore_f1",
        "sas_cross_encoder",
        "pedants_probability",
        "g_eval_answer_correctness",
        "ragchecker_f1",
    ),
    note: str | None = None,
) -> list[str]:
    lines = ["", f"## {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.extend(
        [
            "| Group | Human target | Metric | n | Spearman rho [95% CI] | Kendall tau-b |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for group, value in groups.items():
        correlations = _mapping(_mapping(value).get("human_correlations", {}))
        for target in targets:
            summaries = _mapping(correlations.get(target, {}))
            for metric in metrics:
                summary = _mapping(summaries.get(metric, {}))
                if not summary or not summary.get("n"):
                    continue
                lines.append(
                    f"| {group} | {_TARGET_LABELS.get(target, target)} | `{metric}` | "
                    f"{summary.get('n', 0)} | "
                    f"{_estimate_interval(summary.get('spearman'), summary.get('spearman_ci95'))} "
                    "| "
                    f"{_number(summary.get('kendall_tau_b'))} |"
                )
    return lines


def _metric_difference_section(title: str, pairs: Mapping[str, object]) -> list[str]:
    if not pairs:
        return []
    lines = [
        "",
        f"## {title}",
        "",
        "Each row is Spearman rho(left) minus Spearman rho(right), estimated on identical "
        "units with a joint qid-cluster bootstrap. An interval crossing zero does not establish "
        "a difference in this sample.",
        "",
        "| Metric pair | n | Delta rho [95% CI] |",
        "|---|---:|---:|",
    ]
    for pair, value in pairs.items():
        summary = _mapping(value)
        lines.append(
            f"| {pair} | {summary.get('n', 0)} | "
            f"{_estimate_interval(summary.get('mean_difference'), summary.get('ci95'))} |"
        )
    return lines


def _group_mean_section(
    title: str,
    groups: Mapping[str, object],
    columns: Iterable[str],
    *,
    note: str | None = None,
    human_target: str | None = None,
) -> list[str]:
    columns = tuple(columns)
    lines = ["", f"## {title}", ""]
    if note:
        lines.extend([note, ""])
    human_headers = " | Human yes/partial/no | Human strict yes [95% CI]" if human_target else ""
    lines.append(
        "| Group | n" + human_headers + " | " + " | ".join(f"`{name}`" for name in columns) + " |"
    )
    lines.append("|---|---:|" + ("---:|---:|" if human_target else "") + "---:|" * len(columns))
    for name, result_value in groups.items():
        result = _mapping(result_value)
        means = _mapping(result.get("metric_means", {}))
        cells = [_mean_cell(_mapping(means.get(column, {}))) for column in columns]
        human_cells = ""
        if human_target:
            label_summary = _mapping(
                _mapping(result.get("human_label_summary", {})).get(human_target, {})
            )
            counts = _mapping(label_summary.get("counts", {}))
            strict = _mapping(label_summary.get("strict_yes_rate", {}))
            human_cells = (
                f" | {counts.get('yes', 0)}/{counts.get('partial', 0)}/{counts.get('no', 0)}"
                f" | {_mean_cell(strict)}"
            )
        lines.append(
            f"| {name} | {result.get('units', 0)}{human_cells} | " + " | ".join(cells) + " |"
        )
    return lines


def _paired_section(pairs: Mapping[str, object]) -> list[str]:
    metrics = _available_pair_metrics(pairs)
    lines = [
        "",
        "## Complete-Run Paired System Contrasts",
        "",
        "Each cell is mean(left minus right) with a paired qid-bootstrap 95% interval.",
        "",
        "| Pair | " + " | ".join(f"`{metric}`" for metric in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for pair, values in pairs.items():
        mapped = _mapping(values)
        cells = []
        for metric in metrics:
            summary = _mapping(mapped.get(metric, {}))
            cells.append(_estimate_interval(summary.get("mean_difference"), summary.get("ci95")))
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")
    return lines


def _dataset_paired_sections(datasets: Mapping[str, object]) -> list[str]:
    lines = ["", "## Complete-Run Paired Contrasts by Dataset", ""]
    for dataset, pairs in datasets.items():
        lines.extend([f"### {dataset}", ""])
        lines.extend(_paired_table(_mapping(pairs)))
        lines.append("")
    return lines


def _paired_table(pairs: Mapping[str, object]) -> list[str]:
    metrics = _available_pair_metrics(pairs)
    lines = [
        "| Pair | " + " | ".join(f"`{metric}`" for metric in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for pair, values in pairs.items():
        mapped = _mapping(values)
        cells = [
            _estimate_interval(
                _mapping(mapped.get(metric, {})).get("mean_difference"),
                _mapping(mapped.get(metric, {})).get("ci95"),
            )
            for metric in metrics
        ]
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")
    return lines


def _available_pair_metrics(pairs: Mapping[str, object]) -> tuple[str, ...]:
    available = {metric for values in pairs.values() for metric in _mapping(values)}
    return tuple(metric for metric in _PAIRED_SCORE_COLUMNS if metric in available)


def _sensitivity_section(analysis: Mapping[str, object]) -> list[str]:
    variants = (
        ("Primary gold", _mapping(analysis.get("human_gold_all", {}))),
        (
            "Exclude medium-confidence adjudications",
            _mapping(analysis.get("human_gold_without_medium_confidence_adjudication", {})),
        ),
        (
            "Annotator-agreement fields only",
            _mapping(analysis.get("human_gold_agreement_only", {})),
        ),
    )
    candidate_metrics = (
        "token_f1",
        "bertscore_f1",
        "sas_cross_encoder",
        "pedants_probability",
        "g_eval_answer_correctness",
        "ragchecker_f1",
    )
    primary_answer = _mapping(
        _mapping(_mapping(analysis.get("human_gold_all", {})).get("human_correlations", {})).get(
            "answer_correct",
            {},
        )
    )
    metrics = tuple(metric for metric in candidate_metrics if metric in primary_answer)
    lines = [
        "",
        "## Gold-Provenance Sensitivity",
        "",
        "Answer-correctness Spearman correlations under the preregistered provenance exclusions. "
        "Filtering is field-level: a unit can remain for one target while an excluded field is "
        "absent for another.",
        "",
        "| Gold variant | " + " | ".join(f"`{metric}`" for metric in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for name, population in variants:
        answer = _mapping(
            _mapping(population.get("human_correlations", {})).get("answer_correct", {})
        )
        cells = [
            _estimate_interval(
                _mapping(answer.get(metric, {})).get("spearman"),
                _mapping(answer.get(metric, {})).get("spearman_ci95"),
            )
            for metric in metrics
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _stability_section(stability: Mapping[str, object]) -> list[str]:
    lines = ["", "## LLM-Judge Repeat Stability", ""]
    if stability.get("status") != "complete":
        lines.append("The repeat-stability check was skipped.")
        return lines
    lines.append(
        f"A deterministic hash sample of **{stability.get('sample_size', 0)}** inputs was scored "
        "again with a separate cache."
    )
    lines.extend(
        [
            "",
            "| Diagnostic | Value |",
            "|---|---:|",
            f"| Exact structured-result agreement | "
            f"{_number(stability.get('exact_structured_result_rate'))} |",
        ]
    )
    differences = _mapping(stability.get("mean_absolute_score_difference", {}))
    for name, value in differences.items():
        lines.append(f"| Mean absolute difference: `{name}` | {_number(value)} |")
    return lines


def _mean_cell(summary: Mapping[str, object]) -> str:
    return _estimate_interval(summary.get("mean"), summary.get("ci95"))


def _estimate_interval(estimate: object, interval: object) -> str:
    if not isinstance(estimate, (int, float)):
        return "NA"
    if not isinstance(interval, list) or len(interval) != 2:
        return f"{estimate:.3f}"
    return f"{estimate:.3f} [{float(interval[0]):.3f}, {float(interval[1]):.3f}]"


def _number(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "NA"


def _binary_coverage(summary: Mapping[str, object]) -> str:
    count = summary.get("binary_n")
    prevalence = summary.get("binary_positive_rate")
    if not isinstance(count, int) or not isinstance(prevalence, (int, float)):
        return "NA"
    return f"{count} ({prevalence:.1%})"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
