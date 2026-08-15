from __future__ import annotations

from collections.abc import Mapping


def render_diagnostic_report(analysis: dict[str, object]) -> str:
    suite = _mapping(analysis.get("suite_audit"))
    lines = [
        "# Controlled Diagnostic Meta-Evaluation Results",
        "",
        "> This report evaluates metric behavior on formal contrastive and invariance tests. "
        "It complements, but does not replace, concurrent validity against human judgments.",
        "",
        "## Run Summary",
        "",
        f"- Diagnostic cases: **{suite.get('case_count', 0)}**",
        f"- Behavioral pairs: **{suite.get('pair_count', 0)}**",
        f"- Question IDs: **{suite.get('question_clusters', 0)}**",
        f"- Source scenarios: **{suite.get('source_scenarios', 0)}**",
        f"- Dependency-preserving inference clusters: **{suite.get('inference_clusters', 0)}**",
        f"- Bootstrap replicates: **{analysis.get('bootstrap_samples', 0)}**",
        f"- Computed score columns: **{analysis.get('metric_count', 0)}**",
        "",
        "The primary directional statistic is an equal-phenomenon macro-average of paired "
        "utility (win = 1, tie = 0.5, reversal = 0). Strict consistency follows BUMP and "
        "requires a strict score decrease after the targeted error. Invariance is normalized "
        "absolute score change; lower is better.",
        "",
        "## Formal-Oracle Audit Against Human Gold",
        "",
    ]
    oracle = _mapping(analysis.get("formal_oracle_human_audit"))
    lines.extend(
        [
            f"- Controlled human-rated units: **{oracle.get('controlled_units', 0)}**",
            f"- Comparable field judgments: **{oracle.get('field_comparisons', 0)}**",
            f"- Exact formal-oracle/human agreement: **{_number(oracle.get('exact_agreement'))}** "
            f"(95% Wilson CI {_interval(oracle.get('wilson_ci95'))})",
            "",
            "| Field | n | Exact agreement | 95% Wilson CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for field, value in _mapping(oracle.get("by_field")).items():
        row = _mapping(value)
        lines.append(
            f"| `{field}` | {row.get('n', 0)} | {_number(row.get('exact_agreement'))} | "
            f"{_interval(row.get('wilson_ci95'))} |"
        )

    constructs = _mapping(analysis.get("constructs"))
    for construct, value in constructs.items():
        result = _mapping(value)
        lines.extend(
            [
                "",
                f"## {_title(construct)}",
                "",
                f"Directional pairs: **{result.get('directional_pair_count', 0)}**; "
                f"invariance pairs: **{result.get('invariance_pair_count', 0)}**.",
                "",
            ]
        )
        directional = _mapping(result.get("directional_metrics"))
        directional_rows = [
            (name, _mapping(summary))
            for name, summary in directional.items()
            if _mapping(summary).get("n", 0)
        ]
        if directional_rows:
            lines.extend(
                [
                    "### Directional Discrimination",
                    "",
                    "| Metric | n | Coverage | Strict | Pair acc. | Macro pair acc. | "
                    "95% CI | ROC AUC | Ties |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            directional_rows.sort(
                key=lambda item: float(item[1].get("macro_phenomenon_pairwise_accuracy") or -1),
                reverse=True,
            )
            for name, row in directional_rows:
                lines.append(
                    f"| `{name}` | {row.get('n', 0)} | {_number(row.get('coverage'))} | "
                    f"{_number(row.get('strict_consistency'))} | "
                    f"{_number(row.get('tie_adjusted_pairwise_accuracy'))} | "
                    f"**{_number(row.get('macro_phenomenon_pairwise_accuracy'))}** | "
                    f"{_interval(row.get('macro_phenomenon_ci95'))} | "
                    f"{_number(row.get('roc_auc'))} | {_number(row.get('tie_rate'))} |"
                )
        invariance = _mapping(result.get("invariance_metrics"))
        invariance_rows = [
            (name, _mapping(summary))
            for name, summary in invariance.items()
            if _mapping(summary).get("n", 0)
        ]
        if invariance_rows:
            lines.extend(
                [
                    "",
                    "### Invariance Robustness",
                    "",
                    "| Metric | n | Coverage | Mean change | Macro change | 95% CI | "
                    "Exact | Within 5% |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            invariance_rows.sort(
                key=lambda item: float(item[1].get("macro_phenomenon_absolute_change") or 1)
            )
            for name, row in invariance_rows:
                lines.append(
                    f"| `{name}` | {row.get('n', 0)} | {_number(row.get('coverage'))} | "
                    f"{_number(row.get('mean_normalized_absolute_change'))} | "
                    f"**{_number(row.get('macro_phenomenon_absolute_change'))}** | "
                    f"{_interval(row.get('macro_phenomenon_ci95'))} | "
                    f"{_number(row.get('exact_invariance_rate'))} | "
                    f"{_number(row.get('within_five_percent_range_rate'))} |"
                )
        lines.extend(_comparison_section(result))
        lines.extend(_metric_phenomenon_sections(result))
        lines.extend(_phenomenon_section(result))

    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "1. A formal challenge set establishes behavior on specified constructs; it cannot by "
            "itself establish real-world concurrent validity or replace human judgments.",
            "2. The oracle is strongest where the generator changes one formally checked "
            "component. Results are therefore reported per phenomenon and construct, not as one "
            "universal score.",
            "3. A metric with high pair accuracy but poor invariance is sensitive but unreliable; "
            "a stable metric with chance-level discrimination is reliable but invalid for the "
            "target.",
            "4. Missing scores count against coverage and are never silently imputed.",
            "5. Pairwise significance uses a paired source-scenario-cluster permutation test "
            "on common cases with Holm correction. Non-significant rank differences are "
            "treated as unresolved, not as evidence of equivalence.",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_section(result: dict[str, object]) -> list[str]:
    rows = [
        _mapping(row)
        for row in result.get("directional_pairwise_comparisons", [])
        if isinstance(row, dict) and row.get("significant_at_0_05")
    ]
    if not rows:
        return []
    rows.sort(key=lambda row: abs(float(row["macro_utility_difference"])), reverse=True)
    output = [
        "",
        "### Holm-Significant Directional Contrasts",
        "",
        "| Metric A | Metric B | n | A - B | Bootstrap 95% CI | Holm permutation p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows[:20]:
        output.append(
            f"| `{row['left']}` | `{row['right']}` | {row['n_common_pairs']} | "
            f"{_number(row['macro_utility_difference'])} | {_interval(row.get('ci95'))} | "
            f"{_number(row.get('holm_adjusted_p_value'))} |"
        )
    return output


def _phenomenon_section(result: dict[str, object]) -> list[str]:
    tiers = [_mapping(row) for row in result.get("evidence_tiers", []) if isinstance(row, dict)]
    if not tiers:
        return []
    output = [
        "",
        "### Evidence Profile",
        "",
        "The ordering below prioritizes Holm-significant pairwise wins, then losses and observed "
        "directional performance. It is not an unqualified global leaderboard.",
        "",
        "| Metric | Sig. wins | Sig. losses | Directional macro | Invariance change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tiers:
        output.append(
            f"| `{row['metric']}` | {row['significant_pairwise_wins']} | "
            f"{row['significant_pairwise_losses']} | {_number(row.get('directional_macro'))} | "
            f"{_number(row.get('invariance_macro_change'))} |"
        )
    return output


def _metric_phenomenon_sections(result: dict[str, object]) -> list[str]:
    """Render the construct-level behavior that aggregate ranks would otherwise hide."""

    output: list[str] = []
    directional_rows: list[tuple[str, str, dict[str, object]]] = []
    for metric, summary in _mapping(result.get("directional_metrics")).items():
        for phenomenon, row in _mapping(_mapping(summary).get("per_phenomenon")).items():
            directional_rows.append((str(phenomenon), str(metric), _mapping(row)))
    if directional_rows:
        output.extend(
            [
                "",
                "### Directional Results by Phenomenon",
                "",
                "| Phenomenon | Metric | n | Strict | Pair acc. | ROC AUC | Ties |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        directional_rows.sort(
            key=lambda item: (
                item[0],
                -float(item[2].get("tie_adjusted_pairwise_accuracy") or -1),
                item[1],
            )
        )
        for phenomenon, metric, row in directional_rows:
            output.append(
                f"| `{phenomenon}` | `{metric}` | {row.get('n', 0)} | "
                f"{_number(row.get('strict_consistency'))} | "
                f"{_number(row.get('tie_adjusted_pairwise_accuracy'))} | "
                f"{_number(row.get('roc_auc'))} | {_number(row.get('tie_rate'))} |"
            )

    invariance_rows: list[tuple[str, str, dict[str, object]]] = []
    for metric, summary in _mapping(result.get("invariance_metrics")).items():
        for phenomenon, row in _mapping(_mapping(summary).get("per_phenomenon")).items():
            invariance_rows.append((str(phenomenon), str(metric), _mapping(row)))
    if invariance_rows:
        output.extend(
            [
                "",
                "### Invariance Results by Phenomenon",
                "",
                "| Phenomenon | Metric | n | Normalized change | Exact invariance |",
                "|---|---|---:|---:|---:|",
            ]
        )
        invariance_rows.sort(
            key=lambda item: (
                item[0],
                float(item[2].get("mean_normalized_absolute_change") or 0),
                item[1],
            )
        )
        for phenomenon, metric, row in invariance_rows:
            output.append(
                f"| `{phenomenon}` | `{metric}` | {row.get('n', 0)} | "
                f"{_number(row.get('mean_normalized_absolute_change'))} | "
                f"{_number(row.get('exact_invariance_rate'))} |"
            )
    return output


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "n/a"
    return f"[{float(value[0]):.3f}, {float(value[1]):.3f}]"


def _title(value: str) -> str:
    return value.replace("_", " ").title()
