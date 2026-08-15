from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

PRIMARY_COMPONENTS = {
    "tcred_answer_equivalence": "answer_correct",
    "tcred_semantic_attribution": "evidence_supports_answer",
    "tcred_temporal_attribution": "evidence_supports_answer",
    "tcred_temporal_correctness": "temporal_correct",
    "tcred_citation_quality": "citation_temporally_valid",
    "tcred_graph_answer_coverage": "graph_evidence_sufficient",
    "tcred_response_decision": "response_decision_appropriate",
}

LABEL_TARGET = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _human_scores(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["metric_id"]: row
        for row in _read_jsonl(path)
        if row.get("population") == "human_gold"
    }


def _components(path: Path) -> dict[str, dict[str, Any]]:
    return {row["metric_id"]: row for row in _read_jsonl(path)}


def _gold_units(path: Path) -> dict[str, dict[str, Any]]:
    return {f"gold:{row['unit_id']}": row for row in _read_jsonl(path)}


def _different(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is not right
    return not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _error(score: float | None, label: str | None) -> float | None:
    target = LABEL_TARGET.get(str(label))
    if score is None or target is None:
        return None
    return abs(float(score) - target)


def _evidence_summary(unit: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for item in [*unit.get("cited_evidence", []), *unit.get("retrieved_evidence", [])]:
        evidence_id = str(item.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        output.append(
            {
                "evidence_id": evidence_id,
                "text": item.get("text"),
                "valid_time": item.get("valid_time"),
                "cited": evidence_id in set(unit.get("cited_evidence_ids", [])),
            }
        )
    return output


def analyze(
    *,
    old_dir: Path,
    new_dir: Path,
    gold_path: Path,
) -> dict[str, Any]:
    old_scores = _human_scores(old_dir / "metric_scores_with_tcred.jsonl")
    new_scores = _human_scores(new_dir / "metric_scores_with_tcred.jsonl")
    old_components = _components(old_dir / "tcred_component_results.jsonl")
    new_components = _components(new_dir / "tcred_component_results.jsonl")
    gold = _gold_units(gold_path)
    common = sorted(set(old_scores) & set(new_scores) & set(gold))
    changes = []
    summary: dict[str, dict[str, Any]] = {}
    for metric, field in PRIMARY_COMPONENTS.items():
        counts: Counter[str] = Counter()
        metric_changes = []
        for metric_id in common:
            old = old_scores[metric_id]["scores"].get(metric)
            new = new_scores[metric_id]["scores"].get(metric)
            if not _different(old, new):
                continue
            row = gold[metric_id]
            label = row["gold_labels"].get(field)
            old_error = _error(old, label)
            new_error = _error(new, label)
            if old is None and new is not None:
                outcome = "coverage_added"
            elif old is not None and new is None:
                outcome = "coverage_removed"
            elif old_error is None or new_error is None:
                outcome = "unscored_against_gold"
            elif new_error < old_error - 1e-12:
                outcome = "closer_to_gold"
            elif new_error > old_error + 1e-12:
                outcome = "farther_from_gold"
            else:
                outcome = "equal_gold_error"
            counts[outcome] += 1
            unit = row["unit"]
            change = {
                "metric_id": metric_id,
                "unit_id": row["unit_id"],
                "component": metric,
                "gold_field": field,
                "gold_label": label,
                "outcome": outcome,
                "old_score": old,
                "new_score": new,
                "old_error": old_error,
                "new_error": new_error,
                "dataset_family": row["metadata"].get("dataset_family"),
                "source_kind": row["metadata"].get("source_kind"),
                "variant_type": row["metadata"].get("variant_type"),
                "question": unit.get("question"),
                "reference_answer": unit.get("reference_answer"),
                "candidate_answer": unit.get("answer_text"),
                "cited_evidence_ids": unit.get("cited_evidence_ids", []),
                "evidence": _evidence_summary(unit),
                "old_candidate_claims": old_components[metric_id].get("candidate_claims", []),
                "new_candidate_claims": new_components[metric_id].get("candidate_claims", []),
                "old_coverage": old_components[metric_id].get("coverage", {}),
                "new_coverage": new_components[metric_id].get("coverage", {}),
                "old_audit": old_components[metric_id].get("audit", {}),
                "new_audit": new_components[metric_id].get("audit", {}),
            }
            metric_changes.append(change)
            changes.append(change)
        summary[metric] = {
            "changed": len(metric_changes),
            "outcomes": dict(sorted(counts.items())),
            "old_coverage": sum(
                old_scores[metric_id]["scores"].get(metric) is not None for metric_id in common
            ),
            "new_coverage": sum(
                new_scores[metric_id]["scores"].get(metric) is not None for metric_id in common
            ),
        }
    return {
        "schema_version": "1.0",
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "gold_path": str(gold_path),
        "common_human_units": len(common),
        "summary": summary,
        "changes": changes,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# T-CRED Version-Delta Audit",
        "",
        f"- Old run: `{analysis['old_dir']}`",
        f"- New run: `{analysis['new_dir']}`",
        f"- Common human-gold units: **{analysis['common_human_units']}**",
        "",
        "## Summary",
        "",
        "| Component | Changed | Old coverage | New coverage | Closer | Farther | "
        "Added | Removed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for component, row in analysis["summary"].items():
        outcomes = row["outcomes"]
        lines.append(
            f"| `{component}` | {row['changed']} | {row['old_coverage']} | "
            f"{row['new_coverage']} | {outcomes.get('closer_to_gold', 0)} | "
            f"{outcomes.get('farther_from_gold', 0)} | "
            f"{outcomes.get('coverage_added', 0)} | "
            f"{outcomes.get('coverage_removed', 0)} |"
        )
    lines.extend(["", "## Changed Human-Gold Cases", ""])
    for change in analysis["changes"]:
        lines.extend(
            [
                f"### {change['component']} / {change['unit_id']}",
                "",
                f"- Outcome: `{change['outcome']}`; gold `{change['gold_field']}` = "
                f"`{change['gold_label']}`; score `{_fmt(change['old_score'])}` -> "
                f"`{_fmt(change['new_score'])}`.",
                f"- Dataset/source: `{change['dataset_family']}` / "
                f"`{change['source_kind']}` / `{change['variant_type']}`.",
                f"- Question: {change['question']}",
                f"- Reference: {change['reference_answer']}",
                f"- Candidate: {change['candidate_answer']}",
                f"- Cited evidence: `{', '.join(change['cited_evidence_ids']) or 'none'}`.",
                "- Old claims: " + json.dumps(change["old_candidate_claims"], ensure_ascii=True),
                "- New claims: " + json.dumps(change["new_candidate_claims"], ensure_ascii=True),
            ]
        )
        for evidence in change["evidence"]:
            lines.append(
                f"- Evidence `{evidence['evidence_id']}` (cited={evidence['cited']}, "
                f"valid={evidence['valid_time']}): {evidence['text']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    return "missing" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit T-CRED changes on fixed human gold")
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--gold-units", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(old_dir=args.old_dir, new_dir=args.new_dir, gold_path=args.gold_units)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
