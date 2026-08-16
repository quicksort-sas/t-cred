from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

LABEL_TARGET = {"yes": 1.0, "partial": 0.5, "no": 0.0}
HUMAN_FIELDS = {
    "answer_correctness": (
        "answer_correct",
        "tcred_sl_answer_equivalence_semantic",
        "tcred_answer_equivalence",
    ),
    "evidence_support": (
        "evidence_supports_answer",
        "tcred_sl_evidence_support",
        "tcred_semantic_attribution",
    ),
    "temporal_correctness": (
        "temporal_correct",
        "tcred_sl_temporal_correctness",
        "tcred_temporal_correctness",
    ),
    "temporal_attribution": (
        "temporal_correct",
        "tcred_sl_temporal_attribution",
        "tcred_temporal_attribution",
    ),
    "citation_quality": (
        "citation_temporally_valid",
        "tcred_sl_citation_quality",
        "tcred_citation_quality",
    ),
    "graph_sufficiency": (
        "graph_evidence_sufficient",
        "tcred_sl_graph_sufficiency",
        "tcred_graph_answer_coverage",
    ),
    "response_decision": (
        "response_decision_appropriate",
        "tcred_sl_response_decision",
        "tcred_response_decision",
    ),
}
FORMAL_METRICS = {
    "answer_correctness": (
        "tcred_sl_answer_equivalence_semantic",
        "tcred_answer_equivalence",
    ),
    "evidence_support": (
        "tcred_sl_evidence_support",
        "tcred_semantic_attribution",
    ),
    "temporal_correctness": (
        "tcred_sl_temporal_correctness",
        "tcred_temporal_correctness",
    ),
    "temporal_attribution": (
        "tcred_sl_temporal_attribution",
        "tcred_temporal_attribution",
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
EPSILON = 1e-12


def analyze(
    *,
    score_dir: Path,
    population_dir: Path,
    source_disjoint_root: Path,
) -> dict[str, Any]:
    population_scores = _by_id(score_dir / "population_scores_with_tcred_sl.jsonl")
    population_inputs = _by_id(population_dir / "task_judge_inputs.jsonl")
    formal_scores = _by_id(score_dir / "source_disjoint_scores_with_tcred_sl.jsonl")
    cases = _by_id(source_disjoint_root / "challenge" / "diagnostic_cases.jsonl", "case_id")
    pairs = _read_jsonl(source_disjoint_root / "challenge" / "diagnostic_pairs.jsonl")
    return {
        "schema_version": "tcred-sl-posthoc-error-analysis-v1",
        "analysis_status": "retrospective_descriptive_not_metric_tuning",
        "inputs": {
            "population_scores": _file_record(score_dir / "population_scores_with_tcred_sl.jsonl"),
            "population_inputs": _file_record(population_dir / "task_judge_inputs.jsonl"),
            "formal_scores": _file_record(score_dir / "source_disjoint_scores_with_tcred_sl.jsonl"),
            "diagnostic_cases": _file_record(
                source_disjoint_root / "challenge" / "diagnostic_cases.jsonl"
            ),
            "diagnostic_pairs": _file_record(
                source_disjoint_root / "challenge" / "diagnostic_pairs.jsonl"
            ),
        },
        "human_gold": _human_error_analysis(population_scores, population_inputs),
        "formal_diagnostics": _formal_error_analysis(formal_scores, cases, pairs),
        "interpretation_boundary": (
            "This analysis explains frozen results after evaluation. It must not be used to "
            "modify T-CRED-SL and then claim performance on the same cases as confirmatory."
        ),
    }


def _human_error_analysis(
    scores: dict[str, dict[str, Any]],
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for construct, (field, learned_name, baseline_name) in HUMAN_FIELDS.items():
        comparisons = []
        outcomes: Counter[str] = Counter()
        missing: Counter[str] = Counter()
        for metric_id, record in scores.items():
            if record.get("population") != "human_gold":
                continue
            label = record.get("gold_labels", {}).get(field)
            target = LABEL_TARGET.get(str(label))
            learned = record.get("scores", {}).get(learned_name)
            baseline = record.get("scores", {}).get(baseline_name)
            if target is None:
                missing["no_usable_label"] += 1
                continue
            if learned is None or baseline is None:
                missing["missing_score"] += 1
                continue
            learned_error = abs(float(learned) - target)
            baseline_error = abs(float(baseline) - target)
            delta = learned_error - baseline_error
            if delta > EPSILON:
                outcomes["baseline_closer"] += 1
            elif delta < -EPSILON:
                outcomes["learned_closer"] += 1
            else:
                outcomes["equal_absolute_error"] += 1
            source = inputs[metric_id]
            comparisons.append(
                {
                    "metric_id": metric_id,
                    "dataset_family": source.get("dataset_family"),
                    "qid": source.get("qid"),
                    "gold_label": label,
                    "target": target,
                    "learned_score": learned,
                    "baseline_score": baseline,
                    "learned_minus_baseline_absolute_error": delta,
                    "question": source.get("question"),
                    "reference_answer": source.get("reference_answer"),
                    "candidate_answer": source.get("candidate_answer"),
                    "displayed_evidence_count": len(
                        {
                            item.get("evidence_id")
                            for item in [
                                *source.get("cited_evidence", []),
                                *source.get("retrieved_evidence", []),
                            ]
                        }
                    ),
                    "graph_path_count": len(source.get("graph_paths", [])),
                }
            )
        comparisons.sort(
            key=lambda row: float(row["learned_minus_baseline_absolute_error"]),
            reverse=True,
        )
        output[construct] = {
            "n": len(comparisons),
            "outcomes_by_absolute_error": dict(sorted(outcomes.items())),
            "missing": dict(sorted(missing.items())),
            "largest_learned_regressions": comparisons[:5],
            "largest_learned_improvements": list(reversed(comparisons[-5:])),
        }
    return output


def _formal_error_analysis(
    scores: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["target_construct"])].append(pair)
    output = {}
    for construct, construct_pairs in sorted(grouped.items()):
        learned_name, baseline_name = FORMAL_METRICS[construct]
        directional: Counter[str] = Counter()
        directional_by_phenomenon: defaultdict[str, Counter[str]] = defaultdict(Counter)
        invariance_changes: defaultdict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"learned": [], "baseline": []}
        )
        examples = []
        for pair in construct_pairs:
            left = scores[pair["left_case_id"]]["scores"]
            right = scores[pair["right_case_id"]]["scores"]
            learned_values = (left.get(learned_name), right.get(learned_name))
            baseline_values = (left.get(baseline_name), right.get(baseline_name))
            phenomenon = str(pair["phenomenon"])
            if pair["test_type"] == "directional":
                learned_result = _directional_result(*learned_values)
                baseline_result = _directional_result(*baseline_values)
                transition = f"baseline_{baseline_result}__learned_{learned_result}"
                directional[transition] += 1
                directional_by_phenomenon[phenomenon][transition] += 1
                if baseline_result == "success" and learned_result != "success":
                    examples.append(
                        _pair_example(
                            pair,
                            cases,
                            learned_values,
                            baseline_values,
                            learned_result,
                            baseline_result,
                        )
                    )
            else:
                if None not in learned_values:
                    invariance_changes[phenomenon]["learned"].append(
                        abs(float(learned_values[0]) - float(learned_values[1]))
                    )
                if None not in baseline_values:
                    invariance_changes[phenomenon]["baseline"].append(
                        abs(float(baseline_values[0]) - float(baseline_values[1]))
                    )
        output[construct] = {
            "directional_transitions": dict(sorted(directional.items())),
            "directional_transitions_by_phenomenon": {
                key: dict(sorted(value.items()))
                for key, value in sorted(directional_by_phenomenon.items())
            },
            "invariance_mean_absolute_change_by_phenomenon": {
                phenomenon: {
                    name: mean(values) if values else None for name, values in groups.items()
                }
                for phenomenon, groups in sorted(invariance_changes.items())
            },
            "representative_baseline_success_learned_failure": examples[:10],
        }
    return output


def _directional_result(left: Any, right: Any) -> str:
    if left is None or right is None:
        return "missing"
    difference = float(left) - float(right)
    if difference > EPSILON:
        return "success"
    if difference < -EPSILON:
        return "reversal"
    return "tie"


def _pair_example(
    pair: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    learned: tuple[Any, Any],
    baseline: tuple[Any, Any],
    learned_result: str,
    baseline_result: str,
) -> dict[str, Any]:
    left = cases[pair["left_case_id"]]["task_judge_input"]
    right = cases[pair["right_case_id"]]["task_judge_input"]
    return {
        "pair_id": pair["pair_id"],
        "phenomenon": pair["phenomenon"],
        "learned_result": learned_result,
        "baseline_result": baseline_result,
        "learned_scores": list(learned),
        "baseline_scores": list(baseline),
        "left": {
            "question": left["question"],
            "reference_answer": left["reference_answer"],
            "candidate_answer": left["candidate_answer"],
        },
        "right": {
            "question": right["question"],
            "reference_answer": right["reference_answer"],
            "candidate_answer": right["candidate_answer"],
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _by_id(path: Path, field: str = "metric_id") -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    output = {str(row[field]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate {field} values in {path}")
    return output


def _file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=Path("data/metrics/tcred_sl/seed42-2026-08-16"),
    )
    parser.add_argument(
        "--population-dir",
        type=Path,
        default=Path("data/metrics/tcred_suite/population-v1.4"),
    )
    parser.add_argument(
        "--source-disjoint-root",
        type=Path,
        default=Path("data/validation/tcred_v1_4_source_disjoint"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metrics/tcred_sl/seed42-2026-08-16-posthoc/error_analysis.json"),
    )
    args = parser.parse_args()
    result = analyze(
        score_dir=args.score_dir,
        population_dir=args.population_dir,
        source_disjoint_root=args.source_disjoint_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
