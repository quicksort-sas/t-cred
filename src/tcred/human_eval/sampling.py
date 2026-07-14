from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

from tcred.dataset.io import read_jsonl
from tcred.dataset.models import AnswerVariant, AnswerVariantType, DatasetBundle, Question
from tcred.dataset.text import annotation_plain_text
from tcred.qa.models import ALL_QA_SYSTEMS, QASystemName, SystemOutput

CONTROLLED_VARIANT_WEIGHTS: dict[str, float] = {
    AnswerVariantType.CORRECT_SUPPORTED: 0.20,
    AnswerVariantType.CORRECT_INVALID_EVIDENCE: 0.15,
    AnswerVariantType.STALE_ANSWER: 0.15,
    AnswerVariantType.OUTDATED_SOURCE_ANSWER: 0.15,
    AnswerVariantType.FUTURE_INVALID_ANSWER: 0.10,
    AnswerVariantType.WRONG_OPERATOR_ANSWER: 0.10,
    AnswerVariantType.INVALID_GRAPH_PATH_ANSWER: 0.10,
    AnswerVariantType.PARTIAL_ANSWER: 0.05,
    AnswerVariantType.HALLUCINATED_ANSWER: 0.05,
    AnswerVariantType.OVERCONFIDENT_SHOULD_REFUSE: 0.05,
    AnswerVariantType.CORRECT_REFUSAL: 0.05,
    AnswerVariantType.INAPPROPRIATE_REFUSAL: 0.05,
}


def select_controlled_answers(
    bundle: DatasetBundle,
    *,
    target: int,
    excluded_qids: set[str] | None = None,
) -> list[AnswerVariant]:
    if target < 1:
        return []
    question_by_id = {question.qid: question for question in bundle.questions}
    excluded = set(excluded_qids or ())
    held_out_scenarios = set(bundle.splits.get("test_auto", []))
    excluded_series = {_series_id(question_by_id[qid]) for qid in excluded if qid in question_by_id}
    eligible_qids = _one_qid_per_series(
        [
            question
            for question in bundle.questions
            if question.scenario_id in held_out_scenarios
            and _series_id(question) not in excluded_series
        ]
    )
    available_types = {
        str(answer.variant_type) for answer in bundle.answer_variants if answer.qid in eligible_qids
    }
    weights = {
        variant_type: weight
        for variant_type, weight in CONTROLLED_VARIANT_WEIGHTS.items()
        if str(variant_type) in available_types
    }
    quotas = proportional_quotas(weights, target)
    by_type: dict[str, list[AnswerVariant]] = {}
    for variant_type in sorted(available_types):
        candidates = [
            answer
            for answer in bundle.answer_variants
            if str(answer.variant_type) == variant_type and answer.qid in eligible_qids
        ]
        by_type[variant_type] = _diverse_order(
            candidates,
            question_by_id=question_by_id,
            bundle=bundle,
            item_id=lambda answer: answer.answer_id,
            question_id=lambda answer: answer.qid,
        )

    selected: list[AnswerVariant] = []
    selected_ids: set[str] = set()
    selected_content_keys: set[tuple[object, ...]] = set()
    use_counts: defaultdict[str, int] = defaultdict(int)
    positions = defaultdict(int)
    while len(selected) < target and any(quotas.values()):
        progressed = False
        for variant_type in sorted(quotas):
            if quotas[variant_type] <= 0:
                continue
            candidate = None
            while candidate is None:
                next_candidate, positions[variant_type] = _next_available(
                    by_type.get(variant_type, []),
                    positions[variant_type],
                    use_counts,
                    lambda answer: answer.qid,
                    max_per_question=2,
                )
                if next_candidate is None:
                    break
                if _controlled_content_key(next_candidate, bundle) in selected_content_keys:
                    continue
                candidate = next_candidate
            if candidate is None:
                quotas[variant_type] = 0
                continue
            selected.append(candidate)
            selected_ids.add(candidate.answer_id)
            selected_content_keys.add(_controlled_content_key(candidate, bundle))
            use_counts[candidate.qid] += 1
            quotas[variant_type] -= 1
            progressed = True
        if not progressed:
            break

    if len(selected) < target:
        fallback = _diverse_order(
            [answer for answer in bundle.answer_variants if answer.qid in eligible_qids],
            question_by_id=question_by_id,
            bundle=bundle,
            item_id=lambda answer: answer.answer_id,
            question_id=lambda answer: answer.qid,
        )
        for max_per_question in (2, 3):
            for answer in fallback:
                if (
                    answer.answer_id in selected_ids
                    or _controlled_content_key(answer, bundle) in selected_content_keys
                    or use_counts[answer.qid] >= max_per_question
                ):
                    continue
                selected.append(answer)
                selected_ids.add(answer.answer_id)
                selected_content_keys.add(_controlled_content_key(answer, bundle))
                use_counts[answer.qid] += 1
                if len(selected) == target:
                    break
            if len(selected) == target:
                break
    if len(selected) != target:
        raise ValueError(f"Could select only {len(selected)} of {target} controlled units")
    if len(selected_ids) != len(selected):
        raise RuntimeError("Controlled-answer sampling produced duplicate answer ids")
    return selected


def select_system_outputs(
    *,
    bundle: DatasetBundle,
    family: str,
    system_output_root: Path,
    target: int,
    excluded_qids: set[str] | None = None,
    allow_excluded_fallback: bool = False,
) -> list[SystemOutput]:
    if target < 1:
        return []
    systems = [QASystemName(value) for value in ALL_QA_SYSTEMS]
    if target % len(systems):
        raise ValueError(
            f"Paired system sample target must be divisible by {len(systems)}: {target}"
        )
    question_by_id = {question.qid: question for question in bundle.questions}
    held_out_scenarios = set(bundle.splits.get("test_auto", []))
    excluded = set(excluded_qids or ())
    excluded_series = {_series_id(question_by_id[qid]) for qid in excluded if qid in question_by_id}
    by_system: dict[QASystemName, dict[str, SystemOutput]] = {}
    for system in systems:
        path = system_output_root / family / f"{system}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"System output file is missing: {path}")
        outputs = [SystemOutput.model_validate(row) for row in read_jsonl(path)]
        by_system[system] = {
            output.qid: output
            for output in outputs
            if output.status == "success"
            and output.qid in question_by_id
            and question_by_id[output.qid].scenario_id in held_out_scenarios
        }

    common_qids = set.intersection(*(set(outputs) for outputs in by_system.values()))
    common_qids = {
        qid
        for qid in common_qids
        if len({_system_output_content_key(by_system[system][qid]) for system in systems})
        == len(systems)
    }
    needed_questions = target // len(systems)
    preferred_qids = _one_qid_per_series(
        [
            question_by_id[qid]
            for qid in common_qids - excluded
            if _series_id(question_by_id[qid]) not in excluded_series
        ]
    )
    fallback_qids = (
        _one_qid_per_series([question_by_id[qid] for qid in common_qids & excluded])
        if allow_excluded_fallback
        else set()
    )
    preferred_questions = _diverse_order(
        [question_by_id[qid] for qid in preferred_qids],
        question_by_id=question_by_id,
        bundle=bundle,
        item_id=lambda question: question.qid,
        question_id=lambda question: question.qid,
    )
    fallback_questions = _diverse_order(
        [question_by_id[qid] for qid in fallback_qids],
        question_by_id=question_by_id,
        bundle=bundle,
        item_id=lambda question: question.qid,
        question_id=lambda question: question.qid,
    )
    ordered_questions = [*preferred_questions, *fallback_questions]
    selected_questions = ordered_questions[:needed_questions]
    if len(selected_questions) != needed_questions:
        raise ValueError(
            f"Could select only {len(selected_questions)} of {needed_questions} paired "
            f"questions for {family}"
        )
    return [
        by_system[system][question.qid] for question in selected_questions for system in systems
    ]


def proportional_quotas(weights: dict[str, float], target: int) -> dict[str, int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    if not weights:
        return {}
    total_weight = sum(weights.values())
    exact = {key: target * weight / total_weight for key, weight in weights.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def equal_quotas(keys: list[str], target: int) -> dict[str, int]:
    return proportional_quotas(dict.fromkeys(keys, 1.0), target)


def _diverse_order[ItemT](
    items: list[ItemT],
    *,
    question_by_id: dict[str, Question],
    bundle: DatasetBundle,
    item_id: object,
    question_id: object,
) -> list[ItemT]:
    grouped: defaultdict[tuple[int, str, str, str], list[ItemT]] = defaultdict(list)
    for item in items:
        qid = question_id(item)
        question = question_by_id[qid]
        key = (
            _question_priority(question, bundle),
            str(question.temporal_operator),
            str(question.system_difficulty),
            str(question.eval_difficulty),
        )
        grouped[key].append(item)
    for key, values in grouped.items():
        grouped[key] = sorted(values, key=lambda item: _stable_key(item_id(item)))
    ordered: list[ItemT] = []
    keys = sorted(grouped)
    while any(grouped.values()):
        minimum_priority = min(key[0] for key, values in grouped.items() if values)
        for key in keys:
            if key[0] == minimum_priority and grouped[key]:
                ordered.append(grouped[key].pop(0))
    return ordered


def _question_priority(question: Question, bundle: DatasetBundle) -> int:
    human = set(bundle.splits.get("human_pool", []))
    test = set(bundle.splits.get("test_auto", []))
    if question.scenario_id in human and question.human_pool_candidate:
        return 0
    if question.scenario_id in test and question.human_pool_candidate:
        return 1
    if question.scenario_id in test:
        return 2
    if question.human_pool_candidate:
        return 3
    return 4


def _series_id(question: Question) -> str:
    return question.semantic_series_id or question.qid


def _one_qid_per_series(questions: list[Question]) -> set[str]:
    selected: dict[str, Question] = {}
    for question in sorted(questions, key=lambda item: _stable_key(item.qid)):
        selected.setdefault(_series_id(question), question)
    return {question.qid for question in selected.values()}


def _next_available[ItemT](
    items: list[ItemT],
    position: int,
    use_counts: dict[str, int],
    question_id: object,
    *,
    max_per_question: int,
) -> tuple[ItemT | None, int]:
    while position < len(items):
        item = items[position]
        position += 1
        if use_counts.get(question_id(item), 0) < max_per_question:
            return item, position
    return None, position


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"human-eval-v2:{value}".encode()).hexdigest()


def _controlled_content_key(
    answer: AnswerVariant,
    bundle: DatasetBundle,
) -> tuple[object, ...]:
    path_by_id = {path.pid: path for path in bundle.graph_paths}
    path_signatures = []
    for path_id in answer.graph_path_ids:
        path = path_by_id.get(path_id)
        if path is None:
            path_signatures.append(("missing", path_id))
            continue
        path_signatures.append(
            (
                "path",
                tuple(path.nodes),
                tuple((edge.fact_id, edge.traversal_direction) for edge in path.edges),
            )
        )
    return (
        answer.qid,
        annotation_plain_text(answer.answer_text),
        tuple(answer.cited_evidence_ids),
        tuple(sorted(path_signatures)),
    )


def _system_output_content_key(output: SystemOutput) -> tuple[object, ...]:
    """Return the evaluator-relevant identity of one system response.

    Non-cited retrieval rows are intentionally excluded: the annotation labels concern the
    response, its citations, and any submitted graph path. Repeating an otherwise identical
    response merely because its unused retrieval tail differs would waste annotation effort.
    """

    path_signatures = sorted(
        (
            tuple(path.fact_ids),
            tuple(path.traversal_directions),
            tuple(path.node_ids),
        )
        for path in output.retrieval.graph_paths
    )
    return (
        annotation_plain_text(output.answer_text),
        tuple(output.resolved_cited_evidence_ids),
        tuple(output.unresolved_citation_ids),
        tuple(path_signatures),
        bool(output.retrieval.hits),
    )
