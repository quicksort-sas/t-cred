from __future__ import annotations

import re
from collections.abc import Mapping

from tcred.human_eval.presentation import displayed_evidence
from tcred.human_eval.response import response_decision_kind

JUDGMENT_WEIGHT = 4
EVIDENCE_WEIGHT = 1
PATH_WEIGHT = 2
EDGE_WEIGHT = 1

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_ORDERING_OPERATOR = re.compile(
    r"\b(first|last|latest|previous|next|before|after|most recently|immediately before)\b",
    flags=re.IGNORECASE,
)
_INTERVAL_OPERATOR = re.compile(
    r"\b(during|between|throughout|overlap|interval|from .+ (?:to|through))\b",
    flags=re.IGNORECASE,
)
_LIST_CUE = re.compile(
    r"\b(which|what|name|list)\s+(?:two|three|four|all|both)\b|\bco-directed\b",
    flags=re.IGNORECASE,
)
_MIXED_CUE = re.compile(
    r"\b(but|however|although|yet|on the other hand|alternatively)\b",
    flags=re.IGNORECASE,
)


def annotation_complexity(row: Mapping[str, object]) -> int:
    return sum(annotation_complexity_breakdown(row).values())


def annotation_complexity_breakdown(row: Mapping[str, object]) -> dict[str, int]:
    applicable = row.get("applicable_fields")
    judgment_count = len(applicable) if isinstance(applicable, list) else 0
    evidence = displayed_evidence(row)
    raw_paths = row.get("graph_paths")
    paths = raw_paths if isinstance(raw_paths, list) else []
    edge_count = sum(
        len(path.get("edges", []))
        for path in paths
        if isinstance(path, dict) and isinstance(path.get("edges"), list)
    )
    question = str(row.get("question", ""))
    answer = str(row.get("answer_text", ""))
    text_words = _visible_word_count(question=question, answer=answer, evidence=evidence)
    return {
        "judgments": JUDGMENT_WEIGHT * judgment_count,
        "evidence": EVIDENCE_WEIGHT * len(evidence),
        "paths": PATH_WEIGHT * len(paths),
        "edges": EDGE_WEIGHT * edge_count,
        "reading_load": min(4, text_words // 120),
        "temporal_operator": _temporal_operator_load(question),
        "list_answer": _list_load(question=question, answer=answer),
        "revision_conflict": _revision_conflict_load(question=question, evidence=evidence),
        "mixed_response": _mixed_response_load(answer),
    }


def complexity_manifest() -> dict[str, object]:
    return {
        "score_formula": (
            "4 * applicable_judgments + displayed_evidence + 2 * graph_paths + "
            "graph_path_edges + bounded_reading_load + bounded_temporal_operator + "
            "bounded_list_answer + bounded_revision_conflict + bounded_mixed_response"
        ),
        "secondary_term_bounds": {
            "reading_load": [0, 4],
            "temporal_operator": [0, 2],
            "list_answer": [0, 2],
            "revision_conflict": [0, 2],
            "mixed_response": [0, 2],
        },
        "calibration_status": (
            "Conservative bounded heuristic. The usability pilot supplied aggregate session time "
            "but no per-card timing trace, so fitting coefficients would be unsupported. Validate "
            "the ordering against consented per-card or aggregate timing evidence before changing "
            "weights."
        ),
    }


def _visible_word_count(
    *,
    question: str,
    answer: str,
    evidence: list[dict[str, object]],
) -> int:
    text = " ".join(
        [
            question,
            answer,
            *(str(item.get("text", "")) for item in evidence),
        ]
    )
    return len(_TOKEN.findall(text))


def _temporal_operator_load(question: str) -> int:
    if _ORDERING_OPERATOR.search(question):
        return 2
    if _INTERVAL_OPERATOR.search(question):
        return 1
    return 0


def _list_load(*, question: str, answer: str) -> int:
    if _LIST_CUE.search(question):
        return 2
    if "," in answer or re.search(r"\band\b", answer, flags=re.IGNORECASE):
        return 1
    return 0


def _revision_conflict_load(
    *,
    question: str,
    evidence: list[dict[str, object]],
) -> int:
    if "document revision" not in question.casefold() or len(evidence) < 2:
        return 0
    publication_times = {
        str(item.get("publication_time"))
        for item in evidence
        if item.get("publication_time") is not None
    }
    distinct_text = {" ".join(str(item.get("text", "")).casefold().split()) for item in evidence}
    return 2 if len(publication_times) > 1 and len(distinct_text) > 1 else 0


def _mixed_response_load(answer: str) -> int:
    if response_decision_kind(answer) == "hybrid":
        return 2
    return 1 if _MIXED_CUE.search(answer) else 0
