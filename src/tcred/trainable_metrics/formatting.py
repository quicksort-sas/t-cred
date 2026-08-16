from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

from tcred.trainable_metrics.schema import (
    EvidencePassage,
    GraphPathText,
    SemanticRecord,
    SemanticTask,
)

TASK_TOKENS = {
    SemanticTask.ANSWER: "<tcred_task_answer>",
    SemanticTask.SUPPORT: "<tcred_task_support>",
    SemanticTask.RELEVANCE: "<tcred_task_relevance>",
    SemanticTask.TEMPORAL: "<tcred_task_temporal>",
    SemanticTask.ANSWERABILITY: "<tcred_task_answerability>",
    SemanticTask.CITATION: "<tcred_task_citation>",
}
FIELD_TOKENS = {
    "question": "<tcred_question>",
    "time": "<tcred_time>",
    "operator": "<tcred_operator>",
    "reference": "<tcred_reference>",
    "candidate": "<tcred_candidate>",
    "evidence": "<tcred_evidence>",
    "citation": "<tcred_citation>",
    "path": "<tcred_path>",
}
SPECIAL_TOKENS = tuple([*TASK_TOKENS.values(), *FIELD_TOKENS.values()])


def format_semantic_record(record: SemanticRecord) -> str:
    """Render only model-visible semantic fields in a stable, source-blind order."""

    return format_semantic_fields(
        task=SemanticTask(record.task),
        question=record.question,
        query_time_or_interval=record.query_time_or_interval,
        temporal_operator=record.temporal_operator,
        reference_answers=record.reference_answers,
        candidate_or_claim=record.candidate_or_claim,
        evidence_passages=record.evidence_passages,
        citations=record.citations,
        graph_paths=record.graph_paths,
    )


def format_semantic_fields(
    *,
    task: SemanticTask | str,
    question: str | None,
    query_time_or_interval: str | None,
    temporal_operator: str | None,
    reference_answers: Sequence[str],
    candidate_or_claim: str,
    evidence_passages: Sequence[EvidencePassage],
    citations: Sequence[str],
    graph_paths: Sequence[GraphPathText],
) -> str:
    """Render the exact training-time text contract for supervised or inference inputs."""

    task = SemanticTask(task)
    parts = [TASK_TOKENS[task]]
    _append(parts, "question", question)
    _append(parts, "time", query_time_or_interval)
    _append(parts, "operator", temporal_operator)
    for reference in reference_answers:
        _append(parts, "reference", reference)
    _append(parts, "candidate", candidate_or_claim)
    cited_positions = _citation_positions(citations, evidence_passages)
    if cited_positions:
        _append(parts, "citation", ", ".join(str(position) for position in cited_positions))
    for path in graph_paths:
        _append(parts, "path", path.text)
    for index, evidence in enumerate(evidence_passages, start=1):
        _append(parts, "evidence", f"[{index}] {evidence.text}")
    return " ".join(parts)


def formatted_text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def add_special_tokens(tokenizer: Any) -> int:
    return int(tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)}))


def _append(parts: list[str], field: str, value: str | None) -> None:
    if value and value.strip():
        parts.extend((FIELD_TOKENS[field], " ".join(value.split())))


def _citation_positions(
    citations: Sequence[str],
    evidence_passages: Sequence[EvidencePassage],
) -> list[int | str]:
    by_id = {
        evidence.evidence_id: index
        for index, evidence in enumerate(evidence_passages, start=1)
    }
    return [by_id.get(citation, f"unresolved:{citation}") for citation in citations]


def assert_no_prohibited_metadata(texts: Iterable[str]) -> None:
    prohibited = ("source_dataset", "source_group_id", "label_provenance", "partition=")
    for text in texts:
        lowered = text.casefold()
        hit = next((token for token in prohibited if token in lowered), None)
        if hit:
            raise ValueError(f"Formatted model text exposes prohibited metadata token: {hit}")
