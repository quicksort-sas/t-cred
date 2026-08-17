from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcred.external_evaluations.sabet_tkgqa.label_bundle import (
        TimeQuestionsLabelResolver,
    )
    from tcred.external_evaluations.sabet_tkgqa.schema import SabetPredictionRecord

_WIKIDATA_ENTITY_ID = re.compile(r"Q\d+")


@dataclass(frozen=True)
class ResolvedDisplayLabel:
    answer_id: str
    text: str
    source: str
    original_reference_index: int | None = None
    wikidata_lastrevid: int | None = None
    wikidata_canonical_qid: str | None = None


@dataclass(frozen=True)
class ResolvedAnswerText:
    candidate: ResolvedDisplayLabel | None
    references: tuple[ResolvedDisplayLabel, ...]
    raw_reference_count: int

    @property
    def unreadable_reference_count(self) -> int:
        return self.raw_reference_count - len(self.references)


def is_readable_answer_label(*, dataset: str, answer_id: str, label: str) -> bool:
    """Return whether a released label is suitable as natural-language metric text."""

    stripped = label.strip()
    if not stripped:
        return False
    if dataset == "MultiTQ":
        return True
    namespace, separator, payload = answer_id.partition(":")
    if separator and namespace != "entity":
        return True
    raw_entity_id = payload if separator else answer_id
    return not (
        _WIKIDATA_ENTITY_ID.fullmatch(raw_entity_id) is not None
        and stripped == raw_entity_id
    )


def readable_gold_answers(
    record: SabetPredictionRecord,
) -> list[tuple[int, str, str]]:
    resolved = resolve_answer_text(record)
    return [
        (item.original_reference_index, item.answer_id, item.text)
        for item in resolved.references
        if item.original_reference_index is not None
    ]


def candidate_text_available(record: SabetPredictionRecord) -> bool:
    return resolve_answer_text(record).candidate is not None


def resolve_answer_text(
    record: SabetPredictionRecord,
    *,
    resolver: TimeQuestionsLabelResolver | None = None,
) -> ResolvedAnswerText:
    """Resolve display text without changing answer identity or rank."""

    if resolver is not None:
        resolver.validate_record(record)
    candidate = _resolve_label(
        record,
        answer_id=record.predicted_answer_ids[0],
        exported_label=record.predicted_answer_labels[0],
        role="candidate",
        original_reference_index=None,
        resolver=resolver,
    )
    references = tuple(
        item
        for index, (answer_id, label) in enumerate(
            zip(record.gold_answer_ids, record.gold_answer_labels, strict=True)
        )
        if (
            item := _resolve_label(
                record,
                answer_id=answer_id,
                exported_label=label,
                role="reference",
                original_reference_index=index,
                resolver=resolver,
            )
        )
        is not None
    )
    return ResolvedAnswerText(
        candidate=candidate,
        references=references,
        raw_reference_count=len(record.gold_answer_ids),
    )


def _resolve_label(
    record: SabetPredictionRecord,
    *,
    answer_id: str,
    exported_label: str,
    role: str,
    original_reference_index: int | None,
    resolver: TimeQuestionsLabelResolver | None,
) -> ResolvedDisplayLabel | None:
    if resolver is not None:
        supplemental = resolver.resolve(
            record,
            answer_id=answer_id,
            role="candidate" if role == "candidate" else "reference",
        )
        if supplemental is not None and is_readable_answer_label(
            dataset=record.dataset,
            answer_id=answer_id,
            label=supplemental.text,
        ):
            return ResolvedDisplayLabel(
                answer_id=answer_id,
                text=supplemental.text,
                source=supplemental.source,
                original_reference_index=original_reference_index,
                wikidata_lastrevid=supplemental.wikidata_lastrevid,
                wikidata_canonical_qid=supplemental.wikidata_canonical_qid,
            )
    if is_readable_answer_label(
        dataset=record.dataset,
        answer_id=answer_id,
        label=exported_label,
    ):
        return ResolvedDisplayLabel(
            answer_id=answer_id,
            text=exported_label.strip(),
            source="prediction_export",
            original_reference_index=original_reference_index,
        )
    return None
