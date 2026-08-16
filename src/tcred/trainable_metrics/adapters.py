from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tcred.trainable_metrics.config import SourceConfig
from tcred.trainable_metrics.schema import (
    EvidencePassage,
    SemanticRecord,
    SemanticTarget,
    SemanticTask,
    stable_unit_id,
)

SUPPORT_LABELS = ("entailment", "neutral", "contradiction")
TEMPORAL_LABELS = ("support", "unknown", "contradiction")
CITATION_LABELS = ("appropriate", "incomplete", "inappropriate")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class AdapterResult:
    records: list[SemanticRecord]
    exclusions: list[str]


Adapter = Callable[[Mapping[str, Any], SourceConfig, str], AdapterResult]


def adapt_row(
    row: Mapping[str, Any],
    *,
    source: SourceConfig,
    native_split: str,
) -> AdapterResult:
    try:
        adapter = ADAPTERS[source.adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown semantic adapter: {source.adapter}") from exc
    return adapter(row, source, native_split)


def adapt_nli(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    premise = _text(row, "premise", "sentence1")
    hypothesis = _text(row, "hypothesis", "sentence2")
    label = _nli_label(row.get("label", row.get("gold")))
    if not premise or not hypothesis or label is None:
        return _excluded("missing text or a supported NLI label")
    native_id = _native_id(row, premise, hypothesis)
    group_id = _text_hash(premise)
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.SUPPORT,
        candidate=hypothesis,
        evidence=[premise],
        target=SemanticTarget(class_distribution=_hard_distribution(label)),
        provenance=f"official {source.name} label={label}",
    )
    return _accepted(record)


def adapt_paws(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    reference = _text(row, "sentence1")
    candidate = _text(row, "sentence2")
    label = _binary(row.get("label"))
    if not reference or not candidate or label is None:
        return _excluded("missing sentence pair or binary paraphrase label")
    native_id = _native_id(row, reference, candidate)
    group_id = _unordered_pair_hash(reference, candidate)
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.ANSWER,
        question="Do the reference and candidate express the same proposition?",
        references=[reference],
        candidate=candidate,
        target=SemanticTarget(equivalence=label),
        provenance=f"official PAWS paraphrase label={int(label)}",
    )
    return _accepted(record)


def adapt_fever(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    claim = _text(row, "claim")
    label = _fever_label(row.get("label"))
    evidence = _string_list(row.get("evidence_texts") or row.get("evidence"))
    if label == "neutral":
        return _excluded("FEVER NEI has no released gold evidence passage")
    if not claim or label is None or not evidence:
        return _excluded("missing claim, evidence text, or supported FEVER label")
    native_id = _native_id(row, claim)
    group_id = _text_hash(*sorted(evidence))
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.SUPPORT,
        candidate=claim,
        evidence=evidence,
        target=SemanticTarget(class_distribution=_hard_distribution(label)),
        provenance=f"official FEVER label={label}; reconstructed gold evidence set",
    )
    return _accepted(record)


def adapt_vitaminc_support(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    return _adapt_vitaminc(row, source, native_split, temporal=False)


def adapt_vitaminc_temporal(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    return _adapt_vitaminc(row, source, native_split, temporal=True)


def _adapt_vitaminc(
    row: Mapping[str, Any],
    source: SourceConfig,
    native_split: str,
    *,
    temporal: bool,
) -> AdapterResult:
    claim = _text(row, "claim")
    evidence = _text(row, "evidence")
    label = _fever_label(row.get("label"))
    if not claim or not evidence or label is None:
        return _excluded("missing VitaminC claim, evidence, or label")
    native_id = _native_id(row, claim, evidence)
    group_id = str(row.get("case_id") or row.get("page") or native_id)
    task = SemanticTask.TEMPORAL if temporal else SemanticTask.SUPPORT
    mapped_label = {
        "entailment": "support",
        "neutral": "unknown",
        "contradiction": "contradiction",
    }[label]
    target_label = mapped_label if temporal else label
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=task,
        candidate=claim,
        evidence=[evidence],
        query_time=str(row.get("wiki_revision_id") or "Wikipedia revision context")
        if temporal
        else None,
        temporal_operator="document_revision" if temporal else None,
        target=SemanticTarget(class_distribution=_hard_distribution(target_label)),
        provenance=f"official VitaminC label={row.get('label')}",
        invariance_group_id=group_id,
    )
    return _accepted(record)


def adapt_temporal_nli(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    premise = _text(row, "premise", "sentence1", "context")
    hypothesis = _text(row, "hypothesis", "sentence2")
    class_label, supported = _temporal_nli_target(row.get("label", row.get("gold")))
    if not premise or not hypothesis or (class_label is None and supported is None):
        return _excluded("missing temporal NLI text or label")
    native_id = _native_id(row, premise, hypothesis)
    group_id = _text_hash(premise)
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.TEMPORAL,
        candidate=hypothesis,
        evidence=[premise],
        query_time=_optional_text(row, "query_time", "interval", "time"),
        temporal_operator=_optional_text(row, "relation", "operator", "template"),
        target=SemanticTarget(
            class_distribution=_hard_distribution(class_label) if class_label else {},
            supported=supported,
        ),
        provenance=f"official Temporal NLI recast label={row.get('label', row.get('gold'))}",
        transformation=str(
            row.get("type-of-inference")
            or row.get("template")
            or row.get("source")
            or "temporal_recast"
        ),
    )
    return _accepted(record)


def adapt_mocha(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    question = _text(row, "question")
    reference = _text(row, "reference")
    candidate = _text(row, "candidate")
    context = _text(row, "context")
    score = _float(row.get("score"))
    if not question or not reference or not candidate or score is None:
        return _excluded("missing MOCHA text or score")
    if not 1.0 <= score <= 5.0:
        return _excluded("MOCHA score outside the released 1-5 scale")
    normalized = (score - 1.0) / 4.0
    native_id = _native_id(row, question, candidate)
    group_id = _text_hash(context or question)
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.ANSWER,
        question=question,
        references=[reference],
        candidate=candidate,
        evidence=[context] if context else [],
        target=SemanticTarget(
            answer_u1=min(1.0, 2.0 * normalized),
            answer_u2=max(0.0, 2.0 * normalized - 1.0),
            scalar_rating=normalized,
        ),
        provenance=f"official MOCHA mean human score={score:g}/5",
    )
    return _accepted(record)


def adapt_answer_equivalence(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    question = _text(row, "question")
    reference = _text(row, "reference")
    candidate = _text(row, "candidate")
    context = _text(row, "context")
    equivalence = _answer_equivalence_label(row)
    if not question or not reference or not candidate or equivalence is None:
        return _excluded("missing Answer Equivalence text or adjudicated label")
    native_id = _native_id(row, question, candidate)
    group_id = _text_hash(context or question)
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.ANSWER,
        question=question,
        references=[reference],
        candidate=candidate,
        evidence=[context] if context else [],
        target=SemanticTarget(equivalence=equivalence, answer_u2=equivalence),
        provenance=f"official Answer Equivalence label={int(equivalence)}",
    )
    return _accepted(record)


def adapt_attribution_bench(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    question = _text(row, "question") or "Is the response claim supported by its citation?"
    claim = _text(row, "claim", "claim_raw_string")
    evidence = _attribution_references(row)
    label = str(row.get("attribution_label") or "").strip().lower()
    mapped = {
        "attributable": "appropriate",
        "not attributable": "inappropriate",
        "not_attributable": "inappropriate",
    }.get(label)
    if not claim or not evidence or mapped is None:
        return _excluded("missing AttributionBench claim, references, or binary label")
    native_id = _native_id(row, claim)
    group_id = _text_hash(question, *sorted(evidence))
    citations = _string_list(row.get("citation_links"))
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.CITATION,
        question=question,
        candidate=claim,
        evidence=evidence,
        citations=citations,
        target=SemanticTarget(class_distribution=_hard_distribution(mapped)),
        provenance=f"official AttributionBench label={label}",
    )
    return _accepted(record)


def adapt_ragtruth(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    response = _text(row, "response")
    source_text = _ragtruth_source_text(row)
    question = _ragtruth_question(row)
    if not response or not source_text:
        return _excluded("missing RAGTruth response or source")
    annotations = row.get("labels") if isinstance(row.get("labels"), list) else []
    source_id = str(row.get("source_id") or _text_hash(source_text, question))
    records: list[SemanticRecord] = []
    exclusions: list[str] = []
    for index, (start, end, sentence) in enumerate(_sentence_spans(response)):
        overlaps = [label for label in annotations if _span_overlaps(label, start, end)]
        has_implicit_true = any(
            bool(label.get("implicit_true"))
            for label in overlaps
            if isinstance(label, Mapping)
        )
        if has_implicit_true:
            exclusions.append("sentence overlaps an implicit-true annotation")
            continue
        supported = 0.0 if overlaps else 1.0
        native_id = f"{row.get('id', source_id)}:sentence:{index}"
        records.append(
            _record(
                source=source,
                native_split=native_split,
                native_id=native_id,
                group_id=source_id,
                task=SemanticTask.SUPPORT,
                question=question,
                candidate=sentence,
                evidence=[source_text],
                target=SemanticTarget(supported=supported),
                provenance=(
                    "RAGTruth exhaustive span annotation: overlaps hallucination span"
                    if overlaps
                    else "RAGTruth exhaustive span annotation: no hallucination span"
                ),
            )
        )
    if not records:
        exclusions.append("no eligible sentence after span alignment")
    return AdapterResult(records=records, exclusions=exclusions)


def adapt_torque(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    passage = _text(row, "passage")
    question = _text(row, "question")
    answers = _torque_answers(row)
    if not passage or not question or not answers:
        return _excluded("missing TORQUE passage, question, or agreed answer")
    passage_id = str(row.get("passageID") or row.get("passage_id") or _text_hash(passage))
    native_id = _native_id(row, passage_id, question)
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=passage_id,
        task=SemanticTask.TEMPORAL,
        question=question,
        references=answers,
        candidate="; ".join(answers),
        evidence=[passage],
        temporal_operator="implicit_event_order",
        target=SemanticTarget(class_distribution={"support": 1.0}),
        provenance="official TORQUE agreed answer spans; positive temporal support only",
    )
    return _accepted(record)


def adapt_squad2(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    question = _text(row, "question")
    context = _text(row, "context")
    answers = row.get("answers")
    answer_texts = (
        _string_list(answers.get("text")) if isinstance(answers, Mapping) else _string_list(answers)
    )
    if not question or not context:
        return _excluded("missing SQuAD 2.0 question or context")
    native_id = _native_id(row, question)
    title = str(row.get("title") or "")
    group_id = f"{title}:{_text_hash(context)}"
    record = _record(
        source=source,
        native_split=native_split,
        native_id=native_id,
        group_id=group_id,
        task=SemanticTask.ANSWERABILITY,
        question=question,
        references=answer_texts,
        candidate="Can the question be answered from the supplied evidence?",
        evidence=[context],
        target=SemanticTarget(answerable=float(bool(answer_texts))),
        provenance=f"official SQuAD 2.0 answer span count={len(answer_texts)}",
    )
    return _accepted(record)


def adapt_ms_marco(
    row: Mapping[str, Any], source: SourceConfig, native_split: str
) -> AdapterResult:
    question = _text(row, "query", "question")
    passages = row.get("passages")
    if not question or not isinstance(passages, Mapping):
        return _excluded("missing MS MARCO query or passages")
    texts = _string_list(passages.get("passage_text"))
    selected = list(passages.get("is_selected") or [])
    if len(texts) != len(selected):
        return _excluded("MS MARCO passage_text and is_selected lengths differ")
    positives = [text for text, flag in zip(texts, selected, strict=True) if int(flag) == 1]
    negatives = [text for text, flag in zip(texts, selected, strict=True) if int(flag) == 0]
    if not positives or not negatives:
        return _excluded("query has no released positive/negative passage pair")
    positive = positives[0]
    negative = max(negatives, key=lambda text: (_token_jaccard(question, text), text))
    query_id = str(row.get("query_id") or _text_hash(question))
    pair_id = f"msmarco:{query_id}"
    records = [
        _record(
            source=source,
            native_split=native_split,
            native_id=f"{query_id}:positive",
            group_id=query_id,
            task=SemanticTask.RELEVANCE,
            question=question,
            candidate="Is this passage relevant to the query?",
            evidence=[positive],
            target=SemanticTarget(relevance=1.0, pair_id=pair_id, pair_role="positive"),
            provenance="official MS MARCO is_selected=1",
        ),
        _record(
            source=source,
            native_split=native_split,
            native_id=f"{query_id}:negative",
            group_id=query_id,
            task=SemanticTask.RELEVANCE,
            question=question,
            candidate="Is this passage relevant to the query?",
            evidence=[negative],
            target=SemanticTarget(relevance=0.0, pair_id=pair_id, pair_role="negative"),
            provenance="official MS MARCO is_selected=0; lexical-hardest released negative",
        ),
    ]
    return AdapterResult(records=records, exclusions=[])


ADAPTERS: dict[str, Adapter] = {
    "nli": adapt_nli,
    "paws": adapt_paws,
    "fever": adapt_fever,
    "vitaminc_support": adapt_vitaminc_support,
    "vitaminc_temporal": adapt_vitaminc_temporal,
    "temporal_nli": adapt_temporal_nli,
    "mocha": adapt_mocha,
    "answer_equivalence": adapt_answer_equivalence,
    "attribution_bench": adapt_attribution_bench,
    "ragtruth": adapt_ragtruth,
    "torque": adapt_torque,
    "squad2": adapt_squad2,
    "ms_marco": adapt_ms_marco,
}


def _record(
    *,
    source: SourceConfig,
    native_split: str,
    native_id: str,
    group_id: str,
    task: SemanticTask,
    candidate: str,
    target: SemanticTarget,
    provenance: str,
    question: str | None = None,
    references: list[str] | None = None,
    evidence: list[str] | None = None,
    citations: list[str] | None = None,
    query_time: str | None = None,
    temporal_operator: str | None = None,
    transformation: str | None = None,
    invariance_group_id: str | None = None,
) -> SemanticRecord:
    target = target.model_copy(update={"invariance_group_id": invariance_group_id})
    evidence_rows = [
        EvidencePassage(
            evidence_id=f"{native_id}:evidence:{index}",
            text=text,
            source_id=group_id,
            rank=index + 1,
        )
        for index, text in enumerate(evidence or [])
        if text.strip()
    ]
    return SemanticRecord.create(
        unit_id=stable_unit_id(source.name, native_split, native_id, task),
        source_dataset=source.name,
        source_version=source.revision,
        source_native_split=native_split,
        source_native_id=native_id,
        source_group_id=group_id,
        curriculum_stage=source.curriculum_stage,
        task=task,
        question=question,
        query_time_or_interval=query_time,
        temporal_operator=temporal_operator,
        reference_answers=references or [],
        candidate_or_claim=candidate,
        evidence_passages=evidence_rows,
        citations=citations or [],
        target=target,
        label_provenance=provenance,
        transformation_family=transformation,
        license_record=f"{source.license_id}@{source.revision}",
    )


def _accepted(*records: SemanticRecord) -> AdapterResult:
    return AdapterResult(records=list(records), exclusions=[])


def _excluded(reason: str) -> AdapterResult:
    return AdapterResult(records=[], exclusions=[reason])


def _hard_distribution(label: str) -> dict[str, float]:
    return {label: 1.0}


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _optional_text(row: Mapping[str, Any], *keys: str) -> str | None:
    value = _text(row, *keys)
    return value or None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [" ".join(value.split())] if value.strip() else []
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(" ".join(item.split()))
        elif isinstance(item, Mapping):
            text = _text(item, "text", "content", "reference")
            if text:
                result.append(text)
    return result


def _native_id(row: Mapping[str, Any], *fallback_text: str) -> str:
    for key in ("id", "uid", "pairID", "qid", "unique_id", "query_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return _text_hash(*fallback_text)


def _text_hash(*values: str) -> str:
    return stable_unit_id(*values).removeprefix("sl_")


def _unordered_pair_hash(first: str, second: str) -> str:
    return _text_hash(*sorted((first, second)))


def _binary(value: Any) -> float | None:
    if value in {1, "1", "true", "yes", "equivalent", "positive"}:
        return 1.0
    if value in {0, "0", "false", "no", "not_equivalent", "negative"}:
        return 0.0
    return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nli_label(value: Any) -> str | None:
    if isinstance(value, int):
        return {0: "entailment", 1: "neutral", 2: "contradiction"}.get(value)
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return {
        "e": "entailment",
        "entailment": "entailment",
        "n": "neutral",
        "neutral": "neutral",
        "c": "contradiction",
        "contradiction": "contradiction",
    }.get(normalized)


def _temporal_nli_target(value: Any) -> tuple[str | None, float | None]:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized in {"entailed", "entailment"}:
        return "support", 1.0
    if normalized in {"not-entailed", "not entailed", "non-entailment"}:
        # The release is binary NLI. Non-entailment is not promoted to the
        # finer contradiction/unknown distinction that the source does not label.
        return None, 0.0
    label = _nli_label(value)
    mapped = {
        "entailment": "support",
        "neutral": "unknown",
        "contradiction": "contradiction",
    }.get(label or "")
    return mapped, 1.0 if mapped == "support" else None


def _fever_label(value: Any) -> str | None:
    normalized = str(value or "").strip().upper().replace("_", " ")
    return {
        "SUPPORTS": "entailment",
        "SUPPORTED": "entailment",
        "REFUTES": "contradiction",
        "REFUTED": "contradiction",
        "NOT ENOUGH INFO": "neutral",
        "NEI": "neutral",
    }.get(normalized)


def _answer_equivalence_label(row: Mapping[str, Any]) -> float | None:
    label = str(row.get("label") or "").strip().lower()
    if label == "equivalent":
        return 1.0
    if label in {"not_equivalent", "not equivalent"}:
        return 0.0
    score = _float(row.get("score"))
    return score if score in {0.0, 1.0} else None


def _attribution_references(row: Mapping[str, Any]) -> list[str]:
    references = _string_list(row.get("references"))
    if references:
        return references
    return _string_list(row.get("webpage_references"))


def _ragtruth_source_text(row: Mapping[str, Any]) -> str:
    value = row.get("source_info") or row.get("source")
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        preferred = _text(value, "passages", "context", "source", "text", "response")
        if preferred:
            return preferred
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, list):
        return "\n".join(_string_list(value))
    return ""


def _ragtruth_question(row: Mapping[str, Any]) -> str | None:
    direct = _optional_text(row, "question", "prompt")
    if direct:
        return direct
    source_info = row.get("source_info")
    if isinstance(source_info, Mapping):
        return _optional_text(source_info, "question", "prompt", "query")
    return None


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for part in _SENTENCE_BOUNDARY.split(text):
        start = text.find(part, cursor)
        end = start + len(part)
        sentence = " ".join(part.split())
        if sentence:
            spans.append((start, end, sentence))
        cursor = end
    return spans


def _span_overlaps(label: Any, start: int, end: int) -> bool:
    if not isinstance(label, Mapping):
        return False
    try:
        label_start = int(label.get("start"))
        label_end = int(label.get("end"))
    except (TypeError, ValueError):
        return False
    return label_start < end and label_end > start


def _torque_answers(row: Mapping[str, Any]) -> list[str]:
    answers = row.get("answer") or row.get("answers")
    if isinstance(answers, Mapping):
        return _string_list(answers.get("spans") or answers.get("text"))
    return _string_list(answers)


def _token_jaccard(first: str, second: str) -> float:
    left = {token.lower() for token in _TOKEN.findall(first)}
    right = {token.lower() for token in _TOKEN.findall(second)}
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))
