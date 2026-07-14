from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import orjson

from tcred.dataset.io import load_bundle, read_jsonl
from tcred.qa.corpus import RuntimeCorpus
from tcred.qa.models import SystemOutput

_REFUSAL_MARKERS = (
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "insufficient evidence",
    "not enough evidence",
    "unable to determine",
)


def write_run_diagnostics(*, dataset_root: Path, output_root: Path) -> Path:
    rows: list[dict[str, object]] = []
    for family_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        dataset_dir = dataset_root / family_dir.name
        if not (dataset_dir / "questions.jsonl").exists():
            continue
        bundle = load_bundle(dataset_dir)
        corpus = RuntimeCorpus(bundle)
        question_by_id = {question.qid: question for question in bundle.questions}
        for output_path in sorted(family_dir.glob("*.jsonl")):
            for raw in read_jsonl(output_path):
                output = SystemOutput.model_validate(raw)
                question = question_by_id[output.qid]
                rows.append(_diagnostic_row(output, question, corpus=corpus))

    report = {
        "warning": (
            "These are inexpensive pipeline diagnostics, not the benchmark metrics and not "
            "substitutes for human labels."
        ),
        "evidence_matching": "public semantic fact identity, not scenario-local fact ID",
        "overall": _aggregate(rows),
        "by_family_and_system": _grouped(rows, ("dataset_family", "system_name")),
        "by_system": _grouped(rows, ("system_name",)),
    }
    path = output_root / "diagnostics.json"
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    return path


def _diagnostic_row(
    output: SystemOutput,
    question: object,
    *,
    corpus: RuntimeCorpus,
) -> dict[str, object]:
    answer = _normalize(output.answer_text)
    gold_values = [_normalize(value) for value in question.gold_answer_text]
    should_abstain = bool(question.should_abstain)
    refusal = any(marker in answer for marker in _REFUSAL_MARKERS)
    answer_proxy = (
        refusal if should_abstain else any(value and value in answer for value in gold_values)
    )
    required = {
        corpus.semantic_fact_key(fact_id) for fact_id in question.required_valid_evidence_ids
    }
    retrieved = {corpus.semantic_fact_key(hit.fact_id) for hit in output.retrieval.hits}
    cited = {corpus.semantic_fact_key(fact_id) for fact_id in output.resolved_cited_evidence_ids}
    return {
        "dataset_family": output.dataset_family,
        "system_name": str(output.system_name),
        "status": output.status,
        "answer_proxy": answer_proxy,
        "required_evidence_recall": len(required & retrieved) / len(required) if required else None,
        "citation_required_precision": len(required & cited) / len(cited) if cited else None,
        "has_unresolved_citation": bool(output.unresolved_citation_ids),
        "normalized_answer": answer,
    }


def _grouped(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    groups: defaultdict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return {" | ".join(key): _aggregate(group) for key, group in sorted(groups.items())}


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row["status"] == "success"]
    recall = [
        float(row["required_evidence_recall"])
        for row in successful
        if row["required_evidence_recall"] is not None
    ]
    precision = [
        float(row["citation_required_precision"])
        for row in successful
        if row["citation_required_precision"] is not None
    ]
    wrong_answers = {
        str(row["normalized_answer"])
        for row in successful
        if not row["answer_proxy"] and row["normalized_answer"]
    }
    return {
        "outputs": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "answer_proxy_rate": _mean([bool(row["answer_proxy"]) for row in successful]),
        "required_evidence_recall": _mean(recall),
        "citation_required_precision": _mean(precision),
        "unresolved_citation_rate": _mean(
            [bool(row["has_unresolved_citation"]) for row in successful]
        ),
        "distinct_proxy_wrong_answers": len(wrong_answers),
    }


def _mean(values: list[float | bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", normalized)).strip()
