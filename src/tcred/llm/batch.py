from __future__ import annotations

import re
from pathlib import Path
from shutil import copy2
from typing import Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field

from tcred.llm.paraphrase import _clean_plain_text
from tcred.llm.prompts import load_prompt

BatchProvider = Literal["openai", "anthropic", "mistral", "groq"]
ParaphraseKind = Literal["question", "evidence", "answer"]


class ParaphraseTask(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    custom_id: str = Field(min_length=1, max_length=64)
    kind: ParaphraseKind
    item_id: str
    source_file: str
    source_field: str
    target_field: str
    prompt_name: str
    input_text: str


class ParaphraseImportDecision(BaseModel):
    custom_id: str
    item_id: str
    source_file: str
    target_field: str
    accepted: bool
    output_text: str = ""
    reason: str


def build_paraphrase_tasks(
    dataset_dir: Path,
    *,
    include_questions: bool = True,
    include_evidence: bool = True,
    include_answers: bool = False,
    limit: int | None = None,
) -> list[ParaphraseTask]:
    tasks: list[ParaphraseTask] = []
    if include_questions:
        tasks.extend(_question_tasks(dataset_dir, start_index=len(tasks)))
    if include_evidence:
        tasks.extend(_evidence_tasks(dataset_dir, start_index=len(tasks)))
    if include_answers:
        tasks.extend(_answer_tasks(dataset_dir, start_index=len(tasks)))
    if limit is not None:
        return tasks[:limit]
    return tasks


def write_provider_batch(
    *,
    tasks: list[ParaphraseTask],
    provider: BatchProvider,
    model: str,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "paraphrase_tasks.jsonl"
    _guard(manifest_path, overwrite=overwrite)
    _write_jsonl(manifest_path, [task.model_dump(mode="json") for task in tasks])

    if provider == "openai":
        request_path = output_dir / "openai_responses_batch.jsonl"
        rows = [_openai_batch_row(task, model=model) for task in tasks]
        _guard(request_path, overwrite=overwrite)
        _write_jsonl(request_path, rows)
    elif provider == "anthropic":
        request_path = output_dir / "anthropic_message_batch.json"
        body = {"requests": [_anthropic_batch_request(task, model=model) for task in tasks]}
        _guard(request_path, overwrite=overwrite)
        request_path.write_bytes(orjson.dumps(body, option=orjson.OPT_INDENT_2))
    elif provider == "mistral":
        request_path = output_dir / "mistral_chat_batch.jsonl"
        rows = [_mistral_batch_row(task) for task in tasks]
        _guard(request_path, overwrite=overwrite)
        _write_jsonl(request_path, rows)
    elif provider == "groq":
        request_path = output_dir / "groq_chat_batch.jsonl"
        rows = [_groq_batch_row(task, model=model) for task in tasks]
        _guard(request_path, overwrite=overwrite)
        _write_jsonl(request_path, rows)
    else:
        raise ValueError(f"Unsupported batch provider: {provider}")

    summary_path = output_dir / "batch_manifest.json"
    _guard(summary_path, overwrite=overwrite)
    summary = {
        "provider": provider,
        "model": model,
        "task_count": len(tasks),
        "request_file": request_path.name,
        "task_manifest": manifest_path.name,
        "model_facing_user_content": "Only the source text to rewrite is sent as user content.",
    }
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return {
        "task_manifest": manifest_path,
        "provider_requests": request_path,
        "batch_manifest": summary_path,
    }


def import_paraphrase_results(
    *,
    dataset_dir: Path,
    task_manifest: Path,
    result_file: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        task.custom_id: task
        for task in (ParaphraseTask.model_validate(row) for row in _read_jsonl(task_manifest))
    }
    decisions = [_decision_for_result(row=row, tasks=tasks) for row in _read_jsonl(result_file)]
    accepted = {decision.item_id: decision for decision in decisions if decision.accepted}
    written = _write_dataset_with_imports(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        accepted=accepted,
        overwrite=overwrite,
    )
    report_path = output_dir / "paraphrase_import_report.json"
    _guard(report_path, overwrite=overwrite)
    report = {
        "task_count": len(tasks),
        "result_count": len(decisions),
        "accepted_count": sum(decision.accepted for decision in decisions),
        "rejected_count": sum(not decision.accepted for decision in decisions),
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    written["paraphrase_import_report"] = report_path
    return written


def _question_tasks(dataset_dir: Path, *, start_index: int) -> list[ParaphraseTask]:
    return [
        ParaphraseTask(
            custom_id=_custom_id("q", row["qid"], start_index + index),
            kind="question",
            item_id=row["qid"],
            source_file="questions.jsonl",
            source_field="canonical_question",
            target_field="question",
            prompt_name="question_paraphrase.md",
            input_text=row["canonical_question"],
        )
        for index, row in enumerate(_read_jsonl(dataset_dir / "questions.jsonl"))
    ]


def _evidence_tasks(dataset_dir: Path, *, start_index: int) -> list[ParaphraseTask]:
    return [
        ParaphraseTask(
            custom_id=_custom_id("f", row["fact_id"], start_index + index),
            kind="evidence",
            item_id=row["fact_id"],
            source_file="facts.jsonl",
            source_field="canonical_evidence",
            target_field="paraphrased_evidence",
            prompt_name="evidence_paraphrase.md",
            input_text=row["canonical_evidence"],
        )
        for index, row in enumerate(_read_jsonl(dataset_dir / "facts.jsonl"))
    ]


def _answer_tasks(dataset_dir: Path, *, start_index: int) -> list[ParaphraseTask]:
    return [
        ParaphraseTask(
            custom_id=_custom_id("a", row["answer_id"], start_index + index),
            kind="answer",
            item_id=row["answer_id"],
            source_file="answer_variants.jsonl",
            source_field="answer_text",
            target_field="answer_text",
            prompt_name="answer_variant_style.md",
            input_text=row["answer_text"],
        )
        for index, row in enumerate(_read_jsonl(dataset_dir / "answer_variants.jsonl"))
    ]


def _openai_batch_row(task: ParaphraseTask, *, model: str) -> dict[str, object]:
    return {
        "custom_id": task.custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": load_prompt(task.prompt_name)},
                {"role": "user", "content": task.input_text},
            ],
            "max_output_tokens": 400,
            "reasoning": {"effort": "minimal"},
            "store": False,
        },
    }


def _anthropic_batch_request(task: ParaphraseTask, *, model: str) -> dict[str, object]:
    return {
        "custom_id": task.custom_id,
        "params": {
            "model": model,
            "max_tokens": 400,
            "temperature": 0.2,
            "system": load_prompt(task.prompt_name),
            "messages": [{"role": "user", "content": task.input_text}],
        },
    }


def _mistral_batch_row(task: ParaphraseTask) -> dict[str, object]:
    return {
        "custom_id": task.custom_id,
        "body": {
            "max_tokens": 400,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": load_prompt(task.prompt_name)},
                {"role": "user", "content": task.input_text},
            ],
        },
    }


def _groq_batch_row(task: ParaphraseTask, *, model: str) -> dict[str, object]:
    return {
        "custom_id": task.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": load_prompt(task.prompt_name)},
                {"role": "user", "content": task.input_text},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset artifact is missing: {path}")
    rows = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def _decision_for_result(
    *,
    row: dict[str, object],
    tasks: dict[str, ParaphraseTask],
) -> ParaphraseImportDecision:
    custom_id = str(row.get("custom_id", ""))
    task = tasks.get(custom_id)
    if task is None:
        return ParaphraseImportDecision(
            custom_id=custom_id,
            item_id="",
            source_file="",
            target_field="",
            accepted=False,
            reason="result custom_id not found in task manifest",
        )
    output_text = _extract_result_text(row)
    if not output_text:
        return ParaphraseImportDecision(
            custom_id=custom_id,
            item_id=task.item_id,
            source_file=task.source_file,
            target_field=task.target_field,
            accepted=False,
            reason="no text output found",
        )
    cleaned = _clean_plain_text(output_text)
    missing = _missing_protected_spans(source=task.input_text, candidate=cleaned)
    if missing:
        return ParaphraseImportDecision(
            custom_id=custom_id,
            item_id=task.item_id,
            source_file=task.source_file,
            target_field=task.target_field,
            accepted=False,
            output_text=cleaned,
            reason=f"missing protected spans: {', '.join(missing[:8])}",
        )
    return ParaphraseImportDecision(
        custom_id=custom_id,
        item_id=task.item_id,
        source_file=task.source_file,
        target_field=task.target_field,
        accepted=True,
        output_text=cleaned,
        reason="accepted",
    )


def _extract_result_text(row: dict[str, object]) -> str:
    result = row.get("result")
    if isinstance(result, dict):
        if result.get("type") not in {None, "succeeded"}:
            return ""
        message = result.get("message")
        text = _text_from_anthropic_message(message)
        if text:
            return text

    response = row.get("response")
    if isinstance(response, dict):
        body = response.get("body")
        if isinstance(body, dict):
            text = _text_from_openai_responses_body(body)
            if text:
                return text
            text = _text_from_chat_body(body)
            if text:
                return text

    body = row.get("body")
    if isinstance(body, dict):
        text = _text_from_openai_responses_body(body)
        if text:
            return text
        text = _text_from_chat_body(body)
        if text:
            return text

    output_text = row.get("output_text")
    if isinstance(output_text, str):
        return output_text
    text = row.get("text")
    if isinstance(text, str):
        return text
    return ""


def _text_from_anthropic_message(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    chunks = [
        str(block.get("text", ""))
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(chunks).strip()


def _text_from_openai_responses_body(body: dict[str, object]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text
    chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "".join(chunks).strip()


def _text_from_chat_body(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def _missing_protected_spans(*, source: str, candidate: str) -> list[str]:
    spans = sorted(_protected_spans(source), key=len, reverse=True)
    return [span for span in spans if span not in candidate]


def _protected_spans(text: str) -> set[str]:
    spans: set[str] = set()
    month_names = (
        "January|February|March|April|May|June|July|August|September|October|November|December"
    )
    patterns = [
        rf"\b(?:{month_names})\s+\d{{1,2}},\s+\d{{4}}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}\b",
        r"\b[aqfcp]_[A-Za-z0-9_-]+\b",
        r"\b[A-Z][A-Za-z0-9&.'-]+(?:\s+(?:of|the|and|for|[A-Z][A-Za-z0-9&.'-]+)){1,5}\b",
    ]
    for pattern in patterns:
        spans.update(match.group(0) for match in re.finditer(pattern, text))
    return {span.strip() for span in spans if not span.endswith("?")}


def _write_dataset_with_imports(
    *,
    dataset_dir: Path,
    output_dir: Path,
    accepted: dict[str, ParaphraseImportDecision],
    overwrite: bool,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    jsonl_files = {task.source_file for task in accepted.values()}
    for source_path in dataset_dir.iterdir():
        target_path = output_dir / source_path.name
        if source_path.is_dir():
            continue
        _guard(target_path, overwrite=overwrite)
        if source_path.suffix == ".jsonl" and source_path.name in jsonl_files:
            _write_jsonl(target_path, _updated_rows(source_path, accepted))
        else:
            copy2(source_path, target_path)
        written[source_path.stem] = target_path
    return written


def _updated_rows(
    source_path: Path,
    accepted: dict[str, ParaphraseImportDecision],
) -> list[dict[str, object]]:
    rows = _read_jsonl(source_path)
    id_key = _id_key_for_source(source_path.name)
    for row in rows:
        item_id = str(row.get(id_key, ""))
        decision = accepted.get(item_id)
        if decision and decision.source_file == source_path.name:
            row[decision.target_field] = decision.output_text
    return rows


def _id_key_for_source(source_file: str) -> str:
    if source_file == "questions.jsonl":
        return "qid"
    if source_file == "facts.jsonl":
        return "fact_id"
    if source_file == "answer_variants.jsonl":
        return "answer_id"
    raise ValueError(f"Unsupported paraphrase source file: {source_file}")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def _custom_id(prefix: str, item_id: str, index: int) -> str:
    suffix = f"_{index:05d}"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{prefix}_{item_id}")
    return f"{cleaned[: 64 - len(suffix)]}{suffix}"


def _guard(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
