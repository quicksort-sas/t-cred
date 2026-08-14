from __future__ import annotations

import asyncio
import hashlib
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import httpx
import orjson
from dotenv import load_dotenv
from pydantic import ValidationError

from tcred.metrics.judge import (
    JudgeProvider,
    _AsyncStartRateLimiter,
    _provider_base_url,
    _provider_endpoint,
    _provider_headers,
    _provider_key,
    _remove_titles,
    _retry_after_seconds,
)
from tcred.metrics.task_judge_models import (
    ANSWER_STAGE_FIELDS,
    EVIDENCE_STAGE_FIELDS,
    AnswerStageResult,
    EvidenceStageResult,
    JudgeStage,
    PromptVariant,
    StageCacheRecord,
    TaskJudgeInput,
    TaskJudgeRecord,
    TaskJudgeResult,
)
from tcred.metrics.task_judge_render import render_answer_stage, render_evidence_stage

PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts" / "metrics"
CONTRACT_VERSION = "tcred_task_judge_v1.6"
DEFAULT_MODEL = "mistral-large-2512"
DEFAULT_PROVIDER: JudgeProvider = "mistral"
DEFAULT_REQUESTS_PER_SECOND = 0.24
DEFAULT_RANDOM_SEED = 20260814
MINIMUM_ACCEPTABLE_RPM = 15

_BASE_PROMPTS = {
    "evidence": PROMPT_ROOT / "tcred_evidence_judge_v1.md",
    "answer": PROMPT_ROOT / "tcred_answer_judge_v1.md",
}
_EXAMPLE_PROMPTS = {
    "evidence": PROMPT_ROOT / "tcred_evidence_judge_examples_v1.md",
    "answer": PROMPT_ROOT / "tcred_answer_judge_examples_v1.md",
}
_SCHEMA_NAMES = {
    "evidence": "tcred_evidence_stage_judge_v1",
    "answer": "tcred_answer_stage_judge_v1",
}


@dataclass(frozen=True)
class _RequestSpec:
    row: TaskJudgeInput
    stage: JudgeStage
    rendered_input: str
    input_sha256: str
    prompt: str
    prompt_sha256: str
    schema: dict[str, object]
    schema_sha256: str
    cache_path: Path

    @property
    def judgment_id(self) -> str:
        return f"{self.row.metric_id}:{self.stage}"


class _RetryableTaskJudgeError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


async def score_with_task_judge(
    rows: list[TaskJudgeInput],
    *,
    cache_dir: Path,
    prompt_variant: PromptVariant,
    provider: JudgeProvider = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    requests_per_second: float | None = DEFAULT_REQUESTS_PER_SECOND,
    random_seed: int = DEFAULT_RANDOM_SEED,
    timeout_seconds: float = 180.0,
) -> dict[str, TaskJudgeRecord]:
    """Run the blinded two-stage task judge with resumable schema-validated caches."""

    if provider != "mistral":
        raise ValueError("The task-matched judge currently supports provider=mistral only")
    if requests_per_second is None or requests_per_second <= 0:
        raise ValueError("The Mistral task judge requires an explicit positive request throttle")
    if requests_per_second > DEFAULT_REQUESTS_PER_SECOND:
        raise ValueError(
            "The configured throttle exceeds the verified safe rate of "
            f"{DEFAULT_REQUESTS_PER_SECOND * 60:.1f} requests/minute"
        )
    load_dotenv(dotenv_path=Path(".env"))
    key = _provider_key(provider)
    if not key:
        raise RuntimeError("No Mistral API key is configured for the task judge")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[str, StageCacheRecord] = {}
    pending: list[_RequestSpec] = []
    for row in rows:
        for stage in _required_stages(row):
            spec = _request_spec(
                row,
                stage=stage,
                cache_dir=cache_dir,
                prompt_variant=prompt_variant,
            )
            record = _valid_cache_record(
                spec,
                provider=provider,
                model=model,
                prompt_variant=prompt_variant,
                random_seed=random_seed,
            )
            if record is None:
                pending.append(spec)
            else:
                cached[spec.judgment_id] = record

    if pending:
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        semaphore = asyncio.Semaphore(concurrency)
        rate_limiter = _AsyncStartRateLimiter(requests_per_second)
        completed = 0
        async with httpx.AsyncClient(
            base_url=_provider_base_url(provider),
            headers=_provider_headers(provider, key),
            timeout=timeout_seconds,
            limits=limits,
        ) as client:

            async def run_one(spec: _RequestSpec) -> tuple[str, StageCacheRecord]:
                nonlocal completed
                async with semaphore:
                    record = await _request_stage(
                        client,
                        spec=spec,
                        provider=provider,
                        model=model,
                        prompt_variant=prompt_variant,
                        random_seed=random_seed,
                        rate_limiter=rate_limiter,
                    )
                    _write_cache(spec.cache_path, record)
                    completed += 1
                    if completed % 25 == 0 or completed == len(pending):
                        print(
                            f"task judge: {completed}/{len(pending)} new stage calls; "
                            f"{len(cached)} cached",
                            flush=True,
                        )
                    return spec.judgment_id, record

            for judgment_id, record in await asyncio.gather(*(run_one(spec) for spec in pending)):
                cached[judgment_id] = record

    return {
        row.metric_id: _combine_stages(
            row,
            prompt_variant=prompt_variant,
            provider=provider,
            model=model,
            records=cached,
        )
        for row in rows
    }


def prompt_contract(stage: str, variant: PromptVariant) -> tuple[str, str]:
    if stage not in _BASE_PROMPTS:
        raise ValueError(f"Unknown judge stage: {stage}")
    prompt = _BASE_PROMPTS[stage].read_text(encoding="utf-8").strip()
    if variant == "contrastive_few_shot":
        prompt += "\n\n" + _EXAMPLE_PROMPTS[stage].read_text(encoding="utf-8").strip()
    elif variant != "rubric_only":
        raise ValueError(f"Unknown prompt variant: {variant}")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return prompt, digest


def write_task_judgments(records: dict[str, TaskJudgeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for metric_id in sorted(records):
            stream.write(
                orjson.dumps(
                    records[metric_id].model_dump(mode="json"),
                    option=orjson.OPT_SORT_KEYS,
                )
            )
            stream.write(b"\n")
    temporary.replace(path)


async def _request_stage(
    client: httpx.AsyncClient,
    *,
    spec: _RequestSpec,
    provider: JudgeProvider,
    model: str,
    prompt_variant: PromptVariant,
    random_seed: int,
    rate_limiter: _AsyncStartRateLimiter,
) -> StageCacheRecord:
    result_model = _result_model(spec.stage)
    payload = _mistral_payload(
        model=model,
        prompt=spec.prompt,
        rendered_input=spec.rendered_input,
        schema=spec.schema,
        schema_name=_SCHEMA_NAMES[spec.stage],
        random_seed=random_seed,
        max_tokens=2200 if spec.stage == "evidence" else 1200,
    )
    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            await rate_limiter.wait()
            response = await client.post(_provider_endpoint(provider), json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                if "credit_balance" in response.text or "insufficient_quota" in response.text:
                    raise RuntimeError("Mistral task-judge account has insufficient credit")
                raise _RetryableTaskJudgeError(
                    f"Mistral task judge returned {response.status_code}: {response.text[:500]}",
                    retry_after=_retry_after_seconds(response),
                )
            response.raise_for_status()
            _assert_minimum_rate_limit(response)
            body = response.json()
            result = _parse_structured_result(body, result_model=result_model)
            _validate_stage_result(result, row=spec.row, stage=spec.stage)
            pointer_warnings = _support_pointer_warnings(result, row=spec.row, stage=spec.stage)
            return StageCacheRecord(
                judgment_id=spec.judgment_id,
                metric_id=spec.row.metric_id,
                stage=spec.stage,  # type: ignore[arg-type]
                input_sha256=spec.input_sha256,
                prompt_sha256=spec.prompt_sha256,
                schema_sha256=spec.schema_sha256,
                prompt_variant=prompt_variant,
                contract_version=CONTRACT_VERSION,
                provider=provider,
                model=model,
                random_seed=random_seed,
                response_id=str(body.get("id") or ""),
                attempts=attempt,
                rate_limit_headers=_rate_limit_headers(response),
                support_pointer_warnings=pointer_warnings,
                usage=_parse_mistral_usage(body),
                result=result,
            )
        except (
            httpx.HTTPError,
            ValidationError,
            ValueError,
            KeyError,
            _RetryableTaskJudgeError,
        ) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise RuntimeError(
                    f"Non-retryable Mistral task-judge error: {exc.response.text[:1000]}"
                ) from exc
            attempt_limit = 3 if isinstance(exc, (ValidationError, ValueError, KeyError)) else 8
            if attempt >= attempt_limit:
                break
            retry_after = exc.retry_after if isinstance(exc, _RetryableTaskJudgeError) else 0.0
            if isinstance(exc, (ValidationError, ValueError, KeyError)):
                payload = _payload_with_contract_repair(payload, error=exc)
                await asyncio.sleep(retry_after)
            else:
                await asyncio.sleep(max(retry_after, min(2**attempt + random.random(), 60.0)))
    raise RuntimeError(f"Task judge failed for {spec.judgment_id} after retries") from last_error


def _payload_with_contract_repair(
    payload: dict[str, object],
    *,
    error: ValidationError | ValueError | KeyError,
) -> dict[str, object]:
    """Add neutral contract feedback after an invalid structured response."""

    repaired = deepcopy(payload)
    messages = repaired.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Task-judge request payload has no messages")
    if isinstance(error, ValueError) and "applicability mismatch" in str(error):
        correction = (
            "Your previous response violated the declared applicability contract. For every "
            "field listed in <applicable_fields>, return yes, partial, no, or unjudgeable; use "
            "not_applicable only for fields absent from that list. Re-evaluate from the displayed "
            "material without assuming which substantive label is expected."
        )
    else:
        correction = (
            "Your previous response violated the required JSON schema. Return a fresh judgment "
            "that obeys every enum, string-length, array-length, uniqueness, and identifier "
            "constraint. Preserve your substantive assessment unless the rubric requires a change."
        )
    messages.append({"role": "user", "content": correction})
    return repaired


def _request_spec(
    row: TaskJudgeInput,
    *,
    stage: JudgeStage,
    cache_dir: Path,
    prompt_variant: PromptVariant,
) -> _RequestSpec:
    prompt, prompt_sha256 = prompt_contract(stage, prompt_variant)
    rendered = render_evidence_stage(row) if stage == "evidence" else render_answer_stage(row)
    input_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    schema = _stage_schema(row, stage=stage)
    schema_sha256 = hashlib.sha256(orjson.dumps(schema, option=orjson.OPT_SORT_KEYS)).hexdigest()
    judgment_id = f"{row.metric_id}:{stage}"
    digest = hashlib.sha256(judgment_id.encode("utf-8")).hexdigest()
    return _RequestSpec(
        row=row,
        stage=stage,
        rendered_input=rendered,
        input_sha256=input_sha256,
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        schema=schema,
        schema_sha256=schema_sha256,
        cache_path=cache_dir / prompt_variant / stage / f"{digest}.json",
    )


def _required_stages(row: TaskJudgeInput) -> tuple[str, ...]:
    stages = ["answer"]
    if row.stage_fields("evidence"):
        stages.insert(0, "evidence")
    return tuple(stages)


def _valid_cache_record(
    spec: _RequestSpec,
    *,
    provider: JudgeProvider,
    model: str,
    prompt_variant: PromptVariant,
    random_seed: int,
) -> StageCacheRecord | None:
    if not spec.cache_path.is_file():
        return None
    try:
        record = StageCacheRecord.model_validate_json(spec.cache_path.read_bytes())
    except (ValidationError, ValueError):
        return None
    if (
        record.judgment_id != spec.judgment_id
        or record.metric_id != spec.row.metric_id
        or record.stage != spec.stage
        or record.input_sha256 != spec.input_sha256
        or record.prompt_sha256 != spec.prompt_sha256
        or record.schema_sha256 != spec.schema_sha256
        or record.prompt_variant != prompt_variant
        or record.contract_version != CONTRACT_VERSION
        or record.provider != provider
        or record.model != model
        or record.random_seed != random_seed
    ):
        return None
    _validate_stage_result(record.result, row=spec.row, stage=spec.stage)
    if record.support_pointer_warnings != _support_pointer_warnings(
        record.result, row=spec.row, stage=spec.stage
    ):
        return None
    return record


def _validate_stage_result(
    result: EvidenceStageResult | AnswerStageResult,
    *,
    row: TaskJudgeInput,
    stage: JudgeStage,
) -> None:
    fields = EVIDENCE_STAGE_FIELDS if stage == "evidence" else ANSWER_STAGE_FIELDS
    applicable = set(row.applicable_fields)
    for field in fields:
        judgment = getattr(result, field)
        should_apply = field in applicable
        if should_apply == (judgment.label == "not_applicable"):
            raise ValueError(
                f"Judge applicability mismatch for {row.metric_id}:{field}: {judgment.label}"
            )
        if judgment.label == "not_applicable" and (judgment.evidence_ids or judgment.path_ids):
            raise ValueError(f"Not-applicable field cites support for {row.metric_id}:{field}")


def _support_pointer_warnings(
    result: EvidenceStageResult | AnswerStageResult,
    *,
    row: TaskJudgeInput,
    stage: JudgeStage,
) -> dict[str, dict[str, list[str]]]:
    evidence_ids = {item.evidence_id for item in row.displayed_evidence()} | {
        edge.fact_id for path in row.graph_paths for edge in path.edges if edge.fact_id
    }
    path_ids = {path.path_id for path in row.graph_paths}
    fields = EVIDENCE_STAGE_FIELDS if stage == "evidence" else ANSWER_STAGE_FIELDS
    warnings: dict[str, dict[str, list[str]]] = {}
    for field in fields:
        judgment = getattr(result, field)
        unknown_evidence = sorted(set(judgment.evidence_ids) - evidence_ids)
        unknown_paths = sorted(set(judgment.path_ids) - path_ids)
        if unknown_evidence or unknown_paths:
            warnings[field] = {
                "unknown_evidence_ids": unknown_evidence,
                "unknown_path_ids": unknown_paths,
            }
    return warnings


def migrate_first_attempt_cache(
    source_dir: Path,
    destination_dir: Path,
    *,
    source_contract: str,
    destination_contract: str,
) -> dict[str, int]:
    """Migrate only first-response caches across an output-policy-only contract change."""

    summary = {
        "source_records": 0,
        "migrated_records": 0,
        "skipped_retried_records": 0,
        "skipped_other_contract_records": 0,
    }
    for source_path in sorted(source_dir.rglob("*.json")):
        summary["source_records"] += 1
        record = StageCacheRecord.model_validate_json(source_path.read_bytes())
        if record.contract_version != source_contract:
            summary["skipped_other_contract_records"] += 1
            continue
        if record.attempts != 1:
            summary["skipped_retried_records"] += 1
            continue
        migrated = record.model_copy(
            update={
                "contract_version": destination_contract,
                "support_pointer_warnings": {},
            }
        )
        destination_path = destination_dir / source_path.relative_to(source_dir)
        if destination_path.is_file():
            existing = StageCacheRecord.model_validate_json(destination_path.read_bytes())
            if existing != migrated:
                raise ValueError(f"Conflicting migrated task-judge cache: {destination_path}")
        else:
            _write_cache(destination_path, migrated)
        summary["migrated_records"] += 1
    return summary


def _combine_stages(
    row: TaskJudgeInput,
    *,
    prompt_variant: PromptVariant,
    provider: JudgeProvider,
    model: str,
    records: dict[str, StageCacheRecord],
) -> TaskJudgeRecord:
    answer = records[f"{row.metric_id}:answer"]
    if not isinstance(answer.result, AnswerStageResult):
        raise TypeError(f"Answer-stage cache has wrong result type: {row.metric_id}")
    evidence = records.get(f"{row.metric_id}:evidence")
    if evidence is not None and not isinstance(evidence.result, EvidenceStageResult):
        raise TypeError(f"Evidence-stage cache has wrong result type: {row.metric_id}")
    result = TaskJudgeResult(
        answer_correct=answer.result.answer_correct,
        response_decision_appropriate=answer.result.response_decision_appropriate,
        **(
            {
                "temporal_correct": evidence.result.temporal_correct,
                "evidence_supports_answer": evidence.result.evidence_supports_answer,
                "citation_temporally_valid": evidence.result.citation_temporally_valid,
                "graph_evidence_sufficient": evidence.result.graph_evidence_sufficient,
            }
            if evidence is not None
            else {}
        ),
    )
    return TaskJudgeRecord(
        metric_id=row.metric_id,
        prompt_variant=prompt_variant,
        provider=provider,
        model=model,
        answer_stage=answer,
        evidence_stage=evidence,
        result=result,
    )


def _result_model(stage: JudgeStage) -> type[EvidenceStageResult] | type[AnswerStageResult]:
    return EvidenceStageResult if stage == "evidence" else AnswerStageResult


def _stage_schema(row: TaskJudgeInput, *, stage: JudgeStage) -> dict[str, object]:
    """Inline field schemas so applicability is enforced by provider structured output."""

    schema = _result_model(stage).model_json_schema()
    definitions = schema.pop("$defs", None)
    properties = schema.get("properties")
    if not isinstance(definitions, dict) or not isinstance(properties, dict):
        raise ValueError("Unexpected task-judge Pydantic schema structure")
    field_definition = definitions.get("FieldJudgment")
    if not isinstance(field_definition, dict):
        raise ValueError("Task-judge schema has no FieldJudgment definition")
    applicable = set(row.stage_fields(stage))
    evidence_ids = sorted(
        {item.evidence_id for item in row.displayed_evidence()}
        | {edge.fact_id for path in row.graph_paths for edge in path.edges if edge.fact_id}
    )
    path_ids = sorted(path.path_id for path in row.graph_paths)
    stage_fields = EVIDENCE_STAGE_FIELDS if stage == "evidence" else ANSWER_STAGE_FIELDS
    for field in stage_fields:
        specific = deepcopy(field_definition)
        field_properties = specific.get("properties")
        if not isinstance(field_properties, dict):
            raise ValueError("Task-judge field schema has no properties")
        label = field_properties.get("label")
        if not isinstance(label, dict):
            raise ValueError("Task-judge field schema has no label property")
        if field in applicable:
            label["enum"] = ["yes", "partial", "no", "unjudgeable"]
            for identifier_field, allowed in (
                ("evidence_ids", evidence_ids),
                ("path_ids", path_ids),
            ):
                identifier_schema = field_properties.get(identifier_field)
                if not isinstance(identifier_schema, dict):
                    continue
                identifier_schema["uniqueItems"] = True
                if allowed:
                    items = identifier_schema.get("items")
                    if isinstance(items, dict):
                        items["enum"] = allowed
                else:
                    identifier_schema["maxItems"] = 0
        else:
            label["enum"] = ["not_applicable"]
            for identifier_field in ("evidence_ids", "path_ids"):
                identifier_schema = field_properties.get(identifier_field)
                if isinstance(identifier_schema, dict):
                    identifier_schema["maxItems"] = 0
        properties[field] = specific
    _remove_titles(schema)
    return schema


def _parse_structured_result(
    body: dict[str, object],
    *,
    result_model: type[EvidenceStageResult] | type[AnswerStageResult],
) -> EvidenceStageResult | AnswerStageResult:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Task-judge response contains no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Task-judge response contains no structured text")
    return result_model.model_validate_json(message["content"])


def _mistral_payload(
    *,
    model: str,
    prompt: str,
    rendered_input: str,
    schema: dict[str, object],
    schema_name: str,
    random_seed: int,
    max_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": rendered_input},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        },
        "max_tokens": max_tokens,
        "temperature": 0,
        "random_seed": random_seed,
    }


def _parse_mistral_usage(body: dict[str, object]) -> dict[str, int]:
    raw = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {
        "input_tokens": int(raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("completion_tokens") or 0),
        "reasoning_tokens": 0,
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def _rate_limit_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name.lower(): value
        for name, value in response.headers.items()
        if "ratelimit" in name.lower() or name.lower() == "retry-after"
    }


def _assert_minimum_rate_limit(response: httpx.Response) -> None:
    raw_limit = response.headers.get("x-ratelimit-limit-req-minute")
    if raw_limit is None:
        return
    try:
        limit = int(float(raw_limit))
    except ValueError as exc:
        raise RuntimeError(
            f"Mistral returned an invalid request-rate header: {raw_limit!r}"
        ) from exc
    if limit < MINIMUM_ACCEPTABLE_RPM:
        raise RuntimeError(
            "Mistral account request limit fell below the pre-run gate: "
            f"{limit} RPM < {MINIMUM_ACCEPTABLE_RPM} RPM. Aborting the experiment."
        )


def _write_cache(path: Path, record: StageCacheRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        orjson.dumps(
            record.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
    )
    temporary.replace(path)
