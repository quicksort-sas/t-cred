from __future__ import annotations

import asyncio
import hashlib
import os
import random
from pathlib import Path
from typing import Literal

import httpx
import orjson
from dotenv import load_dotenv
from pydantic import ValidationError

from tcred.metrics.models import ClaimJudgeResult, JudgeCacheRecord, MetricInput

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "metrics" / "rag_claim_judge_v1.md"
JUDGE_SCHEMA_NAME = "tcred_rag_claim_judge_v1"
JUDGE_CONTRACT_VERSION = "5"
JudgeProvider = Literal["openai", "anthropic", "mistral", "groq"]


async def score_with_claim_judge(
    rows: list[MetricInput],
    *,
    cache_dir: Path,
    provider: JudgeProvider,
    model: str,
    concurrency: int = 12,
    requests_per_second: float | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, JudgeCacheRecord]:
    """Score inputs with a resumable, schema-constrained OpenAI judge."""
    load_dotenv()
    key = _provider_key(provider)
    if not key:
        raise RuntimeError(f"No API key is configured for the {provider} metric judge")
    prompt, schema, prompt_sha256 = _judge_contract()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[str, JudgeCacheRecord] = {}
    pending: list[tuple[MetricInput, str, Path]] = []
    for row in rows:
        input_sha256 = _input_sha256(row)
        path = _cache_path(cache_dir, row.metric_id)
        record = _valid_cache_record(
            path,
            metric_id=row.metric_id,
            input_sha256=input_sha256,
            prompt_sha256=prompt_sha256,
            provider=provider,
            model=model,
        )
        if record is not None:
            cached[row.metric_id] = record
        else:
            pending.append((row, input_sha256, path))

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

        async def run_one(item: tuple[MetricInput, str, Path]) -> tuple[str, JudgeCacheRecord]:
            nonlocal completed
            row, input_sha256, path = item
            async with semaphore:
                record = await _request_judgment(
                    client,
                    row=row,
                    input_sha256=input_sha256,
                    prompt=prompt,
                    prompt_sha256=prompt_sha256,
                    provider=provider,
                    model=model,
                    rate_limiter=rate_limiter,
                )
                _atomic_write(path, record)
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"claim judge: {completed}/{len(pending)} new; {len(cached)} cached",
                        flush=True,
                    )
                return row.metric_id, record

        for metric_id, record in await asyncio.gather(*(run_one(item) for item in pending)):
            cached[metric_id] = record
    return cached


async def _request_judgment(
    client: httpx.AsyncClient,
    *,
    row: MetricInput,
    input_sha256: str,
    prompt: str,
    prompt_sha256: str,
    provider: JudgeProvider,
    model: str,
    rate_limiter: _AsyncStartRateLimiter,
) -> JudgeCacheRecord:
    schema = ClaimJudgeResult.model_json_schema()
    _remove_titles(schema)
    payload = _request_payload(
        provider=provider,
        model=model,
        prompt=prompt,
        rendered_input=_render_input(row),
        schema=schema,
    )
    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            await rate_limiter.wait()
            response = await client.post(_provider_endpoint(provider), json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                if "credit_balance" in response.text or "insufficient_quota" in response.text:
                    raise RuntimeError(f"{provider} metric-judge account has insufficient credit")
                raise _RetryableJudgeError(
                    f"{provider} judge returned {response.status_code}: {response.text[:500]}",
                    retry_after=_retry_after_seconds(response),
                )
            response.raise_for_status()
            body = response.json()
            result = _parse_result(body, provider=provider)
            _validate_indices(result, row)
            usage = _parse_usage(body, provider=provider)
            return JudgeCacheRecord(
                metric_id=row.metric_id,
                input_sha256=input_sha256,
                prompt_sha256=prompt_sha256,
                provider=provider,
                model=model,
                response_id=str(body.get("id") or ""),
                usage=usage,
                result=result,
            )
        except (
            httpx.HTTPError,
            ValidationError,
            ValueError,
            KeyError,
            _RetryableJudgeError,
        ) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise RuntimeError(
                    f"Non-retryable {provider} judge error: {exc.response.text[:1000]}"
                ) from exc
            if attempt == 8:
                break
            retry_after = (
                last_error.retry_after if isinstance(last_error, _RetryableJudgeError) else 0
            )
            await asyncio.sleep(max(retry_after, min(2**attempt + random.random(), 60.0)))
    raise RuntimeError(f"Claim judge failed for {row.metric_id} after retries") from last_error


def _parse_result(body: dict[str, object], *, provider: JudgeProvider) -> ClaimJudgeResult:
    if provider == "anthropic":
        content = body.get("content")
        if not isinstance(content, list):
            raise ValueError("Anthropic judge response contains no content")
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == JUDGE_SCHEMA_NAME
                and isinstance(block.get("input"), dict)
            ):
                return ClaimJudgeResult.model_validate(block["input"])
        raise ValueError("Anthropic judge response contains no required tool call")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Judge response contains no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Judge response contains no structured text")
    return ClaimJudgeResult.model_validate_json(message["content"])


def _request_payload(
    *,
    provider: JudgeProvider,
    model: str,
    prompt: str,
    rendered_input: str,
    schema: dict[str, object],
) -> dict[str, object]:
    if provider == "anthropic":
        return {
            "model": model,
            "max_tokens": 2400,
            "temperature": 0,
            "system": prompt,
            "messages": [{"role": "user", "content": rendered_input}],
            "tools": [
                {
                    "name": JUDGE_SCHEMA_NAME,
                    "description": "Return the complete claim-level evaluation.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": JUDGE_SCHEMA_NAME},
        }
    if provider == "mistral":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": rendered_input},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": JUDGE_SCHEMA_NAME, "schema": schema},
            },
            "max_tokens": 2400,
            "temperature": 0,
            "random_seed": 20260813,
        }
    if provider == "groq":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": rendered_input},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": JUDGE_SCHEMA_NAME,
                    "strict": True,
                    "schema": schema,
                },
            },
            "max_completion_tokens": 2400,
            "reasoning_effort": "low",
            "temperature": 0,
            "seed": 20260813,
        }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": rendered_input},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": JUDGE_SCHEMA_NAME, "strict": True, "schema": schema},
        },
        "max_completion_tokens": 1600,
        "reasoning_effort": "low",
        "store": False,
    }


def _parse_usage(body: dict[str, object], *, provider: JudgeProvider) -> dict[str, int]:
    raw = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    if provider == "anthropic":
        input_tokens = int(raw.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "total_tokens": input_tokens + output_tokens,
        }
    details = (
        raw.get("completion_tokens_details")
        if isinstance(raw.get("completion_tokens_details"), dict)
        else {}
    )
    return {
        "input_tokens": int(raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def _provider_key(provider: JudgeProvider) -> str | None:
    if provider == "anthropic":
        return os.getenv("LLM_ANTHROPIC_API_KEY")
    if provider == "mistral":
        return os.getenv("LLM_MISTRAL_API_KEY")
    if provider == "groq":
        return os.getenv("LLM_GROQ_API_KEY")
    return os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def _provider_base_url(provider: JudgeProvider) -> str:
    if provider == "anthropic":
        return "https://api.anthropic.com"
    if provider == "mistral":
        return "https://api.mistral.ai/v1"
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


def _provider_headers(provider: JudgeProvider, key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
    return {"Authorization": f"Bearer {key}"}


def _provider_endpoint(provider: JudgeProvider) -> str:
    return "/v1/messages" if provider == "anthropic" else "/chat/completions"


def _validate_indices(result: ClaimJudgeResult, row: MetricInput) -> None:
    for claim in result.candidate_claims:
        _indices_in_range(
            claim.retrieved_support_indices,
            upper=len(row.retrieved_evidence),
            name="retrieved",
        )
        _indices_in_range(
            claim.cited_support_indices,
            upper=len(row.cited_evidence),
            name="cited",
        )
    for claim in result.reference_claims:
        _indices_in_range(
            claim.retrieved_support_indices,
            upper=len(row.retrieved_evidence),
            name="retrieved",
        )


def _indices_in_range(values: list[int], *, upper: int, name: str) -> None:
    if len(values) != len(set(values)) or any(value < 1 or value > upper for value in values):
        raise ValueError(f"Judge returned invalid {name} evidence indices: {values}")


def _render_input(row: MetricInput) -> str:
    retrieved = _evidence_block(row.retrieved_evidence, prefix="R")
    cited = _evidence_block(row.cited_evidence, prefix="C")
    return (
        "<question>\n"
        f"{row.question}\n"
        "</question>\n\n"
        "<reference_answer>\n"
        f"{row.reference_answer}\n"
        "</reference_answer>\n\n"
        "<candidate_answer>\n"
        f"{row.candidate_answer}\n"
        "</candidate_answer>\n\n"
        "<retrieved_evidence>\n"
        f"{retrieved}\n"
        "</retrieved_evidence>\n\n"
        "<cited_evidence>\n"
        f"{cited}\n"
        "</cited_evidence>"
    )


def _evidence_block(evidence: list[object], *, prefix: str) -> str:
    if not evidence:
        return "(none)"
    return "\n".join(f"[{prefix}{index}] {item.text}" for index, item in enumerate(evidence, 1))


def _input_sha256(row: MetricInput) -> str:
    content = {
        "question": row.question,
        "reference_answer": row.reference_answer,
        "candidate_answer": row.candidate_answer,
        "retrieved_evidence": [item.model_dump(mode="json") for item in row.retrieved_evidence],
        "cited_evidence": [item.model_dump(mode="json") for item in row.cited_evidence],
    }
    return hashlib.sha256(orjson.dumps(content, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _cache_path(cache_dir: Path, metric_id: str) -> Path:
    digest = hashlib.sha256(metric_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _valid_cache_record(
    path: Path,
    *,
    metric_id: str,
    input_sha256: str,
    prompt_sha256: str,
    provider: JudgeProvider,
    model: str,
) -> JudgeCacheRecord | None:
    if not path.is_file():
        return None
    try:
        record = JudgeCacheRecord.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError):
        return None
    if (
        record.metric_id != metric_id
        or record.input_sha256 != input_sha256
        or record.prompt_sha256 != prompt_sha256
        or record.provider != provider
        or record.model != model
    ):
        return None
    return record


def _atomic_write(path: Path, record: JudgeCacheRecord) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        orjson.dumps(
            record.model_dump(mode="json"), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        )
    )
    temporary.replace(path)


def _remove_titles(value: object) -> None:
    if isinstance(value, dict):
        value.pop("title", None)
        for child in value.values():
            _remove_titles(child)
    elif isinstance(value, list):
        for child in value:
            _remove_titles(child)


def _retry_after_seconds(response: httpx.Response) -> float:
    value = response.headers.get("retry-after", "").strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


class _AsyncStartRateLimiter:
    def __init__(self, requests_per_second: float | None) -> None:
        if requests_per_second is not None and requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive when provided")
        self._interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            delay = max(0.0, self._next_start - now)
            if delay:
                await asyncio.sleep(delay)
                now = loop.time()
            self._next_start = max(now, self._next_start) + self._interval


class _RetryableJudgeError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _judge_contract() -> tuple[str, dict[str, object], str]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    schema = ClaimJudgeResult.model_json_schema()
    _remove_titles(schema)
    contract = (
        JUDGE_CONTRACT_VERSION.encode()
        + b"\0"
        + prompt.encode("utf-8")
        + b"\0"
        + orjson.dumps(
            schema,
            option=orjson.OPT_SORT_KEYS,
        )
    )
    return prompt, schema, hashlib.sha256(contract).hexdigest()
