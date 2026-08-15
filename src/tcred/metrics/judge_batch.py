from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import orjson

from tcred.llm.batch_jobs import BatchJobClient
from tcred.metrics.judge import (
    _atomic_write,
    _cache_path,
    _input_sha256,
    _judge_contract,
    _parse_result,
    _parse_usage,
    _render_input,
    _request_payload,
    _valid_cache_record,
    _validate_indices,
    score_with_claim_judge,
)
from tcred.metrics.models import JudgeCacheRecord, MetricInput

_TERMINAL_SUCCESS = {"SUCCESS", "COMPLETED"}
_TERMINAL_FAILURE = {
    "FAILED",
    "TIMEOUT_EXCEEDED",
    "CANCELLED",
    "EXPIRED",
}
BatchJudgeProvider = Literal["mistral", "groq"]


async def score_with_batch_judge(
    rows: list[MetricInput],
    *,
    cache_dir: Path,
    batch_dir: Path,
    provider: BatchJudgeProvider,
    model: str,
    poll_seconds: float = 20.0,
    fallback_concurrency: int = 4,
    fallback_requests_per_second: float | None = 0.24,
) -> dict[str, JudgeCacheRecord]:
    """Score all uncached rows with one resumable provider-native Batch job."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    prompt, schema, prompt_sha256 = _judge_contract()
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    cached, pending = _partition_cache(
        rows,
        cache_dir=cache_dir,
        prompt_sha256=prompt_sha256,
        provider=provider,
        model=model,
    )
    if not pending:
        return cached

    request_rows, mapping_rows = _batch_rows(
        pending,
        prompt=prompt,
        schema=schema,
        prompt_sha256=prompt_sha256,
        provider=provider,
        model=model,
    )
    request_bytes = _jsonl_bytes(request_rows)
    mapping_bytes = _jsonl_bytes(mapping_rows)
    batch_signature = _batch_signature(
        request_bytes=request_bytes,
        mapping_bytes=mapping_bytes,
        model=model,
        prompt_sha256=prompt_sha256,
        provider=provider,
    )
    run_dir = batch_dir / batch_signature[:20]
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "requests.jsonl"
    mapping_path = run_dir / "mapping.jsonl"
    state_path = run_dir / "state.json"
    results_path = run_dir / "results.jsonl"
    state = _read_state(state_path)
    if state and state.get("batch_signature") != batch_signature:
        raise RuntimeError(
            "Existing metric-judge batch state belongs to a different pending input set; "
            "use a separate batch directory or finish/import that job first"
        )
    if not state:
        request_path.write_bytes(request_bytes)
        mapping_path.write_bytes(mapping_bytes)
        client = BatchJobClient(provider=provider, timeout_seconds=180)
        submission = await client.submit(
            request_file=request_path,
            model=model if provider == "mistral" else None,
            endpoint="/v1/chat/completions",
            timeout_hours=24,
            metadata={"project": "tcred", "purpose": "metric-judge"},
        )
        state = {
            "schema_version": "1.0",
            "batch_signature": batch_signature,
            "provider": provider,
            "model": model,
            "prompt_sha256": prompt_sha256,
            "request_count": len(request_rows),
            "batch_id": submission.batch_id,
            "uploaded_file_id": submission.uploaded_file_id,
            "status": submission.status,
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        _write_json(state_path, state)
        print(
            f"claim judge batch: submitted {len(request_rows)} rows as {submission.batch_id}",
            flush=True,
        )

    client = BatchJobClient(provider=provider, timeout_seconds=180)
    while True:
        status = await client.retrieve(str(state["batch_id"]))
        normalized = str(status.status or "").upper()
        state.update(
            status=status.status,
            output_file_id=status.output_file_id,
            error_file_id=status.error_file_id,
            last_checked_at=datetime.now(UTC).isoformat(),
            raw_status=status.raw_response,
        )
        _write_json(state_path, state)
        raw = status.raw_response
        complete, total = _request_progress(raw, default_total=len(request_rows))
        print(f"claim judge batch: {normalized or 'UNKNOWN'} {complete}/{total}", flush=True)
        if normalized in _TERMINAL_SUCCESS:
            if not status.output_file_id:
                raise RuntimeError(f"Successful {provider} batch has no output file")
            await client.download_results(
                output_path=results_path,
                file_id=status.output_file_id,
            )
            break
        if normalized in _TERMINAL_FAILURE:
            raise RuntimeError(
                f"{provider} metric-judge batch ended with status {normalized}: {raw}"
            )
        await asyncio.sleep(poll_seconds)

    imported, failures = _import_results(
        results_path,
        pending=pending,
        mapping_rows=mapping_rows,
        cache_dir=cache_dir,
        prompt_sha256=prompt_sha256,
        provider=provider,
        model=model,
    )
    cached.update(imported)
    state["status"] = "IMPORTED" if not failures else "IMPORTED_WITH_DIRECT_RETRY"
    state["imported_at"] = datetime.now(UTC).isoformat()
    state["imported_count"] = len(imported)
    state["direct_retry_count"] = len(failures)
    state["invalid_rows"] = failures[:50]
    _write_json(state_path, state)
    if failures:
        print(
            f"claim judge batch: {len(failures)} invalid rows will be retried directly",
            flush=True,
        )
    # Cache-only after a clean batch; direct retries only invalid or missing batch rows.
    return await score_with_claim_judge(
        rows,
        cache_dir=cache_dir,
        provider=provider,
        model=model,
        concurrency=fallback_concurrency,
        requests_per_second=fallback_requests_per_second,
    )


def _partition_cache(
    rows: list[MetricInput],
    *,
    cache_dir: Path,
    prompt_sha256: str,
    provider: BatchJudgeProvider,
    model: str,
) -> tuple[dict[str, JudgeCacheRecord], list[MetricInput]]:
    cached: dict[str, JudgeCacheRecord] = {}
    pending: list[MetricInput] = []
    for row in rows:
        record = _valid_cache_record(
            _cache_path(cache_dir, row.metric_id),
            metric_id=row.metric_id,
            input_sha256=_input_sha256(row),
            prompt_sha256=prompt_sha256,
            provider=provider,
            model=model,
        )
        if record is None:
            pending.append(row)
        else:
            cached[row.metric_id] = record
    return cached, pending


def _batch_rows(
    rows: list[MetricInput],
    *,
    prompt: str,
    schema: dict[str, object],
    prompt_sha256: str,
    provider: BatchJudgeProvider,
    model: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    requests: list[dict[str, object]] = []
    mapping: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        custom_id = f"metric-{index:06d}-{hashlib.sha256(row.metric_id.encode()).hexdigest()[:12]}"
        body = _request_payload(
            provider=provider,
            model=model,
            prompt=prompt,
            rendered_input=_render_input(row),
            schema=schema,
        )
        if provider == "mistral":
            body.pop("model", None)
            requests.append({"custom_id": custom_id, "body": body})
        else:
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        mapping.append(
            {
                "custom_id": custom_id,
                "metric_id": row.metric_id,
                "input_sha256": _input_sha256(row),
                "prompt_sha256": prompt_sha256,
            }
        )
    return requests, mapping


def _import_results(
    path: Path,
    *,
    pending: list[MetricInput],
    mapping_rows: list[dict[str, object]],
    cache_dir: Path,
    prompt_sha256: str,
    provider: BatchJudgeProvider,
    model: str,
) -> tuple[dict[str, JudgeCacheRecord], list[str]]:
    by_custom_id = {str(row["custom_id"]): row for row in _read_jsonl(path)}
    row_by_metric_id = {row.metric_id: row for row in pending}
    imported: dict[str, JudgeCacheRecord] = {}
    failures: list[str] = []
    for mapping in mapping_rows:
        custom_id = str(mapping["custom_id"])
        metric_id = str(mapping["metric_id"])
        result_row = by_custom_id.get(custom_id)
        try:
            body = _response_body(result_row)
            result = _parse_result(body, provider=provider)
            source = row_by_metric_id[metric_id]
            _validate_indices(result, source)
            record = JudgeCacheRecord(
                metric_id=metric_id,
                input_sha256=str(mapping["input_sha256"]),
                prompt_sha256=prompt_sha256,
                provider=provider,
                model=model,
                response_id=str(body.get("id") or custom_id),
                usage=_parse_usage(body, provider=provider),
                result=result,
            )
            _atomic_write(_cache_path(cache_dir, metric_id), record)
            imported[metric_id] = record
        except Exception as exc:  # noqa: BLE001 - aggregate provider row failures
            failures.append(f"{metric_id}: {type(exc).__name__}: {exc}")
    return imported, failures


def _response_body(result: dict[str, object] | None) -> dict[str, object]:
    if result is None:
        raise ValueError("batch result is missing")
    if result.get("error"):
        raise ValueError(f"batch request failed: {result['error']}")
    response = result.get("response")
    if isinstance(response, dict):
        status_code = int(response.get("status_code") or 200)
        if status_code >= 400:
            raise ValueError(f"batch request returned HTTP {status_code}: {response}")
        body = response.get("body")
        if isinstance(body, dict):
            return body
        if "choices" in response:
            return response
    body = result.get("body")
    if isinstance(body, dict):
        return body
    raise ValueError(f"unrecognized batch result shape: {result}")


def _batch_signature(
    *,
    request_bytes: bytes,
    mapping_bytes: bytes,
    model: str,
    prompt_sha256: str,
    provider: BatchJudgeProvider,
) -> str:
    digest = hashlib.sha256()
    for value in (
        request_bytes,
        mapping_bytes,
        provider.encode(),
        model.encode(),
        prompt_sha256.encode(),
    ):
        digest.update(value)
        digest.update(b"\0")
    return digest.hexdigest()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n" for row in rows)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def _read_state(path: Path) -> dict[str, object]:
    return orjson.loads(path.read_bytes()) if path.is_file() else {}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def _request_progress(raw: dict[str, object], *, default_total: int) -> tuple[int, int]:
    counts = raw.get("request_counts")
    if isinstance(counts, dict):
        return int(counts.get("completed") or 0), int(counts.get("total") or default_total)
    return int(raw.get("completed_requests") or 0), int(raw.get("total_requests") or default_total)
