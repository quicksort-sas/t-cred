from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import orjson

from tcred.llm.batch_jobs import BatchJobClient
from tcred.metrics.task_judge import (
    CONTRACT_VERSION,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REQUESTS_PER_SECOND,
    _mistral_payload,
    _parse_mistral_usage,
    _parse_structured_result,
    _request_spec,
    _required_stages,
    _result_model,
    _support_pointer_warnings,
    _valid_cache_record,
    _validate_stage_result,
    _write_cache,
    score_with_task_judge,
)
from tcred.metrics.task_judge_models import (
    PromptVariant,
    StageCacheRecord,
    TaskJudgeInput,
    TaskJudgeRecord,
)

_TERMINAL_SUCCESS = {"SUCCESS", "COMPLETED"}
_TERMINAL_FAILURE = {"FAILED", "TIMEOUT_EXCEEDED", "CANCELLED", "EXPIRED"}
_SCHEMA_NAMES = {
    "evidence": "tcred_evidence_stage_judge_v1",
    "answer": "tcred_answer_stage_judge_v1",
}


async def score_with_task_judge_batch(
    rows: list[TaskJudgeInput],
    *,
    cache_dir: Path,
    batch_dir: Path,
    prompt_variant: PromptVariant,
    model: str,
    concurrency: int = 4,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    random_seed: int = DEFAULT_RANDOM_SEED,
    poll_seconds: float = 20.0,
) -> dict[str, TaskJudgeRecord]:
    """Run identical task-judge requests through Mistral Batch, then retry invalid rows directly."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, StageCacheRecord] = {}
    pending = []
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
                provider="mistral",
                model=model,
                prompt_variant=prompt_variant,
                random_seed=random_seed,
            )
            if record is None:
                pending.append(spec)
            else:
                cached[spec.judgment_id] = record

    if pending:
        request_rows, mapping_rows = _batch_rows(
            pending,
            model=model,
            random_seed=random_seed,
        )
        request_bytes = _jsonl_bytes(request_rows)
        mapping_bytes = _jsonl_bytes(mapping_rows)
        signature = _batch_signature(
            request_bytes=request_bytes,
            mapping_bytes=mapping_bytes,
            model=model,
            prompt_variant=prompt_variant,
            random_seed=random_seed,
        )
        run_dir = batch_dir / signature[:20]
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / "requests.jsonl"
        mapping_path = run_dir / "mapping.jsonl"
        state_path = run_dir / "state.json"
        results_path = run_dir / "results.jsonl"
        state = _read_state(state_path)
        if state and state.get("batch_signature") != signature:
            raise RuntimeError("Task-judge batch state signature mismatch")
        if not state:
            request_path.write_bytes(request_bytes)
            mapping_path.write_bytes(mapping_bytes)
            client = BatchJobClient(provider="mistral", timeout_seconds=180)
            submission = await client.submit(
                request_file=request_path,
                model=model,
                endpoint="/v1/chat/completions",
                timeout_hours=24,
                metadata={"project": "tcred", "purpose": "task-judge"},
            )
            state = {
                "schema_version": "1.0",
                "batch_signature": signature,
                "model": model,
                "prompt_variant": prompt_variant,
                "contract_version": CONTRACT_VERSION,
                "random_seed": random_seed,
                "request_count": len(request_rows),
                "batch_id": submission.batch_id,
                "uploaded_file_id": submission.uploaded_file_id,
                "status": submission.status,
                "submitted_at": datetime.now(UTC).isoformat(),
            }
            _write_json(state_path, state)
            print(
                f"task judge batch: submitted {len(request_rows)} stages as "
                f"{submission.batch_id}",
                flush=True,
            )

        client = BatchJobClient(provider="mistral", timeout_seconds=180)
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
            complete, total = _request_progress(
                status.raw_response,
                default_total=len(request_rows),
            )
            print(f"task judge batch: {normalized or 'UNKNOWN'} {complete}/{total}", flush=True)
            if normalized in _TERMINAL_SUCCESS:
                if not status.output_file_id:
                    raise RuntimeError("Successful Mistral task-judge batch has no output file")
                await client.download_results(
                    output_path=results_path,
                    file_id=status.output_file_id,
                )
                break
            if normalized in _TERMINAL_FAILURE:
                raise RuntimeError(
                    f"Mistral task-judge batch ended with status {normalized}: "
                    f"{status.raw_response}"
                )
            await asyncio.sleep(poll_seconds)

        imported, failures = _import_results(
            results_path,
            specs=pending,
            mapping_rows=mapping_rows,
            model=model,
            prompt_variant=prompt_variant,
            random_seed=random_seed,
        )
        cached.update(imported)
        state.update(
            status="IMPORTED" if not failures else "IMPORTED_WITH_DIRECT_RETRY",
            imported_at=datetime.now(UTC).isoformat(),
            imported_count=len(imported),
            direct_retry_count=len(failures),
            invalid_rows=failures[:50],
        )
        _write_json(state_path, state)
        if failures:
            print(
                f"task judge batch: {len(failures)} invalid stages will be retried directly",
                flush=True,
            )

    # This call is cache-only after a clean batch and direct-retries only invalid/missing stages.
    records = await score_with_task_judge(
        rows,
        cache_dir=cache_dir,
        prompt_variant=prompt_variant,
        provider="mistral",
        model=model,
        concurrency=concurrency,
        requests_per_second=requests_per_second,
        random_seed=random_seed,
    )
    return records


def _batch_rows(
    specs: list,
    *,
    model: str,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    requests: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        custom_id = (
            f"task-{index:06d}-"
            f"{hashlib.sha256(spec.judgment_id.encode()).hexdigest()[:12]}"
        )
        body = _mistral_payload(
            model=model,
            prompt=spec.prompt,
            rendered_input=spec.rendered_input,
            schema=spec.schema,
            schema_name=_SCHEMA_NAMES[spec.stage],
            random_seed=random_seed,
            max_tokens=2200 if spec.stage == "evidence" else 1200,
        )
        body.pop("model", None)
        requests.append({"custom_id": custom_id, "body": body})
        mappings.append(
            {
                "custom_id": custom_id,
                "judgment_id": spec.judgment_id,
                "metric_id": spec.row.metric_id,
                "stage": spec.stage,
                "input_sha256": spec.input_sha256,
                "prompt_sha256": spec.prompt_sha256,
                "schema_sha256": spec.schema_sha256,
            }
        )
    return requests, mappings


def _import_results(
    path: Path,
    *,
    specs: list,
    mapping_rows: list[dict[str, object]],
    model: str,
    prompt_variant: PromptVariant,
    random_seed: int,
) -> tuple[dict[str, StageCacheRecord], list[str]]:
    by_custom_id = {str(row["custom_id"]): row for row in _read_jsonl(path)}
    spec_by_judgment = {spec.judgment_id: spec for spec in specs}
    imported: dict[str, StageCacheRecord] = {}
    failures: list[str] = []
    for mapping in mapping_rows:
        custom_id = str(mapping["custom_id"])
        judgment_id = str(mapping["judgment_id"])
        spec = spec_by_judgment[judgment_id]
        try:
            body = _response_body(by_custom_id.get(custom_id))
            result = _parse_structured_result(body, result_model=_result_model(spec.stage))
            _validate_stage_result(result, row=spec.row, stage=spec.stage)
            record = StageCacheRecord(
                judgment_id=judgment_id,
                metric_id=spec.row.metric_id,
                stage=spec.stage,
                input_sha256=spec.input_sha256,
                prompt_sha256=spec.prompt_sha256,
                schema_sha256=spec.schema_sha256,
                prompt_variant=prompt_variant,
                contract_version=CONTRACT_VERSION,
                provider="mistral",
                model=model,
                random_seed=random_seed,
                response_id=str(body.get("id") or custom_id),
                attempts=1,
                rate_limit_headers={},
                support_pointer_warnings=_support_pointer_warnings(
                    result,
                    row=spec.row,
                    stage=spec.stage,
                ),
                usage=_parse_mistral_usage(body),
                result=result,
            )
            _write_cache(spec.cache_path, record)
            imported[judgment_id] = record
        except Exception as exc:  # noqa: BLE001 - preserve all valid rows, retry failures directly
            failures.append(f"{judgment_id}: {type(exc).__name__}: {exc}")
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
    prompt_variant: str,
    random_seed: int,
) -> str:
    digest = hashlib.sha256()
    for value in (
        request_bytes,
        mapping_bytes,
        model.encode(),
        prompt_variant.encode(),
        CONTRACT_VERSION.encode(),
        str(random_seed).encode(),
    ):
        digest.update(value)
        digest.update(b"\0")
    return digest.hexdigest()


def _request_progress(raw: dict[str, object], *, default_total: int) -> tuple[int, int]:
    counts = raw.get("request_counts")
    if isinstance(counts, dict):
        return int(counts.get("completed") or 0), int(counts.get("total") or default_total)
    completed = int(raw.get("completed_requests") or raw.get("succeeded_requests") or 0)
    failed = int(raw.get("failed_requests") or 0)
    return completed + failed, int(raw.get("total_requests") or default_total)


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
