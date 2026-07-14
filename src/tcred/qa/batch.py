from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import orjson

from tcred.dataset.io import read_jsonl, write_json_atomic, write_jsonl_atomic
from tcred.llm.batch_jobs import BatchJobClient
from tcred.qa.checkpoint import (
    build_checkpoint_metadata,
    file_sha256,
    write_checkpoint_metadata,
)
from tcred.qa.corpus import RuntimeCorpus, dataset_content_hash, load_runtime_questions
from tcred.qa.diagnostics import write_run_diagnostics
from tcred.qa.embeddings import EmbeddingClient, load_or_create_embeddings
from tcred.qa.generation import (
    PROMPT_VERSION,
    answer_prompt,
    answer_request_payload,
    canonicalize_inline_citations,
    merge_inline_citation_handles,
    parse_generation_body,
)
from tcred.qa.graph_retrieval import GraphRetriever
from tcred.qa.models import (
    FamilyRunSummary,
    PendingSystemOutput,
    QARunConfig,
    QARunManifest,
    QASystemName,
    SystemOutput,
)
from tcred.qa.pipeline import (
    preflight_qa_config,
    retrieve_for_question,
    system_output_id,
)
from tcred.qa.retrieval import HybridRetriever
from tcred.qa.temporal import TemporalRanker

_ACTIVE_STATUSES = {"queued", "running", "validating", "in_progress", "finalizing"}
_SUCCESS_STATUSES = {"success", "succeeded", "completed"}


async def run_qa_systems_batch(
    config: QARunConfig,
    *,
    poll_seconds: float = 10.0,
) -> QARunManifest:
    preflight_qa_config(config)
    if config.generator_provider != "mistral":
        raise ValueError(
            "The implemented QA batch path currently requires generator_provider=mistral"
        )
    config.output_root.mkdir(parents=True, exist_ok=True)
    state_path = config.output_root / "batch_state.json"
    if state_path.exists() and config.resume and not config.overwrite:
        state = orjson.loads(state_path.read_bytes())
        manifest_path = config.output_root / "run_manifest.json"
        if state.get("status") == "imported" and manifest_path.exists():
            return QARunManifest.model_validate(orjson.loads(manifest_path.read_bytes()))
    else:
        state = await _prepare_and_submit(config)
        write_json_atomic(state_path, state)

    client = BatchJobClient(provider="mistral")
    while True:
        status = await client.retrieve(str(state["batch_id"]))
        normalized = str(status.status or "").casefold()
        state["status"] = status.status
        state["output_file_id"] = status.output_file_id
        state["error_file_id"] = status.error_file_id
        state["last_checked_at"] = datetime.now(UTC).isoformat()
        state["request_counts"] = status.raw_response.get("request_counts")
        write_json_atomic(state_path, state)
        if normalized in _SUCCESS_STATUSES:
            break
        if normalized not in _ACTIVE_STATUSES:
            raise RuntimeError(
                f"Mistral batch {state['batch_id']} ended with status {status.status}: "
                f"{status.raw_response}"
            )
        await asyncio.sleep(poll_seconds)

    if not state.get("output_file_id"):
        raise RuntimeError("Completed Mistral batch did not expose an output file")
    batch_dir = config.output_root / "batch"
    result_path = batch_dir / "results.jsonl"
    await client.download_results(
        output_path=result_path,
        file_id=str(state["output_file_id"]),
    )
    if state.get("error_file_id"):
        await client.download_results(
            output_path=batch_dir / "errors.jsonl",
            file_id=str(state["error_file_id"]),
        )
    manifest = _import_batch_results(
        config=config,
        state=state,
        result_path=result_path,
    )
    state["status"] = "imported"
    state["imported_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(state_path, state)
    return manifest


async def _prepare_and_submit(config: QARunConfig) -> dict[str, object]:
    batch_dir = config.output_root / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    request_path = batch_dir / "requests.jsonl"
    pending_path = batch_dir / "pending_outputs.jsonl"
    run_id = f"qa_batch_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    request_rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, object]] = []
    dataset_hashes: dict[str, str] = {}

    async with EmbeddingClient(
        provider=config.embedding_provider,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        timeout_seconds=config.request_timeout_seconds,
    ) as embedding_client:
        for family in config.families:
            dataset_dir = config.dataset_root / family
            corpus = RuntimeCorpus.from_dataset_dir(dataset_dir)
            questions = load_runtime_questions(
                dataset_dir,
                splits=config.splits,
                limit=config.limit_per_family,
            )
            dataset_hashes[family] = dataset_content_hash(dataset_dir)
            fact_embeddings = await load_or_create_embeddings(
                client=embedding_client,
                ids=[document.fact.fact_id for document in corpus.documents],
                texts=[document.semantic_text for document in corpus.documents],
                cache_dir=config.cache_dir,
                namespace=f"{family}.facts",
            )
            question_embeddings = await load_or_create_embeddings(
                client=embedding_client,
                ids=[question.qid for question in questions],
                texts=[question.question for question in questions],
                cache_dir=config.cache_dir,
                namespace=f"{family}.questions",
            )
            hybrid = HybridRetriever(corpus=corpus, fact_embeddings=fact_embeddings)
            temporal_ranker = TemporalRanker(corpus)
            graph = GraphRetriever(corpus)
            for system_value in config.systems:
                system = QASystemName(system_value)
                for index, question in enumerate(questions):
                    retrieval, context = retrieve_for_question(
                        config=config,
                        question=question,
                        query_embedding=question_embeddings[index],
                        corpus=corpus,
                        hybrid=hybrid,
                        temporal_ranker=temporal_ranker,
                        graph=graph,
                        system_name=system,
                    )
                    output_id = system_output_id(family, system, question.qid)
                    user_prompt, prompt_hash = answer_prompt(
                        question=question.question,
                        context=context,
                    )
                    request_rows.append(
                        {
                            "custom_id": output_id,
                            "body": answer_request_payload(
                                provider=config.generator_provider,
                                model=config.generator_model,
                                reasoning_effort=config.reasoning_effort,
                                seed=config.seed,
                                user_prompt=user_prompt,
                            ),
                        }
                    )
                    pending_rows.append(
                        PendingSystemOutput(
                            output_id=output_id,
                            run_id=run_id,
                            dataset_family=family,
                            qid=question.qid,
                            scenario_id=question.scenario_id,
                            system_name=system,
                            retrieval=retrieval,
                            generator_provider=config.generator_provider,
                            generator_model=config.generator_model,
                            reasoning_effort=config.reasoning_effort,
                            prompt_version=PROMPT_VERSION,
                            prompt_sha256=prompt_hash,
                        ).model_dump(mode="json")
                    )

    write_jsonl_atomic(request_path, request_rows)
    write_jsonl_atomic(pending_path, pending_rows)
    client = BatchJobClient(provider="mistral")
    try:
        submission = await client.submit(
            request_file=request_path,
            model=config.generator_model,
            endpoint="/v1/chat/completions",
            timeout_hours=24,
            metadata={"project": "tcred", "run_id": run_id},
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 402:
            raise RuntimeError(
                "Mistral Batch requires billing access for this account; use "
                "`tcred run-qa-systems` with the same model instead."
            ) from exc
        raise
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "dataset_hashes": dataset_hashes,
        "request_path": str(request_path),
        "pending_path": str(pending_path),
        "request_count": len(request_rows),
        "batch_id": submission.batch_id,
        "uploaded_file_id": submission.uploaded_file_id,
        "status": submission.status,
    }


def _import_batch_results(
    *,
    config: QARunConfig,
    state: dict[str, object],
    result_path: Path,
) -> QARunManifest:
    pending = {
        row.output_id: row
        for row in (
            PendingSystemOutput.model_validate(raw)
            for raw in read_jsonl(Path(str(state["pending_path"])))
        )
    }
    result_by_id = {str(row.get("custom_id", "")): row for row in read_jsonl(result_path)}
    outputs: list[SystemOutput] = []
    for output_id, record in pending.items():
        result = result_by_id.get(output_id)
        try:
            body = _response_body(result)
            payload, usage = parse_generation_body(body)
            supplied_handles = record.retrieval.evidence_handle_map
            structured_citations = list(
                dict.fromkeys(item.strip() for item in payload.cited_evidence_ids if item.strip())
            )
            cited = merge_inline_citation_handles(payload.answer_text, structured_citations)
            outputs.append(
                SystemOutput(
                    **record.model_dump(mode="python"),
                    status="success",
                    answer_text=canonicalize_inline_citations(
                        payload.answer_text.strip(),
                        supplied_handles=supplied_handles,
                        cited_handles=cited,
                    ),
                    cited_evidence_ids=cited,
                    resolved_cited_evidence_ids=[
                        supplied_handles[item] for item in cited if item in supplied_handles
                    ],
                    unresolved_citation_ids=[
                        item for item in cited if item not in supplied_handles
                    ],
                    usage=usage,
                )
            )
        except Exception as exc:  # noqa: BLE001 - retain every failed batch row for audit
            outputs.append(
                SystemOutput(
                    **record.model_dump(mode="python"),
                    status="error",
                    answer_text="",
                    cited_evidence_ids=[],
                    resolved_cited_evidence_ids=[],
                    unresolved_citation_ids=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    grouped: defaultdict[tuple[str, str], list[SystemOutput]] = defaultdict(list)
    for output in outputs:
        grouped[(output.dataset_family, str(output.system_name))].append(output)
    summaries: list[FamilyRunSummary] = []
    for (family, system), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda output: output.qid)
        output_path = config.output_root / family / f"{system}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(output_path, [output.model_dump(mode="json") for output in ordered])
        selected_questions = load_runtime_questions(
            config.dataset_root / family,
            splits=config.splits,
            limit=config.limit_per_family,
        )
        output_qids = {output.qid for output in ordered}
        metadata = build_checkpoint_metadata(
            config=config,
            family=family,
            system_name=QASystemName(system),
            dataset_sha256=str(state["dataset_hashes"][family]),
            questions=[question for question in selected_questions if question.qid in output_qids],
        )
        write_checkpoint_metadata(output_path, metadata, record_count=len(ordered))
        successful = [output for output in ordered if output.status == "success"]
        summaries.append(
            FamilyRunSummary(
                family=family,
                system_name=QASystemName(system),
                output_path=output_path,
                requested=len(ordered),
                succeeded=len(successful),
                failed=len(ordered) - len(successful),
                resumed=0,
                input_tokens=sum(output.usage.input_tokens for output in successful),
                output_tokens=sum(output.usage.output_tokens for output in successful),
                output_sha256=file_sha256(output_path),
            )
        )

    diagnostics_path = write_run_diagnostics(
        dataset_root=config.dataset_root,
        output_root=config.output_root,
    )
    manifest = QARunManifest(
        run_id=str(state["run_id"]),
        config=config,
        dataset_hashes={str(key): str(value) for key, value in state["dataset_hashes"].items()},
        summaries=summaries,
        diagnostics_path=diagnostics_path,
    )
    write_json_atomic(
        config.output_root / "run_manifest.json",
        manifest.model_dump(mode="json"),
    )
    return manifest


def _response_body(result: dict[str, object] | None) -> dict[str, object]:
    if result is None:
        raise ValueError("Batch result is missing")
    if result.get("error"):
        raise ValueError(f"Batch request failed: {result['error']}")
    response = result.get("response")
    if isinstance(response, dict):
        status_code = int(response.get("status_code") or 200)
        if status_code >= 400:
            raise ValueError(f"Batch request returned HTTP {status_code}: {response}")
        body = response.get("body")
        if isinstance(body, dict):
            return body
        if "choices" in response:
            return response
    body = result.get("body")
    if isinstance(body, dict):
        return body
    raise ValueError(f"Unrecognized batch result shape: {result}")
