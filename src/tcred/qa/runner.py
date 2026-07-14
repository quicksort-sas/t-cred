from __future__ import annotations

import asyncio
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson

from tcred.dataset.io import read_jsonl, write_jsonl
from tcred.qa.checkpoint import (
    CheckpointMetadata,
    assert_checkpoint_compatible,
    build_checkpoint_metadata,
    checkpoint_integrity_matches,
    checkpoint_metadata_path,
    file_sha256,
    read_checkpoint_metadata,
    write_checkpoint_metadata,
)
from tcred.qa.corpus import (
    RuntimeCorpus,
    RuntimeQuestion,
    dataset_content_hash,
    load_runtime_questions,
)
from tcred.qa.diagnostics import write_run_diagnostics
from tcred.qa.embeddings import EmbeddingClient, load_or_create_embeddings
from tcred.qa.generation import (
    PROMPT_VERSION,
    ChatAnswerClient,
    answer_prompt,
    canonicalize_inline_citations,
    merge_inline_citation_handles,
)
from tcred.qa.graph_retrieval import GraphRetriever
from tcred.qa.models import (
    FamilyRunSummary,
    QARunConfig,
    QARunManifest,
    QASystemName,
    RetrievalResult,
    SystemOutput,
)
from tcred.qa.pipeline import (
    preflight_qa_config,
    retrieve_for_question,
    system_output_id,
)
from tcred.qa.retrieval import HybridRetriever
from tcred.qa.temporal import TemporalRanker

LOGGER = logging.getLogger(__name__)


async def run_qa_systems(
    config: QARunConfig,
    *,
    revalidate_checkpoints: bool = False,
) -> QARunManifest:
    preflight_qa_config(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"qa_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    summaries: list[FamilyRunSummary] = []
    hashes: dict[str, str] = {}

    async with (
        EmbeddingClient(
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            timeout_seconds=config.request_timeout_seconds,
        ) as embedding_client,
        ChatAnswerClient(
            provider=config.generator_provider,
            model=config.generator_model,
            reasoning_effort=config.reasoning_effort,
            timeout_seconds=config.request_timeout_seconds,
            seed=config.seed,
        ) as answer_client,
    ):
        for family in config.families:
            dataset_dir = config.dataset_root / family
            corpus = RuntimeCorpus.from_dataset_dir(dataset_dir)
            questions = load_runtime_questions(
                dataset_dir,
                splits=config.splits,
                limit=config.limit_per_family,
            )
            hashes[family] = dataset_content_hash(dataset_dir)
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
            for system_name in config.systems:
                summary = await _run_family_system(
                    config=config,
                    run_id=run_id,
                    family=family,
                    dataset_sha256=hashes[family],
                    questions=questions,
                    question_embeddings=question_embeddings,
                    corpus=corpus,
                    hybrid=hybrid,
                    temporal_ranker=temporal_ranker,
                    graph=graph,
                    answer_client=answer_client,
                    system_name=QASystemName(system_name),
                    revalidate_checkpoint=revalidate_checkpoints,
                )
                summaries.append(summary)

    diagnostics_path = write_run_diagnostics(
        dataset_root=config.dataset_root,
        output_root=config.output_root,
    )
    manifest = QARunManifest(
        run_id=run_id,
        config=config,
        dataset_hashes=hashes,
        summaries=summaries,
        diagnostics_path=diagnostics_path,
    )
    manifest_path = config.output_root / "run_manifest.json"
    manifest_path.write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    return manifest


async def _run_family_system(
    *,
    config: QARunConfig,
    run_id: str,
    family: str,
    dataset_sha256: str,
    questions: list[RuntimeQuestion],
    question_embeddings: np.ndarray,
    corpus: RuntimeCorpus,
    hybrid: HybridRetriever,
    temporal_ranker: TemporalRanker,
    graph: GraphRetriever,
    answer_client: ChatAnswerClient,
    system_name: QASystemName,
    revalidate_checkpoint: bool,
) -> FamilyRunSummary:
    output_path = config.output_root / family / f"{system_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_checkpoint_metadata(
        config=config,
        family=family,
        system_name=system_name,
        dataset_sha256=dataset_sha256,
        questions=questions,
    )
    expected_output_ids = {
        system_output_id(family, system_name, question.qid) for question in questions
    }
    if config.overwrite:
        _clear_checkpoint(output_path)
    _recover_temporary_checkpoint(
        output_path,
        expected_ids=expected_output_ids,
    )
    existing = _existing_outputs(output_path, config=config)
    actual_metadata = read_checkpoint_metadata(output_path)
    metadata_compatible = True
    if actual_metadata is not None:
        try:
            assert_checkpoint_compatible(actual_metadata, metadata, output_path=output_path)
        except ValueError:
            if not revalidate_checkpoint:
                raise
            metadata_compatible = False
            LOGGER.warning("Revalidating checkpoint provenance for %s", output_path)
    unexpected = set(existing) - expected_output_ids
    if unexpected and revalidate_checkpoint:
        LOGGER.warning(
            "Pruning %d outputs no longer selected by the current split from %s",
            len(unexpected),
            output_path,
        )
        existing = {
            output_id: output
            for output_id, output in existing.items()
            if output_id in expected_output_ids
        }
    integrity_matches = (
        metadata_compatible
        and actual_metadata is not None
        and checkpoint_integrity_matches(
            actual_metadata,
            output_path,
            record_count=len(existing),
        )
    )
    if existing and not integrity_matches:
        _validate_existing_outputs(
            outputs=existing,
            config=config,
            family=family,
            questions=questions,
            question_embeddings=question_embeddings,
            corpus=corpus,
            hybrid=hybrid,
            temporal_ranker=temporal_ranker,
            graph=graph,
            system_name=system_name,
        )
    if not integrity_matches:
        _write_outputs_checkpoint(
            output_path,
            existing,
            questions,
            family,
            system_name,
            metadata,
        )
    completed = {
        output_id: output for output_id, output in existing.items() if output.status == "success"
    }
    missing = [
        (index, question)
        for index, question in enumerate(questions)
        if system_output_id(family, system_name, question.qid) not in completed
    ]
    LOGGER.info(
        "Running %s on %s: %d pending, %d resumed",
        system_name,
        family,
        len(missing),
        len(completed),
    )
    results = dict(completed)
    batch_size = max(config.concurrency, 1)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        generated = await asyncio.gather(
            *[
                _answer_one(
                    config=config,
                    run_id=run_id,
                    family=family,
                    question=question,
                    query_embedding=question_embeddings[index],
                    corpus=corpus,
                    hybrid=hybrid,
                    temporal_ranker=temporal_ranker,
                    graph=graph,
                    answer_client=answer_client,
                    system_name=system_name,
                )
                for index, question in batch
            ]
        )
        results.update({output.output_id: output for output in generated})
        _write_outputs_checkpoint(
            output_path,
            results,
            questions,
            family,
            system_name,
            metadata,
        )
        LOGGER.info(
            "%s/%s: %d/%d generated",
            family,
            system_name,
            min(start + len(batch), len(missing)),
            len(missing),
        )

    ordered = _write_outputs_checkpoint(
        output_path,
        results,
        questions,
        family,
        system_name,
        metadata,
    )
    successful = [output for output in ordered if output.status == "success"]
    return FamilyRunSummary(
        family=family,
        system_name=system_name,
        output_path=output_path,
        requested=len(questions),
        succeeded=len(successful),
        failed=len(ordered) - len(successful),
        resumed=len(completed),
        input_tokens=sum(output.usage.input_tokens for output in successful),
        output_tokens=sum(output.usage.output_tokens for output in successful),
        output_sha256=file_sha256(output_path),
    )


async def _answer_one(
    *,
    config: QARunConfig,
    run_id: str,
    family: str,
    question: RuntimeQuestion,
    query_embedding: np.ndarray,
    corpus: RuntimeCorpus,
    hybrid: HybridRetriever,
    temporal_ranker: TemporalRanker,
    graph: GraphRetriever,
    answer_client: ChatAnswerClient,
    system_name: QASystemName,
) -> SystemOutput:
    retrieval, context = retrieve_for_question(
        config=config,
        question=question,
        query_embedding=query_embedding,
        corpus=corpus,
        hybrid=hybrid,
        temporal_ranker=temporal_ranker,
        graph=graph,
        system_name=system_name,
    )
    output_id = system_output_id(family, system_name, question.qid)
    try:
        generated = await answer_client.answer(question=question.question, context=context)
        supplied_handles = retrieval.evidence_handle_map
        structured_citations = list(
            dict.fromkeys(
                item.strip() for item in generated.payload.cited_evidence_ids if item.strip()
            )
        )
        cited = merge_inline_citation_handles(
            generated.payload.answer_text,
            structured_citations,
        )
        resolved = [supplied_handles[item] for item in cited if item in supplied_handles]
        unresolved = [item for item in cited if item not in supplied_handles]
        answer_text = canonicalize_inline_citations(
            generated.payload.answer_text.strip(),
            supplied_handles=supplied_handles,
            cited_handles=cited,
        )
        return SystemOutput(
            output_id=output_id,
            run_id=run_id,
            dataset_family=family,
            qid=question.qid,
            scenario_id=question.scenario_id,
            system_name=system_name,
            status="success",
            answer_text=answer_text,
            cited_evidence_ids=cited,
            resolved_cited_evidence_ids=resolved,
            unresolved_citation_ids=unresolved,
            retrieval=retrieval,
            generator_provider=config.generator_provider,
            generator_model=config.generator_model,
            reasoning_effort=config.reasoning_effort,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=generated.prompt_sha256,
            usage=generated.usage,
            latency_ms=generated.latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - preserve per-item failures and continue the run
        LOGGER.exception("QA generation failed for %s", output_id)
        return SystemOutput(
            output_id=output_id,
            run_id=run_id,
            dataset_family=family,
            qid=question.qid,
            scenario_id=question.scenario_id,
            system_name=system_name,
            status="error",
            answer_text="",
            cited_evidence_ids=[],
            resolved_cited_evidence_ids=[],
            unresolved_citation_ids=[],
            retrieval=retrieval,
            generator_provider=config.generator_provider,
            generator_model=config.generator_model,
            reasoning_effort=config.reasoning_effort,
            prompt_version=PROMPT_VERSION,
            prompt_sha256="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _existing_outputs(path: Path, *, config: QARunConfig) -> dict[str, SystemOutput]:
    if not path.exists() or config.overwrite:
        return {}
    if not config.resume:
        raise FileExistsError(f"{path} exists; use --resume or --overwrite")
    outputs = [SystemOutput.model_validate(row) for row in read_jsonl(path)]
    by_id = {output.output_id: output for output in outputs}
    if len(by_id) != len(outputs):
        raise ValueError(f"Duplicate output ids in checkpoint: {path}")
    return by_id


def _clear_checkpoint(path: Path) -> None:
    metadata_path = checkpoint_metadata_path(path)
    for candidate in (
        path,
        path.with_suffix(".jsonl.tmp"),
        metadata_path,
        metadata_path.with_suffix(metadata_path.suffix + ".tmp"),
    ):
        candidate.unlink(missing_ok=True)


def _recover_temporary_checkpoint(path: Path, *, expected_ids: set[str]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    if not temporary.exists():
        return
    temporary_outputs = _read_checkpoint_rows(temporary, expected_ids=expected_ids)
    if not path.exists():
        temporary.replace(path)
        LOGGER.warning("Recovered checkpoint from %s", temporary)
        return

    current_outputs = _read_checkpoint_rows(path, expected_ids=expected_ids)
    temporary_ids = set(temporary_outputs)
    current_ids = set(current_outputs)
    if current_ids <= temporary_ids and all(
        current_outputs[output_id] == temporary_outputs[output_id] for output_id in current_ids
    ):
        temporary.replace(path)
        LOGGER.warning(
            "Recovered %d additional rows from %s",
            len(temporary_ids - current_ids),
            temporary,
        )
        return
    if temporary_ids < current_ids and all(
        temporary_outputs[output_id] == current_outputs[output_id] for output_id in temporary_ids
    ):
        temporary.unlink()
        LOGGER.warning("Removed an older temporary checkpoint: %s", temporary)
        return
    raise ValueError(
        f"Temporary checkpoint conflicts with the committed output: {temporary}. "
        "Inspect the files or use --overwrite."
    )


def _read_checkpoint_rows(path: Path, *, expected_ids: set[str]) -> dict[str, SystemOutput]:
    outputs = [SystemOutput.model_validate(row) for row in read_jsonl(path)]
    by_id = {output.output_id: output for output in outputs}
    if len(by_id) != len(outputs):
        raise ValueError(f"Duplicate output ids in checkpoint: {path}")
    unexpected = set(by_id) - expected_ids
    if unexpected:
        raise ValueError(f"Unexpected output ids in checkpoint {path}: {sorted(unexpected)[:5]}")
    return by_id


def _validate_existing_outputs(
    *,
    outputs: dict[str, SystemOutput],
    config: QARunConfig,
    family: str,
    questions: list[RuntimeQuestion],
    question_embeddings: np.ndarray,
    corpus: RuntimeCorpus,
    hybrid: HybridRetriever,
    temporal_ranker: TemporalRanker,
    graph: GraphRetriever,
    system_name: QASystemName,
) -> None:
    expected = {
        system_output_id(family, system_name, question.qid): (index, question)
        for index, question in enumerate(questions)
    }
    unexpected = set(outputs) - set(expected)
    if unexpected:
        raise ValueError(f"Checkpoint contains unexpected output ids: {sorted(unexpected)[:5]}")
    for output_id, output in outputs.items():
        if output.status != "success":
            continue
        index, question = expected[output_id]
        identity_matches = (
            output.dataset_family == family
            and output.qid == question.qid
            and output.scenario_id == question.scenario_id
            and output.system_name == system_name
            and output.generator_provider == config.generator_provider
            and output.generator_model == config.generator_model
            and output.reasoning_effort == config.reasoning_effort
            and output.prompt_version == PROMPT_VERSION
        )
        if not identity_matches:
            raise ValueError(
                f"Checkpoint output provenance is incompatible: {output_id}. "
                "Use --overwrite to regenerate the shard."
            )
        retrieval, context = retrieve_for_question(
            config=config,
            question=question,
            query_embedding=question_embeddings[index],
            corpus=corpus,
            hybrid=hybrid,
            temporal_ranker=temporal_ranker,
            graph=graph,
            system_name=system_name,
        )
        _, prompt_sha256 = answer_prompt(question=question.question, context=context)
        if not _retrievals_equivalent(output.retrieval, retrieval) or (
            output.prompt_sha256 != prompt_sha256
        ):
            raise ValueError(
                f"Checkpoint retrieval or prompt is stale: {output_id}. "
                "Use --overwrite to regenerate the shard."
            )


def _retrievals_equivalent(actual: object, expected: object) -> bool:
    if not isinstance(actual, RetrievalResult) or not isinstance(expected, RetrievalResult):
        return False
    if (
        actual.system_name != expected.system_name
        or actual.snapshot_id != expected.snapshot_id
        or actual.visible_fact_count != expected.visible_fact_count
        or actual.context_sha256 != expected.context_sha256
        or actual.evidence_handle_map != expected.evidence_handle_map
        or actual.temporal_intent != expected.temporal_intent
        or len(actual.hits) != len(expected.hits)
        or len(actual.graph_paths) != len(expected.graph_paths)
    ):
        return False
    for left, right in zip(actual.hits, expected.hits, strict=True):
        if (
            left.fact_id != right.fact_id
            or left.rank != right.rank
            or not _optional_score_close(left.score, right.score)
            or not _optional_score_close(left.dense_score, right.dense_score)
            or not _optional_score_close(left.lexical_score, right.lexical_score)
            or not _optional_score_close(left.temporal_score, right.temporal_score)
            or not _optional_score_close(left.graph_score, right.graph_score)
        ):
            return False
    for left, right in zip(actual.graph_paths, expected.graph_paths, strict=True):
        if (
            left.path_id != right.path_id
            or left.fact_ids != right.fact_ids
            or left.traversal_directions != right.traversal_directions
            or left.node_ids != right.node_ids
            or left.path_time_status != right.path_time_status
            or left.explanation != right.explanation
            or not _optional_score_close(left.score, right.score)
        ):
            return False
    return True


def _optional_score_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)


def _write_outputs_checkpoint(
    path: Path,
    outputs: dict[str, SystemOutput],
    questions: list[RuntimeQuestion],
    family: str,
    system_name: QASystemName,
    metadata: CheckpointMetadata,
) -> list[SystemOutput]:
    ordered = [
        outputs[output_id]
        for question in questions
        if (output_id := system_output_id(family, system_name, question.qid)) in outputs
    ]
    temporary = path.with_suffix(".jsonl.tmp")
    write_jsonl(temporary, [output.model_dump(mode="json") for output in ordered])
    temporary.replace(path)
    write_checkpoint_metadata(path, metadata, record_count=len(ordered))
    return ordered
