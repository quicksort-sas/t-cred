from __future__ import annotations

import hashlib

import numpy as np

from tcred.dataset.models import DatasetBundle, Question
from tcred.qa.corpus import RuntimeCorpus, RuntimeQuestion, runtime_snapshot_id
from tcred.qa.generation import context_sha256, evidence_handle_map, render_retrieval_context
from tcred.qa.graph_retrieval import GraphRetriever
from tcred.qa.models import (
    QARunConfig,
    QASystemName,
    RetrievalHit,
    RetrievalResult,
    RetrievedGraphPath,
)
from tcred.qa.retrieval import HybridRetriever
from tcred.qa.temporal import TemporalRanker, parse_temporal_intent


def retrieve_for_question(
    *,
    config: QARunConfig,
    question: Question | RuntimeQuestion,
    query_embedding: np.ndarray,
    corpus: RuntimeCorpus,
    hybrid: HybridRetriever,
    temporal_ranker: TemporalRanker,
    graph: GraphRetriever,
    system_name: QASystemName,
) -> tuple[RetrievalResult, str]:
    snapshot_id = runtime_snapshot_id(question)
    initial = hybrid.retrieve(
        question=question.question,
        query_embedding=query_embedding,
        snapshot_id=snapshot_id,
        top_k=config.candidate_k,
        candidate_k=config.candidate_k,
    )
    initial = _deduplicate_hits(initial, corpus=corpus, limit=config.candidate_k)
    intent = None
    paths = []
    if system_name == QASystemName.VECTOR_RAG:
        hits = initial[: config.top_k]
    elif system_name == QASystemName.TEMPORAL_FILTER_RAG:
        intent = parse_temporal_intent(question.question)
        hits = temporal_ranker.rank(
            initial,
            intent=intent,
            top_k=config.top_k,
            snapshot_id=snapshot_id,
        )
    elif system_name == QASystemName.GRAPH_RAG_NO_TIME:
        hits, paths = graph.expand(
            seed_hits=initial,
            snapshot_id=snapshot_id,
            top_k=config.candidate_k,
            seed_k=config.graph_seed_k,
            max_hops=config.graph_max_hops,
            path_limit=config.graph_path_limit,
        )
    else:
        intent = parse_temporal_intent(question.question)
        temporal_seeds = temporal_ranker.rank(
            initial,
            intent=intent,
            top_k=config.candidate_k,
            snapshot_id=snapshot_id,
        )
        hits, paths = graph.expand(
            seed_hits=temporal_seeds,
            snapshot_id=snapshot_id,
            top_k=config.candidate_k,
            seed_k=config.graph_seed_k,
            max_hops=config.graph_max_hops,
            path_limit=config.graph_path_limit,
            temporal_intent=intent,
        )

    if paths:
        paths = _deduplicate_paths(paths, corpus=corpus, limit=config.graph_path_limit)
    preferred_path_facts = [fact_id for path in paths for fact_id in path.fact_ids]
    hits = _deduplicate_hits(
        hits,
        corpus=corpus,
        limit=config.top_k,
        preferred_fact_ids=preferred_path_facts,
    )

    retrieval = RetrievalResult(
        system_name=system_name,
        snapshot_id=snapshot_id,
        visible_fact_count=len(corpus.visible_indices(snapshot_id)),
        hits=hits,
        graph_paths=paths,
        temporal_intent=intent,
        evidence_handle_map={},
        context_sha256="",
    )
    retrieval = retrieval.model_copy(update={"evidence_handle_map": evidence_handle_map(retrieval)})
    context = render_retrieval_context(retrieval=retrieval, corpus=corpus)
    retrieval = retrieval.model_copy(update={"context_sha256": context_sha256(context)})
    return retrieval, context


def _deduplicate_hits(
    hits: list[RetrievalHit],
    *,
    corpus: RuntimeCorpus,
    limit: int,
    preferred_fact_ids: list[str] | None = None,
) -> list[RetrievalHit]:
    preferred_by_key = {
        corpus.semantic_fact_key(fact_id): fact_id for fact_id in preferred_fact_ids or []
    }
    selected: list[RetrievalHit] = []
    seen: set[tuple[str, ...]] = set()
    for hit in hits:
        key = corpus.semantic_fact_key(hit.fact_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            hit.model_copy(
                update={
                    "fact_id": preferred_by_key.get(key, hit.fact_id),
                    "rank": len(selected) + 1,
                }
            )
        )
        if len(selected) == limit:
            break
    return selected


def _deduplicate_paths(
    paths: list[RetrievedGraphPath],
    *,
    corpus: RuntimeCorpus,
    limit: int,
) -> list[RetrievedGraphPath]:
    selected: list[RetrievedGraphPath] = []
    seen: set[tuple[tuple[tuple[str, ...], str], ...]] = set()
    for path in paths:
        directions = path.traversal_directions or ["forward"] * len(path.fact_ids)
        signature = tuple(
            (corpus.semantic_fact_key(fact_id), direction)
            for fact_id, direction in zip(path.fact_ids, directions, strict=True)
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(path)
        if len(selected) == limit:
            break
    return selected


def system_output_id(family: str, system_name: QASystemName, qid: str) -> str:
    digest = hashlib.sha256(f"{family}:{system_name}:{qid}".encode()).hexdigest()[:16]
    return f"so_{digest}"


def preflight_qa_config(config: QARunConfig) -> None:
    if config.top_k < 1 or config.candidate_k < config.top_k:
        raise ValueError("candidate_k must be >= top_k >= 1")
    if config.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not config.splits:
        raise ValueError("At least one evaluation split is required")
    for family in config.families:
        dataset_dir = config.dataset_root / family
        required = (
            dataset_dir / "runtime" / "entities.jsonl",
            dataset_dir / "runtime" / "facts.jsonl",
            dataset_dir / "runtime" / "questions.jsonl",
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Dataset runtime projection is incomplete: {', '.join(map(str, missing))}"
            )


def select_questions(
    bundle: DatasetBundle,
    limit: int | None,
    *,
    splits: list[str] | None = None,
) -> list[Question]:
    requested = splits or ["test_auto"]
    unknown = set(requested) - set(bundle.splits) - {"all"}
    if unknown:
        raise ValueError(f"Unknown dataset split(s): {sorted(unknown)}")
    scenario_ids = (
        {scenario.scenario_id for scenario in bundle.scenarios}
        if "all" in requested
        else {scenario_id for split in requested for scenario_id in bundle.splits.get(split, [])}
    )
    questions = sorted(
        (question for question in bundle.questions if question.scenario_id in scenario_ids),
        key=lambda question: question.qid,
    )
    if limit is None or limit >= len(questions):
        return questions
    groups: dict[tuple[str, str, str], list[Question]] = {}
    for question in questions:
        key = (
            str(question.temporal_operator),
            str(question.system_difficulty),
            str(question.eval_difficulty),
        )
        groups.setdefault(key, []).append(question)
    selected: list[Question] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected
