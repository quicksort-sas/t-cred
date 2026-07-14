from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from tcred.qa.corpus import RuntimeCorpus
from tcred.qa.lexical import BM25Index
from tcred.qa.models import RetrievalHit


class HybridRetriever:
    """Exact dense + BM25 retrieval fused with reciprocal-rank fusion."""

    def __init__(
        self,
        *,
        corpus: RuntimeCorpus,
        fact_embeddings: NDArray[np.float32],
        rrf_constant: int = 60,
    ) -> None:
        if fact_embeddings.shape[0] != len(corpus.documents):
            raise ValueError("Fact embedding count does not match corpus size")
        self.corpus = corpus
        self.fact_embeddings = fact_embeddings
        self.lexical = BM25Index([document.semantic_text for document in corpus.documents])
        self.rrf_constant = rrf_constant

    def retrieve(
        self,
        *,
        question: str,
        query_embedding: NDArray[np.float32],
        snapshot_id: str,
        top_k: int,
        candidate_k: int,
    ) -> list[RetrievalHit]:
        visible = self.corpus.visible_indices(snapshot_id)
        if not visible:
            return []
        visible_array = np.asarray(visible, dtype=np.int64)
        dense_all = self.fact_embeddings @ query_embedding
        lexical_all = self.lexical.scores(question)
        branch_k = min(max(candidate_k, top_k), len(visible))
        dense_ranked = _rank_visible(dense_all, visible_array, branch_k)
        lexical_ranked = _rank_visible(lexical_all, visible_array, branch_k)

        fused: defaultdict[int, float] = defaultdict(float)
        dense_rank = {index: rank for rank, index in enumerate(dense_ranked, start=1)}
        lexical_rank = {index: rank for rank, index in enumerate(lexical_ranked, start=1)}
        for index, rank in dense_rank.items():
            fused[index] += 1.0 / (self.rrf_constant + rank)
        for index, rank in lexical_rank.items():
            fused[index] += 1.0 / (self.rrf_constant + rank)

        ordered = sorted(fused, key=lambda index: (-fused[index], index))[:candidate_k]
        dense_min, dense_max = _range(dense_all[visible_array])
        lexical_min, lexical_max = _range(lexical_all[visible_array])
        hits = []
        for rank, index in enumerate(ordered[:top_k], start=1):
            fact_id = self.corpus.documents[index].fact.fact_id
            hits.append(
                RetrievalHit(
                    fact_id=fact_id,
                    rank=rank,
                    score=fused[index],
                    dense_score=_scale(float(dense_all[index]), dense_min, dense_max),
                    lexical_score=_scale(float(lexical_all[index]), lexical_min, lexical_max),
                )
            )
        return hits


def _rank_visible(
    scores: NDArray[np.float32],
    visible: NDArray[np.int64],
    top_k: int,
) -> list[int]:
    visible_scores = scores[visible]
    if top_k >= len(visible):
        local = np.arange(len(visible), dtype=np.int64)
    else:
        local = np.argpartition(visible_scores, -top_k)[-top_k:]
    return sorted(
        (int(visible[index]) for index in local),
        key=lambda index: (-float(scores[index]), index),
    )


def _range(values: NDArray[np.float32]) -> tuple[float, float]:
    return float(values.min()), float(values.max())


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return (value - low) / (high - low)
