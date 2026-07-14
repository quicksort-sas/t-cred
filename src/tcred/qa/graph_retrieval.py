from __future__ import annotations

from collections import defaultdict, deque

from tcred.dataset.graph import fact_endpoint_ids, graph_path_node_ids
from tcred.dataset.models import GraphPathEdge, PathTimeStatus, Relation, TemporalInterval
from tcred.dataset.solver import path_time_status
from tcred.qa.corpus import RuntimeCorpus, RuntimeFact
from tcred.qa.models import RetrievalHit, RetrievedGraphPath, TemporalIntent
from tcred.qa.temporal import score_temporal_compatibility


class GraphRetriever:
    """Direction-aware KG expansion with explicit inverse traversal records."""

    def __init__(self, corpus: RuntimeCorpus) -> None:
        self.corpus = corpus
        incidence: defaultdict[str, list[str]] = defaultdict(list)
        for fact in corpus.facts:
            source_id, target_id = fact_endpoint_ids(fact)
            if source_id:
                incidence[source_id].append(fact.fact_id)
            if target_id and target_id != source_id:
                incidence[target_id].append(fact.fact_id)
        self.incidence = {
            node_id: tuple(sorted(set(fact_ids))) for node_id, fact_ids in incidence.items()
        }

    def expand(
        self,
        *,
        seed_hits: list[RetrievalHit],
        snapshot_id: str,
        top_k: int,
        seed_k: int,
        max_hops: int,
        path_limit: int,
        temporal_intent: TemporalIntent | None = None,
    ) -> tuple[list[RetrievalHit], list[RetrievedGraphPath]]:
        visible_ids = {
            self.corpus.documents[index].fact.fact_id
            for index in self.corpus.visible_indices(snapshot_id)
        }
        visible_seeds = [hit for hit in seed_hits if hit.fact_id in visible_ids]
        seed_by_id = {hit.fact_id: hit for hit in visible_seeds}
        max_seed_score = max((hit.score for hit in visible_seeds), default=1.0) or 1.0
        temporal = temporal_intent is not None
        groups = self._visible_groups(visible_ids) if temporal else {}

        best_score: dict[str, float] = {}
        best_edges: dict[str, list[GraphPathEdge]] = {}
        state_scores: dict[tuple[str, str], float] = {}
        queue: deque[tuple[str, list[GraphPathEdge], float, int]] = deque()

        for seed in visible_seeds[:seed_k]:
            fact = self.corpus.fact_by_id[seed.fact_id]
            normalized = seed.score / max_seed_score
            forward = self._edge(fact, "forward")
            best_score[seed.fact_id] = normalized
            best_edges[seed.fact_id] = [forward]
            source_id, target_id = fact_endpoint_ids(fact)
            for current_node, direction in ((target_id, "forward"), (source_id, "reverse")):
                if not current_node:
                    continue
                edge = self._edge(fact, direction)
                state_scores[(seed.fact_id, current_node)] = normalized
                queue.append((current_node, [edge], normalized, 0))

        while queue:
            current_node, chain, origin_score, hop = queue.popleft()
            if hop >= max_hops:
                continue
            used_ids = {edge.fact_id for edge in chain}
            for neighbor_id in self.incidence.get(current_node, ()):
                if neighbor_id not in visible_ids or neighbor_id in used_ids:
                    continue
                neighbor = self.corpus.fact_by_id[neighbor_id]
                direction, next_node = self._traversal_from(neighbor, current_node)
                if direction is None or not next_node:
                    continue
                next_edge = self._edge(neighbor, direction)
                next_chain = [*chain, next_edge]
                semantic = seed_by_id.get(neighbor_id)
                semantic_score = semantic.score / max_seed_score if semantic else 0.0
                score = origin_score * (0.72 ** (hop + 1)) + 0.35 * semantic_score
                if temporal_intent is not None:
                    peers = groups[self.corpus.fact_group_key(neighbor)]
                    compatibility = score_temporal_compatibility(
                        neighbor,
                        intent=temporal_intent,
                        peers=peers,
                    )
                    if compatibility <= 0.0:
                        continue
                    score *= 0.55 + 0.45 * compatibility
                    if self._path_status(next_chain, temporal_intent) in {
                        PathTimeStatus.INCOHERENT_EMPTY_INTERSECTION,
                        PathTimeStatus.INCOHERENT_QUERY_TIME,
                        PathTimeStatus.WRONG_ORDER,
                    }:
                        continue
                state_key = (neighbor_id, next_node)
                if score <= state_scores.get(state_key, -1.0):
                    continue
                state_scores[state_key] = score
                if score > best_score.get(neighbor_id, -1.0):
                    best_score[neighbor_id] = score
                    best_edges[neighbor_id] = next_chain
                queue.append((next_node, next_chain, origin_score, hop + 1))

        ordered_ids = sorted(best_score, key=lambda item: (-best_score[item], item))[:top_k]
        hits: list[RetrievalHit] = []
        for rank, fact_id in enumerate(ordered_ids, start=1):
            seed = seed_by_id.get(fact_id)
            temporal_score = None
            if temporal_intent is not None:
                fact = self.corpus.fact_by_id[fact_id]
                temporal_score = score_temporal_compatibility(
                    fact,
                    intent=temporal_intent,
                    peers=groups[self.corpus.fact_group_key(fact)],
                )
            hits.append(
                RetrievalHit(
                    fact_id=fact_id,
                    rank=rank,
                    score=best_score[fact_id],
                    dense_score=seed.dense_score if seed else None,
                    lexical_score=seed.lexical_score if seed else None,
                    temporal_score=temporal_score,
                    graph_score=best_score[fact_id],
                )
            )

        paths = self._paths(
            ordered_ids=ordered_ids,
            best_edges=best_edges,
            best_score=best_score,
            path_limit=path_limit,
            temporal_intent=temporal_intent,
        )
        return hits, paths

    def _visible_groups(
        self,
        visible_ids: set[str],
    ) -> dict[tuple[str, str, str], list[RuntimeFact]]:
        groups: defaultdict[tuple[str, str, str], list[RuntimeFact]] = defaultdict(list)
        for fact_id in visible_ids:
            fact = self.corpus.fact_by_id[fact_id]
            groups[self.corpus.fact_group_key(fact)].append(fact)
        return groups

    def _paths(
        self,
        *,
        ordered_ids: list[str],
        best_edges: dict[str, list[GraphPathEdge]],
        best_score: dict[str, float],
        path_limit: int,
        temporal_intent: TemporalIntent | None,
    ) -> list[RetrievedGraphPath]:
        paths: list[RetrievedGraphPath] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for fact_id in ordered_ids:
            edges = best_edges[fact_id]
            signature = tuple((edge.fact_id, str(edge.traversal_direction)) for edge in edges)
            if signature in seen:
                continue
            seen.add(signature)
            inverse_count = sum(edge.traversal_direction == "reverse" for edge in edges)
            paths.append(
                RetrievedGraphPath(
                    path_id=f"retrieved_path_{len(paths) + 1:02d}",
                    fact_ids=[edge.fact_id for edge in edges],
                    traversal_directions=[edge.traversal_direction for edge in edges],
                    node_ids=graph_path_node_ids(edges, self.corpus.fact_by_id),
                    score=best_score[fact_id],
                    path_time_status=self._path_status(edges, temporal_intent),
                    explanation=(
                        "Direction-continuous KG walk"
                        + (
                            f" with {inverse_count} explicit reverse traversal(s)"
                            if inverse_count
                            else ""
                        )
                        + ("; query-time compatibility enforced." if temporal_intent else ".")
                    ),
                )
            )
            if len(paths) >= path_limit:
                break
        return paths

    def _path_status(
        self,
        edges: list[GraphPathEdge],
        intent: TemporalIntent | None,
    ) -> PathTimeStatus:
        query_time = _intent_interval(intent)
        sequence = bool(edges) and all(edge.relation == Relation.EVENT_PRECEDES for edge in edges)
        return path_time_status(edges, sequence=sequence, query_time=query_time)

    @staticmethod
    def _edge(fact: RuntimeFact, direction: str) -> GraphPathEdge:
        return GraphPathEdge(
            fact_id=fact.fact_id,
            relation=fact.relation,
            valid_time=fact.valid_time,
            traversal_direction=direction,
        )

    @staticmethod
    def _traversal_from(
        fact: RuntimeFact,
        current_node: str,
    ) -> tuple[str | None, str | None]:
        source_id, target_id = fact_endpoint_ids(fact)
        if source_id == current_node:
            return "forward", target_id
        if target_id == current_node:
            return "reverse", source_id
        return None, None


def _intent_interval(intent: TemporalIntent | None) -> TemporalInterval | None:
    if (
        intent is None
        or intent.query_start is None
        or intent.operator not in {"current", "as_of", "during", "between", "effective"}
    ):
        return None
    return TemporalInterval(
        type=(
            "point"
            if not intent.query_end or intent.query_end == intent.query_start
            else "interval"
        ),
        start=intent.query_start,
        end=intent.query_end or intent.query_start,
    )
