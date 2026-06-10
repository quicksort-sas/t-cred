from __future__ import annotations

import re
from datetime import date

from tcred.dataset.graph import fact_answer_id
from tcred.dataset.models import (
    ContextPackType,
    Fact,
    GraphPathEdge,
    PathTimeStatus,
    QuestionProgram,
    TemporalInterval,
    TemporalOperator,
    TimeStatus,
)


def fact_matches_program(fact: Fact, program: QuestionProgram) -> bool:
    return (
        fact.relation == program.relation
        and (program.context_id is None or fact.context_id == program.context_id)
        and (program.object_id is None or fact.object_id == program.object_id)
    )


def fact_visible(fact: Fact, snapshot_id: str) -> bool:
    return _snapshot_rank(fact.snapshot_visible_from) <= _snapshot_rank(snapshot_id)


def query_point(program: QuestionProgram) -> date:
    if program.query_time.start is None:
        raise ValueError("Query program has no normalized start time")
    return program.query_time.start


def path_query_time(program: QuestionProgram) -> TemporalInterval | None:
    """Return an overlap constraint only for operators that require simultaneity."""
    if program.operator in {
        TemporalOperator.CURRENT,
        TemporalOperator.AS_OF,
        TemporalOperator.DURING,
        TemporalOperator.BETWEEN,
        TemporalOperator.EFFECTIVE,
    }:
        return program.query_time
    return None


def time_status_for_fact(
    fact: Fact,
    program: QuestionProgram,
    *,
    selected_fact_ids: set[str] | None = None,
) -> TimeStatus:
    """Classify a fact relative to the complete temporal operator.

    ``selected_fact_ids`` should be supplied when labels or diagnostic context
    packs are generated. It prevents ordinal answers such as ``first``,
    ``previous``, and ``next`` from being mislabeled merely because their valid
    interval does not contain the query point.
    """
    if fact.valid_time.type == "unknown":
        return TimeStatus.UNKNOWN_VALID_TIME
    if not fact_matches_program(fact, program):
        return TimeStatus.IRRELEVANT

    if selected_fact_ids is not None and fact.fact_id in selected_fact_ids:
        return TimeStatus.VALID

    q_start = query_point(program)
    q_end = program.query_time.end or q_start

    if program.operator in {
        TemporalOperator.DURING,
        TemporalOperator.BETWEEN,
    }:
        if fact.valid_time.overlaps(program.query_time):
            return (
                TimeStatus.VALID if selected_fact_ids is None else TimeStatus.TEMPORALLY_AMBIGUOUS
            )
    elif program.operator in {
        TemporalOperator.CURRENT,
        TemporalOperator.AS_OF,
        TemporalOperator.EFFECTIVE,
    } and fact.valid_time.contains(q_start):
        return TimeStatus.VALID

    if program.operator in {TemporalOperator.FIRST, TemporalOperator.NEXT, TemporalOperator.AFTER}:
        if fact.valid_time.start is not None and fact.valid_time.start > q_start:
            return TimeStatus.FUTURE_INVALID
        return TimeStatus.STALE

    if program.operator in {
        TemporalOperator.PREVIOUS,
        TemporalOperator.BEFORE,
        TemporalOperator.EXPIRED,
    }:
        if fact.valid_time.end is not None and fact.valid_time.end < q_start:
            return TimeStatus.STALE
        return TimeStatus.FUTURE_INVALID

    if program.operator in {TemporalOperator.LATEST, TemporalOperator.LAST}:
        if fact.valid_time.start is not None and fact.valid_time.start > q_start:
            return TimeStatus.FUTURE_INVALID
        return TimeStatus.STALE

    if fact.valid_time.end is not None and fact.valid_time.end < q_start:
        return TimeStatus.STALE
    if fact.valid_time.start is not None and fact.valid_time.start > q_end:
        return TimeStatus.FUTURE_INVALID
    return TimeStatus.TEMPORALLY_AMBIGUOUS


class GoldSolver:
    """Deterministic oracle over generated structured facts.

    This solver is intentionally private to dataset construction. It sees normalized
    entities, relations, intervals, and snapshots that evaluated systems do not see.
    """

    def solve(self, facts: list[Fact], program: QuestionProgram) -> tuple[list[str], list[str]]:
        visible = [fact for fact in facts if fact_visible(fact, program.snapshot_id)]
        matching = [fact for fact in visible if fact_matches_program(fact, program)]
        if program.operator in {TemporalOperator.CURRENT, TemporalOperator.AS_OF}:
            valid = [
                fact for fact in matching if time_status_for_fact(fact, program) == TimeStatus.VALID
            ]
        elif program.operator == TemporalOperator.DURING:
            valid = [fact for fact in matching if _covers(fact.valid_time, program.query_time)]
        elif program.operator == TemporalOperator.BETWEEN:
            valid = [fact for fact in matching if fact.valid_time.overlaps(program.query_time)]
        elif program.operator == TemporalOperator.PREVIOUS:
            valid = self._previous(matching, program)
        elif program.operator == TemporalOperator.NEXT:
            valid = self._next(matching, program)
        elif program.operator in {TemporalOperator.LATEST, TemporalOperator.LAST}:
            valid = self._latest(matching, program)
        elif program.operator == TemporalOperator.FIRST:
            valid = self._first(matching)
        elif program.operator == TemporalOperator.BEFORE:
            valid = [
                fact
                for fact in matching
                if fact.valid_time.end is not None and fact.valid_time.end < query_point(program)
            ]
            valid = self._latest_ended_from_candidates(valid)
        elif program.operator == TemporalOperator.AFTER:
            valid = [
                fact
                for fact in matching
                if fact.valid_time.start is not None
                and fact.valid_time.start > query_point(program)
            ]
            valid = self._first_from_candidates(valid)
        elif program.operator == TemporalOperator.EXPIRED:
            valid = [
                fact
                for fact in matching
                if fact.valid_time.end is not None and fact.valid_time.end < query_point(program)
            ]
            valid = self._latest_ended_from_candidates(valid)
        elif program.operator == TemporalOperator.EFFECTIVE:
            valid = [
                fact for fact in matching if time_status_for_fact(fact, program) == TimeStatus.VALID
            ]
        else:
            valid = []

        answer_ids = sorted({fact_answer_id(fact) for fact in valid if fact_answer_id(fact)})
        evidence_ids = [fact.fact_id for fact in valid]
        return answer_ids, evidence_ids

    def _previous(self, facts: list[Fact], program: QuestionProgram) -> list[Fact]:
        current = [
            fact
            for fact in facts
            if fact.valid_time.contains(query_point(program)) and fact.valid_time.start is not None
        ]
        if not current or len({fact_answer_id(fact) for fact in current}) != 1:
            return []
        current_start = min(fact.valid_time.start for fact in current if fact.valid_time.start)
        candidates = [
            fact
            for fact in facts
            if fact.valid_time.end is not None and fact.valid_time.end < current_start
        ]
        return self._latest_ended_from_candidates(candidates)

    def _next(self, facts: list[Fact], program: QuestionProgram) -> list[Fact]:
        current = [
            fact
            for fact in facts
            if fact.valid_time.contains(query_point(program)) and fact.valid_time.end is not None
        ]
        if not current:
            candidates = [
                fact
                for fact in facts
                if fact.valid_time.start is not None
                and fact.valid_time.start > query_point(program)
            ]
            return self._first_from_candidates(candidates)
        if len({fact_answer_id(fact) for fact in current}) != 1:
            return []
        current_end = max(fact.valid_time.end for fact in current if fact.valid_time.end)
        candidates = [
            fact
            for fact in facts
            if fact.valid_time.start is not None and fact.valid_time.start > current_end
        ]
        return self._first_from_candidates(candidates)

    def _latest(self, facts: list[Fact], program: QuestionProgram) -> list[Fact]:
        candidates = [
            fact
            for fact in facts
            if fact.valid_time.start is not None and fact.valid_time.start <= query_point(program)
        ]
        return self._latest_from_candidates(candidates)

    def _first(self, facts: list[Fact]) -> list[Fact]:
        return self._first_from_candidates([fact for fact in facts if fact.valid_time.start])

    @staticmethod
    def _latest_from_candidates(candidates: list[Fact]) -> list[Fact]:
        starts = [fact.valid_time.start for fact in candidates if fact.valid_time.start]
        if not starts:
            return []
        latest = max(starts)
        return [fact for fact in candidates if fact.valid_time.start == latest]

    @staticmethod
    def _latest_ended_from_candidates(candidates: list[Fact]) -> list[Fact]:
        ends = [fact.valid_time.end for fact in candidates if fact.valid_time.end]
        if not ends:
            return []
        latest = max(ends)
        return [fact for fact in candidates if fact.valid_time.end == latest]

    @staticmethod
    def _first_from_candidates(candidates: list[Fact]) -> list[Fact]:
        starts = [fact.valid_time.start for fact in candidates if fact.valid_time.start]
        if not starts:
            return []
        first = min(starts)
        return [fact for fact in candidates if fact.valid_time.start == first]


def _eligible_for_gold(fact: Fact) -> bool:
    """Compatibility hook: diagnostic roles never determine semantic eligibility."""
    del fact
    return True


def _covers(fact_time: TemporalInterval, query_time: TemporalInterval) -> bool:
    if fact_time.start is None or query_time.start is None or query_time.end is None:
        return False
    fact_end = fact_time.end or date.max
    return fact_time.start <= query_time.start and fact_end >= query_time.end


def path_time_status(
    edges: list[GraphPathEdge],
    *,
    sequence: bool = False,
    query_time: TemporalInterval | None = None,
) -> PathTimeStatus:
    if not edges:
        return PathTimeStatus.NOT_APPLICABLE
    if any(edge.valid_time.type == "unknown" for edge in edges):
        return PathTimeStatus.UNKNOWN_EDGE_TIME
    if sequence:
        ordered = True
        last_end = None
        for edge in edges:
            start = edge.valid_time.start
            if start is None:
                return PathTimeStatus.UNKNOWN_EDGE_TIME
            if last_end is not None and start < last_end:
                ordered = False
            last_end = edge.valid_time.end or start
        if not ordered:
            return PathTimeStatus.WRONG_ORDER
        if query_time is not None and not any(
            edge.valid_time.overlaps(query_time) for edge in edges
        ):
            return PathTimeStatus.INCOHERENT_QUERY_TIME
        return PathTimeStatus.COHERENT_SEQUENCE

    intersection_start = max(edge.valid_time.start for edge in edges if edge.valid_time.start)
    ends = [edge.valid_time.end or date.max for edge in edges]
    intersection_end = min(ends)
    if intersection_start <= intersection_end:
        if query_time is not None:
            shared = TemporalInterval(start=intersection_start, end=intersection_end)
            if not shared.overlaps(query_time):
                return PathTimeStatus.INCOHERENT_QUERY_TIME
        return PathTimeStatus.COHERENT_SHARED_INTERVAL
    return PathTimeStatus.INCOHERENT_EMPTY_INTERSECTION


def expected_pack_behavior(pack_type: ContextPackType) -> str:
    return {
        ContextPackType.VALID_ONLY: "answer_from_valid_evidence",
        ContextPackType.STALE_ONLY: "penalize_or_refuse_stale_support",
        ContextPackType.FUTURE_ONLY: "penalize_or_refuse_future_invalid_support",
        ContextPackType.VALID_PLUS_STALE: "prefer_valid_evidence",
        ContextPackType.VALID_PLUS_FUTURE: "prefer_query_valid_evidence",
        ContextPackType.CONFLICT: "resolve_conflict_or_abstain",
        ContextPackType.PUBLICATION_ONLY: "abstain_or_mark_unknown_valid_time",
        ContextPackType.UNKNOWN_TIME: "abstain_or_mark_unknown_valid_time",
        ContextPackType.GRAPH_INCOHERENT: "penalize_invalid_graph_path",
        ContextPackType.INSUFFICIENT: "refuse_due_to_insufficient_temporal_evidence",
    }[pack_type]


def _snapshot_rank(snapshot_id: str) -> int:
    match = re.fullmatch(r"S(\d+)", snapshot_id, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unsupported snapshot id: {snapshot_id!r}")
    return int(match.group(1))
