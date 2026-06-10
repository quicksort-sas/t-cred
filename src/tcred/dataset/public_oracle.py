from __future__ import annotations

import re
from datetime import date

from tcred.dataset.models import Fact, QuestionProgram, TemporalInterval, TemporalOperator


def solve_from_public_facts(
    facts: list[Fact],
    program: QuestionProgram,
) -> tuple[list[str], list[str]]:
    """Independent gold computation using only runtime-visible fact fields."""
    matching = [
        fact
        for fact in facts
        if _fact_visible(fact, program.snapshot_id) and _fact_matches_program(fact, program)
    ]
    operator = program.operator
    if operator in {TemporalOperator.CURRENT, TemporalOperator.AS_OF, TemporalOperator.EFFECTIVE}:
        selected = [fact for fact in matching if fact.valid_time.contains(_query_point(program))]
    elif operator == TemporalOperator.DURING:
        selected = [fact for fact in matching if _covers(fact.valid_time, program.query_time)]
    elif operator == TemporalOperator.BETWEEN:
        selected = [fact for fact in matching if fact.valid_time.overlaps(program.query_time)]
    elif operator == TemporalOperator.PREVIOUS:
        selected = _previous(matching, program)
    elif operator == TemporalOperator.NEXT:
        selected = _next(matching, program)
    elif operator in {TemporalOperator.LATEST, TemporalOperator.LAST}:
        selected = _latest_started(
            [
                fact
                for fact in matching
                if fact.valid_time.start is not None
                and fact.valid_time.start <= _query_point(program)
            ]
        )
    elif operator == TemporalOperator.FIRST:
        selected = _earliest_started(matching)
    elif operator in {TemporalOperator.BEFORE, TemporalOperator.EXPIRED}:
        selected = _latest_ended(
            [
                fact
                for fact in matching
                if fact.valid_time.end is not None and fact.valid_time.end < _query_point(program)
            ]
        )
    elif operator == TemporalOperator.AFTER:
        selected = _earliest_started(
            [
                fact
                for fact in matching
                if fact.valid_time.start is not None
                and fact.valid_time.start > _query_point(program)
            ]
        )
    else:
        selected = []
    return (
        sorted({_fact_answer_id(fact) for fact in selected if _fact_answer_id(fact)}),
        [fact.fact_id for fact in selected],
    )


def _previous(facts: list[Fact], program: QuestionProgram) -> list[Fact]:
    current = [fact for fact in facts if fact.valid_time.contains(_query_point(program))]
    if len({_fact_answer_id(fact) for fact in current}) != 1:
        return []
    starts = [fact.valid_time.start for fact in current if fact.valid_time.start is not None]
    if not starts:
        return []
    start = min(starts)
    return _latest_ended(
        [fact for fact in facts if fact.valid_time.end is not None and fact.valid_time.end < start]
    )


def _next(facts: list[Fact], program: QuestionProgram) -> list[Fact]:
    current = [
        fact
        for fact in facts
        if fact.valid_time.contains(_query_point(program)) and fact.valid_time.end is not None
    ]
    if not current:
        return _earliest_started(
            [
                fact
                for fact in facts
                if fact.valid_time.start is not None
                and fact.valid_time.start > _query_point(program)
            ]
        )
    if len({_fact_answer_id(fact) for fact in current}) != 1:
        return []
    end = max(fact.valid_time.end for fact in current if fact.valid_time.end is not None)
    return _earliest_started(
        [
            fact
            for fact in facts
            if fact.valid_time.start is not None and fact.valid_time.start > end
        ]
    )


def _latest_started(facts: list[Fact]) -> list[Fact]:
    values = [fact.valid_time.start for fact in facts if fact.valid_time.start is not None]
    if not values:
        return []
    latest = max(values)
    return [fact for fact in facts if fact.valid_time.start == latest]


def _earliest_started(facts: list[Fact]) -> list[Fact]:
    values = [fact.valid_time.start for fact in facts if fact.valid_time.start is not None]
    if not values:
        return []
    earliest = min(values)
    return [fact for fact in facts if fact.valid_time.start == earliest]


def _latest_ended(facts: list[Fact]) -> list[Fact]:
    values = [fact.valid_time.end for fact in facts if fact.valid_time.end is not None]
    if not values:
        return []
    latest = max(values)
    return [fact for fact in facts if fact.valid_time.end == latest]


def _covers(fact_time: TemporalInterval, query_time: TemporalInterval) -> bool:
    if fact_time.start is None or query_time.start is None or query_time.end is None:
        return False
    return fact_time.start <= query_time.start and (fact_time.end or date.max) >= query_time.end


def _fact_matches_program(fact: Fact, program: QuestionProgram) -> bool:
    return (
        fact.relation == program.relation
        and (program.context_id is None or fact.context_id == program.context_id)
        and (program.object_id is None or fact.object_id == program.object_id)
    )


def _fact_visible(fact: Fact, snapshot_id: str) -> bool:
    return _snapshot_rank(fact.snapshot_visible_from) <= _snapshot_rank(snapshot_id)


def _snapshot_rank(snapshot_id: str) -> int:
    match = re.fullmatch(r"S(\d+)", snapshot_id, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported snapshot id: {snapshot_id!r}")
    return int(match.group(1))


def _query_point(program: QuestionProgram) -> date:
    if program.query_time.start is None:
        raise ValueError("Public question program has no normalized query start")
    return program.query_time.start


def _fact_answer_id(fact: Fact) -> str:
    return fact.answer_entity_id or fact.subject_id
