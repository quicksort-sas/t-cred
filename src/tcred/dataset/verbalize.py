from __future__ import annotations

from datetime import date

from tcred.dataset.intervals import human_interval
from tcred.dataset.models import Entity, Fact, Question, QuestionProgram, TemporalOperator


def entity_lookup(entities: list[Entity]) -> dict[str, Entity]:
    return {entity.entity_id: entity for entity in entities}


def fact_sentence(fact: Fact, entities: dict[str, Entity]) -> str:
    subject = entities[fact.subject_id].name
    context = entities[fact.context_id].name if fact.context_id else "the relevant context"
    obj = entities[fact.object_id].name if fact.object_id else fact.relation.replace("_", " ")
    interval = human_interval(fact.valid_time)

    if fact.fact_role == "publication_only":
        pub = _fmt_date(fact.publication_time)
        return (
            f"A source published on {pub} stated that {subject} was associated "
            f"with {context}, but it did not state when the claim was valid."
        )
    if fact.valid_time.type == "unknown":
        return (
            f"{subject} was reported as the {obj} for {context}, but the source "
            "did not give a valid-time interval."
        )

    relation_phrase = {
        "held_role": f"held the {obj} role at",
        "member_of": f"was a {obj} for",
        "employed_by": "was employed by",
        "political_affiliation": "was affiliated with",
        "policy_effective": f"was the {obj} for",
        "contract_active": f"was the {obj} for",
        "support_window": f"was the {obj} for",
        "located_at": "was the listed location for",
        "event_occurs": f"was the {obj} in",
        "project_participant": f"served as {obj} for",
        "affiliated_with": "was affiliated with",
        "event_precedes": "preceded",
        "product_version": f"was the {obj} for",
        "document_supports": "was documented as supporting",
        "wikidata_property": "was linked by a Wikidata property to",
    }[str(fact.relation)]
    return f"{subject} {relation_phrase} {context} {interval}."


def canonical_question(program: QuestionProgram, entities: dict[str, Entity]) -> str:
    context = entities[program.context_id].name if program.context_id else "the scenario"
    obj = entities[program.object_id].name if program.object_id else "the target"
    date_text = _fmt_date(program.query_time.start)

    if program.operator == TemporalOperator.CURRENT:
        return f"As of {date_text}, who or what is the current {obj} for {context}?"
    if program.operator == TemporalOperator.AS_OF:
        return f"Who or what was the {obj} for {context} on {date_text}?"
    if program.operator == TemporalOperator.PREVIOUS:
        return (
            f"Who or what held the {obj} for {context} immediately before the "
            f"holder on {date_text}?"
        )
    if program.operator == TemporalOperator.NEXT:
        return (
            f"Who or what held the {obj} for {context} immediately after the holder on {date_text}?"
        )
    if program.operator == TemporalOperator.DURING:
        start = _fmt_date(program.query_time.start)
        end = _fmt_date(program.query_time.end)
        return f"Who or what was the {obj} for {context} during the period from {start} to {end}?"
    if program.operator == TemporalOperator.BETWEEN:
        start = _fmt_date(program.query_time.start)
        end = _fmt_date(program.query_time.end)
        return f"Who or what was the {obj} for {context} between {start} and {end}?"
    if program.operator == TemporalOperator.BEFORE:
        return f"Who or what was the last {obj} for {context} before {date_text}?"
    if program.operator == TemporalOperator.AFTER:
        return f"Who or what was the first {obj} for {context} after {date_text}?"
    if program.operator == TemporalOperator.FIRST:
        return f"Who or what was the first recorded {obj} for {context}?"
    if program.operator in {TemporalOperator.LATEST, TemporalOperator.LAST}:
        return f"Who or what was the latest recorded {obj} for {context} as of {date_text}?"
    if program.operator == TemporalOperator.EXPIRED:
        return f"Who or what was the most recently expired {obj} for {context} by {date_text}?"
    if program.operator == TemporalOperator.EFFECTIVE:
        return f"Which {obj} for {context} was effective on {date_text}?"
    return f"Which answer satisfies the {obj} relation for {context} at {date_text}?"


def naturalize_question(canonical: str, index: int) -> str:
    """Deterministic light paraphrase to avoid one template style.

    LLM paraphrasing is implemented separately so generation remains reproducible.
    """
    variants = (
        canonical,
        canonical.replace("Who or what was", "Which entity was"),
        canonical.replace("Who or what is", "Which entity is"),
        canonical.replace("As of", "Using the evaluation date"),
    )
    return variants[index % len(variants)]


def answer_sentence(question: Question, entities: dict[str, Entity]) -> str:
    if question.should_abstain:
        return "The available evidence is not temporally sufficient to answer confidently."
    names = [entities[entity_id].name for entity_id in question.gold_answer_entity_ids]
    return ", ".join(names)


def _fmt_date(value: date | None) -> str:
    if value is None:
        return "the relevant date"
    return value.strftime("%B %d, %Y").replace(" 0", " ")
