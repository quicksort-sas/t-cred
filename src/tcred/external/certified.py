from __future__ import annotations

import hashlib
import html
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any

from ftfy import fix_text

from tcred.dataset.generator import stable_opaque_id
from tcred.dataset.graph import graph_path_node_ids
from tcred.dataset.intervals import point, unknown_interval
from tcred.dataset.models import (
    AnswerClaim,
    AnswerType,
    AnswerVariant,
    AnswerVariantType,
    ContextPack,
    ContextPackType,
    DatasetBundle,
    DatasetFamily,
    Entity,
    EntityType,
    EvalDifficulty,
    Fact,
    FactRole,
    GraphPath,
    GraphPathEdge,
    Question,
    QuestionProgram,
    Relation,
    Scenario,
    Snapshot,
    SourceProvenance,
    SystemDifficulty,
    TemporalOperator,
)
from tcred.dataset.solver import path_time_status
from tcred.dataset.splits import stable_group_splits
from tcred.dataset.text import normalize_visible_text

PAT_PROPERTY_LABELS = {
    "P6": "head of government",
    "P17": "country",
    "P19": "place of birth",
    "P26": "spouse",
    "P27": "country of citizenship",
    "P35": "head of state",
    "P39": "position held",
    "P54": "member of sports team",
    "P69": "educated at",
    "P102": "political party",
    "P108": "employer",
    "P112": "founded by",
    "P115": "home venue",
    "P127": "owned by",
    "P159": "headquarters location",
    "P169": "chief executive officer",
    "P286": "head coach",
    "P488": "chairperson",
    "P1037": "director or manager",
    "P1365": "replaces",
}


def build_pat_bundle(rows: list[dict[str, Any]]) -> DatasetBundle:
    scenarios: list[Scenario] = []
    entities: list[Entity] = []
    facts: list[Fact] = []
    snapshots: list[Snapshot] = []
    questions: list[Question] = []
    graph_paths: list[GraphPath] = []
    context_packs: list[ContextPack] = []
    answer_variants: list[AnswerVariant] = []
    decoy_answer_by_row = _pat_relation_matched_decoys(rows)

    for row_index, row in enumerate(rows):
        if not pat_row_is_convertible(row):
            continue
        answers = row["answer annotations"]
        relations = [str(value) for value in row["relations"]]
        intermediates = row.get("intermediate entities") or []

        snapshot_name = str(row["_snapshot_name"])
        snapshot_date = row["_snapshot_date"]
        source_group_id = pat_semantic_series_id(row)
        scenario_id = stable_opaque_id(
            "pat",
            snapshot_name,
            row.get("uniq_id", row_index),
            source_group_id,
        )
        qid = f"q_{scenario_id}"
        subject = row.get("subject") or {}
        subject_entity = _source_entity(
            scenario_id=scenario_id,
            source_id=str(subject.get("subject") or "subject"),
            label=clean_english_text(str(subject.get("subLabel") or "PAT subject")),
            suffix="subject",
        )
        intermediate_entities = [
            _source_entity(
                scenario_id=scenario_id,
                source_id=str(value.get("ID") or f"intermediate_{index}"),
                label=clean_english_text(str(value.get("Label") or f"intermediate {index}")),
                suffix=f"intermediate_{index}",
            )
            for index, value in enumerate(intermediates)
        ]
        answer_entities = [
            _source_entity(
                scenario_id=scenario_id,
                source_id=str(value.get("ID") or f"answer_{index}"),
                label=clean_english_text(str(value.get("Label") or f"answer {index}")),
                suffix=f"answer_{index}",
            )
            for index, value in enumerate(answers)
        ]
        scenario_entities = [subject_entity, *intermediate_entities, *answer_entities]
        scenario_facts_by_id: dict[str, Fact] = {}
        path_ids: list[str] = []
        required_ids: list[str] = []

        for answer_index, answer_entity in enumerate(answer_entities):
            chain = [subject_entity, *intermediate_entities, answer_entity]
            path_fact_ids: list[str] = []
            for edge_index, relation_id in enumerate(relations):
                source_entity = chain[edge_index]
                target_entity = chain[edge_index + 1]
                fact_id = stable_opaque_id(
                    "f",
                    scenario_id,
                    relation_id,
                    source_entity.entity_id,
                    target_entity.entity_id,
                )
                path_fact_ids.append(fact_id)
                required_ids.append(fact_id)
                scenario_facts_by_id.setdefault(
                    fact_id,
                    Fact(
                        fact_id=fact_id,
                        scenario_id=scenario_id,
                        subject_id=target_entity.entity_id,
                        relation=Relation.WIKIDATA_PROPERTY,
                        context_id=subject_entity.entity_id,
                        answer_entity_id=(
                            answer_entity.entity_id if edge_index == len(relations) - 1 else None
                        ),
                        graph_source_id=source_entity.entity_id,
                        graph_target_id=target_entity.entity_id,
                        source_relation_id=relation_id,
                        source_relation_label=PAT_PROPERTY_LABELS[relation_id],
                        relation_direction=("symmetric" if relation_id == "P26" else "directed"),
                        source_record_id=str(row.get("uniq_id") or source_group_id),
                        source_revision=snapshot_name,
                        valid_time=unknown_interval(),
                        publication_time=snapshot_date,
                        transaction_time=snapshot_date,
                        snapshot_visible_from="S1",
                        source_type=f"pat_wikidata_snapshot:{snapshot_name}",
                        provenance_reliability="high",
                        fact_role=FactRole.SOURCE_ASSERTION,
                        canonical_evidence=(
                            "In the PAT answer-supporting annotation for the Wikidata "
                            f"snapshot dated {_date_text(snapshot_date)}, {source_entity.name} "
                            f"is linked by {PAT_PROPERTY_LABELS[relation_id]} to "
                            f"{target_entity.name}{_sentence_terminator(target_entity.name)}"
                        ),
                    ),
                )
            edges = [
                GraphPathEdge(
                    fact_id=fact_id,
                    relation=Relation.WIKIDATA_PROPERTY,
                    valid_time=unknown_interval(),
                )
                for fact_id in path_fact_ids
            ]
            fact_lookup = scenario_facts_by_id
            path_id = stable_opaque_id("p", qid, answer_index)
            path_ids.append(path_id)
            graph_paths.append(
                GraphPath(
                    pid=path_id,
                    scenario_id=scenario_id,
                    qid=qid,
                    nodes=graph_path_node_ids(edges, fact_lookup),
                    edges=edges,
                    path_time_status=path_time_status(edges),
                    supports_gold_answer=True,
                    explanation=(
                        "The path preserves every source relation and intermediate entity from "
                        "the PAT annotation. It is a snapshot path; world-valid edge time is "
                        "not asserted."
                    ),
                )
            )

        scenario_facts = list(scenario_facts_by_id.values())
        required_ids = list(dict.fromkeys(required_ids))
        original_question = _pat_question_text(
            clean_english_text(str(row.get("question") or "")),
            relations=relations,
        )
        question_text = (
            f"According to the Wikidata snapshot dated {_date_text(snapshot_date)}, "
            f"{_lower_initial(original_question)}"
        )
        answer_text = [entity.name for entity in answer_entities]
        operator = _pat_operator(original_question, str(row.get("template") or ""))
        program = QuestionProgram(
            operator=operator,
            target="snapshot answer",
            query_time=point(snapshot_date.year, snapshot_date.month, snapshot_date.day),
            relation=Relation.WIKIDATA_PROPERTY,
            context_id=subject_entity.entity_id,
            snapshot_id="S1",
            answer_function="pat_snapshot_answer",
            required_path_semantics="source_relation_chain",
            temporal_basis="snapshot_observation",
        )
        question = Question(
            qid=qid,
            scenario_id=scenario_id,
            dataset_family=DatasetFamily.PAT,
            canonical_question=question_text,
            question=question_text,
            program=program,
            temporal_operator=operator,
            answer_type=AnswerType.LIST if len(answer_text) > 1 else AnswerType.ENTITY,
            gold_answer_entity_ids=[entity.entity_id for entity in answer_entities],
            gold_answer_text=answer_text,
            required_valid_evidence_ids=required_ids,
            should_abstain=False,
            system_difficulty=(
                SystemDifficulty.HARD if len(relations) > 1 else SystemDifficulty.MEDIUM
            ),
            eval_difficulty=(
                EvalDifficulty.HARD if len(answer_text) > 1 else EvalDifficulty.MEDIUM
            ),
            difficulty_provenance="source_heuristic",
            human_pool_candidate=True,
            semantic_series_id=source_group_id,
            template_family_id=f"pat:{row.get('template') or original_question}",
            certification_status="certified",
        )
        questions.append(question)
        facts.extend(scenario_facts)
        entities.extend(scenario_entities)
        snapshot = Snapshot(
            scenario_id=scenario_id,
            snapshot_id="S1",
            snapshot_time=snapshot_date,
            visible_fact_ids=[fact.fact_id for fact in scenario_facts],
            description=f"PAT Wikidata source snapshot {snapshot_name}.",
        )
        snapshots.append(snapshot)
        context_packs.append(
            ContextPack(
                pack_id=f"cp_{qid}_source_path",
                qid=qid,
                scenario_id=scenario_id,
                pack_type=ContextPackType.VALID_ONLY,
                evidence_ids=required_ids,
                expected_behavior="answer_from_complete_snapshot_path",
            )
        )
        answer_variants.append(
            _source_correct_variant(
                question=question,
                answer_text=", ".join(answer_text),
                evidence_ids=required_ids,
                graph_path_ids=path_ids,
                temporal_basis="snapshot_observation",
            )
        )
        answer_variants.append(_source_inappropriate_refusal_variant(question=question))
        if decoy_answer := decoy_answer_by_row.get(row_index):
            answer_variants.append(
                _source_hallucinated_variant(
                    question=question,
                    answer_text=decoy_answer,
                )
            )
        if len(answer_text) > 1:
            answer_variants.append(
                _source_partial_variant(
                    question=question,
                    answer_text=answer_text[0],
                    evidence_ids=[
                        edge.fact_id
                        for edge in next(
                            path for path in graph_paths if path.pid == path_ids[0]
                        ).edges
                    ],
                    graph_path_ids=[path_ids[0]],
                )
            )
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                split_group_id=source_group_id,
                domain="pat_questions",
                blueprint=f"pat_{row.get('_hop_type', 'unknown')}_snapshot_path",
                entities=scenario_entities,
                facts=scenario_facts,
                snapshots=[snapshot],
                question_ids=[qid],
                update_behavior="not_applicable",
                source_provenance=SourceProvenance(
                    source_id=source_group_id,
                    source_family="pat_questions",
                    fidelity="source_record_converted",
                    source_record_ids=[str(row.get("uniq_id") or source_group_id)],
                    source_revision=snapshot_name,
                    source_relation=" -> ".join(
                        f"{relation} ({PAT_PROPERTY_LABELS[relation]})" for relation in relations
                    ),
                    topology_signature=f"{len(relations)}-edge-source-path",
                ),
                notes=(
                    "Converted as a dated Wikidata snapshot question. Snapshot observation time "
                    "is public; no world-valid interval is inferred."
                ),
            )
        )

    return _bundle(
        scenarios=scenarios,
        entities=entities,
        facts=facts,
        snapshots=snapshots,
        questions=questions,
        graph_paths=graph_paths,
        context_packs=context_packs,
        answer_variants=answer_variants,
        namespace="pat-certified",
    )


def build_hoh_bundle(rows: list[dict[str, Any]]) -> DatasetBundle:
    scenarios: list[Scenario] = []
    entities: list[Entity] = []
    facts: list[Fact] = []
    snapshots: list[Snapshot] = []
    questions: list[Question] = []
    context_packs: list[ContextPack] = []
    answer_variants: list[AnswerVariant] = []

    for row_index, row in enumerate(rows):
        current_answer = clean_english_text(str(row.get("answer") or ""))
        current_evidence = clean_english_text(str(row.get("evidence") or ""))
        if not current_answer or not current_evidence:
            continue
        revision_date = _required_date(
            row.get("last_modified_time"),
            field_name="last_modified_time",
        )
        source_group_id = hoh_semantic_series_id(row)
        scenario_id = stable_opaque_id("hoh", source_group_id, row_index)
        qid = f"q_{scenario_id}"
        document = row.get("document") or {}
        document_entity = Entity(
            entity_id=f"e_{scenario_id}_document",
            name=clean_english_text(
                str(document.get("title") or document.get("id") or "source document")
            ),
            entity_type=EntityType.DOCUMENT,
            aliases=[str(document.get("id"))] if document.get("id") else [],
            domain="hoh_document_revision",
        )
        current_entity = Entity(
            entity_id=f"e_{scenario_id}_answer_current",
            name=current_answer,
            entity_type=EntityType.ANSWER_VALUE,
            domain="hoh_document_revision",
        )
        current_fact = _document_fact(
            scenario_id=scenario_id,
            document=document_entity,
            answer=current_entity,
            evidence=current_evidence,
            revision_date=revision_date,
            suffix="current",
            visible_from="S1",
        )
        scenario_entities = [document_entity, current_entity]
        scenario_facts = [current_fact]
        older_rows: list[tuple[Fact, Entity]] = []
        for older_index, older in enumerate(row.get("outdated_infos") or []):
            older_answer = clean_english_text(str(older.get("answer") or ""))
            older_evidence = clean_english_text(str(older.get("evidence") or ""))
            if not older_answer or not older_evidence:
                continue
            older_date = _required_date(
                older.get("last_modified_time"),
                field_name="outdated_infos.last_modified_time",
            )
            older_entity = Entity(
                entity_id=f"e_{scenario_id}_answer_older_{older_index}",
                name=older_answer,
                entity_type=EntityType.ANSWER_VALUE,
                domain="hoh_document_revision",
            )
            older_fact = _document_fact(
                scenario_id=scenario_id,
                document=document_entity,
                answer=older_entity,
                evidence=older_evidence,
                revision_date=older_date,
                suffix=f"older_{older_index}",
                visible_from="S0",
            )
            scenario_entities.append(older_entity)
            scenario_facts.append(older_fact)
            older_rows.append((older_fact, older_entity))

        question_base = clean_english_text(str(row.get("question") or ""))
        question_text = (
            f"According to the document revision for {document_entity.name} dated "
            f"{_date_text(revision_date)}, "
            f"{_lower_initial(question_base)}"
        )
        program = QuestionProgram(
            operator=TemporalOperator.AS_OF,
            target="document-revision answer",
            query_time=point(revision_date.year, revision_date.month, revision_date.day),
            relation=Relation.DOCUMENT_SUPPORTS,
            context_id=document_entity.entity_id,
            snapshot_id="S1",
            answer_function="latest_document_revision_answer",
            required_path_semantics="not_applicable",
            temporal_basis="document_revision",
        )
        question = Question(
            qid=qid,
            scenario_id=scenario_id,
            dataset_family=DatasetFamily.HOH,
            canonical_question=question_text,
            question=question_text,
            program=program,
            temporal_operator=TemporalOperator.AS_OF,
            answer_type=_hoh_answer_type(question_base, current_answer),
            gold_answer_entity_ids=[current_entity.entity_id],
            gold_answer_text=[current_answer],
            required_valid_evidence_ids=[current_fact.fact_id],
            should_abstain=False,
            system_difficulty=SystemDifficulty.MEDIUM,
            eval_difficulty=EvalDifficulty.MEDIUM,
            difficulty_provenance="source_heuristic",
            human_pool_candidate=True,
            semantic_series_id=source_group_id,
            template_family_id="hoh:document_revision",
            certification_status="certified",
        )
        questions.append(question)
        answer_variants.append(
            _source_correct_variant(
                question=question,
                answer_text=current_answer,
                evidence_ids=[current_fact.fact_id],
                graph_path_ids=[],
                temporal_basis="document_revision",
            )
        )
        normalized_current = _normalized(current_answer)
        distinct_older = next(
            (
                (fact, entity)
                for fact, entity in older_rows
                if _normalized(entity.name) != normalized_current
            ),
            None,
        )
        if distinct_older is not None:
            older_fact, older_entity = distinct_older
            answer_variants.append(
                AnswerVariant(
                    answer_id=f"a_{qid}_outdated_source",
                    qid=qid,
                    scenario_id=scenario_id,
                    variant_type=AnswerVariantType.OUTDATED_SOURCE_ANSWER,
                    answer_text=older_entity.name,
                    cited_evidence_ids=[older_fact.fact_id],
                    claims=[
                        AnswerClaim(
                            cid=f"c_{qid}_outdated_source",
                            text=older_entity.name,
                            claim_time=program.query_time,
                            cited_evidence_ids=[older_fact.fact_id],
                            temporally_valid=None,
                        )
                    ],
                    answer_correct="no",
                    temporal_correct="not_applicable",
                    evidence_supports_answer="yes",
                    citation_temporally_valid="not_applicable",
                    graph_path_sufficient="not_applicable",
                    refusal_appropriate="not_applicable",
                )
            )
        s0_fact_ids = [fact.fact_id for fact, _ in older_rows]
        snapshots_for_scenario = [
            Snapshot(
                scenario_id=scenario_id,
                snapshot_id="S0",
                snapshot_time=max(
                    (fact.publication_time for fact, _ in older_rows if fact.publication_time),
                    default=revision_date,
                ),
                visible_fact_ids=s0_fact_ids,
                description="Earlier document revision evidence.",
            ),
            Snapshot(
                scenario_id=scenario_id,
                snapshot_id="S1",
                snapshot_time=revision_date,
                visible_fact_ids=[fact.fact_id for fact in scenario_facts],
                description="Latest document revision with revision history retained.",
            ),
        ]
        snapshots.extend(snapshots_for_scenario)
        facts.extend(scenario_facts)
        entities.extend(scenario_entities)
        context_packs.append(
            ContextPack(
                pack_id=f"cp_{qid}_revision_history",
                qid=qid,
                scenario_id=scenario_id,
                pack_type=ContextPackType.VALID_PLUS_STALE,
                evidence_ids=[fact.fact_id for fact in scenario_facts],
                expected_behavior="answer_from_named_document_revision",
            )
        )
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                split_group_id=source_group_id,
                domain="hoh_document_revision",
                blueprint="dated_document_revision",
                entities=scenario_entities,
                facts=scenario_facts,
                snapshots=snapshots_for_scenario,
                question_ids=[qid],
                update_behavior="not_applicable",
                source_provenance=SourceProvenance(
                    source_id=source_group_id,
                    source_family="hoh_qas",
                    fidelity="source_record_converted",
                    source_record_ids=[str(document.get("id") or source_group_id)],
                    source_revision=revision_date.isoformat(),
                    source_relation="document revision supports answer",
                    topology_signature="revision-history-evidence",
                ),
                notes=(
                    "Converted as document-revision freshness, not world-valid temporal change. "
                    "Graph-path and temporal-validity judgments are intentionally not applicable."
                ),
            )
        )

    return _bundle(
        scenarios=scenarios,
        entities=entities,
        facts=facts,
        snapshots=snapshots,
        questions=questions,
        graph_paths=[],
        context_packs=context_packs,
        answer_variants=answer_variants,
        namespace="hoh-certified",
    )


def clean_english_text(value: str) -> str:
    cleaned = fix_text(html.unescape(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\bthe the\b", "the", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(\bdid\b[^.!?]{0,120})\battended\b",
        r"\1attend",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def pat_row_is_convertible(row: dict[str, Any]) -> bool:
    subject = row.get("subject") or {}
    answers = row.get("answer annotations") or []
    relations = [str(value) for value in row.get("relations") or []]
    intermediates = row.get("intermediate entities") or []
    question = clean_english_text(str(row.get("question") or ""))
    return bool(
        question
        and subject.get("subject")
        and _human_readable_source_label(subject.get("subLabel"))
        and answers
        and all(
            answer.get("ID") and _human_readable_source_label(answer.get("Label"))
            for answer in answers
        )
        and relations
        and all(relation in PAT_PROPERTY_LABELS for relation in relations)
        and len(relations) == len(intermediates) + 1
        and all(
            entity.get("ID") and _human_readable_source_label(entity.get("Label"))
            for entity in intermediates
        )
        and _pat_temporal_semantics_are_explicit(question, relations)
    )


def _pat_temporal_semantics_are_explicit(question: str, relations: list[str]) -> bool:
    """Reject predecessor questions whose graph contains no predecessor relation.

    A dated subject-relation-answer edge establishes a source-snapshot value, but it cannot by
    itself establish that the value was previous. PAT rows with an explicit ``P1365``
    (``replaces``) edge encode that contrast. Other predecessor rows are not certifiable as
    evidence-grounded human-evaluation cards and are conservatively excluded.
    """

    lowered = question.casefold()
    asks_for_previous = "previous" in lowered or "before the current" in lowered
    return not asks_for_previous or "P1365" in relations


def _human_readable_source_label(value: Any) -> bool:
    label = clean_english_text(str(value or ""))
    if not label or len(label) > 200:
        return False
    if re.search(r"(?i)(?:https?://|www\.|\.well-known/genid)", label):
        return False
    if re.fullmatch(r"[PQ]\d+", label, flags=re.IGNORECASE):
        return False
    return label.casefold() not in {"none", "null", "nan", "unknown", "n/a"}


def _pat_question_text(original_question: str, *, relations: list[str]) -> str:
    if relations != ["P102"]:
        return original_question
    current = re.fullmatch(
        r"Which political party does (.+) belong to currently\?",
        original_question,
        flags=re.IGNORECASE,
    )
    if current:
        return f"What is the current political affiliation of {current.group(1)}?"
    previous = re.fullmatch(
        r"Which political party did (.+) belong to before the current political party\?",
        original_question,
        flags=re.IGNORECASE,
    )
    if previous:
        return f"What was the previous political affiliation of {previous.group(1)}?"
    return original_question


def pat_semantic_series_id(row: dict[str, Any]) -> str:
    subject = row.get("subject") or {}
    payload = "\x1f".join(
        [
            clean_english_text(str(row.get("template") or row.get("question") or "")).casefold(),
            str(subject.get("subject") or subject.get("subLabel") or ""),
            ",".join(str(value) for value in row.get("relations") or []),
        ]
    )
    return "pat_series_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def hoh_semantic_series_id(row: dict[str, Any]) -> str:
    document = row.get("document") or {}
    payload = "\x1f".join(
        [
            clean_english_text(str(row.get("question") or "")).casefold(),
            clean_english_text(str(document.get("id") or document.get("title") or "")).casefold(),
        ]
    )
    return "hoh_series_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _source_correct_variant(
    *,
    question: Question,
    answer_text: str,
    evidence_ids: list[str],
    graph_path_ids: list[str],
    temporal_basis: str,
) -> AnswerVariant:
    return AnswerVariant(
        answer_id=f"a_{question.qid}_correct",
        qid=question.qid,
        scenario_id=question.scenario_id,
        variant_type=AnswerVariantType.CORRECT_SUPPORTED,
        answer_text=answer_text,
        cited_evidence_ids=evidence_ids,
        graph_path_ids=graph_path_ids,
        claims=[
            AnswerClaim(
                cid=f"c_{question.qid}_correct",
                text=answer_text,
                claim_time=question.program.query_time,
                cited_evidence_ids=evidence_ids,
                temporally_valid=None,
            )
        ],
        answer_correct="yes",
        temporal_correct="not_applicable",
        evidence_supports_answer="yes",
        citation_temporally_valid="not_applicable",
        graph_path_sufficient="yes" if graph_path_ids else "not_applicable",
        refusal_appropriate="not_applicable",
    )


def _source_partial_variant(
    *,
    question: Question,
    answer_text: str,
    evidence_ids: list[str],
    graph_path_ids: list[str],
) -> AnswerVariant:
    return AnswerVariant(
        answer_id=f"a_{question.qid}_partial",
        qid=question.qid,
        scenario_id=question.scenario_id,
        variant_type=AnswerVariantType.PARTIAL_ANSWER,
        answer_text=answer_text,
        cited_evidence_ids=evidence_ids,
        graph_path_ids=graph_path_ids,
        claims=[
            AnswerClaim(
                cid=f"c_{question.qid}_partial",
                text=answer_text,
                claim_time=question.program.query_time,
                cited_evidence_ids=evidence_ids,
                temporally_valid=None,
            )
        ],
        answer_correct="partial",
        temporal_correct="not_applicable",
        evidence_supports_answer="yes",
        citation_temporally_valid="not_applicable",
        graph_path_sufficient="partial",
        refusal_appropriate="not_applicable",
    )


def _source_inappropriate_refusal_variant(*, question: Question) -> AnswerVariant:
    answer_text = "There is not enough information to answer."
    return AnswerVariant(
        answer_id=f"a_{question.qid}_inappropriate_refusal",
        qid=question.qid,
        scenario_id=question.scenario_id,
        variant_type=AnswerVariantType.INAPPROPRIATE_REFUSAL,
        answer_text=answer_text,
        cited_evidence_ids=[],
        graph_path_ids=[],
        claims=[
            AnswerClaim(
                cid=f"c_{question.qid}_inappropriate_refusal",
                text=answer_text,
                claim_time=question.program.query_time,
                cited_evidence_ids=[],
                temporally_valid=None,
            )
        ],
        answer_correct="no",
        temporal_correct="not_applicable",
        evidence_supports_answer="not_applicable",
        citation_temporally_valid="not_applicable",
        graph_path_sufficient="not_applicable",
        refusal_appropriate="no",
    )


def _source_hallucinated_variant(
    *,
    question: Question,
    answer_text: str,
) -> AnswerVariant:
    return AnswerVariant(
        answer_id=f"a_{question.qid}_unsupported",
        qid=question.qid,
        scenario_id=question.scenario_id,
        variant_type=AnswerVariantType.HALLUCINATED_ANSWER,
        answer_text=answer_text,
        cited_evidence_ids=[],
        graph_path_ids=[],
        claims=[
            AnswerClaim(
                cid=f"c_{question.qid}_unsupported",
                text=answer_text,
                claim_time=question.program.query_time,
                cited_evidence_ids=[],
                temporally_valid=None,
            )
        ],
        answer_correct="no",
        temporal_correct="not_applicable",
        evidence_supports_answer="not_applicable",
        citation_temporally_valid="not_applicable",
        graph_path_sufficient="not_applicable",
        refusal_appropriate="not_applicable",
    )


def _pat_relation_matched_decoys(rows: list[dict[str, Any]]) -> dict[int, str]:
    pools: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        relations = [str(value) for value in row.get("relations") or []]
        if not relations:
            continue
        for answer in row.get("answer annotations") or []:
            label = clean_english_text(str(answer.get("Label") or ""))
            if label and label not in pools[relations[-1]]:
                pools[relations[-1]].append(label)

    result: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        relations = [str(value) for value in row.get("relations") or []]
        gold = {
            normalize_visible_text(clean_english_text(str(answer.get("Label") or "")))
            for answer in row.get("answer annotations") or []
        }
        candidates = [
            label for label in pools[relations[-1]] if normalize_visible_text(label) not in gold
        ]
        if not candidates:
            continue
        digest_input = str(row.get("uniq_id", row_index)).encode()
        offset = int(hashlib.sha256(digest_input).hexdigest()[:8], 16)
        result[row_index] = candidates[offset % len(candidates)]
    return result


def _document_fact(
    *,
    scenario_id: str,
    document: Entity,
    answer: Entity,
    evidence: str,
    revision_date: date,
    suffix: str,
    visible_from: str,
) -> Fact:
    return Fact(
        fact_id=stable_opaque_id("f", scenario_id, "document_revision", suffix),
        scenario_id=scenario_id,
        subject_id=answer.entity_id,
        relation=Relation.DOCUMENT_SUPPORTS,
        context_id=document.entity_id,
        answer_entity_id=answer.entity_id,
        graph_source_id=document.entity_id,
        graph_target_id=answer.entity_id,
        source_relation_id="document_revision_supports_answer",
        source_relation_label="document revision supports answer",
        relation_direction="directed",
        source_record_id=f"{document.entity_id}:{revision_date.isoformat()}:{suffix}",
        source_revision=revision_date.isoformat(),
        valid_time=unknown_interval(),
        publication_time=revision_date,
        transaction_time=revision_date,
        snapshot_visible_from=visible_from,
        source_type="hoh_document_revision",
        provenance_reliability="high",
        fact_role=FactRole.SOURCE_ASSERTION,
        canonical_evidence=evidence,
    )


def _source_entity(*, scenario_id: str, source_id: str, label: str, suffix: str) -> Entity:
    return Entity(
        entity_id=stable_opaque_id("e", scenario_id, suffix, source_id),
        name=label,
        entity_type=EntityType.WIKIDATA_ENTITY,
        aliases=[source_id] if source_id else [],
        domain="pat_questions",
    )


def _bundle(
    *,
    scenarios: list[Scenario],
    entities: list[Entity],
    facts: list[Fact],
    snapshots: list[Snapshot],
    questions: list[Question],
    graph_paths: list[GraphPath],
    context_packs: list[ContextPack],
    answer_variants: list[AnswerVariant],
    namespace: str,
) -> DatasetBundle:
    return DatasetBundle(
        scenarios=scenarios,
        entities=entities,
        facts=facts,
        snapshots=snapshots,
        questions=questions,
        graph_paths=graph_paths,
        context_packs=context_packs,
        answer_variants=answer_variants,
        splits=stable_group_splits(
            {
                scenario.scenario_id: scenario.split_group_id or scenario.scenario_id
                for scenario in scenarios
            },
            namespace=namespace,
        ),
    )


def _required_date(value: object, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    raise ValueError(f"Missing or invalid required source date: {field_name}")


def _hoh_answer_type(question: str, answer: str) -> AnswerType:
    lowered = question.casefold()
    if any(marker in lowered for marker in ("what year", "what date", "when ")):
        return AnswerType.TIME
    if ";" in answer or " and " in answer and "," in answer:
        return AnswerType.LIST
    if lowered.startswith(("who ", "where ", "which ", "what is the name")):
        return AnswerType.ENTITY
    return AnswerType.SPAN


def _pat_operator(question: str, template: str) -> TemporalOperator:
    text = f"{question} {template}".casefold()
    markers = (
        ("previous", TemporalOperator.PREVIOUS),
        ("current", TemporalOperator.CURRENT),
        ("before", TemporalOperator.BEFORE),
        ("after", TemporalOperator.AFTER),
        ("latest", TemporalOperator.LATEST),
        ("first", TemporalOperator.FIRST),
    )
    for marker, operator in markers:
        if marker in text:
            return operator
    return TemporalOperator.AS_OF


def _lower_initial(value: str) -> str:
    if not value:
        return value
    return value[0].lower() + value[1:]


def _date_text(value: date) -> str:
    return value.strftime("%B %d, %Y").replace(" 0", " ")


def _sentence_terminator(value: str) -> str:
    return "" if value.endswith((".", "!", "?")) else "."


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())
