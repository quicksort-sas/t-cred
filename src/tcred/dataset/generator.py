from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from datetime import date

from tcred.dataset.domains import DOMAIN_SPECS, DomainSpec
from tcred.dataset.graph import fact_answer_id, graph_path_node_ids
from tcred.dataset.intervals import make_interval, point, unknown_interval
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
    EvalDifficulty,
    Fact,
    FactRole,
    GraphPath,
    GraphPathEdge,
    PathTimeStatus,
    Question,
    QuestionProgram,
    Scenario,
    Snapshot,
    SourceProvenance,
    SystemDifficulty,
    TemporalInterval,
    TemporalOperator,
    TimeStatus,
)
from tcred.dataset.solver import (
    GoldSolver,
    expected_pack_behavior,
    fact_visible,
    path_query_time,
    path_time_status,
    time_status_for_fact,
)
from tcred.dataset.splits import stable_group_splits
from tcred.dataset.text import normalize_visible_text
from tcred.dataset.verbalize import (
    answer_sentence,
    canonical_question,
    entity_lookup,
    fact_sentence,
    naturalize_question,
)

FIRST_NAMES = (
    "Mira",
    "Jon",
    "Leah",
    "Arun",
    "Nadia",
    "Theo",
    "Iris",
    "Samir",
    "Elena",
    "Noah",
    "Priya",
    "Mateo",
    "Lina",
    "Owen",
    "Clara",
    "Dalia",
)
LAST_NAMES = (
    "Chen",
    "Vale",
    "Stone",
    "Rao",
    "Keller",
    "Okafor",
    "Silva",
    "Moreau",
    "Nakamura",
    "Haddad",
    "Voss",
    "Grant",
)
ORG_PREFIXES = (
    "Orion",
    "Helio",
    "Northstar",
    "Cedar",
    "Bluefield",
    "Aster",
    "Harbor",
    "Summit",
    "Vela",
    "Keystone",
)
VERSION_NAMES = ("1.0", "1.4", "2.0", "2.5", "3.0", "Aurora", "Beacon", "Cascade")
LOCATIONS = ("Riverton", "Northbridge", "Eastmere", "Lakeview", "Stoneport", "Redwood")


class SyntheticDatasetGenerator:
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.solver = GoldSolver()

    def generate(self, scenario_count: int, questions_per_scenario: int = 4) -> DatasetBundle:
        scenarios: list[Scenario] = []
        entities: list[Entity] = []
        facts: list[Fact] = []
        snapshots: list[Snapshot] = []
        questions: list[Question] = []
        graph_paths: list[GraphPath] = []
        context_packs: list[ContextPack] = []
        answer_variants: list[AnswerVariant] = []

        for index in range(scenario_count):
            spec = DOMAIN_SPECS[index % len(DOMAIN_SPECS)]
            scenario_data = self._generate_scenario(index=index, spec=spec)
            scenario_entities = scenario_data["entities"]
            scenario_facts = scenario_data["facts"]
            scenario_snapshots = scenario_data["snapshots"]
            scenario_questions = self._generate_questions(
                index=index,
                spec=spec,
                entities=scenario_entities,
                facts=scenario_facts,
                questions_per_scenario=questions_per_scenario,
            )
            scenario_graph_paths = self._generate_graph_paths(
                questions=scenario_questions,
                facts=scenario_facts,
                scenario_id=scenario_data["scenario_id"],
            )
            scenario_contexts = self._generate_context_packs(
                questions=scenario_questions,
                facts=scenario_facts,
            )
            lookup = entity_lookup(scenario_entities)
            scenario_answers = self._generate_answer_variants(
                questions=scenario_questions,
                facts=scenario_facts,
                entities=lookup,
                paths=scenario_graph_paths,
            )

            scenario = Scenario(
                scenario_id=scenario_data["scenario_id"],
                split_group_id=scenario_data["scenario_id"],
                domain=spec.domain,
                blueprint=spec.blueprint,
                entities=scenario_entities,
                facts=scenario_facts,
                snapshots=scenario_snapshots,
                question_ids=[question.qid for question in scenario_questions],
                update_behavior=scenario_data["update_behavior"],
                source_provenance=_source_provenance(scenario_data),
                notes=_scenario_notes(scenario_data),
            )

            scenarios.append(scenario)
            entities.extend(scenario_entities)
            facts.extend(scenario_facts)
            snapshots.extend(scenario_snapshots)
            questions.extend(scenario_questions)
            graph_paths.extend(scenario_graph_paths)
            context_packs.extend(scenario_contexts)
            answer_variants.extend(scenario_answers)

        splits = self._split_scenarios([scenario.scenario_id for scenario in scenarios])
        return DatasetBundle(
            scenarios=scenarios,
            entities=entities,
            facts=facts,
            snapshots=snapshots,
            questions=questions,
            graph_paths=graph_paths,
            context_packs=context_packs,
            answer_variants=answer_variants,
            splits=splits,
        )

    def _generate_scenario(self, index: int, spec: DomainSpec) -> dict[str, object]:
        scenario_id = f"synth_{index:04d}"
        context = self._entity(
            scenario_id,
            f"ctx_{index}",
            self._context_name(spec),
            spec.context_type,
            spec.domain,
        )
        obj = self._entity(
            scenario_id,
            f"obj_{index}",
            self.rng.choice(spec.object_names),
            spec.object_type,
            spec.domain,
        )
        answer_entities = [
            self._answer_entity(scenario_id, index, candidate_index, spec)
            for candidate_index in range(5)
        ]
        hard_negative_context = self._entity(
            scenario_id,
            f"negctx_{index}",
            self._context_name(spec),
            spec.context_type,
            spec.domain,
        )

        update_behavior = _update_behavior_for_index(index)
        entities = [context, obj, hard_negative_context, *answer_entities]
        facts: list[Fact] = []
        intervals = _timeline_interval_profile(index)
        roles = [
            FactRole.STALE_DISTRACTOR,
            FactRole.STALE_DISTRACTOR,
            FactRole.STALE_DISTRACTOR,
            FactRole.VALID_SUPPORT,
            FactRole.FUTURE_DISTRACTOR,
        ]
        for candidate_index, answer in enumerate(answer_entities):
            fact = Fact(
                fact_id=stable_opaque_id("f", scenario_id, "timeline", candidate_index),
                scenario_id=scenario_id,
                subject_id=answer.entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=context.entity_id,
                valid_time=intervals[candidate_index],
                publication_time=(
                    date(2023, 11, 15)
                    if candidate_index == 4
                    else date(intervals[candidate_index].start.year, 2, 15)
                    if intervals[candidate_index].start
                    else None
                ),
                transaction_time=(
                    date(2023, 12, 1)
                    if candidate_index == 4
                    else date(intervals[candidate_index].start.year, 3, 1)
                    if intervals[candidate_index].start
                    else None
                ),
                snapshot_visible_from=_snapshot_visible_from_for_timeline(
                    update_behavior=update_behavior,
                    candidate_index=candidate_index,
                ),
                source_type=spec.source_type,
                provenance_reliability="high",
                fact_role=roles[candidate_index],
                canonical_evidence="",
            )
            facts.append(fact)

        facts.append(
            Fact(
                fact_id=stable_opaque_id("f", scenario_id, "conflict"),
                scenario_id=scenario_id,
                subject_id=answer_entities[2].entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=context.entity_id,
                valid_time=make_interval(2024, None),
                publication_time=date(2024, 5, 1),
                transaction_time=date(2024, 5, 2),
                snapshot_visible_from="S1",
                source_type="conflicting_report",
                provenance_reliability="low",
                fact_role=FactRole.CONTRADICTORY,
                canonical_evidence="",
            )
        )
        facts.append(
            Fact(
                fact_id=stable_opaque_id("f", scenario_id, "unknown"),
                scenario_id=scenario_id,
                subject_id=answer_entities[1].entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=context.entity_id,
                valid_time=unknown_interval(),
                publication_time=date(2022, 9, 1),
                transaction_time=date(2022, 9, 1),
                snapshot_visible_from="S0",
                source_type="undated_note",
                provenance_reliability="medium",
                fact_role=FactRole.UNKNOWN_TIME,
                canonical_evidence="",
            )
        )
        facts.append(
            Fact(
                fact_id=stable_opaque_id("f", scenario_id, "publication_only"),
                scenario_id=scenario_id,
                subject_id=answer_entities[0].entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=context.entity_id,
                valid_time=unknown_interval(),
                publication_time=date(2024, 4, 20),
                transaction_time=date(2024, 4, 21),
                snapshot_visible_from="S1",
                source_type="news_article",
                provenance_reliability="medium",
                fact_role=FactRole.PUBLICATION_ONLY,
                canonical_evidence="",
            )
        )
        facts.append(
            Fact(
                fact_id=stable_opaque_id("f", scenario_id, "hard_negative"),
                scenario_id=scenario_id,
                subject_id=answer_entities[3].entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=hard_negative_context.entity_id,
                valid_time=make_interval(2024, None),
                publication_time=date(2024, 1, 5),
                transaction_time=date(2024, 1, 5),
                snapshot_visible_from="S1",
                source_type=spec.source_type,
                provenance_reliability="high",
                fact_role=FactRole.HARD_NEGATIVE,
                canonical_evidence="",
            )
        )
        facts.append(
            Fact(
                fact_id=stable_opaque_id("f", scenario_id, "historical_same_answer"),
                scenario_id=scenario_id,
                subject_id=answer_entities[3].entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=context.entity_id,
                valid_time=make_interval(2019, 2020),
                publication_time=date(2020, 1, 8),
                transaction_time=date(2020, 1, 8),
                snapshot_visible_from="S0",
                source_type=spec.source_type,
                provenance_reliability="high",
                fact_role=FactRole.BACKGROUND,
                canonical_evidence="",
            )
        )
        facts.append(
            Fact(
                fact_id=stable_opaque_id("f", scenario_id, "path_bad"),
                scenario_id=scenario_id,
                subject_id=answer_entities[2].entity_id,
                relation=spec.relation,
                object_id=obj.entity_id,
                context_id=hard_negative_context.entity_id,
                valid_time=make_interval(2019, 2020),
                publication_time=date(2020, 1, 8),
                transaction_time=date(2020, 1, 8),
                snapshot_visible_from="S0",
                source_type=spec.source_type,
                provenance_reliability="high",
                fact_role=FactRole.GRAPH_INCOHERENT,
                canonical_evidence="",
            )
        )

        lookup = entity_lookup(entities)
        facts = [
            fact.model_copy(update={"canonical_evidence": fact_sentence(fact, lookup)})
            for fact in facts
        ]

        s0_visible = [fact.fact_id for fact in facts if fact.snapshot_visible_from == "S0"]
        s1_visible = [fact.fact_id for fact in facts if fact_visible(fact, "S1")]
        snapshots = [
            Snapshot(
                scenario_id=scenario_id,
                snapshot_id="S0",
                snapshot_time=date(2023, 12, 31),
                visible_fact_ids=s0_visible,
                description="Initial snapshot before the latest update.",
            ),
            Snapshot(
                scenario_id=scenario_id,
                snapshot_id="S1",
                snapshot_time=date(2024, 6, 1),
                visible_fact_ids=s1_visible,
                description=(
                    "Snapshot after current valid fact and conflicting update become visible."
                ),
            ),
        ]

        return {
            "scenario_id": scenario_id,
            "entities": entities,
            "facts": facts,
            "snapshots": snapshots,
            "update_behavior": update_behavior,
        }

    def _generate_questions(
        self,
        *,
        index: int,
        spec: DomainSpec,
        entities: list[Entity],
        facts: list[Fact],
        questions_per_scenario: int,
    ) -> list[Question]:
        lookup = entity_lookup(entities)
        context = next(entity for entity in entities if "_ctx_" in entity.entity_id)
        obj = next(entity for entity in entities if "_obj_" in entity.entity_id)
        scenario_id = facts[0].scenario_id
        operator_cycle = [
            TemporalOperator.CURRENT,
            TemporalOperator.AS_OF,
            TemporalOperator.PREVIOUS,
            TemporalOperator.NEXT,
            TemporalOperator.DURING,
            TemporalOperator.BETWEEN,
            TemporalOperator.BEFORE,
            TemporalOperator.AFTER,
            TemporalOperator.FIRST,
            TemporalOperator.LATEST,
            TemporalOperator.LAST,
            TemporalOperator.EFFECTIVE,
            TemporalOperator.EXPIRED,
        ]
        questions: list[Question] = []
        for q_index in range(questions_per_scenario):
            operator = operator_cycle[(index + q_index) % len(operator_cycle)]
            query_time = self._query_interval_for_operator(operator)
            snapshot_id = "S1"
            if operator == TemporalOperator.AS_OF and index % 4 == 0:
                snapshot_id = "S0"
            if index % 5 == 0 and q_index == questions_per_scenario - 1:
                operator = TemporalOperator.CURRENT
                query_time = point(2026, 6, 1)
                snapshot_id = "S0"
            program = QuestionProgram(
                operator=operator,
                target=spec.question_noun,
                query_time=query_time,
                relation=spec.relation,
                context_id=context.entity_id,
                object_id=obj.entity_id,
                snapshot_id=snapshot_id,
                answer_function=f"{operator}_entity_validity",
                required_path_semantics="shared_interval"
                if operator in {TemporalOperator.CURRENT, TemporalOperator.AS_OF}
                else "single_fact",
            )
            answer_ids, evidence_ids = self.solver.solve(facts, program)
            should_abstain = len(answer_ids) == 0
            canonical = canonical_question(program, lookup)
            qid = f"q_{scenario_id}_{q_index:02d}"
            question = Question(
                qid=qid,
                scenario_id=scenario_id,
                dataset_family=DatasetFamily.SYNTH,
                canonical_question=canonical,
                question=naturalize_question(canonical, index + q_index),
                program=program,
                temporal_operator=operator,
                answer_type=AnswerType.REFUSAL if should_abstain else AnswerType.ENTITY,
                gold_answer_entity_ids=answer_ids,
                gold_answer_text=[lookup[entity_id].name for entity_id in answer_ids],
                required_valid_evidence_ids=evidence_ids,
                should_abstain=should_abstain,
                system_difficulty=self._system_difficulty(operator, q_index),
                eval_difficulty=self._eval_difficulty(operator, index, q_index),
                human_pool_candidate=q_index < 2,
            )
            questions.append(question)
        return questions

    def _generate_graph_paths(
        self,
        *,
        questions: list[Question],
        facts: list[Fact],
        scenario_id: str,
    ) -> list[GraphPath]:
        paths: list[GraphPath] = []
        fact_by_id = {fact.fact_id: fact for fact in facts}
        for question in questions:
            if not question.required_valid_evidence_ids:
                continue
            for evidence_index, evidence_id in enumerate(question.required_valid_evidence_ids):
                valid_fact = fact_by_id[evidence_id]
                direct_edges = [
                    GraphPathEdge(
                        fact_id=valid_fact.fact_id,
                        relation=valid_fact.relation,
                        valid_time=valid_fact.valid_time,
                    )
                ]
                paths.append(
                    GraphPath(
                        pid=stable_opaque_id("p", question.qid, "direct", evidence_index),
                        scenario_id=scenario_id,
                        qid=question.qid,
                        nodes=graph_path_node_ids(direct_edges, fact_by_id),
                        edges=direct_edges,
                        path_time_status=path_time_status(
                            direct_edges,
                            query_time=path_query_time(question.program),
                        ),
                        supports_gold_answer=True,
                        explanation=(
                            "This directed source fact supports one complete answer claim."
                        ),
                    )
                )
            query_time = path_query_time(question.program)
            negative_fact = None
            negative_status = None
            if query_time is not None:
                for candidate in facts:
                    if candidate.fact_role not in {
                        FactRole.HARD_NEGATIVE,
                        FactRole.GRAPH_INCOHERENT,
                    } or not fact_visible(candidate, question.program.snapshot_id):
                        continue
                    candidate_edge = GraphPathEdge(
                        fact_id=candidate.fact_id,
                        relation=candidate.relation,
                        valid_time=candidate.valid_time,
                    )
                    candidate_status = path_time_status(
                        [candidate_edge],
                        query_time=query_time,
                    )
                    if candidate_status == PathTimeStatus.COHERENT_SHARED_INTERVAL:
                        negative_fact = candidate
                        negative_status = candidate_status
                        break
            if negative_fact is not None and negative_status is not None:
                edges = [
                    GraphPathEdge(
                        fact_id=negative_fact.fact_id,
                        relation=negative_fact.relation,
                        valid_time=negative_fact.valid_time,
                    ),
                ]
                paths.append(
                    GraphPath(
                        pid=stable_opaque_id("p", question.qid, "wrong_context"),
                        scenario_id=scenario_id,
                        qid=question.qid,
                        nodes=graph_path_node_ids(edges, fact_by_id),
                        edges=edges,
                        path_time_status=negative_status,
                        supports_gold_answer=False,
                        explanation=(
                            "The edge belongs to a different context and cannot support "
                            "the requested answer."
                        ),
                    )
                )
        return paths

    def _generate_context_packs(
        self, *, questions: list[Question], facts: list[Fact]
    ) -> list[ContextPack]:
        packs: list[ContextPack] = []
        for question in questions:
            visible_facts = [
                fact for fact in facts if fact_visible(fact, question.program.snapshot_id)
            ]
            by_status: dict[TimeStatus, list[str]] = defaultdict(list)
            for fact in visible_facts:
                by_status[
                    time_status_for_fact(
                        fact,
                        question.program,
                        selected_fact_ids=set(question.required_valid_evidence_ids),
                    )
                ].append(fact.fact_id)
            valid = question.required_valid_evidence_ids
            stale = by_status[TimeStatus.STALE]
            future = by_status[TimeStatus.FUTURE_INVALID]
            unknown = by_status[TimeStatus.UNKNOWN_VALID_TIME]
            publication = by_status[TimeStatus.PUBLICATION_ONLY]
            irrelevant = by_status[TimeStatus.IRRELEVANT]
            graph_incoherent = [
                fact.fact_id
                for fact in visible_facts
                if fact.fact_role == FactRole.GRAPH_INCOHERENT
            ]
            candidates = [
                (ContextPackType.VALID_ONLY, valid[:3]),
                (ContextPackType.STALE_ONLY, stale[:3]),
                (ContextPackType.FUTURE_ONLY, future[:3]),
                (ContextPackType.VALID_PLUS_STALE, (valid[:2] + stale[:2])),
                (ContextPackType.VALID_PLUS_FUTURE, (valid[:2] + future[:2])),
                (ContextPackType.PUBLICATION_ONLY, publication[:2]),
                (ContextPackType.UNKNOWN_TIME, unknown[:2]),
                (ContextPackType.GRAPH_INCOHERENT, graph_incoherent[:2]),
                (ContextPackType.INSUFFICIENT, irrelevant[:2]),
            ]
            if by_status[TimeStatus.VALID] and len(by_status[TimeStatus.VALID]) > len(valid):
                candidates.append((ContextPackType.CONFLICT, by_status[TimeStatus.VALID][:4]))
            for pack_type, evidence_ids in candidates:
                if not evidence_ids:
                    continue
                packs.append(
                    ContextPack(
                        pack_id=f"cp_{question.qid}_{pack_type}",
                        qid=question.qid,
                        scenario_id=question.scenario_id,
                        pack_type=pack_type,
                        evidence_ids=evidence_ids,
                        expected_behavior=expected_pack_behavior(pack_type),
                    )
                )
        return packs

    def _generate_answer_variants(
        self,
        *,
        questions: list[Question],
        facts: list[Fact],
        entities: dict[str, Entity],
        paths: list[GraphPath],
    ) -> list[AnswerVariant]:
        variants: list[AnswerVariant] = []
        for question in questions:
            visible_facts = [
                fact for fact in facts if fact_visible(fact, question.program.snapshot_id)
            ]
            fact_by_status: dict[TimeStatus, list[Fact]] = defaultdict(list)
            for fact in visible_facts:
                status = time_status_for_fact(
                    fact,
                    question.program,
                    selected_fact_ids=set(question.required_valid_evidence_ids),
                )
                fact_by_status[status].append(fact)
            answer_specs = self._answer_specs(
                question,
                fact_by_status,
                paths,
                entities,
                visible_facts,
            )
            normalized_gold = {normalize_visible_text(value) for value in question.gold_answer_text}
            answer_specs = [
                spec
                for spec in answer_specs
                if not (
                    spec[3]["answer_correct"] == "no"
                    and normalize_visible_text(spec[0]) in normalized_gold
                )
            ]
            for index, spec in enumerate(answer_specs):
                answer_text, cited, variant_type, labels = spec
                graph_path_ids = self._variant_path_ids(
                    question=question,
                    variant_type=variant_type,
                    cited_evidence_ids=cited,
                    paths=paths,
                )
                claim = AnswerClaim(
                    cid=f"c_{question.qid}_{index:02d}",
                    text=answer_text,
                    claim_time=question.program.query_time,
                    cited_evidence_ids=cited,
                    temporally_valid=(
                        True
                        if labels["temporal_correct"] == "yes"
                        else False
                        if labels["temporal_correct"] == "no"
                        else None
                    ),
                )
                variants.append(
                    AnswerVariant(
                        answer_id=f"a_{question.qid}_{index:02d}",
                        qid=question.qid,
                        scenario_id=question.scenario_id,
                        variant_type=variant_type,
                        alternate_operator=(
                            self._alternate_operator(question.temporal_operator)
                            if variant_type == AnswerVariantType.WRONG_OPERATOR_ANSWER
                            else None
                        ),
                        answer_text=answer_text,
                        cited_evidence_ids=cited,
                        graph_path_ids=graph_path_ids,
                        claims=[claim],
                        **labels,
                    )
                )
        return variants

    def _answer_specs(
        self,
        question: Question,
        fact_by_status: dict[TimeStatus, list[Fact]],
        paths: list[GraphPath],
        entities: dict[str, Entity],
        visible_facts: list[Fact],
    ) -> list[
        tuple[
            str,
            list[str],
            AnswerVariantType,
            dict[str, str],
        ]
    ]:
        correct = answer_sentence(question, entities)
        if question.should_abstain:
            return self._abstention_answer_specs(question, fact_by_status, entities)
        specs = [
            (
                correct,
                question.required_valid_evidence_ids,
                AnswerVariantType.CORRECT_SUPPORTED,
                {
                    "answer_correct": "yes",
                    "temporal_correct": "yes",
                    "evidence_supports_answer": "yes",
                    "citation_temporally_valid": "yes"
                    if question.required_valid_evidence_ids
                    else "not_applicable",
                    "graph_path_sufficient": "yes",
                },
            )
        ]
        stale = fact_by_status[TimeStatus.STALE]
        future = fact_by_status[TimeStatus.FUTURE_INVALID]
        supportive_stale_by_answer = {
            answer_id: next(
                (fact for fact in stale if fact_answer_id(fact) == answer_id),
                None,
            )
            for answer_id in question.gold_answer_entity_ids
        }
        stale_wrong = next(
            (fact for fact in stale if fact_answer_id(fact) not in question.gold_answer_entity_ids),
            None,
        )
        future_wrong = next(
            (
                fact
                for fact in future
                if fact_answer_id(fact) not in question.gold_answer_entity_ids
            ),
            None,
        )
        if supportive_stale_by_answer and all(supportive_stale_by_answer.values()):
            supportive_stale = [
                supportive_stale_by_answer[answer_id]
                for answer_id in question.gold_answer_entity_ids
            ]
            specs.append(
                (
                    correct,
                    [fact.fact_id for fact in supportive_stale if fact is not None],
                    AnswerVariantType.CORRECT_INVALID_EVIDENCE,
                    {
                        "answer_correct": "yes",
                        "temporal_correct": "yes",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "no",
                    },
                )
            )
        if stale_wrong:
            specs.append(
                (
                    entities[fact_answer_id(stale_wrong)].name,
                    [stale_wrong.fact_id],
                    AnswerVariantType.STALE_ANSWER,
                    {
                        "answer_correct": "no",
                        "temporal_correct": "no",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "no",
                    },
                )
            )
        if future_wrong:
            specs.append(
                (
                    entities[fact_answer_id(future_wrong)].name,
                    [future_wrong.fact_id],
                    AnswerVariantType.FUTURE_INVALID_ANSWER,
                    {
                        "answer_correct": "no",
                        "temporal_correct": "no",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "no",
                    },
                )
            )
        alternate = self._alternate_operator_answer(question, visible_facts, entities)
        if alternate is not None:
            alternate_text, alternate_evidence = alternate
            specs.append(
                (
                    alternate_text,
                    alternate_evidence,
                    AnswerVariantType.WRONG_OPERATOR_ANSWER,
                    {
                        "answer_correct": "no",
                        "temporal_correct": "no",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "no",
                    },
                )
            )
        invalid_path = next(
            (path for path in paths if path.qid == question.qid and not path.supports_gold_answer),
            None,
        )
        if invalid_path and question.gold_answer_entity_ids:
            specs.append(
                (
                    correct,
                    [edge.fact_id for edge in invalid_path.edges],
                    AnswerVariantType.INVALID_GRAPH_PATH_ANSWER,
                    {
                        "answer_correct": "yes",
                        "temporal_correct": "yes",
                        "evidence_supports_answer": "no",
                        "citation_temporally_valid": "yes",
                        "graph_path_sufficient": "no",
                    },
                )
            )
        if question.should_abstain:
            return self._abstention_answer_specs(question, fact_by_status, entities)

        specs.append(
            (
                "There is not enough information to answer.",
                [],
                AnswerVariantType.INAPPROPRIATE_REFUSAL,
                {
                    "answer_correct": "no",
                    "temporal_correct": "not_applicable",
                    "evidence_supports_answer": "not_applicable",
                    "citation_temporally_valid": "not_applicable",
                    "refusal_appropriate": "no",
                },
            )
        )
        if len(question.gold_answer_text) > 1:
            first_answer_id = question.gold_answer_entity_ids[0]
            visible_fact_by_id = {fact.fact_id: fact for fact in visible_facts}
            partial_evidence = [
                fact_id
                for fact_id in question.required_valid_evidence_ids
                if fact_answer_id(visible_fact_by_id[fact_id]) == first_answer_id
            ]
            if not partial_evidence:
                raise RuntimeError(
                    f"No answer-aligned evidence for partial variant of {question.qid}"
                )
            specs.append(
                (
                    question.gold_answer_text[0],
                    partial_evidence,
                    AnswerVariantType.PARTIAL_ANSWER,
                    {
                        "answer_correct": "partial",
                        "temporal_correct": "yes",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "yes",
                    },
                )
            )
        return [(a, c, t, self._complete_labels(labels)) for a, c, t, labels in specs]

    def _abstention_answer_specs(
        self,
        question: Question,
        fact_by_status: dict[TimeStatus, list[Fact]],
        entities: dict[str, Entity],
    ) -> list[
        tuple[
            str,
            list[str],
            AnswerVariantType,
            dict[str, str],
        ]
    ]:
        stale = fact_by_status[TimeStatus.STALE]
        future = fact_by_status[TimeStatus.FUTURE_INVALID]
        wrong_fact = (stale or future)[0] if (stale or future) else None
        specs = [
            (
                "The available evidence is not temporally sufficient to answer confidently.",
                [],
                AnswerVariantType.CORRECT_REFUSAL,
                {
                    "answer_correct": "yes",
                    "temporal_correct": "not_applicable",
                    "evidence_supports_answer": "not_applicable",
                    "citation_temporally_valid": "not_applicable",
                    "refusal_appropriate": "yes",
                },
            )
        ]
        if wrong_fact:
            specs.append(
                (
                    entities[fact_answer_id(wrong_fact)].name,
                    [wrong_fact.fact_id],
                    AnswerVariantType.OVERCONFIDENT_SHOULD_REFUSE,
                    {
                        "answer_correct": "no",
                        "temporal_correct": "no",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "no",
                        "refusal_appropriate": "no",
                    },
                )
            )
        specs.append(
            (
                self._hallucinated_answer(question, entities),
                [],
                AnswerVariantType.HALLUCINATED_ANSWER,
                {
                    "answer_correct": "no",
                    "temporal_correct": "no",
                    "evidence_supports_answer": "not_applicable",
                    "citation_temporally_valid": "not_applicable",
                    "refusal_appropriate": "no",
                },
            )
        )
        if future:
            specs.append(
                (
                    entities[fact_answer_id(future[0])].name,
                    [future[0].fact_id],
                    AnswerVariantType.FUTURE_INVALID_ANSWER,
                    {
                        "answer_correct": "no",
                        "temporal_correct": "no",
                        "evidence_supports_answer": "yes",
                        "citation_temporally_valid": "no",
                        "refusal_appropriate": "no",
                    },
                )
            )
        return [(a, c, t, self._complete_labels(labels)) for a, c, t, labels in specs]

    @staticmethod
    def _complete_labels(labels: dict[str, str]) -> dict[str, str]:
        labels = dict(labels)
        labels.setdefault("evidence_supports_answer", "not_applicable")
        labels.setdefault("graph_path_sufficient", "not_applicable")
        labels.setdefault("refusal_appropriate", "not_applicable")
        return labels

    def _alternate_operator_answer(
        self,
        question: Question,
        facts: list[Fact],
        entities: dict[str, Entity],
    ) -> tuple[str, list[str]] | None:
        alternate_operator = self._alternate_operator(question.temporal_operator)
        if alternate_operator is None:
            return None
        alternate_program = question.program.model_copy(update={"operator": alternate_operator})
        answer_ids, evidence_ids = self.solver.solve(facts, alternate_program)
        if not answer_ids or set(answer_ids) & set(question.gold_answer_entity_ids):
            return None
        return ", ".join(entities[entity_id].name for entity_id in answer_ids), evidence_ids

    @staticmethod
    def _alternate_operator(
        operator: TemporalOperator,
    ) -> TemporalOperator | None:
        alternate_by_operator = {
            TemporalOperator.CURRENT: TemporalOperator.PREVIOUS,
            TemporalOperator.AS_OF: TemporalOperator.PREVIOUS,
            TemporalOperator.EFFECTIVE: TemporalOperator.PREVIOUS,
            TemporalOperator.PREVIOUS: TemporalOperator.CURRENT,
            TemporalOperator.NEXT: TemporalOperator.CURRENT,
            TemporalOperator.FIRST: TemporalOperator.LATEST,
            TemporalOperator.LATEST: TemporalOperator.FIRST,
            TemporalOperator.LAST: TemporalOperator.FIRST,
            TemporalOperator.BEFORE: TemporalOperator.AFTER,
            TemporalOperator.AFTER: TemporalOperator.BEFORE,
            TemporalOperator.DURING: TemporalOperator.CURRENT,
            TemporalOperator.BETWEEN: TemporalOperator.CURRENT,
            TemporalOperator.EXPIRED: TemporalOperator.CURRENT,
        }
        return alternate_by_operator.get(operator)

    @staticmethod
    def _variant_path_ids(
        *,
        question: Question,
        variant_type: AnswerVariantType,
        cited_evidence_ids: list[str],
        paths: list[GraphPath],
    ) -> list[str]:
        cited = set(cited_evidence_ids)
        candidates = [path for path in paths if path.qid == question.qid]
        if variant_type == AnswerVariantType.INVALID_GRAPH_PATH_ANSWER:
            return [path.pid for path in candidates if not path.supports_gold_answer][:1]
        if variant_type == AnswerVariantType.CORRECT_SUPPORTED:
            return [
                path.pid
                for path in candidates
                if path.supports_gold_answer
                and {edge.fact_id for edge in path.edges}.issubset(cited)
            ]
        return []

    @staticmethod
    def _hallucinated_answer(
        question: Question,
        entities: dict[str, Entity],
    ) -> str:
        expected_types = {
            entities[entity_id].entity_type
            for entity_id in question.gold_answer_entity_ids
            if entity_id in entities
        }
        candidates = sorted(
            entity.name
            for entity in entities.values()
            if entity.entity_id not in question.gold_answer_entity_ids
            and entity.entity_id not in {question.program.context_id, question.program.object_id}
            and (not expected_types or entity.entity_type in expected_types)
        )
        if not candidates:
            return "No verified answer"
        index = int(hashlib.sha256(question.qid.encode()).hexdigest()[:8], 16) % len(candidates)
        return candidates[index]

    def _query_interval_for_operator(self, operator: TemporalOperator) -> TemporalInterval:
        if operator == TemporalOperator.AS_OF:
            return point(2019, 6, 1)
        if operator == TemporalOperator.DURING:
            return TemporalInterval(
                type="interval",
                start=date(2021, 1, 1),
                end=date(2022, 12, 31),
            )
        if operator == TemporalOperator.BETWEEN:
            return TemporalInterval(
                type="interval",
                start=date(2018, 1, 1),
                end=date(2023, 12, 31),
            )
        if operator == TemporalOperator.BEFORE:
            return point(2024, 1, 1)
        if operator == TemporalOperator.AFTER:
            return point(2023, 12, 31)
        if operator == TemporalOperator.NEXT:
            return point(2019, 6, 1)
        if operator == TemporalOperator.EXPIRED:
            return point(2024, 6, 1)
        if operator == TemporalOperator.EFFECTIVE:
            return point(2024, 6, 1)
        return point(2024, 6, 1)

    def _entity(
        self,
        scenario_id: str,
        suffix: str,
        name: str,
        entity_type: str,
        domain: str,
    ) -> Entity:
        return Entity(
            entity_id=f"e_{scenario_id}_{suffix}",
            name=name,
            entity_type=entity_type,
            aliases=[],
            domain=domain,
        )

    def _answer_entity(
        self,
        scenario_id: str,
        scenario_index: int,
        candidate_index: int,
        spec: DomainSpec,
    ) -> Entity:
        if spec.answer_type == "person":
            name = (
                f"{FIRST_NAMES[(scenario_index + candidate_index) % len(FIRST_NAMES)]} "
                f"{LAST_NAMES[(scenario_index * 3 + candidate_index) % len(LAST_NAMES)]}"
            )
        elif spec.answer_type == "product_version":
            version = VERSION_NAMES[candidate_index % len(VERSION_NAMES)]
            name = f"{self._context_seed(scenario_index)} {version}"
        elif spec.answer_type == "location":
            name = LOCATIONS[(scenario_index + candidate_index) % len(LOCATIONS)]
        elif spec.answer_type == "event":
            object_name = spec.object_names[candidate_index % len(spec.object_names)].title()
            name = f"{self._context_seed(scenario_index)} {object_name} {candidate_index + 1}"
        else:
            name = (
                f"{self._context_seed(scenario_index)} "
                f"{spec.question_noun.title()} {candidate_index + 1}"
            )
        return self._entity(
            scenario_id,
            f"ans_{candidate_index}",
            name,
            spec.answer_type,
            spec.domain,
        )

    def _context_name(self, spec: DomainSpec) -> str:
        return f"{self.rng.choice(ORG_PREFIXES)} {self.rng.choice(spec.context_nouns)}"

    @staticmethod
    def _context_seed(index: int) -> str:
        return ORG_PREFIXES[index % len(ORG_PREFIXES)]

    @staticmethod
    def _system_difficulty(operator: TemporalOperator, q_index: int) -> SystemDifficulty:
        if operator in {TemporalOperator.CURRENT, TemporalOperator.AS_OF} and q_index == 0:
            return SystemDifficulty.EASY
        if operator in {
            TemporalOperator.PREVIOUS,
            TemporalOperator.NEXT,
            TemporalOperator.DURING,
            TemporalOperator.BETWEEN,
            TemporalOperator.EXPIRED,
        }:
            return SystemDifficulty.HARD
        return SystemDifficulty.MEDIUM

    @staticmethod
    def _eval_difficulty(
        operator: TemporalOperator, scenario_index: int, q_index: int
    ) -> EvalDifficulty:
        if operator in {
            TemporalOperator.CURRENT,
            TemporalOperator.PREVIOUS,
            TemporalOperator.NEXT,
            TemporalOperator.EXPIRED,
        }:
            return EvalDifficulty.HARD
        if (scenario_index + q_index) % 4 == 0:
            return EvalDifficulty.EASY
        return EvalDifficulty.MEDIUM

    def _split_scenarios(self, scenario_ids: list[str]) -> dict[str, list[str]]:
        return stable_group_splits(
            {scenario_id: scenario_id for scenario_id in scenario_ids},
            namespace=f"synthetic:{self.seed}",
        )


def _scenario_notes(scenario_data: dict[str, object]) -> str:
    source_id = scenario_data.get("source_subgraph_id")
    source_notes = scenario_data.get("source_notes")
    if source_id:
        source_family = scenario_data.get("source_family")
        source_relation = scenario_data.get("source_relation")
        source_path_relation = scenario_data.get("source_path_relation")
        source_topology = scenario_data.get("source_topology")
        source_fidelity = scenario_data.get("source_fidelity", "pattern_only")
        return (
            "Scenario generated from a normalized temporal source pattern "
            f"{source_id}. Source family: {source_family}; main relation: "
            f"{source_relation}; path relation: {source_path_relation}; topology: "
            f"{source_topology}; fidelity: {source_fidelity}. {source_notes} "
            "Only records marked source_extracted preserve source claim times exactly. "
            "LLM paraphrasing, if used, must "
            "preserve protected slots."
        )
    return (
        "Synthetic scenario generated from executable temporal facts; "
        "LLM paraphrasing, if used, must preserve protected slots."
    )


def _source_provenance(
    scenario_data: dict[str, object],
) -> SourceProvenance | None:
    source_id = scenario_data.get("source_subgraph_id")
    if not source_id:
        return None
    return SourceProvenance(
        source_id=str(source_id),
        source_family=str(scenario_data.get("source_family") or "unknown"),
        fidelity=str(scenario_data.get("source_fidelity") or "pattern_only"),
        source_record_ids=[str(value) for value in scenario_data.get("source_record_ids", [])],
        source_revision=(
            str(scenario_data["source_revision"]) if scenario_data.get("source_revision") else None
        ),
        source_relation=str(scenario_data.get("source_relation") or "") or None,
        source_path_relation=(str(scenario_data.get("source_path_relation") or "") or None),
        topology_signature=str(scenario_data.get("source_topology") or "") or None,
    )


def _update_behavior_for_index(index: int) -> str:
    # Update stability requires paired questions over materialized before/after
    # snapshots. The current generator does not implement that experiment and must
    # not imply otherwise through metadata.
    del index
    return "not_applicable"


def _timeline_interval_profile(index: int) -> list[TemporalInterval]:
    """Return varied stale/current/future intervals for one scenario.

    The benchmark still preserves a stable semantic role order for diagnostics, but the
    calendar shape varies so systems cannot rely on one fixed 2015/2018/2021/2024/2027
    pattern.
    """

    profiles = (
        ((2015, 2017), (2018, 2020), (2021, 2023), (2024, None), (2027, None)),
        ((2012, 2014), (2015, 2016), (2017, 2019), (2020, 2025), (2028, None)),
        ((2014, 2016), (2017, 2018), (2019, 2021), (2022, None), (2026, None)),
        ((2016, 2017), (2018, 2019), (2020, 2022), (2023, None), (2029, None)),
    )
    return [make_interval(start, end) for start, end in profiles[index % len(profiles)]]


def _timeline_year_profile(index: int) -> list[tuple[int, int | None]]:
    intervals = _timeline_interval_profile(index)
    return [
        (
            interval.start.year if interval.start else 2024,
            interval.end.year if interval.end else None,
        )
        for interval in intervals
    ]


def _snapshot_visible_from_for_timeline(
    *,
    update_behavior: str,
    candidate_index: int,
) -> str:
    del update_behavior
    if candidate_index == 3:
        return "S1"
    return "S0"


def stable_opaque_id(prefix: str, *parts: object) -> str:
    """Return a stable identifier that does not expose a fact's semantic role."""
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"
