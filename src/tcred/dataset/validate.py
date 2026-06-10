from __future__ import annotations

import re
from collections import Counter, defaultdict

from ftfy import fix_encoding

from tcred.dataset.graph import fact_answer_id, fact_endpoint_ids, graph_path_node_ids
from tcred.dataset.models import (
    AnswerVariantType,
    DatasetBundle,
    DatasetFamily,
    Relation,
    TemporalOperator,
    TimeStatus,
)
from tcred.dataset.public_oracle import solve_from_public_facts
from tcred.dataset.solver import (
    GoldSolver,
    fact_visible,
    path_query_time,
    path_time_status,
    time_status_for_fact,
)
from tcred.dataset.splits import BASE_SPLITS
from tcred.dataset.text import normalize_visible_text


class DatasetValidationError(ValueError):
    pass


def validate_bundle(bundle: DatasetBundle) -> list[str]:
    """Validate internal consistency and return non-fatal warnings."""
    warnings: list[str] = []
    fact_ids = {fact.fact_id for fact in bundle.facts}
    fact_by_id = {fact.fact_id: fact for fact in bundle.facts}
    entity_ids = {entity.entity_id for entity in bundle.entities}
    entity_by_id = {entity.entity_id: entity for entity in bundle.entities}
    scenario_ids = {scenario.scenario_id for scenario in bundle.scenarios}
    qids = {question.qid for question in bundle.questions}
    question_by_id = {question.qid: question for question in bundle.questions}

    _require_unique("scenario_id", [scenario.scenario_id for scenario in bundle.scenarios])
    _require_unique("entity_id", [entity.entity_id for entity in bundle.entities])
    _require_unique("fact_id", [fact.fact_id for fact in bundle.facts])
    _require_unique("qid", [question.qid for question in bundle.questions])
    _require_unique("answer_id", [answer.answer_id for answer in bundle.answer_variants])
    _require_unique("pid", [path.pid for path in bundle.graph_paths])
    _require_unique("pack_id", [pack.pack_id for pack in bundle.context_packs])

    gold_solver = GoldSolver()

    for scenario in bundle.scenarios:
        snapshot_ids = {snapshot.snapshot_id for snapshot in scenario.snapshots}
        embedded_entity_ids = {entity.entity_id for entity in scenario.entities}
        missing_entities = embedded_entity_ids - entity_ids
        if missing_entities:
            raise DatasetValidationError(
                f"Scenario {scenario.scenario_id} embeds missing entities "
                f"{sorted(missing_entities)}"
            )
        embedded_fact_ids = {fact.fact_id for fact in scenario.facts}
        missing_facts = embedded_fact_ids - fact_ids
        if missing_facts:
            raise DatasetValidationError(
                f"Scenario {scenario.scenario_id} embeds missing facts {sorted(missing_facts)}"
            )
        wrong_scenario_facts = [
            fact.fact_id for fact in scenario.facts if fact.scenario_id != scenario.scenario_id
        ]
        if wrong_scenario_facts:
            raise DatasetValidationError(
                f"Scenario {scenario.scenario_id} embeds facts from another scenario "
                f"{wrong_scenario_facts[:10]}"
            )
        missing_questions = set(scenario.question_ids) - qids
        if missing_questions:
            raise DatasetValidationError(
                f"Scenario {scenario.scenario_id} references missing questions "
                f"{sorted(missing_questions)}"
            )
        missing_snapshot_facts = {
            fact_id
            for snapshot in scenario.snapshots
            for fact_id in snapshot.visible_fact_ids
            if fact_id not in fact_ids
        }
        if missing_snapshot_facts:
            raise DatasetValidationError(
                f"Scenario {scenario.scenario_id} snapshots reference missing facts "
                f"{sorted(missing_snapshot_facts)}"
            )
        unavailable_visibility = {
            fact.snapshot_visible_from
            for fact in scenario.facts
            if fact.snapshot_visible_from not in snapshot_ids
        }
        if unavailable_visibility:
            raise DatasetValidationError(
                f"Scenario {scenario.scenario_id} facts reference unmaterialized snapshots "
                f"{sorted(unavailable_visibility)}"
            )
        for snapshot in scenario.snapshots:
            future_transactions = [
                fact.fact_id
                for fact in scenario.facts
                if fact.fact_id in snapshot.visible_fact_ids
                and fact.transaction_time is not None
                and fact.transaction_time > snapshot.snapshot_time
            ]
            if future_transactions:
                raise DatasetValidationError(
                    f"Snapshot {scenario.scenario_id}/{snapshot.snapshot_id} exposes facts "
                    f"before transaction time: {future_transactions[:10]}"
                )
        if (
            scenario.source_provenance is not None
            and scenario.source_provenance.fidelity == "source_extracted"
        ):
            incomplete = [
                fact.fact_id
                for fact in scenario.facts
                if not fact.source_record_id or not fact.source_revision
            ]
            if incomplete:
                raise DatasetValidationError(
                    f"Source-extracted scenario {scenario.scenario_id} has unpinned facts "
                    f"{incomplete[:10]}"
                )

    for fact in bundle.facts:
        if fact.scenario_id not in scenario_ids:
            raise DatasetValidationError(f"Fact {fact.fact_id} references missing scenario")
        if fact.subject_id not in entity_ids:
            raise DatasetValidationError(f"Fact {fact.fact_id} references missing subject")
        if fact.object_id and fact.object_id not in entity_ids:
            raise DatasetValidationError(f"Fact {fact.fact_id} references missing object")
        if fact.context_id and fact.context_id not in entity_ids:
            raise DatasetValidationError(f"Fact {fact.fact_id} references missing context")
        for field, value in (
            ("answer_entity_id", fact.answer_entity_id),
            ("graph_source_id", fact.graph_source_id),
            ("graph_target_id", fact.graph_target_id),
        ):
            if value and value not in entity_ids:
                raise DatasetValidationError(f"Fact {fact.fact_id} has missing {field} {value}")
        if bool(fact.graph_source_id) != bool(fact.graph_target_id):
            raise DatasetValidationError(
                f"Fact {fact.fact_id} must define both explicit graph endpoints"
            )
        if bool(fact.source_relation_id) != bool(fact.source_relation_label):
            raise DatasetValidationError(
                f"Fact {fact.fact_id} must pair source relation ID and English label"
            )

    for question in bundle.questions:
        if question.scenario_id not in scenario_ids:
            raise DatasetValidationError(f"Question {question.qid} references missing scenario")
        if question.qid not in set(
            qid
            for scenario in bundle.scenarios
            if scenario.scenario_id == question.scenario_id
            for qid in scenario.question_ids
        ):
            raise DatasetValidationError(
                f"Question {question.qid} is missing from scenario question_ids"
            )
        missing = set(question.required_valid_evidence_ids) - fact_ids
        if missing:
            raise DatasetValidationError(
                f"Question {question.qid} references missing evidence {sorted(missing)}"
            )
        wrong_scenario_evidence = [
            fact_id
            for fact_id in question.required_valid_evidence_ids
            if fact_by_id[fact_id].scenario_id != question.scenario_id
        ]
        if wrong_scenario_evidence:
            raise DatasetValidationError(
                f"Question {question.qid} uses evidence from another scenario "
                f"{wrong_scenario_evidence}"
            )
        invisible_gold = [
            fact_id
            for fact_id in question.required_valid_evidence_ids
            if not fact_visible(fact_by_id[fact_id], question.program.snapshot_id)
        ]
        if invisible_gold:
            raise DatasetValidationError(
                f"Question {question.qid} requires evidence unavailable at its snapshot: "
                f"{invisible_gold}"
            )
        if (
            question.dataset_family == DatasetFamily.PAT
            and question.temporal_operator == TemporalOperator.PREVIOUS
            and "P1365"
            not in {
                fact_by_id[fact_id].source_relation_id
                for fact_id in question.required_valid_evidence_ids
            }
        ):
            raise DatasetValidationError(
                f"PAT previous question {question.qid} lacks explicit P1365 predecessor evidence"
            )
        if (
            question.certification_status == "certified"
            and question.program.temporal_basis == "world_valid_time"
        ):
            scenario_facts = [
                fact for fact in bundle.facts if fact.scenario_id == question.scenario_id
            ]
            answer_ids, evidence_ids = solve_from_public_facts(
                scenario_facts,
                question.program,
            )
            if answer_ids != question.gold_answer_entity_ids or evidence_ids != (
                question.required_valid_evidence_ids
            ):
                raise DatasetValidationError(
                    f"Question {question.qid} disagrees with the independent public oracle"
                )
        elif question.dataset_family == DatasetFamily.SYNTH:
            scenario_facts = [
                fact for fact in bundle.facts if fact.scenario_id == question.scenario_id
            ]
            answer_ids, evidence_ids = gold_solver.solve(scenario_facts, question.program)
            if answer_ids != question.gold_answer_entity_ids or evidence_ids != (
                question.required_valid_evidence_ids
            ):
                raise DatasetValidationError(
                    f"Question {question.qid} disagrees with a fresh oracle solve"
                )
        if question.program.temporal_basis == "world_valid_time":
            unknown_required = [
                fact_id
                for fact_id in question.required_valid_evidence_ids
                if fact_by_id[fact_id].valid_time.type == "unknown"
            ]
            if unknown_required:
                raise DatasetValidationError(
                    f"World-time question {question.qid} requires unknown-time evidence "
                    f"{unknown_required}"
                )
        if question.program.temporal_basis in {
            "snapshot_observation",
            "document_revision",
        }:
            start = question.program.query_time.start
            if start is None or str(start.year) not in question.question:
                raise DatasetValidationError(
                    f"Snapshot/revision question {question.qid} does not expose its date"
                )

    _validate_public_question_uniqueness(bundle)
    _validate_update_pairs(bundle)
    _validate_text_hygiene(bundle)

    for pack in bundle.context_packs:
        if pack.qid not in qids:
            raise DatasetValidationError(f"Context pack {pack.pack_id} references missing qid")
        question = question_by_id[pack.qid]
        if pack.scenario_id != question.scenario_id:
            raise DatasetValidationError(
                f"Context pack {pack.pack_id} scenario does not match question {pack.qid}"
            )
        missing = set(pack.evidence_ids) - fact_ids
        if missing:
            raise DatasetValidationError(
                f"Context pack {pack.pack_id} references missing evidence {sorted(missing)}"
            )

    for graph_path in bundle.graph_paths:
        if graph_path.qid not in qids:
            raise DatasetValidationError(f"Graph path {graph_path.pid} references missing qid")
        question = question_by_id[graph_path.qid]
        if graph_path.scenario_id != question.scenario_id:
            raise DatasetValidationError(
                f"Graph path {graph_path.pid} scenario does not match question {graph_path.qid}"
            )
        missing_nodes = set(graph_path.nodes) - entity_ids
        if missing_nodes:
            raise DatasetValidationError(
                f"Graph path {graph_path.pid} references missing nodes {sorted(missing_nodes)}"
            )
        missing = {edge.fact_id for edge in graph_path.edges} - fact_ids
        if missing:
            raise DatasetValidationError(
                f"Graph path {graph_path.pid} references missing facts {sorted(missing)}"
            )
        if len(graph_path.nodes) != len(set(graph_path.nodes)):
            raise DatasetValidationError(f"Graph path {graph_path.pid} contains duplicate nodes")
        for edge in graph_path.edges:
            fact = fact_by_id[edge.fact_id]
            if edge.relation != fact.relation:
                raise DatasetValidationError(
                    f"Graph path {graph_path.pid} edge {edge.fact_id} relation mismatch"
                )
            if edge.valid_time != fact.valid_time:
                raise DatasetValidationError(
                    f"Graph path {graph_path.pid} edge {edge.fact_id} valid_time mismatch"
                )
        expected_nodes = graph_path_node_ids(graph_path.edges, fact_by_id)
        if graph_path.nodes != expected_nodes:
            raise DatasetValidationError(
                f"Graph path {graph_path.pid} nodes are not the ordered traversal nodes"
            )
        for left, right in zip(graph_path.edges, graph_path.edges[1:], strict=False):
            left_fact = fact_by_id[left.fact_id]
            right_fact = fact_by_id[right.fact_id]
            _, left_target = fact_endpoint_ids(
                left_fact,
                traversal_direction=left.traversal_direction,
            )
            right_source, _ = fact_endpoint_ids(
                right_fact,
                traversal_direction=right.traversal_direction,
            )
            if left_target != right_source:
                raise DatasetValidationError(
                    f"Graph path {graph_path.pid} is directionally discontinuous"
                )
        sequence = bool(graph_path.edges) and all(
            edge.relation == Relation.EVENT_PRECEDES for edge in graph_path.edges
        )
        expected_status = path_time_status(
            graph_path.edges,
            sequence=sequence,
            query_time=path_query_time(question.program),
        )
        if graph_path.path_time_status != expected_status:
            raise DatasetValidationError(f"Graph path {graph_path.pid} has stale temporal status")
        if graph_path.supports_gold_answer and question.required_valid_evidence_ids:
            path_fact_ids = {edge.fact_id for edge in graph_path.edges}
            if path_fact_ids.isdisjoint(question.required_valid_evidence_ids):
                raise DatasetValidationError(
                    f"Graph path {graph_path.pid} claims gold support without required evidence"
                )

    for answer in bundle.answer_variants:
        if answer.qid not in qids:
            raise DatasetValidationError(f"Answer {answer.answer_id} references missing qid")
        question = question_by_id[answer.qid]
        if answer.scenario_id != question.scenario_id:
            raise DatasetValidationError(
                f"Answer {answer.answer_id} scenario does not match question {answer.qid}"
            )
        missing = set(answer.cited_evidence_ids) - fact_ids
        if missing:
            raise DatasetValidationError(
                f"Answer {answer.answer_id} references missing citations {sorted(missing)}"
            )
        invisible = [
            fact_id
            for fact_id in answer.cited_evidence_ids
            if not fact_visible(fact_by_id[fact_id], question.program.snapshot_id)
        ]
        if invisible:
            raise DatasetValidationError(
                f"Answer {answer.answer_id} cites evidence unavailable at the snapshot: {invisible}"
            )
        path_ids = {path.pid for path in bundle.graph_paths if path.qid == answer.qid}
        missing_paths = set(answer.graph_path_ids) - path_ids
        if missing_paths:
            raise DatasetValidationError(
                f"Answer {answer.answer_id} references missing or wrong-question paths "
                f"{sorted(missing_paths)}"
            )
        for claim in answer.claims:
            missing_claim_citations = set(claim.cited_evidence_ids) - fact_ids
            if missing_claim_citations:
                raise DatasetValidationError(
                    f"Answer {answer.answer_id} claim {claim.cid} references missing citations "
                    f"{sorted(missing_claim_citations)}"
                )
        normalized_answer = normalize_visible_text(answer.answer_text)
        normalized_gold = {normalize_visible_text(value) for value in question.gold_answer_text}
        if answer.answer_correct == "no" and normalized_answer in normalized_gold:
            raise DatasetValidationError(
                f"Answer {answer.answer_id} is labeled wrong but exactly matches gold"
            )
        if answer.variant_type == AnswerVariantType.CORRECT_SUPPORTED and (
            answer.answer_correct != "yes" or answer.evidence_supports_answer != "yes"
        ):
            raise DatasetValidationError(
                f"Correct-supported answer {answer.answer_id} has contradictory labels"
            )
        if answer.variant_type in {
            AnswerVariantType.CORRECT_SUPPORTED,
            AnswerVariantType.CORRECT_INVALID_EVIDENCE,
        }:
            cited_answer_ids = {
                fact_answer_id(fact_by_id[fact_id]) for fact_id in answer.cited_evidence_ids
            }
            missing_answer_support = set(question.gold_answer_entity_ids) - cited_answer_ids
            if missing_answer_support:
                raise DatasetValidationError(
                    f"Fully correct answer {answer.answer_id} lacks evidence for gold entities "
                    f"{sorted(missing_answer_support)}"
                )
        if answer.variant_type == AnswerVariantType.CORRECT_INVALID_EVIDENCE and (
            answer.answer_correct != "yes"
            or answer.temporal_correct != "yes"
            or answer.evidence_supports_answer != "yes"
            or answer.citation_temporally_valid != "no"
        ):
            raise DatasetValidationError(
                f"Correct-answer/invalid-evidence variant {answer.answer_id} conflates "
                "answer and citation validity"
            )
        if answer.variant_type == AnswerVariantType.CORRECT_INVALID_EVIDENCE:
            cited_facts = [fact_by_id[fact_id] for fact_id in answer.cited_evidence_ids]
            if not cited_facts or any(
                fact_answer_id(fact) not in question.gold_answer_entity_ids for fact in cited_facts
            ):
                raise DatasetValidationError(
                    f"Temporal-only evidence variant {answer.answer_id} is not semantically "
                    "supportive"
                )
            if any(
                time_status_for_fact(
                    fact,
                    question.program,
                    selected_fact_ids=set(question.required_valid_evidence_ids),
                )
                == TimeStatus.VALID
                for fact in cited_facts
            ):
                raise DatasetValidationError(
                    f"Temporal-only evidence variant {answer.answer_id} cites valid evidence"
                )
        if answer.variant_type == AnswerVariantType.PARTIAL_ANSWER:
            matching_gold_ids = {
                entity_id
                for entity_id in question.gold_answer_entity_ids
                if normalize_visible_text(entity_by_id[entity_id].name) == normalized_answer
            }
            cited_answer_ids = {
                fact_answer_id(fact_by_id[fact_id]) for fact_id in answer.cited_evidence_ids
            }
            if (
                answer.answer_correct != "partial"
                or not matching_gold_ids
                or not matching_gold_ids.intersection(cited_answer_ids)
            ):
                raise DatasetValidationError(
                    f"Partial answer {answer.answer_id} is not supported for its selected "
                    "gold entity"
                )
        if answer.variant_type == AnswerVariantType.WRONG_OPERATOR_ANSWER:
            if answer.alternate_operator is None:
                raise DatasetValidationError(
                    f"Wrong-operator answer {answer.answer_id} has no named intervention"
                )
            if question.dataset_family == DatasetFamily.SYNTH:
                scenario_facts = [
                    fact for fact in bundle.facts if fact.scenario_id == question.scenario_id
                ]
                alternate_program = question.program.model_copy(
                    update={"operator": answer.alternate_operator}
                )
                alternate_ids, alternate_evidence = gold_solver.solve(
                    scenario_facts,
                    alternate_program,
                )
                alternate_text = normalize_visible_text(
                    ", ".join(entity_by_id[entity_id].name for entity_id in alternate_ids)
                )
                if normalized_answer != alternate_text or set(answer.cited_evidence_ids) != set(
                    alternate_evidence
                ):
                    raise DatasetValidationError(
                        f"Wrong-operator answer {answer.answer_id} does not match its named "
                        "alternate solve"
                    )
        if answer.graph_path_sufficient == "yes" and not answer.graph_path_ids:
            raise DatasetValidationError(
                f"Answer {answer.answer_id} claims path sufficiency without a path"
            )
        if answer.graph_path_sufficient == "yes":
            selected_paths = [
                path for path in bundle.graph_paths if path.pid in answer.graph_path_ids
            ]
            covered_facts = {edge.fact_id for path in selected_paths for edge in path.edges}
            required = set(question.required_valid_evidence_ids)
            if not required.issubset(covered_facts):
                raise DatasetValidationError(
                    f"Answer {answer.answer_id} claims path sufficiency but omits required "
                    f"support {sorted(required - covered_facts)}"
                )
        if answer.variant_type == AnswerVariantType.INVALID_GRAPH_PATH_ANSWER:
            selected_paths = [
                path for path in bundle.graph_paths if path.pid in answer.graph_path_ids
            ]
            if not selected_paths or any(path.supports_gold_answer for path in selected_paths):
                raise DatasetValidationError(
                    f"Invalid-path answer {answer.answer_id} does not reference a negative path"
                )
            if (
                answer.answer_correct != "yes"
                or answer.temporal_correct != "yes"
                or answer.evidence_supports_answer != "no"
                or answer.citation_temporally_valid != "yes"
                or answer.graph_path_sufficient != "no"
            ):
                raise DatasetValidationError(
                    f"Invalid-path answer {answer.answer_id} does not isolate graph support"
                )
            if any(
                path.path_time_status
                not in {
                    "coherent_shared_interval",
                    "coherent_sequence",
                }
                for path in selected_paths
            ):
                raise DatasetValidationError(
                    f"Invalid-path answer {answer.answer_id} also has invalid path time"
                )

    question_domains = Counter(scenario.domain for scenario in bundle.scenarios)
    if question_domains:
        max_domain_share = max(question_domains.values()) / sum(question_domains.values())
        if max_domain_share > 0.20 and len(question_domains) >= 5:
            warnings.append(f"One domain exceeds 20% of scenarios: {question_domains}")

    operator_counts = Counter(question.temporal_operator for question in bundle.questions)
    if operator_counts:
        max_operator_share = max(operator_counts.values()) / sum(operator_counts.values())
        if max_operator_share > 0.35:
            warnings.append(f"One temporal operator exceeds 35% of questions: {operator_counts}")

    variant_by_type = Counter(answer.variant_type for answer in bundle.answer_variants)
    required_variants = {
        "correct_supported",
        "stale_answer",
        "future_invalid_answer",
        "invalid_graph_path_answer",
    }
    source_extracted_synth = any(
        question.dataset_family == DatasetFamily.SYNTH for question in bundle.questions
    ) and all(
        scenario.source_provenance is not None
        and scenario.source_provenance.fidelity == "source_extracted"
        for scenario in bundle.scenarios
    )
    if not source_extracted_synth:
        required_variants.add("correct_answer_invalid_evidence")
    missing_variants = required_variants - set(variant_by_type)
    if missing_variants:
        warnings.append(f"Missing expected answer variant types: {sorted(missing_variants)}")

    split_membership: dict[str, list[str]] = defaultdict(list)
    for split_name, ids in bundle.splits.items():
        missing_split_ids = set(ids) - scenario_ids
        if missing_split_ids:
            raise DatasetValidationError(
                f"Split {split_name} references missing scenarios {sorted(missing_split_ids)}"
            )
        if split_name == "human_pool":
            continue
        for scenario_id in ids:
            split_membership[scenario_id].append(split_name)
    leaked = {sid: names for sid, names in split_membership.items() if len(names) > 1}
    if leaked:
        raise DatasetValidationError(f"Scenario split leakage detected: {leaked}")
    missing_partition = scenario_ids - set(split_membership)
    if missing_partition:
        raise DatasetValidationError(
            f"Scenarios missing from base partitions: {sorted(missing_partition)[:10]}"
        )
    unexpected_base = set(bundle.splits) - {*BASE_SPLITS, "human_pool"}
    if unexpected_base:
        warnings.append(f"Non-standard split views present: {sorted(unexpected_base)}")
    if not set(bundle.splits.get("human_pool", [])).issubset(
        set(bundle.splits.get("test_auto", []))
    ):
        raise DatasetValidationError("human_pool must be a subset of test_auto")

    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in bundle.scenarios}
    for scenario_id, split_names in split_membership.items():
        group_id = scenario_by_id[scenario_id].split_group_id or scenario_id
        group_splits[group_id].update(split_names)
    crossing_groups = {
        group_id: sorted(names) for group_id, names in group_splits.items() if len(names) > 1
    }
    if crossing_groups:
        raise DatasetValidationError(
            f"Related source groups cross base splits: {dict(list(crossing_groups.items())[:10])}"
        )

    leaking_ids = [
        fact.fact_id
        for fact in bundle.facts
        if re.search(
            r"(?:conflict|unknown|publication|negative|bridge|path_bad|stale|future|current)",
            fact.fact_id,
            flags=re.IGNORECASE,
        )
    ]
    if leaking_ids:
        raise DatasetValidationError(
            f"Fact identifiers expose hidden semantic roles: {leaking_ids[:10]}"
        )

    return warnings


def _validate_public_question_uniqueness(bundle: DatasetBundle) -> None:
    references: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    for question in bundle.questions:
        references[normalize_visible_text(question.question)].add(
            tuple(sorted(normalize_visible_text(value) for value in question.gold_answer_text))
        )
    conflicts = {question: values for question, values in references.items() if len(values) > 1}
    if conflicts:
        raise DatasetValidationError(
            "Identical public questions have incompatible references: "
            f"{dict(list(conflicts.items())[:5])}"
        )


def _validate_update_pairs(bundle: DatasetBundle) -> None:
    questions_by_scenario: defaultdict[str, list[object]] = defaultdict(list)
    for question in bundle.questions:
        questions_by_scenario[question.scenario_id].append(question)
    for scenario in bundle.scenarios:
        if scenario.update_behavior == "not_applicable":
            continue
        by_series: defaultdict[str, list[object]] = defaultdict(list)
        for question in questions_by_scenario[scenario.scenario_id]:
            if question.semantic_series_id:
                by_series[question.semantic_series_id].append(question)
        pairs = [
            values
            for values in by_series.values()
            if len(values) == 2 and {value.program.snapshot_id for value in values} == {"S0", "S1"}
        ]
        if len(pairs) != 1:
            raise DatasetValidationError(
                f"Update scenario {scenario.scenario_id} must contain exactly one question pair"
            )
        before, after = sorted(pairs[0], key=lambda item: item.program.snapshot_id)
        if [before.program.snapshot_id, after.program.snapshot_id] != ["S0", "S1"]:
            raise DatasetValidationError(
                f"Update pair in {scenario.scenario_id} must compare S0 with S1"
            )
        before_answer = (tuple(before.gold_answer_entity_ids), before.should_abstain)
        after_answer = (tuple(after.gold_answer_entity_ids), after.should_abstain)
        if scenario.update_behavior == "answer_should_change" and before_answer == after_answer:
            raise DatasetValidationError(
                f"Declared answer-changing update does not change {scenario.scenario_id}"
            )
        if scenario.update_behavior == "answer_should_stay" and before_answer != after_answer:
            raise DatasetValidationError(
                f"Declared answer-preserving update changes {scenario.scenario_id}"
            )
        s0 = next(snapshot for snapshot in scenario.snapshots if snapshot.snapshot_id == "S0")
        s1 = next(snapshot for snapshot in scenario.snapshots if snapshot.snapshot_id == "S1")
        added = set(s1.visible_fact_ids) - set(s0.visible_fact_ids)
        removed = set(s0.visible_fact_ids) - set(s1.visible_fact_ids)
        if len(added) != 1 or removed:
            raise DatasetValidationError(
                f"Update scenario {scenario.scenario_id} is not a one-fact intervention"
            )


def _validate_text_hygiene(bundle: DatasetBundle) -> None:
    for entity in bundle.entities:
        if re.search(r"(?i)(?:https?://|www\.|\.well-known/genid)", entity.name):
            raise DatasetValidationError(
                f"Entity name is a machine-readable URL rather than a label: {entity.name!r}"
            )
        if re.fullmatch(r"[PQ]\d+", entity.name, flags=re.IGNORECASE):
            raise DatasetValidationError(
                f"Entity name is an unexplained Wikidata identifier: {entity.name!r}"
            )

    values = [
        *(entity.name for entity in bundle.entities),
        *(question.question for question in bundle.questions),
        *(
            question_text
            for question in bundle.questions
            for question_text in question.gold_answer_text
        ),
        *(fact.canonical_evidence for fact in bundle.facts),
        *(
            fact.source_relation_label
            for fact in bundle.facts
            if fact.source_relation_label is not None
        ),
        *(answer.answer_text for answer in bundle.answer_variants),
    ]
    for value in values:
        if fix_encoding(value) != value:
            raise DatasetValidationError(f"Text contains repairable mojibake: {value[:100]!r}")
        if re.search(r"&(?:[A-Za-z]+|#\d+);", value):
            raise DatasetValidationError(f"Text contains a raw HTML entity: {value[:100]!r}")
        if re.search(r"\bP\d{1,5}\b", value):
            raise DatasetValidationError(
                f"Public text contains an unexplained Wikidata property ID: {value[:100]!r}"
            )
        if re.search(r"\bdid\b[^?]{0,120}\battended\b", value, flags=re.IGNORECASE):
            raise DatasetValidationError(f"Text contains 'did ... attended': {value[:100]!r}")


def _require_unique(name: str, values: list[str]) -> None:
    counts = Counter(values)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        raise DatasetValidationError(f"Duplicate {name} values: {duplicates[:10]}")


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return [value for value, count in counts.items() if count > 1]
