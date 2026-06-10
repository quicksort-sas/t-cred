from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import orjson

from tcred.dataset.graph import graph_path_node_ids, path_edge_payload
from tcred.dataset.io import load_bundle, write_json_atomic, write_jsonl_atomic
from tcred.dataset.models import (
    AnswerVariant,
    DatasetBundle,
    Entity,
    Fact,
    GraphPath,
    GraphPathEdge,
    Question,
    TemporalOperator,
)
from tcred.dataset.solver import fact_matches_program, fact_visible, query_point
from tcred.dataset.text import annotation_plain_text
from tcred.human_eval.assignments import (
    DEFAULT_ASSIGNMENT_SEED,
    assign_annotation_units,
    assignment_manifest_metadata,
)
from tcred.human_eval.package import artifact_hashes
from tcred.human_eval.presentation import (
    ANNOTATION_TEXT_REPAIR_VERSION,
    annotation_question_text,
    annotation_text_quality_issues,
    displayed_evidence,
    normalize_annotation_payload,
    require_visible_citations,
)
from tcred.human_eval.protocol import (
    CATEGORICAL_FIELDS,
    PROTOCOL_VERSION,
    annotation_fields_manifest,
)
from tcred.human_eval.response import response_decision_kind

ANNOTATION_FIELDS = annotation_fields_manifest()
_ORDERING_OPERATORS = {
    TemporalOperator.PREVIOUS,
    TemporalOperator.NEXT,
    TemporalOperator.FIRST,
    TemporalOperator.LATEST,
    TemporalOperator.LAST,
    TemporalOperator.BEFORE,
    TemporalOperator.AFTER,
    TemporalOperator.EXPIRED,
}
_BUNDLE_INDEX_CACHE: dict[
    int,
    tuple[
        DatasetBundle,
        dict[str, Fact],
        dict[str, Entity],
        dict[str, list[Fact]],
        dict[str, list[GraphPath]],
    ],
] = {}


def export_human_eval(
    *,
    dataset_dir: Path,
    output_dir: Path,
    target_units: int = 160,
    annotators: int = 24,
    assignments_per_annotator: int = 10,
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED,
    overwrite: bool = False,
) -> dict[str, Path]:
    bundle = load_bundle(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = output_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_units(bundle, target_units=target_units)
    unit_rows = [controlled_public_unit(bundle, answer) for answer in selected]
    key_rows = [controlled_private_key(answer, bundle=bundle) for answer in selected]
    validate_annotation_contract(unit_rows=unit_rows, key_rows=key_rows)
    plan = assign_annotation_units(
        unit_rows=unit_rows,
        key_rows=key_rows,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        seed=assignment_seed,
    )
    assignment_rows = plan.assignments

    units_path = output_dir / "human_eval_units.jsonl"
    key_path = output_dir / "human_eval_key.jsonl"
    manifest_path = output_dir / "assignment_manifest.json"
    _guard(units_path, overwrite=overwrite)
    _guard(key_path, overwrite=overwrite)
    _guard(manifest_path, overwrite=overwrite)
    write_jsonl_atomic(units_path, unit_rows)
    write_jsonl_atomic(key_path, key_rows)

    assignment_paths: list[Path] = []
    for annotator_id, rows in assignment_rows.items():
        path = assignments_dir / f"{annotator_id}.jsonl"
        _guard(path, overwrite=overwrite)
        write_jsonl_atomic(path, rows)
        assignment_paths.append(path)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "target_units_requested": target_units,
        "unique_units_exported": len(unit_rows),
        "annotators": annotators,
        "assignments_per_annotator": assignments_per_annotator,
        "total_assignments": sum(len(rows) for rows in assignment_rows.values()),
        **assignment_manifest_metadata(plan, key_rows=key_rows),
        "annotation_fields": ANNOTATION_FIELDS,
        "annotation_protocol_version": PROTOCOL_VERSION,
        "annotation_text_repair_version": ANNOTATION_TEXT_REPAIR_VERSION,
        "annotator_files": [str(path) for path in assignment_paths],
        "artifact_sha256": artifact_hashes(
            root=output_dir,
            paths=[units_path, key_path, *assignment_paths],
        ),
        "blind_annotation_note": (
            "Assignment files contain a plain-text reference answer for the trusted server. The "
            "annotation API withholds it until evidence-stage judgments are locked, and omits "
            "question programs, fact roles, systems, models, and metric labels."
        ),
    }
    write_json_atomic(manifest_path, manifest, sort_keys=True)
    return {
        "human_eval_units": units_path,
        "human_eval_key": key_path,
        "assignment_manifest": manifest_path,
    }


def export_multi_dataset_human_eval(
    *,
    dataset_dirs: Mapping[str, Path],
    output_dir: Path,
    target_units_by_family: Mapping[str, int],
    annotators: int = 36,
    assignments_per_annotator: int = 20,
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = output_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    unit_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    selected_counts: dict[str, int] = {}
    requested_counts: dict[str, int] = {}

    for family, dataset_dir in sorted(dataset_dirs.items()):
        bundle = load_bundle(dataset_dir)
        requested = target_units_by_family.get(family, 0)
        requested_counts[family] = requested
        selected = _select_units(bundle, target_units=requested)
        selected_counts[family] = len(selected)
        for answer in selected:
            public = controlled_public_unit(bundle, answer)
            private = controlled_private_key(answer, bundle=bundle)
            private["dataset_family"] = family
            public["_ordering_family"] = family
            unit_rows.append(public)
            key_rows.append(private)

    unit_rows = _family_interleaved(unit_rows)
    for row in unit_rows:
        row.pop("_ordering_family", None)
    key_by_unit_id = {str(row["unit_id"]): row for row in key_rows}
    key_rows = [key_by_unit_id[str(row["unit_id"])] for row in unit_rows]
    validate_annotation_contract(unit_rows=unit_rows, key_rows=key_rows)
    plan = assign_annotation_units(
        unit_rows=unit_rows,
        key_rows=key_rows,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        seed=assignment_seed,
    )
    assignment_rows = plan.assignments

    units_path = output_dir / "human_eval_units.jsonl"
    key_path = output_dir / "human_eval_key.jsonl"
    manifest_path = output_dir / "assignment_manifest.json"
    _guard(units_path, overwrite=overwrite)
    _guard(key_path, overwrite=overwrite)
    _guard(manifest_path, overwrite=overwrite)
    write_jsonl_atomic(units_path, unit_rows)
    write_jsonl_atomic(key_path, key_rows)

    assignment_paths: list[Path] = []
    for annotator_id, rows in assignment_rows.items():
        path = assignments_dir / f"{annotator_id}.jsonl"
        _guard(path, overwrite=overwrite)
        write_jsonl_atomic(path, rows)
        assignment_paths.append(path)

    manifest = {
        "dataset_dirs": {family: str(path) for family, path in sorted(dataset_dirs.items())},
        "target_units_requested": requested_counts,
        "unique_units_exported": len(unit_rows),
        "unique_units_by_family": selected_counts,
        "annotators": annotators,
        "assignments_per_annotator": assignments_per_annotator,
        "total_assignments": sum(len(rows) for rows in assignment_rows.values()),
        **assignment_manifest_metadata(plan, key_rows=key_rows),
        "annotation_fields": ANNOTATION_FIELDS,
        "annotation_protocol_version": PROTOCOL_VERSION,
        "annotation_text_repair_version": ANNOTATION_TEXT_REPAIR_VERSION,
        "annotator_files": [str(path) for path in assignment_paths],
        "artifact_sha256": artifact_hashes(
            root=output_dir,
            paths=[units_path, key_path, *assignment_paths],
        ),
        "blind_annotation_note": (
            "Assignment files contain a plain-text reference answer for the trusted server. The "
            "annotation API withholds it until evidence-stage judgments are locked, and omits "
            "question programs, fact roles, systems, models, and metric labels."
        ),
    }
    write_json_atomic(manifest_path, manifest, sort_keys=True)
    return {
        "human_eval_units": units_path,
        "human_eval_key": key_path,
        "assignment_manifest": manifest_path,
    }


def _select_units(bundle: DatasetBundle, *, target_units: int) -> list[AnswerVariant]:
    if target_units < 1:
        return []
    human_scenarios = set(bundle.splits.get("human_pool", []))
    test_scenarios = set(bundle.splits.get("test_auto", []))
    question_tiers = [
        {
            question.qid
            for question in bundle.questions
            if question.scenario_id in human_scenarios and question.human_pool_candidate
        },
        {question.qid for question in bundle.questions if question.scenario_id in human_scenarios},
        {
            question.qid
            for question in bundle.questions
            if question.scenario_id in test_scenarios and question.human_pool_candidate
        },
        {question.qid for question in bundle.questions if question.scenario_id in test_scenarios},
    ]
    candidates: list[AnswerVariant] = []
    seen_answer_ids: set[str] = set()
    for question_ids in question_tiers:
        for answer in sorted(bundle.answer_variants, key=lambda item: item.answer_id):
            if answer.qid in question_ids and answer.answer_id not in seen_answer_ids:
                candidates.append(answer)
                seen_answer_ids.add(answer.answer_id)
        if len(candidates) >= target_units:
            break
    if len(candidates) < target_units:
        raise ValueError(
            f"Requested {target_units} annotation units but only {len(candidates)} held-out "
            "answer variants are available"
        )

    by_type: dict[str, list[AnswerVariant]] = defaultdict(list)
    for answer in candidates:
        by_type[str(answer.variant_type)].append(answer)

    selected: list[AnswerVariant] = []
    while len(selected) < target_units and any(by_type.values()):
        for variant_type in sorted(by_type):
            if by_type[variant_type] and len(selected) < target_units:
                selected.append(by_type[variant_type].pop(0))
    return selected


def _family_interleaved(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["_ordering_family"])].append(row)

    ordered: list[dict[str, object]] = []
    families = sorted(by_family)
    while any(by_family.values()):
        for family in families:
            if by_family[family]:
                ordered.append(by_family[family].pop(0))
    return ordered


def controlled_public_unit(
    bundle: DatasetBundle,
    answer: AnswerVariant,
) -> dict[str, object]:
    public, _ = controlled_public_unit_with_key(bundle, answer)
    return public


def controlled_public_unit_with_key(
    bundle: DatasetBundle,
    answer: AnswerVariant,
) -> tuple[dict[str, object], dict[str, object]]:
    """Render one controlled case and return its local, non-semantic ID map.

    The map is kept outside the displayed payload. Evaluation tooling uses it
    only to calculate formal retrieval/citation labels after blinding.
    """
    internal = _controlled_internal_unit(bundle, answer)
    return blind_public_payload(internal)


def controlled_internal_unit(
    bundle: DatasetBundle,
    answer: AnswerVariant,
) -> dict[str, object]:
    """Render the unblinded payload for trusted, local transformation tooling."""
    return _controlled_internal_unit(bundle, answer)


def _controlled_internal_unit(
    bundle: DatasetBundle,
    answer: AnswerVariant,
) -> dict[str, object]:
    question = _question_by_id(bundle, answer.qid)
    fact_by_id, entity_by_id, facts_by_scenario, _ = _bundle_indexes(bundle)
    scenario_facts = facts_by_scenario.get(answer.scenario_id, [])
    cited_evidence = [
        public_evidence(fact_by_id[fact_id])
        for fact_id in answer.cited_evidence_ids
        if fact_id in fact_by_id
    ]
    retrieved_fact_ids = _controlled_retrieved_fact_ids(
        question=question,
        answer=answer,
        facts=scenario_facts,
    )
    retrieved_evidence = [
        public_evidence(fact_by_id[fact_id])
        for fact_id in retrieved_fact_ids
        if fact_id in fact_by_id
    ]
    graph_paths = _controlled_graph_paths(
        bundle=bundle,
        question=question,
        answer=answer,
        fact_by_id=fact_by_id,
        entity_by_id=entity_by_id,
        retrieved_fact_ids=retrieved_fact_ids,
    )
    response_kind = response_decision_kind(answer.answer_text)
    applicable_fields = _applicable_fields(
        has_support_evidence=bool(retrieved_evidence),
        has_citations=bool(cited_evidence),
        graph_applicable=bool(graph_paths),
        response_kind=response_kind,
        should_abstain=question.should_abstain,
        world_temporal=question.program.temporal_basis == "world_valid_time",
    )
    internal = {
        "unit_id": _opaque_unit_id(answer.answer_id),
        "dataset_family": question.dataset_family,
        "scenario_id": answer.scenario_id,
        "qid": question.qid,
        "answer_id": _opaque_candidate_id(answer.answer_id),
        "answer_source": "anonymous",
        "question": annotation_question_text(
            question.question,
            dataset_family=str(question.dataset_family),
        ),
        "reference_answer": _reference_answer(question),
        "answer_text": answer.answer_text,
        "cited_evidence_ids": [row["evidence_id"] for row in cited_evidence],
        "cited_evidence": cited_evidence,
        "retrieved_evidence": retrieved_evidence,
        "graph_paths": graph_paths,
        "context_note": _controlled_context_note(question=question, answer=answer),
        "applicable_fields": applicable_fields,
    }
    return normalize_annotation_payload(
        internal,
        dataset_family=str(question.dataset_family),
    )


def controlled_private_key(
    answer: AnswerVariant,
    *,
    bundle: DatasetBundle | None = None,
) -> dict[str, object]:
    blinding: dict[str, object] = {}
    question: Question | None = None
    if bundle is not None:
        question = _question_by_id(bundle, answer.qid)
        _, blinding = blind_public_payload(_controlled_internal_unit(bundle, answer))
    semantic_oracle_labels = _semantic_oracle_labels(answer)
    expected_labels = dict(semantic_oracle_labels)
    expected_reasons: dict[str, str] = {}
    if (
        str(answer.variant_type) in {"invalid_graph_path_answer", "unsupported_hallucinated_answer"}
        and expected_labels["temporal_correct"] != "not_applicable"
    ):
        expected_labels["temporal_correct"] = "unjudgeable"
        expected_reasons["temporal_correct"] = "candidate_time_not_established"
    return {
        "unit_id": _opaque_unit_id(answer.answer_id),
        "source_kind": "controlled_variant",
        "answer_id": answer.answer_id,
        "qid": answer.qid,
        "scenario_id": answer.scenario_id,
        "variant_type": answer.variant_type,
        "alternate_operator": answer.alternate_operator,
        **expected_labels,
        "semantic_oracle_labels": semantic_oracle_labels,
        "expected_reasons": expected_reasons,
        "response_decision_kind": response_decision_kind(answer.answer_text),
        "should_abstain": question.should_abstain if question is not None else None,
        "claim_labels": [
            {
                "cid": claim.cid,
                "temporally_valid": claim.temporally_valid,
                "cited_evidence_ids": claim.cited_evidence_ids,
            }
            for claim in answer.claims
        ],
        "blinding_map": blinding,
    }


def _bundle_indexes(
    bundle: DatasetBundle,
) -> tuple[
    dict[str, Fact],
    dict[str, Entity],
    dict[str, list[Fact]],
    dict[str, list[GraphPath]],
]:
    """Build immutable lookup indexes once per loaded dataset bundle."""
    cache_key = id(bundle)
    cached = _BUNDLE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is bundle:
        return cached[1], cached[2], cached[3], cached[4]
    facts_by_scenario: defaultdict[str, list[Fact]] = defaultdict(list)
    for fact in bundle.facts:
        facts_by_scenario[fact.scenario_id].append(fact)
    paths_by_qid: defaultdict[str, list[GraphPath]] = defaultdict(list)
    for path in bundle.graph_paths:
        paths_by_qid[path.qid].append(path)
    indexed = (
        bundle,
        {fact.fact_id: fact for fact in bundle.facts},
        {entity.entity_id: entity for entity in bundle.entities},
        dict(facts_by_scenario),
        dict(paths_by_qid),
    )
    _BUNDLE_INDEX_CACHE[cache_key] = indexed
    return indexed[1], indexed[2], indexed[3], indexed[4]


def _semantic_oracle_labels(answer: AnswerVariant) -> dict[str, str]:
    return {
        "answer_correct": str(answer.answer_correct),
        "temporal_correct": str(answer.temporal_correct),
        "evidence_supports_answer": str(answer.evidence_supports_answer),
        "citation_temporally_valid": str(answer.citation_temporally_valid),
        "graph_evidence_sufficient": str(answer.graph_path_sufficient),
        "response_decision_appropriate": str(answer.refusal_appropriate),
    }


def public_evidence(fact: Fact) -> dict[str, object]:
    return {
        "evidence_id": fact.fact_id,
        "text": fact.paraphrased_evidence or fact.canonical_evidence,
        "publication_time": fact.publication_time.isoformat() if fact.publication_time else None,
        "valid_time": fact.valid_time.model_dump(mode="json"),
    }


def _public_path(
    path: GraphPath,
    *,
    fact_by_id: dict[str, Fact],
    entity_by_id: dict[str, Entity],
) -> dict[str, object]:
    return {
        "path_id": path.pid,
        "path_source": "dataset_graph_paths",
        "nodes": [
            public_path_node(node_id, entity_by_id=entity_by_id)
            for node_id in _path_node_ids(path, fact_by_id=fact_by_id)
        ],
        "edges": [
            public_path_edge(edge, fact_by_id=fact_by_id, entity_by_id=entity_by_id)
            for edge in path.edges
        ],
    }


def _path_node_ids(path: GraphPath, *, fact_by_id: dict[str, Fact]) -> list[str]:
    edge_nodes = graph_path_node_ids(path.edges, fact_by_id)
    return list(dict.fromkeys([*path.nodes, *edge_nodes]))


def _controlled_retrieved_fact_ids(
    *,
    question: Question,
    answer: AnswerVariant,
    facts: list[Fact],
) -> list[str]:
    fact_ids = list(answer.cited_evidence_ids)
    variant_type = str(answer.variant_type)
    if variant_type == "inappropriate_refusal":
        fact_ids.extend(question.required_valid_evidence_ids)
    if variant_type == "correct_answer_invalid_evidence":
        # The valid uncited fact makes claim-time correctness decidable while the
        # candidate's stale/invalid citation remains independently judgeable.
        fact_ids.extend(question.required_valid_evidence_ids)
    if _needs_complete_ordering_context(question=question, answer=answer):
        fact_ids.extend(_ordering_comparison_fact_ids(question=question, facts=facts))
    return list(dict.fromkeys(fact_ids))


def _needs_complete_ordering_context(
    *,
    question: Question,
    answer: AnswerVariant,
) -> bool:
    """Return whether temporal judgment requires comparison against all eligible facts."""
    return (
        question.temporal_operator in _ORDERING_OPERATORS
        and str(answer.temporal_correct) != "not_applicable"
    )


def _ordering_comparison_fact_ids(*, question: Question, facts: list[Fact]) -> list[str]:
    matching = [
        fact
        for fact in facts
        if fact_visible(fact, question.program.snapshot_id)
        and fact_matches_program(fact, question.program)
    ]
    operator = question.temporal_operator
    point = query_point(question.program)
    candidates: list[Fact]
    if operator == TemporalOperator.FIRST:
        candidates = [fact for fact in matching if fact.valid_time.start is not None]
    elif operator in {TemporalOperator.LATEST, TemporalOperator.LAST}:
        candidates = [
            fact
            for fact in matching
            if fact.valid_time.start is not None and fact.valid_time.start <= point
        ]
    elif operator in {TemporalOperator.BEFORE, TemporalOperator.EXPIRED}:
        candidates = [
            fact
            for fact in matching
            if fact.valid_time.end is not None and fact.valid_time.end < point
        ]
    elif operator == TemporalOperator.AFTER:
        candidates = [
            fact
            for fact in matching
            if fact.valid_time.start is not None and fact.valid_time.start > point
        ]
    elif operator == TemporalOperator.PREVIOUS:
        current = [fact for fact in matching if fact.valid_time.contains(point)]
        starts = [fact.valid_time.start for fact in current if fact.valid_time.start is not None]
        boundary = min(starts) if starts else point
        candidates = [
            *current,
            *[
                fact
                for fact in matching
                if fact.valid_time.end is not None and fact.valid_time.end < boundary
            ],
        ]
    elif operator == TemporalOperator.NEXT:
        current = [
            fact
            for fact in matching
            if fact.valid_time.contains(point) and fact.valid_time.end is not None
        ]
        ends = [fact.valid_time.end for fact in current if fact.valid_time.end is not None]
        boundary = max(ends) if ends else point
        candidates = [
            *current,
            *[
                fact
                for fact in matching
                if fact.valid_time.start is not None and fact.valid_time.start > boundary
            ],
        ]
    else:
        candidates = []
    selected = set(question.required_valid_evidence_ids)
    ordered = sorted(
        {fact.fact_id: fact for fact in candidates}.values(),
        key=lambda fact: (
            0 if fact.fact_id in selected else 1,
            fact.valid_time.start or point,
            fact.valid_time.end or point,
            fact.fact_id,
        ),
    )
    return [fact.fact_id for fact in ordered]


def _controlled_graph_paths(
    *,
    bundle: DatasetBundle,
    question: Question,
    answer: AnswerVariant,
    fact_by_id: dict[str, Fact],
    entity_by_id: dict[str, Entity],
    retrieved_fact_ids: list[str],
) -> list[dict[str, object]]:
    variant_type = str(answer.variant_type)
    _, _, _, paths_by_qid = _bundle_indexes(bundle)
    question_paths = paths_by_qid.get(question.qid, [])
    if variant_type == "inappropriate_refusal":
        selected_paths = [path for path in question_paths if path.supports_gold_answer]
    else:
        explicit_ids = set(answer.graph_path_ids)
        selected_paths = [path for path in question_paths if path.pid in explicit_ids]
    public_paths = [
        _public_path(path, fact_by_id=fact_by_id, entity_by_id=entity_by_id)
        for path in selected_paths
    ]
    represented_fact_ids = {
        str(edge.get("fact_id"))
        for path in public_paths
        for edge in path.get("edges", [])
        if isinstance(edge, dict) and edge.get("fact_id")
    }
    needs_complete_comparison = (
        variant_type == "correct_supported" and question.temporal_operator in _ORDERING_OPERATORS
    )
    if variant_type == "inappropriate_refusal" or needs_complete_comparison:
        for fact_id in retrieved_fact_ids:
            fact = fact_by_id.get(fact_id)
            if fact is None or fact_id in represented_fact_ids:
                continue
            public_paths.append(
                _public_single_fact_path(
                    fact,
                    fact_by_id=fact_by_id,
                    entity_by_id=entity_by_id,
                )
            )
            represented_fact_ids.add(fact_id)
    return public_paths


def _public_single_fact_path(
    fact: Fact,
    *,
    fact_by_id: dict[str, Fact],
    entity_by_id: dict[str, Entity],
) -> dict[str, object]:
    edge = GraphPathEdge(
        fact_id=fact.fact_id,
        relation=fact.relation,
        valid_time=fact.valid_time,
    )
    node_ids = graph_path_node_ids([edge], fact_by_id)
    return {
        "path_id": f"context_{fact.fact_id}",
        "path_source": "dataset_graph_paths",
        "nodes": [public_path_node(node_id, entity_by_id=entity_by_id) for node_id in node_ids],
        "edges": [public_path_edge(edge, fact_by_id=fact_by_id, entity_by_id=entity_by_id)],
    }


def _controlled_context_note(*, question: Question, answer: AnswerVariant) -> str:
    notes: list[str] = []
    if str(answer.variant_type) == "inappropriate_refusal":
        notes.append(
            "The displayed supporting material was available to the candidate when it decided "
            "whether to answer."
        )
    if _needs_complete_ordering_context(question=question, answer=answer):
        notes.append(
            "For this ordering question, the displayed support includes the complete visible "
            "comparison set eligible for the requested operator."
        )
    return " ".join(notes)


def public_path_edge(
    edge: GraphPathEdge,
    *,
    fact_by_id: dict[str, Fact],
    entity_by_id: dict[str, Entity],
) -> dict[str, object]:
    return path_edge_payload(edge, fact_by_id, entity_by_id)


def public_path_node(node_id: str, *, entity_by_id: dict[str, Entity]) -> dict[str, str]:
    entity = entity_by_id.get(node_id)
    return {
        "id": node_id,
        "label": getattr(entity, "name", node_id) if entity else node_id,
        "type": str(getattr(entity, "entity_type", "")) if entity else "",
    }


def blind_public_payload(
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Remove experimental-condition metadata and replace internal identifiers.

    Handles are local to one unit, so neither annotators nor learned evaluators
    can recover global fact roles, source families, or scenario neighborhoods.
    """
    copied = orjson.loads(orjson.dumps(payload))
    fact_ids: list[str] = []
    for key in ("cited_evidence", "retrieved_evidence"):
        for evidence in copied.get(key, []):
            if isinstance(evidence, dict) and evidence.get("evidence_id"):
                fact_ids.append(str(evidence["evidence_id"]))
    for path in copied.get("graph_paths", []):
        if not isinstance(path, dict):
            continue
        for edge in path.get("edges", []):
            if isinstance(edge, dict) and edge.get("fact_id"):
                fact_ids.append(str(edge["fact_id"]))
    fact_ids = list(dict.fromkeys(fact_ids))
    fact_handles = {fact_id: f"E{index}" for index, fact_id in enumerate(fact_ids, start=1)}

    node_ids: list[str] = []
    for path in copied.get("graph_paths", []):
        if not isinstance(path, dict):
            continue
        for node in path.get("nodes", []):
            if isinstance(node, dict) and node.get("id"):
                node_ids.append(str(node["id"]))
        for edge in path.get("edges", []):
            if not isinstance(edge, dict):
                continue
            for endpoint in ("source", "target"):
                node = edge.get(endpoint)
                if isinstance(node, dict) and node.get("id"):
                    node_ids.append(str(node["id"]))
    node_handles = {
        node_id: f"N{index}" for index, node_id in enumerate(dict.fromkeys(node_ids), start=1)
    }

    for key in ("cited_evidence", "retrieved_evidence"):
        blinded_evidence = []
        for evidence in copied.get(key, []):
            if not isinstance(evidence, dict):
                continue
            internal_id = str(evidence.get("evidence_id", ""))
            blinded_evidence.append(
                {
                    field: value
                    for field, value in evidence.items()
                    if field not in {"source_type", "provenance_reliability", "fact_role"}
                }
                | {"evidence_id": fact_handles.get(internal_id, "")}
            )
        copied[key] = blinded_evidence

    copied["cited_evidence_ids"] = [
        fact_handles[fact_id]
        for fact_id in copied.get("cited_evidence_ids", [])
        if fact_id in fact_handles
    ]
    for path_index, path in enumerate(copied.get("graph_paths", []), start=1):
        if not isinstance(path, dict):
            continue
        path["path_id"] = f"path_{path_index:02d}"
        path["path_source"] = "candidate_path"
        for node in path.get("nodes", []):
            if isinstance(node, dict):
                node["id"] = node_handles.get(str(node.get("id", "")), "")
        for edge in path.get("edges", []):
            if not isinstance(edge, dict):
                continue
            internal_id = str(edge.get("fact_id", ""))
            edge["fact_id"] = fact_handles.get(internal_id, "")
            edge["id"] = edge["fact_id"]
            for endpoint in ("source", "target"):
                node = edge.get(endpoint)
                if isinstance(node, dict):
                    node["id"] = node_handles.get(str(node.get("id", "")), "")
            for field in ("source_type", "role", "object_id"):
                edge.pop(field, None)

    answer_text = str(copied.get("answer_text", ""))
    for internal_id, handle in fact_handles.items():
        answer_text = answer_text.replace(internal_id, handle)
    copied["answer_text"] = annotation_plain_text(answer_text)
    require_visible_citations(copied)
    for field in ("dataset_family", "scenario_id", "qid"):
        copied.pop(field, None)
    return copied, {
        "evidence_handles": {handle: fact_id for fact_id, handle in fact_handles.items()},
        "node_handles": {handle: node_id for node_id, handle in node_handles.items()},
    }


def _reference_answer(question: Question) -> str:
    if question.should_abstain:
        return "No answer is supported by the available evidence at the requested time."
    return ", ".join(question.gold_answer_text)


def _applicable_fields(
    *,
    has_support_evidence: bool,
    has_citations: bool,
    graph_applicable: bool,
    response_kind: str,
    should_abstain: bool,
    world_temporal: bool,
) -> list[str]:
    fields = ["answer_correct"]
    candidate_makes_claim = response_kind != "refusal"
    if candidate_makes_claim:
        if world_temporal:
            fields.append("temporal_correct")
        if has_support_evidence:
            fields.append("evidence_supports_answer")
        if has_citations and world_temporal:
            fields.append("citation_temporally_valid")
        if graph_applicable:
            fields.append("graph_evidence_sufficient")
    if should_abstain or response_kind != "answer":
        fields.append("response_decision_appropriate")
    return fields


def validate_annotation_contract(
    *,
    unit_rows: list[dict[str, object]],
    key_rows: list[dict[str, object]],
) -> None:
    key_by_id = {str(row.get("unit_id", "")): row for row in key_rows}
    if len(key_by_id) != len(key_rows):
        raise ValueError("Human-evaluation private keys contain duplicate or blank unit ids")
    if {str(row.get("unit_id", "")) for row in unit_rows} != set(key_by_id):
        raise ValueError("Human-evaluation public units and private keys do not align")

    for unit in unit_rows:
        require_visible_citations(unit)
        unit_id = str(unit["unit_id"])
        key = key_by_id[unit_id]
        applicable = {
            str(field) for field in unit.get("applicable_fields", []) if isinstance(field, str)
        }
        unknown = applicable - set(CATEGORICAL_FIELDS)
        if unknown:
            raise ValueError(f"Human-evaluation unit {unit_id} has unknown fields: {unknown}")
        text_issues = annotation_text_quality_issues(unit)
        if text_issues:
            raise ValueError(
                f"Human-evaluation unit {unit_id} exposes annotation text defects: "
                f"{', '.join(text_issues)}"
            )
        if key.get("source_kind") == "controlled_variant":
            for field in CATEGORICAL_FIELDS:
                expected = str(key.get(field, "not_applicable")) != "not_applicable"
                if (field in applicable) != expected:
                    raise ValueError(
                        f"Human-evaluation unit {unit_id} disagrees with its controlled key for "
                        f"{field}"
                    )
        response_expected = bool(
            key.get("should_abstain")
            or str(key.get("response_decision_kind", "answer")) != "answer"
        )
        if ("response_decision_appropriate" in applicable) != response_expected:
            raise ValueError(
                f"Human-evaluation unit {unit_id} has inconsistent response-decision applicability"
            )
        if key.get("variant_type") == "inappropriate_refusal" and (
            not displayed_evidence(unit)
            or "available to the candidate" not in str(unit.get("context_note", ""))
        ):
            raise ValueError(
                f"Inappropriate-refusal unit {unit_id} does not display ignored support"
            )
        if key.get("variant_type") == "correct_answer_invalid_evidence":
            visible_ids = {
                str(evidence.get("evidence_id", "")) for evidence in displayed_evidence(unit)
            }
            cited_ids = {str(evidence_id) for evidence_id in unit.get("cited_evidence_ids", [])}
            if not visible_ids - cited_ids:
                raise ValueError(f"Invalid-citation unit {unit_id} lacks uncited valid support")
        if "complete visible comparison set" in str(
            unit.get("context_note", "")
        ) and not displayed_evidence(unit):
            raise ValueError(f"Ordering unit {unit_id} lacks comparison evidence")
        if (
            "complete visible comparison set" in str(unit.get("context_note", ""))
            and "graph_evidence_sufficient" in applicable
            and not unit.get("graph_paths")
        ):
            raise ValueError(
                f"Ordering unit {unit_id} lacks graph witnesses for an applicable graph judgment"
            )


def _question_by_id(bundle: DatasetBundle, qid: str) -> Question:
    for question in bundle.questions:
        if question.qid == qid:
            return question
    raise KeyError(f"Question not found: {qid}")


def _guard(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def _opaque_unit_id(source_id: str) -> str:
    return f"heu_{hashlib.sha256(f'unit:{source_id}'.encode()).hexdigest()[:20]}"


def _opaque_candidate_id(source_id: str) -> str:
    return f"candidate_{hashlib.sha256(f'candidate:{source_id}'.encode()).hexdigest()[:16]}"
