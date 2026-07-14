from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import orjson

from tcred.dataset.io import (
    load_bundle,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from tcred.dataset.models import DatasetBundle, Entity, Fact, GraphPathEdge
from tcred.human_eval.assignments import (
    DEFAULT_ASSIGNMENT_SEED,
    AnnotationAssignmentPlan,
    assign_annotation_units,
    assignment_manifest_metadata,
)
from tcred.human_eval.export import (
    ANNOTATION_FIELDS,
    _applicable_fields,
    _reference_answer,
    blind_public_payload,
    controlled_private_key,
    controlled_public_unit,
    public_evidence,
    public_path_edge,
    public_path_node,
    validate_annotation_contract,
)
from tcred.human_eval.package import artifact_hashes, package_sha256
from tcred.human_eval.presentation import (
    ANNOTATION_TEXT_REPAIR_VERSION,
    annotation_question_text,
    normalize_annotation_payload,
)
from tcred.human_eval.protocol import PROTOCOL_VERSION
from tcred.human_eval.response import response_decision_kind
from tcred.human_eval.sampling import select_controlled_answers, select_system_outputs
from tcred.qa.checkpoint import (
    checkpoint_integrity_matches,
    qa_implementation_sha256,
    read_checkpoint_metadata,
)
from tcred.qa.corpus import dataset_content_hash, load_runtime_questions
from tcred.qa.generation import INTERNAL_PATH_REFERENCE
from tcred.qa.models import ALL_QA_SYSTEMS, SystemOutput


def export_augmented_human_eval(
    *,
    dataset_dirs: Mapping[str, Path],
    system_output_root: Path,
    output_dir: Path,
    target_units_by_family: Mapping[str, int],
    annotators: int = 36,
    assignments_per_annotator: int = 20,
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED,
    controlled_fraction: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Path]:
    if controlled_fraction != 0.5:
        raise ValueError(
            "The preregistered v1 human-evaluation mix requires controlled_fraction=0.5"
        )
    system_output_provenance = _validate_system_output_release(
        dataset_dirs=dataset_dirs,
        system_output_root=system_output_root,
    )
    unit_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for family, dataset_dir in sorted(dataset_dirs.items()):
        target = int(target_units_by_family.get(family, 0))
        if target % 2:
            raise ValueError(f"Human-evaluation target must be even for {family}: {target}")
        bundle = load_bundle(dataset_dir)
        controlled_target = target // 2
        controlled = select_controlled_answers(bundle, target=controlled_target)
        controlled_qids = {answer.qid for answer in controlled}
        system_outputs = select_system_outputs(
            bundle=bundle,
            family=family,
            system_output_root=system_output_root,
            target=target - controlled_target,
            excluded_qids=controlled_qids,
            allow_excluded_fallback=True,
        )
        for answer in controlled:
            public = controlled_public_unit(bundle, answer)
            private = controlled_private_key(answer, bundle=bundle)
            _finalize_identity(
                public=public,
                private=private,
                family=family,
                source_id=answer.answer_id,
            )
            private["dataset_family"] = family
            unit_rows.append(public)
            key_rows.append(private)
        for output in system_outputs:
            public, private = _system_unit(bundle, output)
            unit_rows.append(public)
            key_rows.append(private)

    validate_annotation_contract(unit_rows=unit_rows, key_rows=key_rows)
    _assert_unique_evaluation_cards(unit_rows, key_rows)
    unit_rows = _interleave_units(unit_rows, key_rows)
    key_by_id = {str(row["unit_id"]): row for row in key_rows}
    key_rows = [key_by_id[str(row["unit_id"])] for row in unit_rows]
    plan = assign_annotation_units(
        unit_rows=unit_rows,
        key_rows=key_rows,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        seed=assignment_seed,
    )
    assignments = plan.assignments

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir = output_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    units_path = output_dir / "human_eval_units.jsonl"
    key_path = output_dir / "human_eval_key.jsonl"
    manifest_path = output_dir / "assignment_manifest.json"
    assignment_paths = [assignments_dir / f"{annotator_id}.jsonl" for annotator_id in assignments]
    for path in [units_path, key_path, manifest_path, *assignment_paths]:
        _guard(path, overwrite=overwrite)

    write_jsonl_atomic(units_path, unit_rows)
    write_jsonl_atomic(key_path, key_rows)
    for annotator_id, rows in assignments.items():
        write_jsonl_atomic(assignments_dir / f"{annotator_id}.jsonl", rows)
    manifest = _manifest(
        dataset_dirs=dataset_dirs,
        system_output_root=system_output_root,
        target_units_by_family=target_units_by_family,
        unit_rows=unit_rows,
        key_rows=key_rows,
        assignment_plan=plan,
        annotators=annotators,
        assignments_per_annotator=assignments_per_annotator,
        assignment_paths=assignment_paths,
        units_path=units_path,
        key_path=key_path,
        system_output_provenance=system_output_provenance,
    )
    write_json_atomic(manifest_path, manifest)
    return {
        "human_eval_units": units_path,
        "human_eval_key": key_path,
        "assignment_manifest": manifest_path,
    }


def system_public_unit(bundle: DatasetBundle, output: SystemOutput) -> dict[str, object]:
    """Build the blinded evaluation card used for a QA-system response."""

    public, _private = _system_unit(bundle, output)
    return public


def _system_unit(
    bundle: DatasetBundle,
    output: SystemOutput,
) -> tuple[dict[str, object], dict[str, object]]:
    question_by_id = {question.qid: question for question in bundle.questions}
    fact_by_id = {fact.fact_id: fact for fact in bundle.facts}
    entity_by_id = {entity.entity_id: entity for entity in bundle.entities}
    question = question_by_id[output.qid]
    retrieved_ids = [hit.fact_id for hit in output.retrieval.hits if hit.fact_id in fact_by_id]
    cited_ids = [fact_id for fact_id in output.resolved_cited_evidence_ids if fact_id in fact_by_id]
    graph_paths = _system_paths(
        output,
        fact_by_id=fact_by_id,
        entity_by_id=entity_by_id,
    )
    response_kind = response_decision_kind(output.answer_text)
    applicable_fields = _applicable_fields(
        has_support_evidence=bool(retrieved_ids),
        has_citations=bool(cited_ids),
        graph_applicable=bool(graph_paths),
        response_kind=response_kind,
        should_abstain=question.should_abstain,
        world_temporal=question.program.temporal_basis == "world_valid_time",
    )
    public_internal: dict[str, object] = {
        "unit_id": "",
        "dataset_family": output.dataset_family,
        "scenario_id": output.scenario_id,
        "qid": output.qid,
        "answer_id": "",
        "answer_source": "anonymous",
        "question": annotation_question_text(
            question.question,
            dataset_family=str(question.dataset_family),
        ),
        "reference_answer": _reference_answer(question),
        "answer_text": output.answer_text,
        "cited_evidence_ids": cited_ids,
        "cited_evidence": [public_evidence(fact_by_id[fact_id]) for fact_id in cited_ids],
        "retrieved_evidence": [public_evidence(fact_by_id[fact_id]) for fact_id in retrieved_ids],
        "unresolved_citation_ids": output.unresolved_citation_ids,
        "graph_paths": graph_paths,
        "context_note": (
            "The displayed evidence and graph paths are the retrieved context supplied to this "
            "candidate."
        ),
        "applicable_fields": applicable_fields,
    }
    public_internal = normalize_annotation_payload(
        public_internal,
        dataset_family=str(output.dataset_family),
    )
    public, blinding_map = blind_public_payload(public_internal)
    private: dict[str, object] = {
        "unit_id": "",
        "source_kind": "system_output",
        "dataset_family": output.dataset_family,
        "system_name": output.system_name,
        "system_output_id": output.output_id,
        "run_id": output.run_id,
        "qid": output.qid,
        "scenario_id": output.scenario_id,
        "generator_provider": output.generator_provider,
        "generator_model": output.generator_model,
        "prompt_version": output.prompt_version,
        "response_decision_kind": response_kind,
        "should_abstain": question.should_abstain,
        "retrieval_context_sha256": output.retrieval.context_sha256,
        "blinding_map": blinding_map,
    }
    _finalize_identity(
        public=public,
        private=private,
        family=output.dataset_family,
        source_id=output.output_id,
    )
    return public, private


def _system_paths(
    output: SystemOutput,
    *,
    fact_by_id: dict[str, Fact],
    entity_by_id: dict[str, Entity],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, path in enumerate(output.retrieval.graph_paths, start=1):
        edges = []
        directions = path.traversal_directions or ["forward"] * len(path.fact_ids)
        for fact_id, direction in zip(path.fact_ids, directions, strict=True):
            fact = fact_by_id.get(fact_id)
            if fact is None:
                continue
            edge = GraphPathEdge(
                fact_id=fact_id,
                relation=fact.relation,
                valid_time=fact.valid_time,
                traversal_direction=direction,
            )
            edges.append(public_path_edge(edge, fact_by_id=fact_by_id, entity_by_id=entity_by_id))
        if not edges:
            continue
        node_ids = list(path.node_ids)
        for edge in edges:
            node_ids.extend([str(edge["source"]["id"]), str(edge["target"]["id"])])
        rows.append(
            {
                "path_id": f"path_{index:02d}",
                "path_source": "candidate_path",
                "nodes": [
                    public_path_node(node_id, entity_by_id=entity_by_id)
                    for node_id in dict.fromkeys(node_ids)
                    if node_id
                ],
                "edges": edges,
            }
        )
    return rows


def _finalize_identity(
    *,
    public: dict[str, object],
    private: dict[str, object],
    family: str,
    source_id: str,
) -> None:
    digest = hashlib.sha256(f"human-eval-v2:{family}:{source_id}".encode()).hexdigest()
    public["unit_id"] = f"heu_{digest[:20]}"
    public["answer_id"] = f"candidate_{digest[20:36]}"
    public["answer_source"] = "anonymous"
    private["unit_id"] = public["unit_id"]


def _interleave_units(
    units: list[dict[str, object]],
    keys: list[dict[str, object]],
) -> list[dict[str, object]]:
    key_by_id = {str(row["unit_id"]): row for row in keys}
    groups: defaultdict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for unit in units:
        key = key_by_id[str(unit["unit_id"])]
        source_kind = str(key["source_kind"])
        detail = str(key.get("system_name") or key.get("variant_type") or "")
        groups[(str(key["dataset_family"]), source_kind, detail)].append(unit)
    for group_key, rows in groups.items():
        groups[group_key] = sorted(rows, key=lambda row: str(row["unit_id"]))
    ordered: list[dict[str, object]] = []
    group_keys = sorted(groups)
    while any(groups.values()):
        for group_key in group_keys:
            if groups[group_key]:
                ordered.append(groups[group_key].pop(0))
    return ordered


def _assert_unique_evaluation_cards(
    units: list[dict[str, object]],
    keys: list[dict[str, object]],
) -> None:
    """Reject cards that ask annotators to judge the same observable object twice."""

    key_by_id = {str(row["unit_id"]): row for row in keys}
    seen: dict[str, str] = {}
    for unit in units:
        unit_id = str(unit["unit_id"])
        if INTERNAL_PATH_REFERENCE.search(str(unit.get("answer_text", ""))):
            raise ValueError(
                f"Human-evaluation card {unit_id} exposes an internal retrieval-path label"
            )
        fingerprint = _evaluation_card_fingerprint(unit)
        previous = seen.get(fingerprint)
        if previous is not None:
            previous_key = key_by_id[previous]
            current_key = key_by_id[unit_id]
            raise ValueError(
                "Human-evaluation sampling produced evaluator-equivalent duplicate cards: "
                f"{previous} ({previous_key.get('source_kind')}) and "
                f"{unit_id} ({current_key.get('source_kind')})"
            )
        seen[fingerprint] = unit_id


def _evaluation_card_fingerprint(unit: dict[str, object]) -> str:
    def evidence_view(row: object) -> dict[str, object]:
        evidence = dict(row)  # type: ignore[arg-type]
        evidence.pop("evidence_id", None)
        return evidence

    def endpoint_view(row: object) -> dict[str, object]:
        endpoint = dict(row)  # type: ignore[arg-type]
        endpoint.pop("id", None)
        return endpoint

    def edge_view(row: object) -> dict[str, object]:
        edge = dict(row)  # type: ignore[arg-type]
        for key in ("id", "fact_id"):
            edge.pop(key, None)
        edge["source"] = endpoint_view(edge["source"])
        edge["target"] = endpoint_view(edge["target"])
        return edge

    paths = []
    for raw_path in unit.get("graph_paths", []):  # type: ignore[union-attr]
        path = dict(raw_path)
        paths.append(
            {
                "path_source": path.get("path_source"),
                "nodes": sorted(
                    (endpoint_view(node) for node in path.get("nodes", [])),
                    key=lambda node: orjson.dumps(node, option=orjson.OPT_SORT_KEYS),
                ),
                "edges": [edge_view(edge) for edge in path.get("edges", [])],
            }
        )
    paths.sort(key=lambda path: orjson.dumps(path, option=orjson.OPT_SORT_KEYS))
    payload = {
        "question": unit.get("question"),
        "reference_answer": unit.get("reference_answer"),
        "answer_text": " ".join(str(unit.get("answer_text", "")).split()),
        "cited_evidence": [
            evidence_view(row)
            for row in unit.get("cited_evidence", [])  # type: ignore[union-attr]
        ],
        "unresolved_citation_ids": unit.get("unresolved_citation_ids", []),
        "graph_paths": paths,
        "applicable_fields": unit.get("applicable_fields", []),
    }
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _manifest(
    *,
    dataset_dirs: Mapping[str, Path],
    system_output_root: Path,
    target_units_by_family: Mapping[str, int],
    unit_rows: list[dict[str, object]],
    key_rows: list[dict[str, object]],
    assignment_plan: AnnotationAssignmentPlan,
    annotators: int,
    assignments_per_annotator: int,
    assignment_paths: list[Path],
    units_path: Path,
    key_path: Path,
    system_output_provenance: dict[str, object],
) -> dict[str, object]:
    family_counts = Counter(str(row["dataset_family"]) for row in key_rows)
    source_counts = Counter(str(row["source_kind"]) for row in key_rows)
    system_counts = Counter(
        str(row["system_name"]) for row in key_rows if row["source_kind"] == "system_output"
    )
    variant_counts = Counter(
        str(row["variant_type"]) for row in key_rows if row["source_kind"] == "controlled_variant"
    )
    artifact_paths = [units_path, key_path, *assignment_paths]
    hashes = artifact_hashes(root=units_path.parent, paths=artifact_paths)
    return {
        "dataset_dirs": {family: str(path) for family, path in sorted(dataset_dirs.items())},
        "system_output_root": str(system_output_root),
        "system_output_provenance": system_output_provenance,
        "target_units_requested": dict(target_units_by_family),
        "unique_units_exported": len(unit_rows),
        "unique_units_by_family": dict(sorted(family_counts.items())),
        "answer_source_counts": dict(sorted(source_counts.items())),
        "system_output_counts": dict(sorted(system_counts.items())),
        "controlled_variant_counts": dict(sorted(variant_counts.items())),
        "annotators": annotators,
        "assignments_per_annotator": assignments_per_annotator,
        "total_assignments": annotators * assignments_per_annotator,
        **assignment_manifest_metadata(assignment_plan, key_rows=key_rows),
        "annotation_fields": ANNOTATION_FIELDS,
        "annotation_protocol_version": PROTOCOL_VERSION,
        "annotation_text_repair_version": ANNOTATION_TEXT_REPAIR_VERSION,
        "annotator_files": [str(path) for path in assignment_paths],
        "artifact_sha256": hashes,
        "package_sha256": package_sha256(hashes),
        "blind_annotation_note": (
            "Public rows use opaque candidate identities and omit source kind, system identity, "
            "model identity, controlled labels, fact roles, symbolic programs, and computed "
            "graph-path verdicts. The trusted assignment file contains the reference answer, "
            "but the annotation API withholds it until evidence-stage judgments are locked."
        ),
    }


def _validate_system_output_release(
    *,
    dataset_dirs: Mapping[str, Path],
    system_output_root: Path,
) -> dict[str, object]:
    release_manifest_sha256: str | None = None
    release_roots = {path.resolve().parent for path in dataset_dirs.values()}
    if len(release_roots) == 1:
        release_manifest_path = next(iter(release_roots)) / "release_manifest.json"
        if release_manifest_path.exists():
            release_manifest = orjson.loads(release_manifest_path.read_bytes())
            if release_manifest.get("release_status") != "certified":
                raise ValueError(
                    "QA outputs cannot be promoted to human evaluation from a non-certified "
                    f"release: {release_manifest_path}"
                )
            release_manifest_sha256 = hashlib.sha256(release_manifest_path.read_bytes()).hexdigest()
    current_implementation = qa_implementation_sha256()
    configuration_hashes: set[str] = set()
    output_run_ids: set[str] = set()
    dataset_hashes: dict[str, str] = {}
    private_dataset_hashes: dict[str, str] = {}
    shard_count = 0
    for family, dataset_dir in sorted(dataset_dirs.items()):
        expected_qids = {
            question.qid
            for question in load_runtime_questions(
                dataset_dir,
                splits=["test_auto"],
                limit=None,
            )
        }
        dataset_hash = dataset_content_hash(dataset_dir)
        dataset_hashes[family] = dataset_hash
        dataset_manifest_path = dataset_dir / "dataset_manifest.json"
        dataset_manifest = orjson.loads(dataset_manifest_path.read_bytes())
        private_dataset_hashes[family] = str(dataset_manifest["private_payload_sha256"])
        for system_name in ALL_QA_SYSTEMS:
            output_path = system_output_root / family / f"{system_name}.jsonl"
            metadata = read_checkpoint_metadata(output_path)
            if metadata is None:
                raise ValueError(f"QA output checkpoint metadata is missing: {output_path}")
            rows = [SystemOutput.model_validate(row) for row in read_jsonl(output_path)]
            qids = {row.qid for row in rows}
            output_ids = {row.output_id for row in rows}
            valid = (
                metadata.dataset_family == family
                and metadata.system_name == system_name
                and metadata.dataset_sha256 == dataset_hash
                and metadata.implementation_sha256 == current_implementation
                and len(rows) == len(output_ids) == len(expected_qids)
                and qids == expected_qids
                and all(row.status == "success" for row in rows)
                and all(row.dataset_family == family for row in rows)
                and all(row.system_name == system_name for row in rows)
                and checkpoint_integrity_matches(
                    metadata,
                    output_path,
                    record_count=len(rows),
                )
            )
            if not valid:
                raise ValueError(
                    f"QA output shard failed provenance or integrity validation: {output_path}"
                )
            configuration_hashes.add(metadata.configuration_sha256)
            output_run_ids.update(row.run_id for row in rows)
            shard_count += 1
    if len(configuration_hashes) != 1:
        raise ValueError("QA output shards were produced with different run configurations")
    return {
        "release_manifest_sha256": release_manifest_sha256,
        "validated_shards": shard_count,
        "implementation_sha256": current_implementation,
        "configuration_sha256": next(iter(configuration_hashes)),
        "dataset_sha256": dataset_hashes,
        "private_dataset_sha256": private_dataset_hashes,
        "output_run_ids": sorted(output_run_ids),
    }


def _guard(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
