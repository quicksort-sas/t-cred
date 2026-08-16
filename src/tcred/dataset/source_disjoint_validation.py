from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import orjson

from tcred.dataset.extracted_source import (
    ExtractedSourceDatasetGenerator,
    ExtractedTemporalSource,
    load_extracted_sources,
    reconstruct_wikidata_temporal_sources_from_cache,
)
from tcred.dataset.models import DatasetBundle
from tcred.dataset.validate import validate_bundle
from tcred.dataset.writer import DatasetWriter
from tcred.metrics.diagnostic_builder import build_diagnostic_suite
from tcred.metrics.diagnostic_models import (
    DiagnosticPair,
    DiagnosticSuite,
    diagnostic_inference_cluster_ids,
)
from tcred.metrics.task_judge_models import JudgeGraphPath

PROTOCOL_ID = "tcred-v1.4-source-disjoint-validation-v1"
SOURCE_SPLIT = "source_disjoint_validation"
DEFAULT_OUTPUT_ROOT = Path("data/validation/tcred_v1_4_source_disjoint")

_FROZEN_FILE_BY_PROTOCOL_KEY = {
    "wikidata_source_catalog_sha256": Path(
        "data/external/wikidata/temporal_subgraphs.jsonl"
    ),
    "wikidata_source_manifest_sha256": Path(
        "data/external/wikidata/temporal_subgraphs.manifest.json"
    ),
    "wikidata_candidate_cache_sha256": Path(
        "data/cache/source_extraction/wikidata_temporal_candidates.json"
    ),
    "wikidata_context_cache_sha256": Path(
        "data/cache/source_extraction/wikidata_temporal_context_entities.json"
    ),
    "wikidata_answer_cache_sha256": Path(
        "data/cache/source_extraction/wikidata_temporal_answer_entities.json"
    ),
    "tcred_suite_sha256": Path("src/tcred/metrics/tcred_suite.py"),
    "tcred_claims_sha256": Path("src/tcred/metrics/tcred_claims.py"),
    "tcred_temporal_sha256": Path("src/tcred/metrics/tcred_temporal.py"),
    "tcred_semantic_worker_sha256": Path("src/tcred/metrics/tcred_semantic_worker.py"),
    "tcred_diagnostic_runner_sha256": Path(
        "src/tcred/metrics/tcred_diagnostic_runner.py"
    ),
    "ragchecker_prompt_sha256": Path("prompts/metrics/rag_claim_judge_v1.md"),
}


def prepare_source_disjoint_validation(
    *,
    repository_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    protocol_path: Path = Path(
        "docs/protocols/tcred-v1.4-source-disjoint-validation-v1.json"
    ),
    lock_path: Path = Path(
        "docs/protocols/tcred-v1.4-source-disjoint-validation-v1.lock.json"
    ),
) -> dict[str, Path]:
    """Build and preflight the preregistered validation set without reading scores."""

    repository_root = repository_root.resolve()
    output_root = _resolve(repository_root, output_root)
    protocol_path = _resolve(repository_root, protocol_path)
    lock_path = _resolve(repository_root, lock_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Validation output already exists: {output_root}. Preserve it or choose a new path."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = load_and_verify_locked_protocol(
        repository_root=repository_root,
        protocol_path=protocol_path,
        lock_path=lock_path,
    )
    source_contract = _mapping(protocol, "source_sampling")
    challenge_contract = _mapping(protocol, "challenge_generation")
    inference_contract = _mapping(protocol, "inference")

    source_manifest_path = repository_root / _FROZEN_FILE_BY_PROTOCOL_KEY[
        "wikidata_source_manifest_sha256"
    ]
    source_manifest = orjson.loads(source_manifest_path.read_bytes())
    extraction_time = datetime.fromisoformat(str(source_manifest["extraction_time"]))
    population = reconstruct_wikidata_temporal_sources_from_cache(
        candidate_cache_path=repository_root
        / _FROZEN_FILE_BY_PROTOCOL_KEY["wikidata_candidate_cache_sha256"],
        context_cache_path=repository_root
        / _FROZEN_FILE_BY_PROTOCOL_KEY["wikidata_context_cache_sha256"],
        answer_cache_path=repository_root
        / _FROZEN_FILE_BY_PROTOCOL_KEY["wikidata_answer_cache_sha256"],
        extraction_time=extraction_time,
    )
    _verify_population_counts(population, source_manifest=source_manifest)
    released = load_extracted_sources(
        repository_root / _FROZEN_FILE_BY_PROTOCOL_KEY["wikidata_source_catalog_sha256"]
    )
    selected, selection_audit = select_source_disjoint_sources(
        population,
        released=released,
        target=int(source_contract["target_source_series"]),
        salt=str(source_contract["sampling_salt"]),
    )

    source_path = output_root / "source_sample.jsonl"
    _write_jsonl(
        source_path,
        [source.model_dump(mode="json") for source in selected],
    )
    source_manifest_output = output_root / "source_sample_manifest.json"
    _write_json(
        source_manifest_output,
        {
            "schema_version": "1.0",
            "protocol_id": PROTOCOL_ID,
            "source_snapshot_time": extraction_time.isoformat(),
            "selection": selection_audit,
            "source_sample": _file_record(source_path, relative_to=output_root),
        },
    )

    dataset_root = output_root / "dataset"
    dataset_dir = dataset_root / "tcred_synth"
    generator = ExtractedSourceDatasetGenerator(
        sources=selected,
        seed=int(source_contract["generator_seed"]),
        scenario_prefix=str(challenge_contract["scenario_prefix"]),
    )
    bundle = generator.generate(
        scenario_count=len(selected),
        questions_per_scenario=int(challenge_contract["questions_per_source_series"]),
    )
    bundle = bundle.model_copy(
        update={"splits": {SOURCE_SPLIT: sorted(s.scenario_id for s in bundle.scenarios)}}
    )
    validation_warnings = validate_bundle(bundle)
    DatasetWriter(dataset_dir).write_bundle(bundle)

    raw_suite = build_diagnostic_suite(
        dataset_root,
        seed=int(challenge_contract["diagnostic_seed"]),
        pair_cap_per_phenomenon=int(challenge_contract["pair_cap_per_phenomenon"]),
        source_split=SOURCE_SPLIT,
    )
    suite = exclude_structurally_confounded_pairs(raw_suite)
    challenge_dir = output_root / "challenge"
    challenge_dir.mkdir(parents=True, exist_ok=True)
    challenge_paths = write_diagnostic_suite(suite, output_dir=challenge_dir)

    preflight = audit_source_disjoint_validation(
        selected=selected,
        released=released,
        bundle=bundle,
        suite=suite,
        selection_audit=selection_audit,
        validation_warnings=validation_warnings,
        required_constructs=[str(value) for value in challenge_contract["required_constructs"]],
        minimum_pairs=int(inference_contract["minimum_common_pairs"]),
        minimum_clusters=int(inference_contract["minimum_common_source_clusters"]),
    )
    preflight_path = output_root / "preflight_audit.json"
    _write_json(preflight_path, preflight)
    report_path = output_root / "preflight_report.md"
    report_path.write_text(
        render_preflight_report(preflight),
        encoding="utf-8",
        newline="\n",
    )
    if preflight["status"] != "pass":
        raise RuntimeError(
            f"Score-blind preflight failed; inspect {preflight_path}. No scoring is permitted."
        )

    implementation_lock_path = output_root / "implementation_lock.json"
    locked_paths = [
        protocol_path,
        lock_path,
        repository_root / "src/tcred/dataset/extracted_source.py",
        repository_root / "src/tcred/dataset/source_disjoint_validation.py",
        repository_root / "src/tcred/dataset/generator.py",
        repository_root / "src/tcred/dataset/public_oracle.py",
        repository_root / "src/tcred/dataset/solver.py",
        repository_root / "src/tcred/dataset/validate.py",
        repository_root / "src/tcred/metrics/diagnostic_builder.py",
        source_path,
        source_manifest_output,
        dataset_dir / "dataset_manifest.json",
        *challenge_paths.values(),
        preflight_path,
        report_path,
    ]
    _write_json(
        implementation_lock_path,
        {
            "schema_version": "1.0",
            "protocol_id": PROTOCOL_ID,
            "status": "score_blind_preflight_passed_and_locked",
            "metric_scores_read_during_construction": False,
            "files": [
                _absolute_file_record(path, repository_root=repository_root)
                for path in locked_paths
            ],
        },
    )
    return {
        "source_sample": source_path,
        "source_manifest": source_manifest_output,
        "dataset": dataset_dir,
        "challenge": challenge_dir,
        "preflight": preflight_path,
        "report": report_path,
        "implementation_lock": implementation_lock_path,
    }


def load_and_verify_locked_protocol(
    *,
    repository_root: Path,
    protocol_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    if not protocol_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("Validation protocol or lock file is missing")
    protocol = orjson.loads(protocol_path.read_bytes())
    lock = orjson.loads(lock_path.read_bytes())
    if protocol.get("protocol_id") != PROTOCOL_ID or lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Validation protocol ID does not match the implementation")
    expected_protocol_hash = next(
        (
            str(row["sha256"])
            for row in lock.get("files", [])
            if row.get("path") == protocol_path.relative_to(repository_root).as_posix()
        ),
        None,
    )
    if expected_protocol_hash is None or _sha256(protocol_path) != expected_protocol_hash:
        raise ValueError("Machine-readable protocol no longer matches its preregistration lock")
    frozen_inputs = _mapping(protocol, "frozen_inputs")
    for key, relative_path in _FROZEN_FILE_BY_PROTOCOL_KEY.items():
        expected = str(frozen_inputs.get(key) or "")
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Frozen protocol input is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen protocol input changed: {relative_path} "
                f"(expected {expected}, got {actual})"
            )
    return protocol


def select_source_disjoint_sources(
    population: list[ExtractedTemporalSource],
    *,
    released: list[ExtractedTemporalSource],
    target: int,
    salt: str,
) -> tuple[list[ExtractedTemporalSource], dict[str, object]]:
    if target < 1:
        raise ValueError("target must be positive")
    released_source_ids = {source.source_id for source in released}
    released_context_ids = {source.context_source_id for source in released}
    released_statement_ids = {
        claim.statement_id for source in released for claim in source.claims
    }
    eligible = [
        source
        for source in population
        if source.source_id not in released_source_ids
        and source.context_source_id not in released_context_ids
        and not (
            {claim.statement_id for claim in source.claims} & released_statement_ids
        )
    ]
    queues: defaultdict[str, list[ExtractedTemporalSource]] = defaultdict(list)
    for source in eligible:
        queues[source.property_id].append(source)
    ordered_queues = {
        property_id: sorted(
            sources,
            key=lambda source: _salted_source_hash(
                salt,
                property_id,
                source.source_id,
            ),
        )
        for property_id, sources in queues.items()
    }
    positions = dict.fromkeys(ordered_queues, 0)
    selected: list[ExtractedTemporalSource] = []
    while len(selected) < target:
        progressed = False
        for property_id in sorted(ordered_queues):
            if len(selected) >= target:
                break
            position = positions[property_id]
            queue = ordered_queues[property_id]
            if position >= len(queue):
                continue
            selected.append(queue[position])
            positions[property_id] += 1
            progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise ValueError(
            f"Only {len(selected)} source-disjoint series are available; protocol requires {target}"
        )
    selected_ids = [source.source_id for source in selected]
    selected_contexts = [source.context_source_id for source in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Source selection contains duplicate source IDs")
    if len(selected_contexts) != len(set(selected_contexts)):
        raise ValueError("Source selection contains duplicate context entities")
    return selected, {
        "population_size": len(population),
        "released_size": len(released),
        "eligible_size_after_all_exclusions": len(eligible),
        "selected_size": len(selected),
        "salt": salt,
        "method": "salted_property_queue_round_robin",
        "population_by_property": _property_counts(population),
        "eligible_by_property": _property_counts(eligible),
        "selected_by_property": _property_counts(selected),
        "excluded_released_source_ids": len(
            [source for source in population if source.source_id in released_source_ids]
        ),
        "excluded_released_context_ids": len(
            [
                source
                for source in population
                if source.source_id not in released_source_ids
                and source.context_source_id in released_context_ids
            ]
        ),
        "excluded_released_statement_overlap": len(
            [
                source
                for source in population
                if source.source_id not in released_source_ids
                and source.context_source_id not in released_context_ids
                and ({claim.statement_id for claim in source.claims} & released_statement_ids)
            ]
        ),
    }


def write_diagnostic_suite(
    suite: DiagnosticSuite,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "cases": output_dir / "diagnostic_cases.jsonl",
        "pairs": output_dir / "diagnostic_pairs.jsonl",
        "metric_inputs": output_dir / "metric_inputs.jsonl",
        "task_inputs": output_dir / "task_judge_inputs.jsonl",
    }
    _write_jsonl(paths["cases"], [row.model_dump(mode="json") for row in suite.cases])
    _write_jsonl(paths["pairs"], [row.model_dump(mode="json") for row in suite.pairs])
    _write_jsonl(
        paths["metric_inputs"],
        [row.metric_input.model_dump(mode="json") for row in suite.cases],
    )
    _write_jsonl(
        paths["task_inputs"],
        [row.task_judge_input.model_dump(mode="json") for row in suite.cases],
    )
    return paths


def audit_source_disjoint_validation(
    *,
    selected: list[ExtractedTemporalSource],
    released: list[ExtractedTemporalSource],
    bundle: DatasetBundle,
    suite: DiagnosticSuite,
    selection_audit: dict[str, object],
    validation_warnings: list[str],
    required_constructs: list[str],
    minimum_pairs: int,
    minimum_clusters: int,
) -> dict[str, object]:
    failures: list[str] = []
    released_sources = {source.source_id for source in released}
    released_contexts = {source.context_source_id for source in released}
    released_statements = {
        claim.statement_id for source in released for claim in source.claims
    }
    selected_sources = {source.source_id for source in selected}
    selected_contexts = {source.context_source_id for source in selected}
    selected_statements = {
        claim.statement_id for source in selected for claim in source.claims
    }
    overlaps = {
        "source_ids": sorted(selected_sources & released_sources),
        "context_ids": sorted(selected_contexts & released_contexts),
        "statement_ids": sorted(selected_statements & released_statements),
    }
    if any(overlaps.values()):
        failures.append("released/new source disjointness failed")

    scenario_sources = {
        str(scenario.split_group_id) for scenario in bundle.scenarios if scenario.split_group_id
    }
    if scenario_sources != selected_sources:
        failures.append("generated scenarios do not map one-to-one to the selected source set")
    scenario_ids = {scenario.scenario_id for scenario in bundle.scenarios}
    if set(bundle.splits.get(SOURCE_SPLIT, [])) != scenario_ids:
        failures.append("dedicated validation split does not contain every generated scenario")

    statement_map = {
        claim.statement_id: (source, claim)
        for source in selected
        for claim in source.claims
    }
    fidelity_errors: list[str] = []
    for fact in bundle.facts:
        record_id = str(fact.source_record_id or "")
        source_claim = statement_map.get(record_id)
        if source_claim is None:
            fidelity_errors.append(f"{fact.fact_id}: unknown source statement {record_id}")
            continue
        source, claim = source_claim
        if fact.source_revision != source.source_revision:
            fidelity_errors.append(f"{fact.fact_id}: source revision changed")
        if fact.source_relation_id != source.property_id:
            fidelity_errors.append(f"{fact.fact_id}: source relation changed")
        if fact.relation_direction != "directed":
            fidelity_errors.append(f"{fact.fact_id}: direction is not preserved as directed")
        if fact.valid_time.model_dump(mode="json") != claim.interval().model_dump(mode="json"):
            fidelity_errors.append(f"{fact.fact_id}: valid-time qualifiers changed")
    if fidelity_errors:
        failures.append("source-to-fact fidelity checks failed")

    pair_declaration_errors = _pair_declaration_errors(suite)
    if pair_declaration_errors:
        failures.append("diagnostic pairs contain undeclared top-level input changes")

    present_constructs = {str(pair.target_construct) for pair in suite.pairs}
    missing_constructs = sorted(set(required_constructs) - present_constructs)
    if missing_constructs:
        failures.append("one or more required constructs are absent")

    structural_power = _structural_power_audit(
        suite.pairs,
        minimum_pairs=minimum_pairs,
        minimum_clusters=minimum_clusters,
    )
    underpowered = [
        key for key, value in structural_power.items() if value["status"] != "pass"
    ]
    if underpowered:
        failures.append("one or more construct/test dimensions fail preregistered size floors")

    answer_counts = Counter(
        claim.answer_source_id for source in selected for claim in source.claims
    )
    spans = [
        (claim.end or claim.start).year - claim.start.year
        for source in selected
        for claim in source.claims
    ]
    return {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "status": "pass" if not failures else "fail",
        "score_blind": True,
        "failures": failures,
        "source_selection": selection_audit,
        "disjointness": {
            "released_new_overlap_counts": {
                key: len(value) for key, value in overlaps.items()
            },
            "overlap_examples": {key: value[:10] for key, value in overlaps.items()},
            "within_sample_unique_contexts": len(selected_contexts),
            "selected_sources": len(selected_sources),
        },
        "source_fidelity": {
            "facts_checked": len(bundle.facts),
            "error_count": len(fidelity_errors),
            "error_examples": fidelity_errors[:25],
            "all_relations_directional": all(
                fact.relation_direction == "directed" for fact in bundle.facts
            ),
        },
        "dataset": {
            "scenarios": len(bundle.scenarios),
            "questions": len(bundle.questions),
            "facts": len(bundle.facts),
            "answer_variants": len(bundle.answer_variants),
            "graph_paths": len(bundle.graph_paths),
            "validation_warnings": validation_warnings,
        },
        "diagnostic_suite": {
            "cases": len(suite.cases),
            "pairs": len(suite.pairs),
            "constructs_present": sorted(present_constructs),
            "constructs_missing": missing_constructs,
            "pairs_by_construct": dict(
                sorted(Counter(str(pair.target_construct) for pair in suite.pairs).items())
            ),
            "pairs_by_phenomenon": dict(
                sorted(Counter(pair.phenomenon for pair in suite.pairs).items())
            ),
            "declaration_error_count": len(pair_declaration_errors),
            "declaration_error_examples": pair_declaration_errors[:25],
            "score_blind_pair_filter": suite.audit.get("score_blind_pair_filter", {}),
            "structural_power": structural_power,
        },
        "descriptive_source_properties": {
            "selected_by_property": _property_counts(selected),
            "distinct_answer_entities": len(answer_counts),
            "answer_entity_repeat_instances": sum(count - 1 for count in answer_counts.values()),
            "answer_entities_in_multiple_claims": sum(
                count > 1 for count in answer_counts.values()
            ),
            "claim_span_years": {
                "n": len(spans),
                "mean": mean(spans) if spans else None,
                "median": median(spans) if spans else None,
                "minimum": min(spans) if spans else None,
                "maximum": max(spans) if spans else None,
            },
        },
        "interpretation_boundary": (
            "Passing this audit licenses scoring of a source-disjoint formal replication. It "
            "does not turn formal oracle labels into human judgments or establish open-world "
            "external validity."
        ),
    }


def verify_implementation_lock(*, repository_root: Path, lock_path: Path) -> dict[str, Any]:
    lock = orjson.loads(lock_path.read_bytes())
    if lock.get("status") != "score_blind_preflight_passed_and_locked":
        raise ValueError("Validation implementation lock is not complete")
    for record in lock.get("files", []):
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"Score-blind implementation artifact changed after lock: {path}")
    return lock


def render_preflight_report(audit: dict[str, object]) -> str:
    selection = audit["source_selection"]
    dataset = audit["dataset"]
    diagnostics = audit["diagnostic_suite"]
    lines = [
        "# T-CRED v1.4 Source-Disjoint Validation Preflight",
        "",
        f"- Status: **{str(audit['status']).upper()}**",
        f"- Protocol: `{audit['protocol_id']}`",
        "- Construction boundary: metric scores were not read.",
        "",
        "## Source sample",
        "",
        f"- Reconstructed usable population: **{selection['population_size']}** series",
        "- Eligible after all release exclusions: "
        f"**{selection['eligible_size_after_all_exclusions']}**",
        f"- Deterministically selected: **{selection['selected_size']}**",
        f"- Property allocation: `{selection['selected_by_property']}`",
        "- Released/new source, context, and statement overlaps: **0 / 0 / 0**",
        "",
        "## Generated data",
        "",
        f"- Scenarios: **{dataset['scenarios']}**",
        f"- Questions: **{dataset['questions']}**",
        f"- Facts: **{dataset['facts']}**",
        f"- Answer variants: **{dataset['answer_variants']}**",
        f"- Graph paths: **{dataset['graph_paths']}**",
        "",
        "## Formal challenge",
        "",
        f"- Cases: **{diagnostics['cases']}**",
        f"- Pairs: **{diagnostics['pairs']}**",
        f"- Pairs by construct: `{diagnostics['pairs_by_construct']}`",
        "",
        "| Construct / test type | Potential pairs | Source clusters | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, value in diagnostics["structural_power"].items():
        lines.append(
            f"| {key} | {value['pairs']} | {value['source_clusters']} | {value['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(audit["interpretation_boundary"]),
            "",
        ]
    )
    if audit["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in audit["failures"])
        lines.append("")
    return "\n".join(lines)


def _structural_power_audit(
    pairs: list[DiagnosticPair],
    *,
    minimum_pairs: int,
    minimum_clusters: int,
) -> dict[str, dict[str, object]]:
    grouped: defaultdict[tuple[str, str], list[DiagnosticPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(str(pair.target_construct), pair.test_type)].append(pair)
    output: dict[str, dict[str, object]] = {}
    for (construct, test_type), rows in sorted(grouped.items()):
        clusters = {f"{row.dataset_family}:{row.scenario_id}" for row in rows}
        output[f"{construct}/{test_type}"] = {
            "pairs": len(rows),
            "source_clusters": len(clusters),
            "minimum_pairs": minimum_pairs,
            "minimum_source_clusters": minimum_clusters,
            "status": (
                "pass"
                if len(rows) >= minimum_pairs and len(clusters) >= minimum_clusters
                else "fail"
            ),
        }
    return output


def _pair_declaration_errors(suite: DiagnosticSuite) -> list[str]:
    by_id = {case.case_id: case.task_judge_input for case in suite.cases}
    allowed = {
        "question": {"question_snapshot"},
        "reference_answer": {"reference_answer", "question_snapshot"},
        "candidate_answer": {
            "candidate_answer",
            "response_decision",
            "question_snapshot",
        },
        "retrieved_evidence": {
            "available_evidence",
            "retrieved_evidence",
            "evidence_order",
            "evidence_time_metadata",
            "context_note",
            "question_snapshot",
        },
        "cited_evidence": {
            "available_evidence",
            "cited_evidence",
            "evidence_order",
            "evidence_time_metadata",
            "context_note",
            "question_snapshot",
            # A bare refusal has no claims to cite. Citation omission is a
            # deterministic projection of the declared response decision, not
            # a second evidence intervention.
            "response_decision",
        },
        "graph_paths": {
            "available_evidence",
            "graph_paths",
            "path_order",
            "evidence_time_metadata",
            "question_snapshot",
        },
    }
    errors: list[str] = []
    for pair in suite.pairs:
        left = by_id[pair.left_case_id]
        right = by_id[pair.right_case_id]
        declared = set(pair.changed_components)
        for field, declarations in allowed.items():
            left_value = _semantic_field_value(field, getattr(left, field))
            right_value = _semantic_field_value(field, getattr(right, field))
            if left_value != right_value and not (declared & declarations):
                errors.append(f"{pair.pair_id}: {field} changed but was not declared")
    return errors


def exclude_structurally_confounded_pairs(suite: DiagnosticSuite) -> DiagnosticSuite:
    """Remove score-blind pairs whose payload changes exceed their declaration."""

    by_id = {case.case_id: case for case in suite.cases}
    kept: list[DiagnosticPair] = []
    excluded: list[tuple[DiagnosticPair, list[str]]] = []
    for pair in suite.pairs:
        pair_suite = suite.model_copy(
            update={
                "cases": [by_id[pair.left_case_id], by_id[pair.right_case_id]],
                "pairs": [pair],
            }
        )
        errors = _pair_declaration_errors(pair_suite)
        if errors:
            excluded.append((pair, errors))
        else:
            kept.append(pair)
    retained_case_ids = {
        case_id
        for pair in kept
        for case_id in (pair.left_case_id, pair.right_case_id)
    }
    cases = [case for case in suite.cases if case.case_id in retained_case_ids]
    clusters = diagnostic_inference_cluster_ids(cases, kept)
    audit = dict(suite.audit)
    audit.update(
        {
            "case_count": len(cases),
            "pair_count": len(kept),
            "question_clusters": len(
                {(case.metric_input.dataset_family, case.metric_input.qid) for case in cases}
            ),
            "source_scenarios": len(
                {
                    (case.metric_input.dataset_family, case.metric_input.scenario_id)
                    for case in cases
                }
            ),
            "inference_clusters": len(set(clusters.values())),
            "pair_counts_by_test_type": dict(
                sorted(Counter(pair.test_type for pair in kept).items())
            ),
            "pair_counts_by_construct": dict(
                sorted(Counter(str(pair.target_construct) for pair in kept).items())
            ),
            "pair_counts_by_phenomenon": dict(
                sorted(Counter(pair.phenomenon for pair in kept).items())
            ),
            "pair_counts_by_dataset": dict(
                sorted(Counter(pair.dataset_family for pair in kept).items())
            ),
            "selection_policy": (
                f"{audit['selection_policy']} Pairs with undeclared semantic payload changes "
                "were removed before any metric score was read."
            ),
            "score_blind_pair_filter": {
                "raw_pairs": len(suite.pairs),
                "retained_pairs": len(kept),
                "excluded_pairs": len(excluded),
                "excluded_by_phenomenon": dict(
                    sorted(Counter(pair.phenomenon for pair, _errors in excluded).items())
                ),
                "excluded_by_reason": dict(
                    sorted(
                        Counter(
                            error.split(": ", 1)[1]
                            for _pair, errors in excluded
                            for error in errors
                        ).items()
                    )
                ),
                "metric_scores_read": False,
                "rule": (
                    "Retain only pairs whose semantic top-level input differences are covered "
                    "by changed_components. Local evidence/path identifiers are ignored when "
                    "their semantic payload is unchanged."
                ),
            },
        }
    )
    return suite.model_copy(update={"cases": cases, "pairs": kept, "audit": audit})


def _semantic_field_value(field: str, value: object) -> object:
    if field != "graph_paths":
        return value
    return tuple(_semantic_path_value(path) for path in value)  # type: ignore[union-attr]


def _semantic_path_value(path: JudgeGraphPath) -> tuple[object, ...]:
    edges = path.edges
    return tuple(
        (
            str(edge.relation),
            edge.relation_label,
            edge.traversal_direction,
            edge.directional,
            edge.symmetric,
            edge.evidence_text,
            edge.valid_time.model_dump(mode="json") if edge.valid_time else None,
            (edge.source.label, str(edge.source.type)),
            (edge.target.label, str(edge.target.type)),
        )
        for edge in edges
    )


def _verify_population_counts(
    population: list[ExtractedTemporalSource],
    *,
    source_manifest: dict[str, object],
) -> None:
    actual = _property_counts(population)
    expected = {
        str(key): int(value)
        for key, value in _mapping(source_manifest, "usable_source_counts").items()
    }
    if actual != expected:
        raise ValueError(
            "Offline Wikidata reconstruction drifted from the frozen manifest: "
            f"{actual} != {expected}"
        )


def _property_counts(sources: list[ExtractedTemporalSource]) -> dict[str, int]:
    return dict(sorted(Counter(source.property_id for source in sources).items()))


def _salted_source_hash(salt: str, property_id: str, source_id: str) -> str:
    return hashlib.sha256(f"{salt}|{property_id}|{source_id}".encode()).hexdigest()


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Protocol field {key!r} must be an object")
    return result


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    temporary.replace(path)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _absolute_file_record(path: Path, *, repository_root: Path) -> dict[str, object]:
    try:
        display = path.relative_to(repository_root).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {
        "path": display,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
