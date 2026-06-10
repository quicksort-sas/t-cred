from __future__ import annotations

from collections import Counter
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict

from tcred.dataset.io import load_bundle
from tcred.dataset.models import ContextPackType, DatasetFamily, TemporalOperator
from tcred.dataset.validate import validate_bundle


class DatasetAuditReport(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    scenario_count: int
    question_count: int
    fact_count: int
    answer_variant_count: int
    graph_path_count: int
    context_pack_count: int
    domain_counts: dict[str, int]
    temporal_operator_counts: dict[str, int]
    system_difficulty_counts: dict[str, int]
    eval_difficulty_counts: dict[str, int]
    answer_variant_counts: dict[str, int]
    context_pack_counts: dict[str, int]
    split_counts: dict[str, int]
    source_fidelity_counts: dict[str, int]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.warnings


def audit_dataset_dir(dataset_dir: Path) -> DatasetAuditReport:
    bundle = load_bundle(dataset_dir)
    warnings = validate_bundle(bundle)
    warnings.extend(_coverage_warnings(bundle))
    return DatasetAuditReport(
        scenario_count=len(bundle.scenarios),
        question_count=len(bundle.questions),
        fact_count=len(bundle.facts),
        answer_variant_count=len(bundle.answer_variants),
        graph_path_count=len(bundle.graph_paths),
        context_pack_count=len(bundle.context_packs),
        domain_counts=dict(Counter(scenario.domain for scenario in bundle.scenarios)),
        temporal_operator_counts=dict(Counter(q.temporal_operator for q in bundle.questions)),
        system_difficulty_counts=dict(Counter(q.system_difficulty for q in bundle.questions)),
        eval_difficulty_counts=dict(Counter(q.eval_difficulty for q in bundle.questions)),
        answer_variant_counts=dict(Counter(a.variant_type for a in bundle.answer_variants)),
        context_pack_counts=dict(Counter(pack.pack_type for pack in bundle.context_packs)),
        split_counts={name: len(ids) for name, ids in bundle.splits.items()},
        source_fidelity_counts=dict(
            Counter(
                scenario.source_provenance.fidelity if scenario.source_provenance else "untracked"
                for scenario in bundle.scenarios
            )
        ),
        warnings=warnings,
    )


def write_audit_report(dataset_dir: Path, output_path: Path | None = None) -> Path:
    report = audit_dataset_dir(dataset_dir)
    path = output_path or dataset_dir / "dataset_audit.json"
    path.write_bytes(orjson.dumps(report.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
    return path


def _coverage_warnings(bundle) -> list[str]:  # noqa: ANN001 - internal helper over pydantic bundle
    warnings: list[str] = []
    operators = {question.temporal_operator for question in bundle.questions}
    expected_operators = {
        TemporalOperator.CURRENT,
        TemporalOperator.AS_OF,
        TemporalOperator.BEFORE,
        TemporalOperator.AFTER,
        TemporalOperator.DURING,
        TemporalOperator.PREVIOUS,
        TemporalOperator.NEXT,
        TemporalOperator.LATEST,
        TemporalOperator.FIRST,
        TemporalOperator.LAST,
        TemporalOperator.BETWEEN,
        TemporalOperator.EXPIRED,
        TemporalOperator.EFFECTIVE,
    }
    missing_operators = expected_operators - operators
    if missing_operators:
        warnings.append(
            "Missing temporal operators: "
            + ", ".join(sorted(str(operator) for operator in missing_operators))
        )

    pack_types = {pack.pack_type for pack in bundle.context_packs}
    source_extracted_synth = any(
        question.dataset_family == DatasetFamily.SYNTH for question in bundle.questions
    ) and all(
        scenario.source_provenance is not None
        and scenario.source_provenance.fidelity == "source_extracted"
        for scenario in bundle.scenarios
    )
    expected_pack_types = {
        ContextPackType.VALID_ONLY,
        ContextPackType.STALE_ONLY,
        ContextPackType.FUTURE_ONLY,
        ContextPackType.VALID_PLUS_STALE,
        ContextPackType.VALID_PLUS_FUTURE,
        ContextPackType.INSUFFICIENT,
    }
    if not source_extracted_synth:
        expected_pack_types.update(
            {
                ContextPackType.UNKNOWN_TIME,
                ContextPackType.GRAPH_INCOHERENT,
            }
        )
    missing_pack_types = expected_pack_types - pack_types
    if missing_pack_types:
        warnings.append(
            "Missing context pack types: "
            + ", ".join(sorted(str(pack_type) for pack_type in missing_pack_types))
        )

    if not any(question.should_abstain for question in bundle.questions):
        warnings.append("No abstention-required questions found")
    if not any(path.supports_gold_answer is False for path in bundle.graph_paths):
        warnings.append("No invalid graph paths found")
    if any(question.dataset_family == DatasetFamily.SYNTH for question in bundle.questions):
        fidelity_counts = Counter(
            scenario.source_provenance.fidelity if scenario.source_provenance else "untracked"
            for scenario in bundle.scenarios
        )
        non_extracted = sum(
            count for fidelity, count in fidelity_counts.items() if fidelity != "source_extracted"
        )
        if non_extracted:
            warnings.append(
                "Synthetic source fidelity is not source_extracted for "
                f"{non_extracted} scenarios: {dict(fidelity_counts)}"
            )
        entity_by_id = {entity.entity_id: entity for entity in bundle.entities}
        contexts: dict[str, set[str]] = {}
        for question in bundle.questions:
            context_id = question.program.context_id
            if context_id and context_id in entity_by_id:
                name = entity_by_id[context_id].name.casefold()
                contexts.setdefault(name, set()).add(context_id)
        collisions = {name: ids for name, ids in contexts.items() if len(ids) > 1}
        if collisions:
            warnings.append(
                "Synthetic question contexts contain "
                f"{len(collisions)} cross-scenario name collisions"
            )
    return warnings
