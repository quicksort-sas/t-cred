from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from tcred.dataset.extracted_source import (
    ExtractedSourceDatasetGenerator,
    ExtractedTemporalSource,
)
from tcred.dataset.intervals import human_interval
from tcred.dataset.models import DatasetBundle, Entity, Fact, GraphPath, Question, Scenario
from tcred.dataset.solver import fact_visible
from tcred.trainable_metrics.schema import (
    CurriculumStage,
    EvidencePassage,
    GraphPathText,
    SemanticRecord,
    SemanticTarget,
    SemanticTask,
    stable_unit_id,
)
from tcred.trainable_metrics.source_exclusions import (
    SourceExclusionLedger,
    source_disjointness_violations,
)

FORMAL_SOURCE_VERSION = "tcred-project-formal-v1"
FORMAL_ORACLE_VERSION = "tcred-public-oracle-v1"
FORMAL_LICENSE = "CC0-1.0-Wikidata-plus-project-code@tcred-project-formal-v1"
FORMAL_SCENARIO_ROWS = 30
FORMAL_CATEGORY_ROWS_PER_FOUR_SCENARIOS = {
    "temporal": 24,
    "graph": 24,
    "citation": 18,
    "answerability": 18,
    "update": 24,
    "answer": 12,
}


@dataclass(frozen=True)
class FormalExample:
    record: SemanticRecord
    trace: dict[str, Any]


@dataclass(frozen=True)
class _ScenarioContext:
    scenario: Scenario
    scenario_index: int
    questions: tuple[Question, ...]
    answerable: tuple[Question, ...]
    facts: tuple[Fact, ...]
    entities: dict[str, Entity]
    paths_by_question: dict[str, tuple[GraphPath, ...]]


def build_project_formal_dataset(
    *,
    sources: list[ExtractedTemporalSource],
    output_dir: Path,
    exclusion_ledger: SourceExclusionLedger,
    scenario_count: int = 4_000,
    seed: int = 20260816,
    overwrite: bool = False,
) -> dict[str, Any]:
    if scenario_count < 4 or scenario_count % 4:
        raise ValueError("scenario_count must be a positive multiple of four")
    violations = source_disjointness_violations(sources, ledger=exclusion_ledger)
    if violations:
        preview = ", ".join(row["source_id"] for row in violations[:5])
        raise ValueError(f"Fresh source catalog overlaps protected artifacts: {preview}")
    if len(sources) < scenario_count:
        raise ValueError(f"Need {scenario_count} disjoint source series, found {len(sources)}")
    owned_artifacts = (
        "records.jsonl",
        "derivation_trace.jsonl",
        "source_catalog.jsonl",
        "manifest.json",
    )
    existing = [name for name in owned_artifacts if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Project-formal output artifacts already exist in {output_dir}: {existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = ExtractedSourceDatasetGenerator(
        sources=sources,
        seed=seed,
        scenario_prefix="tr",
    ).generate(scenario_count=scenario_count, questions_per_scenario=4)

    records_path = output_dir / "records.jsonl"
    traces_path = output_dir / "derivation_trace.jsonl"
    source_path = output_dir / "source_catalog.jsonl"
    record_tmp = records_path.with_suffix(".jsonl.tmp")
    trace_tmp = traces_path.with_suffix(".jsonl.tmp")
    source_tmp = source_path.with_suffix(".jsonl.tmp")

    categories: Counter[str] = Counter()
    transformations: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    update_types: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    content_hashes: set[str] = set()
    record_hashes: set[str] = set()
    with record_tmp.open("wb") as record_handle, trace_tmp.open("wb") as trace_handle:
        for example in iter_project_formal_examples(bundle):
            record = example.record
            category = _primary_category(record.transformation_family)
            if record.content_hash in content_hashes:
                raise RuntimeError(f"Duplicate model-visible formal input: {record.unit_id}")
            if record.record_hash in record_hashes:
                raise RuntimeError(f"Duplicate formal record: {record.unit_id}")
            content_hashes.add(record.content_hash)
            record_hashes.add(record.record_hash)
            categories[category] += 1
            transformations[str(record.transformation_family)] += 1
            tasks[str(record.task)] += 1
            groups[record.source_group_id] += 1
            if category == "update":
                update_types[str(record.transformation_family).split(":", 1)[0]] += 1
            record_handle.write(
                orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE)
            )
            trace_handle.write(orjson.dumps(example.trace, option=orjson.OPT_APPEND_NEWLINE))
    record_tmp.replace(records_path)
    trace_tmp.replace(traces_path)

    selected_sources = {
        scenario.source_provenance.source_id
        for scenario in bundle.scenarios
        if scenario.source_provenance is not None
    }
    with source_tmp.open("wb") as source_handle:
        for source in sorted(sources, key=lambda row: row.source_id):
            if source.source_id in selected_sources:
                source_handle.write(
                    orjson.dumps(
                        source.model_dump(mode="json"),
                        option=orjson.OPT_APPEND_NEWLINE,
                    )
                )
    source_tmp.replace(source_path)

    _validate_formal_counts(
        scenario_count=scenario_count,
        categories=categories,
        update_types=update_types,
        groups=groups,
    )
    manifest = {
        "schema_version": "tcred-project-formal-manifest-v1",
        "source_version": FORMAL_SOURCE_VERSION,
        "oracle_version": FORMAL_ORACLE_VERSION,
        "seed": seed,
        "scenario_count": scenario_count,
        "record_count": sum(categories.values()),
        "category_counts": dict(sorted(categories.items())),
        "task_counts": dict(sorted(tasks.items())),
        "transformation_counts": dict(sorted(transformations.items())),
        "update_type_counts": dict(sorted(update_types.items())),
        "group_size_counts": {
            str(size): count for size, count in sorted(Counter(groups.values()).items())
        },
        "max_rows_per_group": max(groups.values()),
        "source_disjointness": {
            "protected_entity_ids": len(exclusion_ledger.entity_ids),
            "protected_source_ids": len(exclusion_ledger.source_ids),
            "violations": 0,
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (records_path, traces_path, source_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def iter_project_formal_examples(bundle: DatasetBundle) -> Iterator[FormalExample]:
    contexts = _scenario_contexts(bundle)
    for context in contexts:
        citation_count = 5 if context.scenario_index % 2 == 0 else 4
        answerability_count = 9 - citation_count
        examples = [
            *_temporal_examples(context),
            *_graph_examples(context),
            *_citation_examples(context, count=citation_count),
            *_answerability_examples(context, count=answerability_count),
            *_update_examples(context),
            *_answer_examples(context),
        ]
        expected = 36 if context.scenario.update_behavior == "answer_should_change" else 28
        if len(examples) != expected:
            raise RuntimeError(
                f"Scenario {context.scenario.scenario_id} emitted {len(examples)} rows; "
                f"expected {expected}"
            )
        yield from examples


def _scenario_contexts(bundle: DatasetBundle) -> list[_ScenarioContext]:
    questions_by_scenario: defaultdict[str, list[Question]] = defaultdict(list)
    facts_by_scenario: defaultdict[str, list[Fact]] = defaultdict(list)
    paths_by_question: defaultdict[str, list[GraphPath]] = defaultdict(list)
    for question in bundle.questions:
        questions_by_scenario[question.scenario_id].append(question)
    for fact in bundle.facts:
        facts_by_scenario[fact.scenario_id].append(fact)
    for path in bundle.graph_paths:
        paths_by_question[path.qid].append(path)
    entity_by_id = {entity.entity_id: entity for entity in bundle.entities}

    contexts: list[_ScenarioContext] = []
    for index, scenario in enumerate(sorted(bundle.scenarios, key=lambda row: row.scenario_id)):
        questions = tuple(
            sorted(questions_by_scenario[scenario.scenario_id], key=lambda row: row.qid)
        )
        answerable = tuple(
            question
            for question in questions
            if not question.should_abstain and question.required_valid_evidence_ids
        )
        if len(questions) != 4 or len(answerable) < 3:
            raise RuntimeError(
                f"Scenario {scenario.scenario_id} does not have four questions and at least "
                "three evidence-bearing answers"
            )
        contexts.append(
            _ScenarioContext(
                scenario=scenario,
                scenario_index=index,
                questions=questions,
                answerable=answerable,
                facts=tuple(facts_by_scenario[scenario.scenario_id]),
                entities=entity_by_id,
                paths_by_question={
                    question.qid: tuple(paths_by_question[question.qid])
                    for question in questions
                },
            )
        )
    return contexts


def _temporal_examples(context: _ScenarioContext) -> list[FormalExample]:
    rows: list[FormalExample] = []
    for index, question in enumerate(context.answerable[:3]):
        gold_facts = _gold_facts(context, question)
        wrong_fact = _wrong_fact(context, question)
        pair_id = f"{context.scenario.scenario_id}:temporal:{index}"
        rows.append(
            _example(
                context,
                question=question,
                suffix=f"temporal:{index}:positive",
                task=SemanticTask.TEMPORAL,
                candidate=_answer_claim(question),
                evidence_facts=gold_facts,
                target=SemanticTarget(
                    class_distribution={"support": 1.0},
                    supported=1.0,
                    pair_id=pair_id,
                    pair_role="positive",
                ),
                transformation="temporal:valid_operator_interval",
            )
        )
        rows.append(
            _example(
                context,
                question=question,
                suffix=f"temporal:{index}:negative",
                task=SemanticTask.TEMPORAL,
                candidate=_fact_answer_claim(context, wrong_fact),
                evidence_facts=[wrong_fact],
                target=SemanticTarget(
                    supported=0.0,
                    pair_id=pair_id,
                    pair_role="negative",
                ),
                transformation="temporal:wrong_time_or_context",
            )
        )
    return rows


def _graph_examples(context: _ScenarioContext) -> list[FormalExample]:
    rows: list[FormalExample] = []
    for index, question in enumerate(context.answerable[:3]):
        path = _positive_path(context, question)
        facts = [_fact_by_id(context, edge.fact_id) for edge in path.edges]
        pair_id = f"{context.scenario.scenario_id}:graph:{index}"
        rows.append(
            _example(
                context,
                question=question,
                suffix=f"graph:{index}:forward",
                task=SemanticTask.SUPPORT,
                candidate=_answer_claim(question),
                evidence_facts=facts,
                graph_paths=[_graph_path_text(context, path, reverse=False)],
                target=SemanticTarget(
                    class_distribution={"entailment": 1.0},
                    supported=1.0,
                    pair_id=pair_id,
                    pair_role="positive",
                ),
                transformation="graph:directed_path_forward",
            )
        )
        rows.append(
            _example(
                context,
                question=question,
                suffix=f"graph:{index}:reversed",
                task=SemanticTask.SUPPORT,
                candidate=_answer_claim(question),
                evidence_facts=facts,
                graph_paths=[_graph_path_text(context, path, reverse=True)],
                target=SemanticTarget(
                    supported=0.0,
                    pair_id=pair_id,
                    pair_role="negative",
                ),
                transformation="graph:directed_path_reversal",
            )
        )
    return rows


def _citation_examples(context: _ScenarioContext, *, count: int) -> list[FormalExample]:
    q0, q1, q2 = context.answerable[:3]
    candidates = [
        _citation_example(context, q0, "0:appropriate", "appropriate"),
        _citation_example(context, q0, "0:incomplete", "incomplete"),
        _citation_example(context, q1, "1:inappropriate", "inappropriate"),
        _citation_example(context, q1, "1:appropriate", "appropriate"),
        _citation_example(context, q2, "2:incomplete", "incomplete"),
    ]
    return candidates[:count]


def _citation_example(
    context: _ScenarioContext,
    question: Question,
    suffix: str,
    label: str,
) -> FormalExample:
    facts = _gold_facts(context, question)
    if label == "inappropriate":
        facts = [_wrong_fact(context, question)]
    citation_indexes = list(range(len(facts))) if label != "incomplete" else []
    return _example(
        context,
        question=question,
        suffix=f"citation:{suffix}",
        task=SemanticTask.CITATION,
        candidate=_answer_claim(question),
        evidence_facts=facts,
        citation_indexes=citation_indexes,
        target=SemanticTarget(class_distribution={label: 1.0}),
        transformation=f"citation:{label}",
    )


def _answerability_examples(
    context: _ScenarioContext,
    *,
    count: int,
) -> list[FormalExample]:
    q0, q1, q2 = context.answerable[:3]
    candidates = [
        _answerability_example(context, q0, "0:complete", complete=True),
        _answerability_example(context, q0, "0:missing", complete=False, no_evidence=True),
        _answerability_example(context, q1, "1:complete", complete=True),
        _answerability_example(context, q1, "1:wrong", complete=False),
        _answerability_example(context, q2, "2:complete", complete=True),
    ]
    return candidates[:count]


def _answerability_example(
    context: _ScenarioContext,
    question: Question,
    suffix: str,
    *,
    complete: bool,
    no_evidence: bool = False,
) -> FormalExample:
    if complete:
        facts = _gold_facts(context, question)
    elif no_evidence:
        facts = []
    else:
        facts = [_wrong_fact(context, question)]
    return _example(
        context,
        question=question,
        suffix=f"answerability:{suffix}",
        task=SemanticTask.ANSWERABILITY,
        candidate=_answer_claim(question),
        evidence_facts=facts,
        target=SemanticTarget(answerable=1.0 if complete else 0.0),
        transformation=(
            "answerability:complete_evidence"
            if complete
            else "answerability:required_evidence_omitted"
        ),
    )


def _answer_examples(context: _ScenarioContext) -> list[FormalExample]:
    q0, q1 = context.answerable[:2]
    wrong_fact = _wrong_fact(context, q0)
    pair_id = f"{context.scenario.scenario_id}:answer:0"
    return [
        _example(
            context,
            question=q0,
            suffix="answer:0:correct",
            task=SemanticTask.ANSWER,
            candidate=_answer_claim(q0),
            evidence_facts=_gold_facts(context, q0),
            target=SemanticTarget(
                answer_u1=1.0,
                answer_u2=1.0,
                equivalence=1.0,
                pair_id=pair_id,
                pair_role="positive",
            ),
            transformation="answer:correct_entity_role",
        ),
        _example(
            context,
            question=q0,
            suffix="answer:0:wrong_role",
            task=SemanticTask.ANSWER,
            candidate=_fact_answer_claim(context, wrong_fact),
            evidence_facts=[wrong_fact],
            target=SemanticTarget(
                answer_u1=0.0,
                answer_u2=0.0,
                equivalence=0.0,
                pair_id=pair_id,
                pair_role="negative",
            ),
            transformation="answer:wrong_entity_or_role",
        ),
        _example(
            context,
            question=q1,
            suffix="answer:1:natural",
            task=SemanticTask.ANSWER,
            candidate=_natural_answer(q1),
            evidence_facts=_gold_facts(context, q1),
            target=SemanticTarget(answer_u1=1.0, answer_u2=1.0, equivalence=1.0),
            transformation="answer:meaning_preserving_rendering",
        ),
    ]


def _update_examples(context: _ScenarioContext) -> list[FormalExample]:
    if context.scenario.update_behavior == "answer_should_change":
        return _relevant_update_examples(context)
    return _irrelevant_update_examples(context)


def _relevant_update_examples(context: _ScenarioContext) -> list[FormalExample]:
    before, after = context.questions[:2]
    if not before.should_abstain or after.should_abstain:
        raise RuntimeError(
            f"Relevant update contract is broken for {context.scenario.scenario_id}"
        )
    gold_facts = _gold_facts(context, after)
    wrong_fact = _wrong_fact(context, after)
    answer = _answer_claim(after)
    refusal = "The available evidence is insufficient to determine an answer."
    pairs: list[FormalExample] = []

    def add_pair(
        name: str,
        task: SemanticTask,
        positive_target: SemanticTarget,
        negative_target: SemanticTarget,
        *,
        positive_candidate: str = answer,
        negative_candidate: str = answer,
        positive_citations: list[int] | None = None,
        negative_citations: list[int] | None = None,
    ) -> None:
        pair_id = f"{context.scenario.scenario_id}:update-relevant:{name}"
        pairs.extend(
            [
                _example(
                    context,
                    question=after,
                    suffix=f"update:relevant:{name}:after",
                    task=task,
                    candidate=positive_candidate,
                    evidence_facts=gold_facts,
                    evidence_prefix="After the relevant update",
                    citation_indexes=positive_citations,
                    target=positive_target.model_copy(
                        update={"pair_id": pair_id, "pair_role": "positive"}
                    ),
                    transformation=f"update_relevant:{name}",
                ),
                _example(
                    context,
                    question=after,
                    suffix=f"update:relevant:{name}:before",
                    task=task,
                    candidate=negative_candidate,
                    evidence_facts=[wrong_fact],
                    evidence_prefix="Before the relevant update",
                    citation_indexes=negative_citations,
                    target=negative_target.model_copy(
                        update={"pair_id": pair_id, "pair_role": "negative"}
                    ),
                    transformation=f"update_relevant:{name}",
                ),
            ]
        )

    add_pair(
        "answer",
        SemanticTask.ANSWER,
        SemanticTarget(answer_u1=1.0, answer_u2=1.0, equivalence=1.0),
        SemanticTarget(answer_u1=0.0, answer_u2=0.0, equivalence=0.0),
        negative_candidate=refusal,
    )
    add_pair(
        "temporal",
        SemanticTask.TEMPORAL,
        SemanticTarget(class_distribution={"support": 1.0}, supported=1.0),
        SemanticTarget(supported=0.0),
    )
    add_pair(
        "relevance",
        SemanticTask.RELEVANCE,
        SemanticTarget(relevance=1.0),
        SemanticTarget(relevance=0.0),
    )
    add_pair(
        "answerability",
        SemanticTask.ANSWERABILITY,
        SemanticTarget(answerable=1.0),
        SemanticTarget(answerable=0.0),
    )
    add_pair(
        "citation",
        SemanticTask.CITATION,
        SemanticTarget(class_distribution={"appropriate": 1.0}),
        SemanticTarget(class_distribution={"inappropriate": 1.0}),
        positive_citations=list(range(len(gold_facts))),
        negative_citations=[0],
    )
    add_pair(
        "support",
        SemanticTask.SUPPORT,
        SemanticTarget(class_distribution={"entailment": 1.0}, supported=1.0),
        SemanticTarget(supported=0.0),
    )
    return pairs


def _irrelevant_update_examples(context: _ScenarioContext) -> list[FormalExample]:
    question = context.answerable[0]
    gold_facts = _gold_facts(context, question)
    irrelevant = _wrong_fact(context, question, prefer_wrong_context=True)
    pair_id = f"{context.scenario.scenario_id}:update-irrelevant:answer"
    invariance_id = f"{context.scenario.scenario_id}:irrelevant-update"
    base_target = SemanticTarget(
        answer_u1=1.0,
        answer_u2=1.0,
        equivalence=1.0,
        pair_id=pair_id,
        pair_role="invariant_a",
        invariance_group_id=invariance_id,
    )
    return [
        _example(
            context,
            question=question,
            suffix="update:irrelevant:answer:before",
            task=SemanticTask.ANSWER,
            candidate=_answer_claim(question),
            evidence_facts=gold_facts,
            evidence_prefix="Before an unrelated update",
            target=base_target,
            transformation="update_irrelevant:answer_invariance",
        ),
        _example(
            context,
            question=question,
            suffix="update:irrelevant:answer:after",
            task=SemanticTask.ANSWER,
            candidate=_answer_claim(question),
            evidence_facts=[*gold_facts, irrelevant],
            evidence_prefix="After an unrelated update",
            target=base_target.model_copy(update={"pair_role": "invariant_b"}),
            transformation="update_irrelevant:answer_invariance",
        ),
        _example(
            context,
            question=question,
            suffix="update:irrelevant:relevance",
            task=SemanticTask.RELEVANCE,
            candidate=irrelevant.canonical_evidence,
            evidence_facts=[irrelevant],
            evidence_prefix="Newly added evidence",
            target=SemanticTarget(relevance=0.0),
            transformation="update_irrelevant:evidence_relevance",
        ),
        _example(
            context,
            question=question,
            suffix="update:irrelevant:citation",
            task=SemanticTask.CITATION,
            candidate=_answer_claim(question),
            evidence_facts=[*gold_facts, irrelevant],
            evidence_prefix="After an unrelated update",
            citation_indexes=list(range(len(gold_facts))),
            target=SemanticTarget(class_distribution={"appropriate": 1.0}),
            transformation="update_irrelevant:citation_stability",
        ),
    ]


def _example(
    context: _ScenarioContext,
    *,
    question: Question,
    suffix: str,
    task: SemanticTask,
    candidate: str,
    evidence_facts: Iterable[Fact],
    target: SemanticTarget,
    transformation: str,
    citation_indexes: list[int] | None = None,
    graph_paths: list[GraphPathText] | None = None,
    evidence_prefix: str | None = None,
) -> FormalExample:
    native_id = f"{context.scenario.scenario_id}:{suffix}"
    facts = list(evidence_facts)
    evidence = [
        EvidencePassage(
            evidence_id=f"{native_id}:evidence:{index}",
            text=(
                f"{evidence_prefix}: {fact.canonical_evidence}"
                if evidence_prefix
                else fact.canonical_evidence
            ),
            source_id=fact.source_record_id or fact.fact_id,
            rank=index + 1,
        )
        for index, fact in enumerate(facts)
    ]
    citations = [
        evidence[index].evidence_id
        for index in (citation_indexes or [])
        if 0 <= index < len(evidence)
    ]
    record = SemanticRecord.create(
        unit_id=stable_unit_id("project_formal_fresh", native_id, task),
        source_dataset="project_formal_fresh",
        source_version=FORMAL_SOURCE_VERSION,
        source_native_split="formal_generation",
        source_native_id=native_id,
        source_group_id=context.scenario.split_group_id or context.scenario.scenario_id,
        curriculum_stage=CurriculumStage.TASK_MATCHED,
        task=task,
        question=question.question,
        query_time_or_interval=human_interval(question.program.query_time),
        temporal_operator=str(question.temporal_operator),
        reference_answers=question.gold_answer_text,
        candidate_or_claim=candidate,
        evidence_passages=evidence,
        citations=citations,
        graph_paths=graph_paths or [],
        target=target,
        label_provenance=(
            f"deterministic {FORMAL_ORACLE_VERSION}; source-grounded Wikidata scenario; "
            f"transformation={transformation}"
        ),
        transformation_family=transformation,
        license_record=FORMAL_LICENSE,
    )
    trace = {
        "unit_id": record.unit_id,
        "record_hash": record.record_hash,
        "source_id": context.scenario.source_provenance.source_id
        if context.scenario.source_provenance
        else None,
        "source_revision": context.scenario.source_provenance.source_revision
        if context.scenario.source_provenance
        else None,
        "scenario_id": context.scenario.scenario_id,
        "question_id": question.qid,
        "oracle_version": FORMAL_ORACLE_VERSION,
        "transformation_family": transformation,
        "evidence_fact_ids": [fact.fact_id for fact in facts],
        "graph_path_ids": [path.path_id for path in graph_paths or []],
    }
    return FormalExample(record=record, trace=trace)


def _gold_facts(context: _ScenarioContext, question: Question) -> list[Fact]:
    return [_fact_by_id(context, fact_id) for fact_id in question.required_valid_evidence_ids]


def _fact_by_id(context: _ScenarioContext, fact_id: str) -> Fact:
    try:
        return next(fact for fact in context.facts if fact.fact_id == fact_id)
    except StopIteration as exc:
        raise RuntimeError(f"Unknown fact {fact_id} in {context.scenario.scenario_id}") from exc


def _wrong_fact(
    context: _ScenarioContext,
    question: Question,
    *,
    prefer_wrong_context: bool = False,
) -> Fact:
    gold_ids = set(question.gold_answer_entity_ids)
    gold_names = {value.casefold() for value in question.gold_answer_text}

    def eligible(fact: Fact) -> bool:
        answer_id = fact.answer_entity_id or fact.subject_id
        entity = context.entities.get(answer_id)
        return (
            fact.fact_id not in question.required_valid_evidence_ids
            and answer_id not in gold_ids
            and entity is not None
            and entity.name.casefold() not in gold_names
            and fact_visible(fact, question.program.snapshot_id)
        )

    wrong_context = [
        fact
        for fact in context.facts
        if eligible(fact) and fact.context_id != question.program.context_id
    ]
    same_context = [
        fact
        for fact in context.facts
        if eligible(fact)
        and fact.context_id == question.program.context_id
        and fact.relation == question.program.relation
        and fact.object_id == question.program.object_id
    ]
    ordered = (
        wrong_context + same_context
        if prefer_wrong_context
        else same_context + wrong_context
    )
    if not ordered:
        raise RuntimeError(f"No valid counterfactual fact for {question.qid}")
    return sorted(ordered, key=lambda fact: fact.fact_id)[0]


def _positive_path(context: _ScenarioContext, question: Question) -> GraphPath:
    paths = [
        path
        for path in context.paths_by_question[question.qid]
        if path.supports_gold_answer and path.edges
    ]
    if not paths:
        raise RuntimeError(f"No positive graph path for {question.qid}")
    return sorted(paths, key=lambda path: path.pid)[0]


def _graph_path_text(
    context: _ScenarioContext,
    path: GraphPath,
    *,
    reverse: bool,
) -> GraphPathText:
    segments: list[str] = []
    edges = list(reversed(path.edges)) if reverse else list(path.edges)
    for edge in edges:
        fact = _fact_by_id(context, edge.fact_id)
        source_id = fact.graph_source_id or fact.context_id or fact.subject_id
        target_id = fact.graph_target_id or fact.answer_entity_id or fact.subject_id
        if reverse:
            source_id, target_id = target_id, source_id
        source = context.entities[source_id].name
        target = context.entities[target_id].name
        relation = fact.source_relation_label or str(fact.relation).replace("_", " ")
        segments.append(
            f"{source} --[{relation}; valid {human_interval(fact.valid_time)}]--> {target}"
        )
    mode = "reverse" if reverse else "forward"
    return GraphPathText(
        path_id=f"{path.pid}:{mode}",
        text=" ; ".join(segments),
        fact_ids=[edge.fact_id for edge in edges],
    )


def _answer_claim(question: Question) -> str:
    if question.should_abstain:
        return "The available evidence is insufficient to determine an answer."
    return f"For the requested time, the answer is {_join_answers(question.gold_answer_text)}."


def _natural_answer(question: Question) -> str:
    return f"Based on the stated evidence, {_join_answers(question.gold_answer_text)}."


def _fact_answer_claim(context: _ScenarioContext, fact: Fact) -> str:
    answer_id = fact.answer_entity_id or fact.subject_id
    return f"For the requested time, the answer is {context.entities[answer_id].name}."


def _join_answers(answers: list[str]) -> str:
    if len(answers) == 1:
        return answers[0]
    if len(answers) == 2:
        return f"{answers[0]} and {answers[1]}"
    return f"{', '.join(answers[:-1])}, and {answers[-1]}"


def _primary_category(transformation: str | None) -> str:
    if not transformation:
        raise RuntimeError("Project-formal row has no transformation family")
    prefix = transformation.split(":", 1)[0]
    return "update" if prefix in {"update_relevant", "update_irrelevant"} else prefix


def _validate_formal_counts(
    *,
    scenario_count: int,
    categories: Counter[str],
    update_types: Counter[str],
    groups: Counter[str],
) -> None:
    multiplier = scenario_count // 4
    expected_categories = {
        key: value * multiplier
        for key, value in FORMAL_CATEGORY_ROWS_PER_FOUR_SCENARIOS.items()
    }
    if dict(categories) != expected_categories:
        raise RuntimeError(
            f"Project-formal category quotas differ: {dict(categories)} != {expected_categories}"
        )
    expected_updates = {"update_irrelevant": 12 * multiplier, "update_relevant": 12 * multiplier}
    if dict(update_types) != expected_updates:
        raise RuntimeError(
            f"Project-formal update quotas differ: {dict(update_types)} != {expected_updates}"
        )
    if len(groups) != scenario_count or max(groups.values()) > 36:
        raise RuntimeError("Project-formal source grouping contract was violated")
    if sum(categories.values()) != scenario_count * FORMAL_SCENARIO_ROWS:
        raise RuntimeError("Project-formal total row count is inconsistent")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
