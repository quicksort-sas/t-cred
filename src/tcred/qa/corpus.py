from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tcred.dataset.graph import fact_endpoint_ids, fact_node_ids, relation_label
from tcred.dataset.io import read_jsonl
from tcred.dataset.models import (
    DatasetBundle,
    DatasetFamily,
    EntityType,
    Question,
    Relation,
    TemporalInterval,
)
from tcred.dataset.text import normalize_visible_text
from tcred.qa.models import FactPromptView

_SNAPSHOT_PATTERN = re.compile(r"^S(?P<rank>\d+)$", re.IGNORECASE)


class RuntimeEntity(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    entity_id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)


class RuntimeFact(BaseModel):
    """Fact fields available to retrieval and answer generation."""

    model_config = ConfigDict(use_enum_values=True)

    fact_id: str
    subject_id: str
    relation: Relation
    object_id: str | None = None
    context_id: str | None = None
    answer_entity_id: str | None = None
    graph_source_id: str | None = None
    graph_target_id: str | None = None
    source_relation_id: str | None = None
    source_relation_label: str | None = None
    relation_direction: str | None = None
    source_record_id: str | None = None
    source_revision: str | None = None
    valid_time: TemporalInterval
    publication_time: date | None = None
    transaction_time: date | None = None
    snapshot_visible_from: str = "S0"
    source_type: str = "source_record"
    canonical_evidence: str
    paraphrased_evidence: str | None = None


class RuntimeQuestion(BaseModel):
    """Question projection with no gold answer or evaluator-only labels."""

    model_config = ConfigDict(use_enum_values=True)

    qid: str
    scenario_id: str
    dataset_family: DatasetFamily
    question: str
    snapshot_id: str
    splits: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CorpusDocument:
    index: int
    fact: RuntimeFact
    semantic_text: str


class RuntimeCorpus:
    """Runtime-safe fact projection used by every QA baseline.

    Production QA entry points construct this object from ``runtime/`` files;
    the bundle constructor remains only for unit-level tests and immediately
    projects private models onto the same restricted schema.
    """

    def __init__(self, bundle: DatasetBundle) -> None:
        self._initialize(
            entities=[
                RuntimeEntity(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    aliases=entity.aliases,
                )
                for entity in bundle.entities
            ],
            facts=[
                RuntimeFact(
                    fact_id=fact.fact_id,
                    subject_id=fact.subject_id,
                    relation=fact.relation,
                    object_id=fact.object_id,
                    context_id=fact.context_id,
                    answer_entity_id=fact.answer_entity_id,
                    graph_source_id=fact.graph_source_id,
                    graph_target_id=fact.graph_target_id,
                    source_relation_id=fact.source_relation_id,
                    source_relation_label=fact.source_relation_label,
                    relation_direction=fact.relation_direction,
                    source_record_id=fact.source_record_id,
                    source_revision=fact.source_revision,
                    valid_time=fact.valid_time,
                    publication_time=fact.publication_time,
                    transaction_time=fact.transaction_time,
                    snapshot_visible_from=fact.snapshot_visible_from,
                    source_type="source_record",
                    canonical_evidence=fact.canonical_evidence,
                    paraphrased_evidence=fact.paraphrased_evidence,
                )
                for fact in bundle.facts
            ],
        )

    @classmethod
    def from_dataset_dir(cls, dataset_dir: Path) -> RuntimeCorpus:
        runtime_dir = dataset_dir / "runtime"
        entities_path = runtime_dir / "entities.jsonl"
        facts_path = runtime_dir / "facts.jsonl"
        if not entities_path.exists() or not facts_path.exists():
            raise FileNotFoundError(
                f"Runtime projection is missing in {runtime_dir}; regenerate the dataset"
            )
        instance = cls.__new__(cls)
        instance._initialize(
            entities=[RuntimeEntity.model_validate(row) for row in read_jsonl(entities_path)],
            facts=[RuntimeFact.model_validate(row) for row in read_jsonl(facts_path)],
        )
        return instance

    def _initialize(
        self,
        *,
        entities: list[RuntimeEntity],
        facts: list[RuntimeFact],
    ) -> None:
        self.entities = {entity.entity_id: entity for entity in entities}
        self.facts = facts
        self.fact_by_id = {fact.fact_id: fact for fact in self.facts}
        if len(self.fact_by_id) != len(self.facts):
            raise ValueError("Runtime projection contains duplicate fact IDs")
        self.index_by_fact_id = {fact.fact_id: index for index, fact in enumerate(self.facts)}
        self.documents = [
            CorpusDocument(index=index, fact=fact, semantic_text=self._semantic_text(fact))
            for index, fact in enumerate(self.facts)
        ]

    def visible_indices(self, snapshot_id: str) -> list[int]:
        requested_rank = snapshot_rank(snapshot_id)
        return [
            document.index
            for document in self.documents
            if snapshot_rank(document.fact.snapshot_visible_from) <= requested_rank
        ]

    def prompt_view(self, fact_id: str) -> FactPromptView:
        fact = self.fact_by_id[fact_id]
        source_id, target_id = fact_endpoint_ids(fact)
        qualifier = self.entities.get(fact.object_id or "")
        return FactPromptView(
            fact_id=fact.fact_id,
            evidence_text=fact.paraphrased_evidence or fact.canonical_evidence,
            source_type=fact.source_type,
            subject_label=self._entity_label(source_id),
            relation_label=fact.source_relation_label
            or relation_label(
                fact.relation,
                object_label=qualifier.name if qualifier else "",
            ),
            target_label=self._entity_label(target_id),
            qualifier_label=qualifier.name if qualifier else "",
            valid_time=fact.valid_time,
            publication_time=fact.publication_time,
        )

    def graph_node_ids(self, fact_id: str) -> list[str]:
        return fact_node_ids(self.fact_by_id[fact_id], include_object=False)

    def fact_group_key(self, fact: RuntimeFact) -> tuple[str, str, str]:
        return (
            str(fact.relation),
            fact.context_id or fact.object_id or "ungrouped",
            fact.object_id or "",
        )

    def semantic_fact_key(self, fact_id: str) -> tuple[str, ...]:
        """Public-content identity used to remove replicated retrieval chunks."""
        fact = self.fact_by_id[fact_id]
        source_id, target_id = fact_endpoint_ids(fact)
        qualifier = self.entities.get(fact.object_id or "")
        return (
            normalize_visible_text(self._entity_label(source_id)),
            normalize_visible_text(
                fact.source_relation_label
                or relation_label(
                    fact.relation,
                    object_label=qualifier.name if qualifier else "",
                )
            ),
            normalize_visible_text(self._entity_label(target_id)),
            normalize_visible_text(qualifier.name if qualifier else ""),
            normalize_visible_text(fact.paraphrased_evidence or fact.canonical_evidence),
            str(fact.valid_time.type),
            str(fact.valid_time.start or ""),
            str(fact.valid_time.end or ""),
            str(fact.valid_time.granularity),
            str(fact.publication_time or ""),
            str(fact.transaction_time or ""),
        )

    def _semantic_text(self, fact: RuntimeFact) -> str:
        source_id, target_id = fact_endpoint_ids(fact)
        qualifier = self.entities.get(fact.object_id or "")
        labels = [
            self._entity_label(source_id),
            fact.source_relation_label
            or relation_label(fact.relation, object_label=qualifier.name if qualifier else ""),
            self._entity_label(target_id),
        ]
        if qualifier:
            labels.append(qualifier.name)
        evidence = fact.paraphrased_evidence or fact.canonical_evidence
        return f"{evidence} Entities and relation: {' | '.join(filter(None, labels))}."

    def _entity_label(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        if entity is None:
            return entity_id
        aliases = ", ".join(entity.aliases)
        return f"{entity.name} ({aliases})" if aliases else entity.name


def load_runtime_questions(
    dataset_dir: Path,
    *,
    splits: list[str],
    limit: int | None,
) -> list[RuntimeQuestion]:
    path = dataset_dir / "runtime" / "questions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Runtime question projection is missing: {path}; regenerate the dataset"
        )
    questions = [RuntimeQuestion.model_validate(row) for row in read_jsonl(path)]
    available = {split for question in questions for split in question.splits}
    unknown = set(splits) - available - {"all"}
    if unknown:
        raise ValueError(f"Unknown dataset split(s): {sorted(unknown)}")
    selected = [
        question
        for question in questions
        if "all" in splits or set(question.splits).intersection(splits)
    ]
    selected.sort(key=lambda question: question.qid)
    if limit is None or limit >= len(selected):
        return selected
    return sorted(
        selected,
        key=lambda question: hashlib.sha256(
            f"runtime-question-v1:{question.qid}".encode()
        ).hexdigest(),
    )[:limit]


def runtime_snapshot_id(question: Question | RuntimeQuestion) -> str:
    """Return the corpus version for the evaluation request."""
    if isinstance(question, RuntimeQuestion):
        return question.snapshot_id
    return question.program.snapshot_id


def snapshot_rank(snapshot_id: str) -> int:
    match = _SNAPSHOT_PATTERN.fullmatch(snapshot_id.strip())
    if not match:
        raise ValueError(f"Unsupported snapshot id: {snapshot_id!r}")
    return int(match.group("rank"))


def dataset_content_hash(dataset_dir: Path) -> str:
    """Hash only the inputs available to a QA system."""
    digest = hashlib.sha256()
    runtime_dir = dataset_dir / "runtime"
    for name in ("entities.jsonl", "facts.jsonl", "questions.jsonl"):
        path = runtime_dir / name
        digest.update(f"runtime/{name}".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
