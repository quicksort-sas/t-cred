from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DATASET_SCHEMA_VERSION = "3.0"


class DatasetRecord(BaseModel):
    """Base class for persisted dataset records.

    The default keeps older releases readable while ensuring every newly written
    record identifies the schema contract that produced it.
    """

    schema_version: str = DATASET_SCHEMA_VERSION


class DatasetFamily(StrEnum):
    SYNTH = "tcred_synth"
    PAT = "tcred_pat"
    HOH = "tcred_hoh"


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    TEAM = "team"
    POLICY = "policy"
    CONTRACT = "contract"
    PRODUCT = "product"
    PRODUCT_VERSION = "product_version"
    LOCATION = "location"
    EVENT = "event"
    PROJECT = "project"
    ROLE = "role"
    WIKIDATA_ENTITY = "wikidata_entity"
    DOCUMENT = "document"
    ANSWER_VALUE = "answer_value"


class Relation(StrEnum):
    HELD_ROLE = "held_role"
    MEMBER_OF = "member_of"
    EMPLOYED_BY = "employed_by"
    POLITICAL_AFFILIATION = "political_affiliation"
    POLICY_EFFECTIVE = "policy_effective"
    CONTRACT_ACTIVE = "contract_active"
    PRODUCT_VERSION = "product_version"
    SUPPORT_WINDOW = "support_window"
    LOCATED_AT = "located_at"
    EVENT_OCCURS = "event_occurs"
    EVENT_PRECEDES = "event_precedes"
    PROJECT_PARTICIPANT = "project_participant"
    AFFILIATED_WITH = "affiliated_with"
    WIKIDATA_PROPERTY = "wikidata_property"
    DOCUMENT_SUPPORTS = "document_supports"


class Granularity(StrEnum):
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    UNKNOWN = "unknown"


class FactRole(StrEnum):
    SOURCE_ASSERTION = "source_assertion"
    VALID_SUPPORT = "valid_support"
    STALE_DISTRACTOR = "stale_distractor"
    FUTURE_DISTRACTOR = "future_distractor"
    CONTRADICTORY = "contradictory"
    GRAPH_INCOHERENT = "graph_incoherent"
    HARD_NEGATIVE = "hard_negative"
    PUBLICATION_ONLY = "publication_only"
    UNKNOWN_TIME = "unknown_time"
    UPDATE_SPECIFIC = "update_specific"
    BACKGROUND = "background"


class TimeStatus(StrEnum):
    VALID = "valid"
    STALE = "stale"
    FUTURE_INVALID = "future_invalid"
    TEMPORALLY_AMBIGUOUS = "temporally_ambiguous"
    UNKNOWN_VALID_TIME = "unknown_valid_time"
    PUBLICATION_ONLY = "publication_only"
    IRRELEVANT = "irrelevant"


class TemporalOperator(StrEnum):
    CURRENT = "current"
    AS_OF = "as_of"
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    PREVIOUS = "previous"
    NEXT = "next"
    LATEST = "latest"
    FIRST = "first"
    LAST = "last"
    BETWEEN = "between"
    EXPIRED = "expired"
    EFFECTIVE = "effective"
    UPDATE_STABILITY = "update_stability"


class AnswerType(StrEnum):
    ENTITY = "entity"
    TIME = "time"
    SPAN = "span"
    LIST = "list"
    COMPARISON = "comparison"
    REFUSAL = "refusal"


class SystemDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvalDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PathTimeStatus(StrEnum):
    COHERENT_SHARED_INTERVAL = "coherent_shared_interval"
    COHERENT_SEQUENCE = "coherent_sequence"
    INCOHERENT_EMPTY_INTERSECTION = "incoherent_empty_intersection"
    WRONG_ORDER = "wrong_order"
    INCOHERENT_QUERY_TIME = "incoherent_query_time"
    UNKNOWN_EDGE_TIME = "unknown_edge_time"
    NOT_APPLICABLE = "not_applicable"


class ContextPackType(StrEnum):
    VALID_ONLY = "valid_only"
    STALE_ONLY = "stale_only"
    FUTURE_ONLY = "future_only"
    VALID_PLUS_STALE = "valid_plus_stale"
    VALID_PLUS_FUTURE = "valid_plus_future"
    CONFLICT = "conflict"
    PUBLICATION_ONLY = "publication_only"
    UNKNOWN_TIME = "unknown_time"
    GRAPH_INCOHERENT = "graph_incoherent"
    INSUFFICIENT = "insufficient"


class AnswerVariantType(StrEnum):
    CORRECT_SUPPORTED = "correct_supported"
    CORRECT_INVALID_EVIDENCE = "correct_answer_invalid_evidence"
    STALE_ANSWER = "stale_answer"
    OUTDATED_SOURCE_ANSWER = "outdated_source_answer"
    FUTURE_INVALID_ANSWER = "future_invalid_answer"
    WRONG_OPERATOR_ANSWER = "wrong_operator_answer"
    INVALID_GRAPH_PATH_ANSWER = "invalid_graph_path_answer"
    PARTIAL_ANSWER = "partial_answer"
    HALLUCINATED_ANSWER = "unsupported_hallucinated_answer"
    OVERCONFIDENT_SHOULD_REFUSE = "overconfident_should_refuse"
    CORRECT_REFUSAL = "correct_refusal"
    INAPPROPRIATE_REFUSAL = "inappropriate_refusal"


class TemporalInterval(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    type: Literal["point", "interval", "open_interval", "unknown"] = "interval"
    start: date | None = None
    end: date | None = None
    granularity: Granularity = Granularity.DAY

    @field_validator("end")
    @classmethod
    def validate_end(cls, end: date | None, info: Any) -> date | None:
        start = info.data.get("start")
        interval_type = info.data.get("type")
        if start and end and start > end:
            raise ValueError("Temporal interval start must be <= end")
        if interval_type == "point" and start and end and start != end:
            raise ValueError("Point interval must have matching start/end")
        return end

    def contains(self, day: date) -> bool:
        if self.type == "unknown" or self.start is None:
            return False
        if self.end is None:
            return self.start <= day
        return self.start <= day <= self.end

    def overlaps(self, other: TemporalInterval) -> bool:
        if self.start is None or other.start is None:
            return False
        left_end = self.end or date.max
        right_end = other.end or date.max
        return self.start <= right_end and other.start <= left_end

    def ends_before(self, day: date) -> bool:
        return self.end is not None and self.end < day

    def starts_after(self, day: date) -> bool:
        return self.start is not None and self.start > day


class Entity(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    entity_id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    domain: str


class Fact(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    fact_id: str
    scenario_id: str
    subject_id: str
    relation: Relation
    object_id: str | None = None
    context_id: str | None = None
    # Answer identity and graph orientation are explicit. Legacy generators may
    # omit these fields; helpers then fall back to the historical subject-based
    # representation.
    answer_entity_id: str | None = None
    graph_source_id: str | None = None
    graph_target_id: str | None = None
    source_relation_id: str | None = None
    source_relation_label: str | None = None
    relation_direction: Literal["directed", "symmetric"] | None = None
    source_record_id: str | None = None
    source_revision: str | None = None
    valid_time: TemporalInterval
    publication_time: date | None = None
    transaction_time: date | None = None
    snapshot_visible_from: str = "S0"
    source_type: str
    provenance_reliability: Literal["high", "medium", "low"] = "high"
    fact_role: FactRole
    canonical_evidence: str
    paraphrased_evidence: str | None = None


class Snapshot(DatasetRecord):
    scenario_id: str
    snapshot_id: str
    snapshot_time: date
    visible_fact_ids: list[str]
    description: str


class QuestionProgram(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    operator: TemporalOperator
    target: str
    query_time: TemporalInterval
    relation: Relation
    context_id: str | None = None
    object_id: str | None = None
    reference_entity_id: str | None = None
    snapshot_id: str = "S1"
    answer_function: str
    required_path_semantics: str = "single_fact"
    temporal_basis: Literal[
        "world_valid_time",
        "snapshot_observation",
        "document_revision",
        "not_temporal",
    ] = "world_valid_time"


class Question(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    qid: str
    scenario_id: str
    dataset_family: DatasetFamily = DatasetFamily.SYNTH
    canonical_question: str
    question: str
    program: QuestionProgram
    temporal_operator: TemporalOperator
    answer_type: AnswerType
    gold_answer_entity_ids: list[str]
    gold_answer_text: list[str]
    required_valid_evidence_ids: list[str]
    should_abstain: bool = False
    system_difficulty: SystemDifficulty
    eval_difficulty: EvalDifficulty
    difficulty_provenance: Literal["generator_heuristic", "source_heuristic"] = (
        "generator_heuristic"
    )
    human_pool_candidate: bool = False
    semantic_series_id: str | None = None
    template_family_id: str | None = None
    certification_status: Literal["candidate", "certified", "rejected"] = "candidate"


class GraphPathEdge(DatasetRecord):
    fact_id: str
    relation: Relation
    valid_time: TemporalInterval
    traversal_direction: Literal["forward", "reverse"] = "forward"


class GraphPath(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    pid: str
    scenario_id: str
    qid: str
    nodes: list[str]
    edges: list[GraphPathEdge]
    path_time_status: PathTimeStatus
    supports_gold_answer: bool
    explanation: str


class ContextPack(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    pack_id: str
    qid: str
    scenario_id: str
    pack_type: ContextPackType
    evidence_ids: list[str]
    expected_behavior: str


class AnswerClaim(DatasetRecord):
    cid: str
    text: str
    claim_time: TemporalInterval
    cited_evidence_ids: list[str] = Field(default_factory=list)
    temporally_valid: bool | None


class AnswerVariant(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    answer_id: str
    qid: str
    scenario_id: str
    variant_type: AnswerVariantType
    alternate_operator: TemporalOperator | None = None
    answer_text: str
    cited_evidence_ids: list[str]
    graph_path_ids: list[str] = Field(default_factory=list)
    claims: list[AnswerClaim]
    answer_correct: Literal["yes", "partial", "no", "unjudgeable"]
    temporal_correct: Literal["yes", "partial", "no", "not_applicable", "unjudgeable"]
    evidence_supports_answer: Literal["yes", "partial", "no", "not_applicable", "unjudgeable"]
    citation_temporally_valid: Literal["yes", "partial", "no", "not_applicable", "unjudgeable"]
    graph_path_sufficient: Literal["yes", "partial", "no", "not_applicable", "unjudgeable"] = (
        "not_applicable"
    )
    refusal_appropriate: Literal["yes", "partial", "no", "not_applicable", "unjudgeable"] = (
        "not_applicable"
    )


class SourceProvenance(DatasetRecord):
    source_id: str
    source_family: str
    fidelity: Literal["pattern_only", "source_extracted", "source_record_converted"]
    source_record_ids: list[str] = Field(default_factory=list)
    source_revision: str | None = None
    source_relation: str | None = None
    source_path_relation: str | None = None
    topology_signature: str | None = None


class Scenario(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)

    scenario_id: str
    split_group_id: str | None = None
    domain: str
    blueprint: str
    entities: list[Entity]
    facts: list[Fact]
    snapshots: list[Snapshot]
    question_ids: list[str]
    update_behavior: Literal[
        "answer_should_change",
        "answer_should_stay",
        "answer_should_abstain",
        "not_applicable",
    ]
    source_provenance: SourceProvenance | None = None
    notes: str


class DatasetBundle(DatasetRecord):
    scenarios: list[Scenario]
    entities: list[Entity]
    facts: list[Fact]
    snapshots: list[Snapshot]
    questions: list[Question]
    graph_paths: list[GraphPath]
    context_packs: list[ContextPack]
    answer_variants: list[AnswerVariant]
    splits: dict[str, list[str]]
