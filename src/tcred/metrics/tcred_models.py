from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QueryParseStatus = Literal[
    "exact",
    "partial",
    "missing_temporal_anchor",
    "ambiguous_temporal_operator",
    "unparseable",
]
TemporalBasis = Literal[
    "world_valid_time",
    "snapshot_observation",
    "document_revision",
    "not_temporal",
]
EvidenceTimeStatus = Literal[
    "valid",
    "stale",
    "future_invalid",
    "temporally_ambiguous",
    "unknown_valid_time",
    "publication_only",
    "not_temporal",
]


class NormalizedTemporalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: QueryParseStatus
    operator: str
    basis: TemporalBasis
    query_start: date | None = None
    query_end: date | None = None
    evaluation_time: date | None = None
    granularity: Literal["day", "month", "year", "unknown"] = "unknown"
    interval_requires_coverage: bool = False
    explanation: str


class SemanticPairScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_source: Literal["candidate", "reference"]
    claim_index: int = Field(ge=0)
    claim: str
    evidence_kind: Literal["retrieved", "cited", "path"]
    evidence_id: str
    evidence_text_sha256: str
    entailment: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    contradiction: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> SemanticPairScore:
        if abs(self.entailment + self.neutral + self.contradiction - 1.0) > 1e-4:
            raise ValueError("Semantic probabilities must sum to one")
        return self


class TCredSemanticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    metric_id: str
    input_sha256: str
    model: str
    class_mapping: dict[str, int]
    candidate_claims: list[str]
    reference_claims: list[str]
    pairs: list[SemanticPairScore]


class EvidenceTemporalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    compatibility: float | None = Field(default=None, ge=0, le=1)
    exact_validity: float | None = Field(default=None, ge=0, le=1)
    near_miss: float | None = Field(default=None, ge=0, le=1)
    status: EvidenceTimeStatus
    effective_start: date | None = None
    effective_end: date | None = None
    temporal_source: Literal["valid_time", "publication_time", "none"]
    explanation: str


class ClaimEvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_index: int
    claim: str
    evidence_id: str
    entailment: float = Field(ge=0, le=1)
    contradiction: float = Field(ge=0, le=1)
    temporal_compatibility: float | None = Field(default=None, ge=0, le=1)
    exact_temporal_validity: float | None = Field(default=None, ge=0, le=1)
    claim_temporal_compatibility: float | None = Field(default=None, ge=0, le=1)
    provenance: float = Field(ge=0, le=1)
    link_support: float = Field(ge=0, le=1)
    link_conflict: float = Field(ge=0, le=1)


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_index: int
    claim: str
    claim_time_status: str = "absent"
    claim_time_mode: str = "none"
    claim_time_start: date | None = None
    claim_time_end: date | None = None
    claim_query_compatibility: float | None = Field(default=None, ge=0, le=1)
    semantic_attribution: float = Field(ge=0, le=1)
    temporal_attribution: float | None = Field(default=None, ge=0, le=1)
    conflict_exposure: float | None = Field(default=None, ge=0, le=1)
    global_conflict_sensitive_attribution: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    best_support_evidence_id: str | None = None
    best_conflict_evidence_id: str | None = None
    links: list[ClaimEvidenceAssessment]


class PathAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    structural_valid: bool
    answer_endpoint_label: str = ""
    answer_endpoint_labels: list[str] = Field(default_factory=list)
    question_anchor: float = Field(ge=0, le=1)
    answer_anchor: float = Field(ge=0, le=1)
    semantic_relevance: float = Field(ge=0, le=1)
    path_relevance: float = Field(ge=0, le=1)
    path_time: float | None = Field(default=None, ge=0, le=1)
    path_provenance: float = Field(ge=0, le=1)
    coherence: float | None = Field(default=None, ge=0, le=1)
    temporal_target_edge_id: str | None = None
    temporal_competitor_count: int = Field(default=0, ge=0)
    temporal_tied_answer_count: int = Field(default=0, ge=0)
    supporting_edge_ids: list[str] = Field(default_factory=list)
    ignored_edge_ids: list[str] = Field(default_factory=list)
    extraneous_invalid_edge_rate: float | None = Field(default=None, ge=0, le=1)
    status: str
    explanation: str


class TCredMetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    metric_id: str
    mode: Literal["automatic", "oracle"] = "automatic"
    query: NormalizedTemporalQuery
    candidate_claims: list[str]
    reference_claims: list[str]
    evidence_time: list[EvidenceTemporalAssessment]
    claim_assessments: list[ClaimAssessment]
    path_assessments: list[PathAssessment]
    scores: dict[str, float | None]
    coverage: dict[str, bool]
    audit: dict[str, object]
