from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    text: str


class MetricInput(BaseModel):
    """Canonical evaluator input shared by every metric implementation."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    population: Literal["human_gold", "system_full", "diagnostic_challenge"]
    dataset_family: str
    source_kind: str
    system_name: str | None = None
    unit_id: str | None = None
    qid: str
    scenario_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    retrieved_evidence: list[EvidenceText] = Field(default_factory=list)
    cited_evidence: list[EvidenceText] = Field(default_factory=list)
    gold_labels: dict[str, str] = Field(default_factory=dict)
    retrieval_metrics: dict[str, float | None] = Field(default_factory=dict)
    citation_metrics: dict[str, float | None] = Field(default_factory=dict)
    unresolved_citation_count: int = 0
    gold_provenance: dict[str, dict[str, object]] = Field(default_factory=dict)


class CandidateClaimJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    reference_supported: bool
    retrieved_support_indices: list[int]
    cited_support_indices: list[int]


class ReferenceClaimJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    candidate_supported: bool
    retrieved_support_indices: list[int]


class ClaimJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_correctness: int = Field(ge=0, le=4)
    answer_relevance: int = Field(ge=0, le=4)
    candidate_claims: list[CandidateClaimJudgment] = Field(min_length=1, max_length=16)
    reference_claims: list[ReferenceClaimJudgment] = Field(min_length=1, max_length=16)
    rationale: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def unique_nonempty_claims(self) -> ClaimJudgeResult:
        for claims in (self.candidate_claims, self.reference_claims):
            normalized = [claim.claim.strip().casefold() for claim in claims]
            if any(not value for value in normalized):
                raise ValueError("Judge claims must be non-empty")
            if len(normalized) != len(set(normalized)):
                raise ValueError("Judge returned duplicate claims")
        return self


class JudgeCacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    metric_id: str
    input_sha256: str
    prompt_sha256: str
    provider: Literal["openai", "anthropic", "mistral", "groq"]
    model: str
    response_id: str
    usage: dict[str, int]
    result: ClaimJudgeResult


class MetricScoreRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str
    population: str
    dataset_family: str
    source_kind: str
    system_name: str | None
    unit_id: str | None
    qid: str
    scenario_id: str
    gold_labels: dict[str, str]
    scores: dict[str, float | None]
    gold_provenance: dict[str, dict[str, object]] = Field(default_factory=dict)
    metric_metadata: dict[str, object] = Field(default_factory=dict)
