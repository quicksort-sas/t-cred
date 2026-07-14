from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tcred.dataset.models import PathTimeStatus, TemporalInterval


class QASystemName(StrEnum):
    VECTOR_RAG = "vector_rag"
    TEMPORAL_FILTER_RAG = "temporal_filter_rag"
    GRAPH_RAG_NO_TIME = "graph_rag_no_time"
    TEMPORAL_GRAPH_RAG = "temporal_graph_rag"


ALL_QA_SYSTEMS = tuple(QASystemName)


class TemporalIntent(BaseModel):
    operator: str
    query_start: date | None = None
    query_end: date | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    explanation: str

    @model_validator(mode="after")
    def ordered_query_interval(self) -> TemporalIntent:
        if self.query_start and self.query_end and self.query_start > self.query_end:
            raise ValueError("Temporal intent query_start must be <= query_end")
        return self


class RetrievalHit(BaseModel):
    fact_id: str
    rank: int
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    temporal_score: float | None = None
    graph_score: float | None = None


class RetrievedGraphPath(BaseModel):
    path_id: str
    fact_ids: list[str]
    traversal_directions: list[Literal["forward", "reverse"]] = Field(default_factory=list)
    node_ids: list[str]
    score: float
    path_time_status: PathTimeStatus
    explanation: str


class RetrievalResult(BaseModel):
    system_name: QASystemName
    snapshot_id: str
    visible_fact_count: int
    hits: list[RetrievalHit]
    graph_paths: list[RetrievedGraphPath] = Field(default_factory=list)
    temporal_intent: TemporalIntent | None = None
    evidence_handle_map: dict[str, str] = Field(default_factory=dict)
    context_sha256: str


class LLMAnswerPayload(BaseModel):
    """The only structure requested from the answer model."""

    model_config = ConfigDict(extra="forbid")

    answer_text: str
    cited_evidence_ids: list[str]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class SystemOutput(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    output_id: str
    run_id: str
    dataset_family: str
    qid: str
    scenario_id: str
    system_name: QASystemName
    status: Literal["success", "error"]
    answer_text: str
    cited_evidence_ids: list[str]
    resolved_cited_evidence_ids: list[str]
    unresolved_citation_ids: list[str]
    retrieval: RetrievalResult
    generator_provider: str
    generator_model: str
    reasoning_effort: str
    prompt_version: str
    prompt_sha256: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class PendingSystemOutput(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    output_id: str
    run_id: str
    dataset_family: str
    qid: str
    scenario_id: str
    system_name: QASystemName
    retrieval: RetrievalResult
    generator_provider: str
    generator_model: str
    reasoning_effort: str
    prompt_version: str
    prompt_sha256: str


class QARunConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    dataset_root: Path
    output_root: Path
    cache_dir: Path = Path("data/cache/qa")
    systems: list[QASystemName] = Field(default_factory=lambda: list(ALL_QA_SYSTEMS))
    families: list[str] = Field(default_factory=lambda: ["tcred_synth", "tcred_pat", "tcred_hoh"])
    splits: list[str] = Field(default_factory=lambda: ["test_auto"])
    embedding_provider: Literal["mistral", "openai"] = "mistral"
    embedding_model: str = "mistral-embed"
    embedding_dimensions: int = 1024
    generator_provider: Literal["groq", "openai", "mistral"] = "groq"
    generator_model: str = "openai/gpt-oss-20b"
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    top_k: int = 10
    candidate_k: int = 80
    graph_seed_k: int = 8
    graph_max_hops: int = 2
    graph_path_limit: int = 4
    concurrency: int = 12
    request_timeout_seconds: float = 90.0
    seed: int = 7
    limit_per_family: int | None = None
    overwrite: bool = False
    resume: bool = True


class FamilyRunSummary(BaseModel):
    family: str
    system_name: QASystemName
    output_path: Path
    requested: int
    succeeded: int
    failed: int
    resumed: int
    input_tokens: int
    output_tokens: int
    output_sha256: str = ""


class QARunManifest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: QARunConfig
    dataset_hashes: dict[str, str]
    summaries: list[FamilyRunSummary]
    diagnostics_path: Path | None = None


class FactPromptView(BaseModel):
    fact_id: str
    evidence_text: str
    source_type: str
    subject_label: str
    relation_label: str
    target_label: str
    qualifier_label: str = ""
    valid_time: TemporalInterval
    publication_time: date | None = None
