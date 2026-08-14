from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JudgeProvider = Literal["openai", "anthropic", "mistral", "groq"]
JudgeLabel = Literal["yes", "partial", "no", "unjudgeable", "not_applicable"]
JudgeStage = Literal["evidence", "answer"]
PromptVariant = Literal["rubric_only", "contrastive_few_shot"]

JUDGED_FIELDS = (
    "answer_correct",
    "temporal_correct",
    "evidence_supports_answer",
    "citation_temporally_valid",
    "graph_evidence_sufficient",
    "response_decision_appropriate",
)
EVIDENCE_STAGE_FIELDS = (
    "temporal_correct",
    "evidence_supports_answer",
    "citation_temporally_valid",
    "graph_evidence_sufficient",
)
ANSWER_STAGE_FIELDS = (
    "answer_correct",
    "response_decision_appropriate",
)


class VisibleInterval(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "unknown"
    start: str | None = None
    end: str | None = None
    granularity: str | None = None


class JudgeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    text: str
    publication_time: str | None = None
    valid_time: VisibleInterval = Field(default_factory=VisibleInterval)


class JudgePathNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    type: str = ""


class JudgePathEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fact_id: str = ""
    relation: str = ""
    relation_label: str = ""
    source: JudgePathNode
    target: JudgePathNode
    valid_time: VisibleInterval = Field(default_factory=VisibleInterval)
    traversal_direction: Literal["forward", "reverse"] = "forward"
    directional: bool = True
    symmetric: bool = False
    evidence_text: str = ""


class JudgeGraphPath(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path_id: str
    path_source: str = "candidate_path"
    nodes: list[JudgePathNode] = Field(default_factory=list)
    edges: list[JudgePathEdge] = Field(default_factory=list)


class TaskJudgeInput(BaseModel):
    """One blinded response card plus private analysis metadata never sent to the judge."""

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
    cited_evidence_ids: list[str] = Field(default_factory=list)
    cited_evidence: list[JudgeEvidence] = Field(default_factory=list)
    retrieved_evidence: list[JudgeEvidence] = Field(default_factory=list)
    graph_paths: list[JudgeGraphPath] = Field(default_factory=list)
    context_note: str = ""
    applicable_fields: list[str]
    gold_labels: dict[str, str] = Field(default_factory=dict)
    gold_provenance: dict[str, dict[str, object]] = Field(default_factory=dict)
    source_question_sha256: str
    source_reference_answer_sha256: str
    source_candidate_answer_sha256: str
    presentation_changed_fields: list[str] = Field(default_factory=list)
    presentation_contract: str

    @field_validator("applicable_fields")
    @classmethod
    def validate_applicable_fields(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value))
        unknown = set(normalized) - set(JUDGED_FIELDS)
        if unknown:
            raise ValueError(f"Unknown task-judge fields: {sorted(unknown)}")
        if "answer_correct" not in normalized:
            raise ValueError("answer_correct must be applicable for every task-judge input")
        return normalized

    def displayed_evidence(self) -> list[JudgeEvidence]:
        rows: list[JudgeEvidence] = []
        seen: set[str] = set()
        for evidence in [*self.cited_evidence, *self.retrieved_evidence]:
            if evidence.evidence_id in seen:
                continue
            seen.add(evidence.evidence_id)
            rows.append(evidence)
        return rows

    def stage_fields(self, stage: JudgeStage) -> tuple[str, ...]:
        candidates = EVIDENCE_STAGE_FIELDS if stage == "evidence" else ANSWER_STAGE_FIELDS
        applicable = set(self.applicable_fields)
        return tuple(field for field in candidates if field in applicable)


class FieldJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: JudgeLabel
    confidence: int = Field(ge=0, le=100)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    path_ids: list[str] = Field(default_factory=list, max_length=10)
    rationale: str = Field(max_length=500)

    @field_validator("evidence_ids", "path_ids")
    @classmethod
    def unique_identifiers(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Returned identifiers must be unique")
        return value

    @model_validator(mode="after")
    def require_substantive_rationale(self) -> FieldJudgment:
        self.rationale = self.rationale.strip()
        if self.label != "not_applicable" and not self.rationale:
            raise ValueError("A substantive judgment requires a non-empty rationale")
        return self


def _not_applicable_field() -> FieldJudgment:
    return FieldJudgment(
        label="not_applicable",
        confidence=100,
        evidence_ids=[],
        path_ids=[],
        rationale="This field is not applicable to the displayed response.",
    )


class EvidenceStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporal_correct: FieldJudgment
    evidence_supports_answer: FieldJudgment
    citation_temporally_valid: FieldJudgment
    graph_evidence_sufficient: FieldJudgment


class AnswerStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_correct: FieldJudgment
    response_decision_appropriate: FieldJudgment


class TaskJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_correct: FieldJudgment
    temporal_correct: FieldJudgment = Field(default_factory=_not_applicable_field)
    evidence_supports_answer: FieldJudgment = Field(default_factory=_not_applicable_field)
    citation_temporally_valid: FieldJudgment = Field(default_factory=_not_applicable_field)
    graph_evidence_sufficient: FieldJudgment = Field(default_factory=_not_applicable_field)
    response_decision_appropriate: FieldJudgment = Field(default_factory=_not_applicable_field)

    def field(self, name: str) -> FieldJudgment:
        if name not in JUDGED_FIELDS:
            raise KeyError(name)
        return getattr(self, name)


class StageCacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    judgment_id: str
    metric_id: str
    stage: JudgeStage
    input_sha256: str
    prompt_sha256: str
    schema_sha256: str
    prompt_variant: PromptVariant
    contract_version: str
    provider: JudgeProvider
    model: str
    random_seed: int
    response_id: str
    attempts: int = Field(ge=1)
    rate_limit_headers: dict[str, str] = Field(default_factory=dict)
    support_pointer_warnings: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    usage: dict[str, int]
    result: EvidenceStageResult | AnswerStageResult


class TaskJudgeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    metric_id: str
    prompt_variant: PromptVariant
    provider: JudgeProvider
    model: str
    answer_stage: StageCacheRecord
    evidence_stage: StageCacheRecord | None = None
    result: TaskJudgeResult


class JudgeSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    seed: int
    target_calibration_fraction: float
    gold_input_sha256: str
    calibration_metric_ids: list[str]
    held_out_metric_ids: list[str]
    balance_summary: dict[str, object]


class PromptSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_version: str
    selected_variant: PromptVariant
    selection_rule: str
    calibration_metric_ids_sha256: str
    candidates: dict[str, dict[str, object]]
