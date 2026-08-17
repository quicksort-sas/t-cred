from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReleasedEvaluation(BaseModel):
    """One evaluation block parsed from an author-provided training log."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    source_sha256: str
    dataset: str
    artifact_label: str
    config: dict[str, str]
    split: str
    ordinal: int = Field(ge=0)
    hits_at_1: float = Field(ge=0, le=1)
    hits_at_10: float = Field(ge=0, le=1)
    group_scores_at_1: dict[str, float] = Field(default_factory=dict)
    group_scores_at_10: dict[str, float] = Field(default_factory=dict)
    group_counts: dict[str, int] = Field(default_factory=dict)
    inferred_example_count: int | None = Field(default=None, ge=1)


class SabetPredictionRecord(BaseModel):
    """Lossless per-question export from an instrumented SABET-QA run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = "1.1"
    run_id: str
    dataset: str
    split: str
    model: str
    variant: Literal["standard", "hard"]
    seed: int
    source_index: int = Field(ge=0)
    qid: str
    question: str
    question_type: str
    answer_type: Literal["entity", "time"]
    gold_answer_ids: list[str]
    gold_answer_labels: list[str]
    predicted_answer_ids: list[str] = Field(min_length=10)
    predicted_answer_labels: list[str] = Field(min_length=10)
    predicted_scores: list[float] = Field(min_length=10)
    source_record_sha256: str

    @model_validator(mode="after")
    def aligned_ranked_outputs(self) -> SabetPredictionRecord:
        lengths = {
            len(self.predicted_answer_ids),
            len(self.predicted_answer_labels),
            len(self.predicted_scores),
        }
        if len(lengths) != 1:
            raise ValueError("Prediction IDs, labels, and scores must have equal lengths")
        if len(self.gold_answer_ids) != len(self.gold_answer_labels):
            raise ValueError("Gold answer IDs and labels must have equal lengths")
        if len(set(self.predicted_answer_ids)) != len(self.predicted_answer_ids):
            raise ValueError("Ranked prediction IDs must be unique")
        if any(not value.strip() for value in [*self.gold_answer_ids, *self.predicted_answer_ids]):
            raise ValueError("Answer IDs must be non-empty")
        return self


class AnswerMetricRecord(BaseModel):
    """Answer-compatible metric scores with an explicit formal benchmark oracle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    dataset: str
    model: str
    variant: Literal["standard", "hard"]
    seed: int
    qid: str
    source_index: int = Field(ge=0)
    question_type: str
    answer_type: Literal["entity", "time"]
    native_hit_at_1: float = Field(ge=0, le=1)
    native_hit_at_10: float = Field(ge=0, le=1)
    scores: dict[str, float | None]
    winning_reference_by_metric: dict[str, int]
    applicability: dict[str, bool]
