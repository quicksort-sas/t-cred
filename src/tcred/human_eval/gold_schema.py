from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tcred.human_eval.protocol import CATEGORICAL_FIELDS, VISIBLE_LABEL_OPTIONS

GOLD_SCHEMA_VERSION = "1.0"
GOLD_POLICY_VERSION = "1.0"

ResolutionMethod = Literal["annotator_agreement", "adjudication"]
FieldStatus = Literal["resolved", "insufficient_support", "non_hard_adjudication"]


class FieldResolution(BaseModel):
    """Provenance for one resolved field in the published gold dataset."""

    model_config = ConfigDict(extra="forbid")

    field: str
    status: FieldStatus
    gold_label: str = ""
    gold_reason: str = ""
    resolution_method: ResolutionMethod | None = None
    raw_judgment_count: int = Field(ge=0)
    agreeing_annotator_count: int = Field(ge=0)
    raw_vote_distribution: dict[str, int]
    raw_reason_distribution: dict[str, int]
    adjudication_target_id: str = ""
    adjudication_disposition: str = ""
    adjudication_confidence: str = ""
    adjudication_decision_relation: str = ""
    adjudication_rationale: str = ""

    @model_validator(mode="after")
    def resolution_is_consistent(self) -> FieldResolution:
        if self.field not in CATEGORICAL_FIELDS:
            raise ValueError(f"Unsupported gold field: {self.field}")
        if self.status == "resolved":
            if self.gold_label not in VISIBLE_LABEL_OPTIONS:
                raise ValueError("A resolved field requires a visible gold label")
            if self.resolution_method is None:
                raise ValueError("A resolved field requires a resolution method")
        elif self.gold_label or self.resolution_method is not None:
            raise ValueError("An unresolved field cannot carry a gold label or method")
        if self.resolution_method == "annotator_agreement":
            if self.agreeing_annotator_count < 2:
                raise ValueError("Annotator-agreement gold requires at least two matching labels")
            if self.adjudication_target_id:
                raise ValueError("Agreement-only provenance cannot name an adjudication target")
        if self.resolution_method == "adjudication" and not self.adjudication_target_id:
            raise ValueError("Adjudicated gold requires an adjudication target")
        return self


class GoldLabelRecord(BaseModel):
    """Validated label row from the published final gold dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GOLD_SCHEMA_VERSION
    gold_policy_version: str = GOLD_POLICY_VERSION
    unit_id: str
    applicable_fields: tuple[str, ...]
    gold_labels: dict[str, str]
    gold_reasons: dict[str, str]
    field_provenance: dict[str, FieldResolution]
    annotation_count: int = Field(ge=2)
    metadata: dict[str, str]

    @model_validator(mode="after")
    def fields_are_complete(self) -> GoldLabelRecord:
        expected = set(self.applicable_fields)
        if set(self.gold_labels) != expected or set(self.field_provenance) != expected:
            raise ValueError("A gold unit must resolve every applicable field")
        if set(self.gold_reasons) != expected:
            raise ValueError("Gold reasons must cover every applicable field")
        if any(row.status != "resolved" for row in self.field_provenance.values()):
            raise ValueError("A gold unit cannot contain a non-resolved field")
        return self
