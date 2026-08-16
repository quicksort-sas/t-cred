from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "tcred-sl-semantic-record-v1"


class SemanticTask(StrEnum):
    ANSWER = "answer"
    SUPPORT = "support"
    RELEVANCE = "relevance"
    TEMPORAL = "temporal"
    ANSWERABILITY = "answerability"
    CITATION = "citation"


class CurriculumStage(StrEnum):
    BROAD = "stage_a"
    TASK_MATCHED = "stage_b"


class DatasetPartition(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    INTERNAL_TEST = "internal_test"
    EXTERNAL_TEST = "external_test"
    RETROSPECTIVE_TEST = "retrospective_test"


class EvidencePassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    text: str
    source_id: str | None = None
    rank: int | None = Field(default=None, ge=1)


class GraphPathText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str
    text: str
    fact_ids: list[str] = Field(default_factory=list)


class SemanticTarget(BaseModel):
    """Sparse target contract.

    A row activates one semantic task, and only labels justified by its source are
    populated. Missing labels are masks, not negative examples.
    """

    model_config = ConfigDict(extra="forbid")

    class_distribution: dict[str, float] = Field(default_factory=dict)
    answer_u1: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_u2: float | None = Field(default=None, ge=0.0, le=1.0)
    equivalence: float | None = Field(default=None, ge=0.0, le=1.0)
    supported: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    answerable: float | None = Field(default=None, ge=0.0, le=1.0)
    scalar_rating: float | None = Field(default=None, ge=0.0, le=1.0)
    pair_id: str | None = None
    pair_role: Literal["positive", "negative", "invariant_a", "invariant_b"] | None = None
    invariance_group_id: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if (
            self.answer_u1 is not None
            and self.answer_u2 is not None
            and self.answer_u2 > self.answer_u1 + 1e-8
        ):
            raise ValueError("answer_u2 must not exceed answer_u1")
        if self.class_distribution:
            if any(not key.strip() for key in self.class_distribution):
                raise ValueError("class_distribution contains an empty label")
            if any(
                not math.isfinite(value) or value < 0
                for value in self.class_distribution.values()
            ):
                raise ValueError("class probabilities must be finite and non-negative")
            total = sum(self.class_distribution.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError("class_distribution must sum to one")
        if (self.pair_id is None) != (self.pair_role is None):
            raise ValueError("pair_id and pair_role must be set together")
        if not self.has_supervision:
            raise ValueError("semantic target has no supervised value")
        return self

    @property
    def has_supervision(self) -> bool:
        values = (
            self.answer_u1,
            self.answer_u2,
            self.equivalence,
            self.supported,
            self.relevance,
            self.answerable,
            self.scalar_rating,
        )
        return bool(self.class_distribution) or any(value is not None for value in values)


class SemanticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    schema_version: Literal["tcred-sl-semantic-record-v1"] = SCHEMA_VERSION
    unit_id: str
    source_dataset: str
    source_version: str
    source_native_split: str
    source_native_id: str
    source_group_id: str
    curriculum_stage: CurriculumStage
    task: SemanticTask
    question: str | None = None
    query_time_or_interval: str | None = None
    temporal_operator: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    candidate_or_claim: str
    evidence_passages: list[EvidencePassage] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    graph_paths: list[GraphPathText] = Field(default_factory=list)
    target: SemanticTarget
    label_provenance: str
    transformation_family: str | None = None
    language: Literal["en"] = "en"
    license_record: str
    partition: DatasetPartition | None = None
    content_hash: str
    record_hash: str

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        required = {
            "unit_id": self.unit_id,
            "source_dataset": self.source_dataset,
            "source_version": self.source_version,
            "source_native_split": self.source_native_split,
            "source_native_id": self.source_native_id,
            "source_group_id": self.source_group_id,
            "candidate_or_claim": self.candidate_or_claim,
            "label_provenance": self.label_provenance,
            "license_record": self.license_record,
        }
        empty = [name for name, value in required.items() if not value.strip()]
        if empty:
            raise ValueError(f"required text fields are empty: {', '.join(empty)}")
        if self.task == SemanticTask.ANSWER and not self.question:
            raise ValueError("answer rows require a question")
        if (
            self.task in {SemanticTask.SUPPORT, SemanticTask.TEMPORAL}
            and not self.evidence_passages
        ):
            raise ValueError(f"{self.task} rows require evidence")
        if self.task == SemanticTask.RELEVANCE and (
            not self.question or not self.evidence_passages
        ):
            raise ValueError("relevance rows require a question and evidence")
        if self.task == SemanticTask.ANSWERABILITY and not self.question:
            raise ValueError("answerability rows require a question")
        if self.task == SemanticTask.CITATION and (
            not self.question or not self.evidence_passages
        ):
            raise ValueError("citation rows require question and evidence")
        dumped = self.model_dump(mode="json", exclude={"content_hash", "record_hash"})
        expected_content = semantic_content_hash(dumped)
        if self.content_hash != expected_content:
            raise ValueError("content_hash does not match normalized semantic content")
        expected_record = semantic_record_hash(dumped)
        if self.record_hash != expected_record:
            raise ValueError("record_hash does not match the normalized training record")
        return self

    @classmethod
    def create(cls, **values: Any) -> SemanticRecord:
        values = dict(values)
        values.pop("content_hash", None)
        values.pop("record_hash", None)
        normalized = cls.model_construct(**values).model_dump(
            mode="json",
            exclude={"content_hash", "record_hash"},
        )
        values["content_hash"] = semantic_content_hash(normalized)
        values["record_hash"] = semantic_record_hash(normalized)
        return cls.model_validate(values)


def semantic_content_hash(values: dict[str, Any]) -> str:
    """Hash only model-visible inputs, deliberately excluding supervision.

    This hash is used for leakage and conflicting-label checks. Two records with
    identical inputs but different labels must collide here.
    """
    payload = _semantic_payload(values)
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_record_hash(values: dict[str, Any]) -> str:
    """Hash immutable provenance and supervision while excluding split assignment."""
    excluded = {"content_hash", "record_hash", "partition"}
    payload = _normalize_json({key: value for key, value in values.items() if key not in excluded})
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_unit_id(*parts: object) -> str:
    joined = "\x1f".join(str(part).strip() for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
    return f"sl_{digest}"


def stable_group_bucket(group_id: str, *, salt: str, modulus: int = 10_000) -> int:
    if modulus < 1:
        raise ValueError("modulus must be positive")
    digest = hashlib.sha256(f"{salt}\x1f{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _semantic_payload(values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": values.get("task"),
        "question": values.get("question"),
        "query_time_or_interval": values.get("query_time_or_interval"),
        "temporal_operator": values.get("temporal_operator"),
        "reference_answers": values.get("reference_answers", []),
        "candidate_or_claim": values.get("candidate_or_claim", ""),
        "evidence_texts": _visible_texts(values.get("evidence_passages", [])),
        "citation_targets": _citation_targets(
            values.get("citations", []),
            values.get("evidence_passages", []),
        ),
        "graph_path_texts": _visible_texts(values.get("graph_paths", [])),
        "language": values.get("language", "en"),
    }
    return _normalize_json(payload)


def _visible_texts(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        if isinstance(value, BaseModel):
            text = getattr(value, "text", "")
        elif isinstance(value, dict):
            text = value.get("text", "")
        else:
            text = ""
        if isinstance(text, str) and text.strip():
            result.append(text)
    return result


def _citation_targets(citations: Any, evidence: Any) -> list[int | str]:
    evidence_ids: dict[str, int] = {}
    for index, value in enumerate(evidence or []):
        if isinstance(value, BaseModel):
            evidence_id = getattr(value, "evidence_id", "")
        elif isinstance(value, dict):
            evidence_id = value.get("evidence_id", "")
        else:
            evidence_id = ""
        if isinstance(evidence_id, str) and evidence_id:
            evidence_ids[evidence_id] = index
    return [
        evidence_ids.get(str(citation), f"unresolved:{citation}")
        for citation in citations or []
    ]


def _normalize_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize_json(value.model_dump(mode="json"))
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value
