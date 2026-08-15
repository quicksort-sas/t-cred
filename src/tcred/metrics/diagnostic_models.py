from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tcred.metrics.models import MetricInput
from tcred.metrics.task_judge_models import TaskJudgeInput

DiagnosticTestType = Literal["directional", "invariance"]
DiagnosticConstruct = Literal[
    "answer_correctness",
    "temporal_correctness",
    "temporal_attribution",
    "evidence_support",
    "citation_correctness",
    "graph_sufficiency",
    "response_decision",
    "retrieval_quality",
]


class DiagnosticCase(BaseModel):
    """One fully rendered evaluator input with a formal, traceable oracle."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    source_answer_id: str | None = None
    transformation: str
    oracle_labels: dict[str, str]
    oracle_basis: str
    changed_components: list[str] = Field(default_factory=list)
    metric_input: MetricInput
    task_judge_input: TaskJudgeInput

    @model_validator(mode="after")
    def aligned_inputs(self) -> DiagnosticCase:
        if self.metric_input.metric_id != self.case_id:
            raise ValueError("Metric input ID does not match diagnostic case ID")
        if self.task_judge_input.metric_id != self.case_id:
            raise ValueError("Task-judge input ID does not match diagnostic case ID")
        shared = (
            "population",
            "dataset_family",
            "source_kind",
            "qid",
            "scenario_id",
            "question",
            "reference_answer",
            "candidate_answer",
        )
        for field in shared:
            if getattr(self.metric_input, field) != getattr(self.task_judge_input, field):
                raise ValueError(f"Diagnostic inputs disagree on {field}: {self.case_id}")
        if self.metric_input.population != "diagnostic_challenge":
            raise ValueError("Diagnostic cases must use population=diagnostic_challenge")
        return self


class DiagnosticPair(BaseModel):
    """A local behavioral test whose expected score relation is known in advance."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    test_type: DiagnosticTestType
    target_construct: DiagnosticConstruct
    phenomenon: str
    dataset_family: str
    qid: str
    scenario_id: str
    left_case_id: str
    right_case_id: str
    expected_relation: Literal["left_higher", "equal"]
    target_fields: list[str]
    changed_components: list[str]
    severity_left: int | None = None
    severity_right: int | None = None
    oracle_basis: str

    @model_validator(mode="after")
    def validate_relation(self) -> DiagnosticPair:
        if self.left_case_id == self.right_case_id:
            raise ValueError("A diagnostic pair must contain two different cases")
        if self.test_type == "directional" and self.expected_relation != "left_higher":
            raise ValueError("Directional tests must expect left_higher")
        if self.test_type == "invariance" and self.expected_relation != "equal":
            raise ValueError("Invariance tests must expect equality")
        return self


class DiagnosticSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    seed: int
    source_split: str = "test_auto"
    pair_cap_per_phenomenon: int
    dataset_content_hashes: dict[str, str]
    cases: list[DiagnosticCase]
    pairs: list[DiagnosticPair]
    audit: dict[str, object]

    @model_validator(mode="after")
    def validate_references(self) -> DiagnosticSuite:
        if not self.source_split.strip():
            raise ValueError("Diagnostic source_split cannot be empty")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Diagnostic suite contains duplicate case IDs")
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Diagnostic suite contains duplicate pair IDs")
        known = set(case_ids)
        for pair in self.pairs:
            if pair.left_case_id not in known or pair.right_case_id not in known:
                raise ValueError(f"Diagnostic pair references an unknown case: {pair.pair_id}")
        return self


def diagnostic_inference_cluster_ids(
    cases: list[DiagnosticCase],
    pairs: list[DiagnosticPair],
) -> dict[str, str]:
    """Group pairs by connected source scenarios for dependency-preserving inference."""

    case_nodes = {
        case.case_id: (
            case.metric_input.dataset_family,
            case.metric_input.scenario_id,
        )
        for case in cases
    }
    parent = {node: node for node in case_nodes.values()}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            node, parent[node] = parent[node], root
        return root

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    for pair in pairs:
        left = case_nodes[pair.left_case_id]
        right = case_nodes[pair.right_case_id]
        if left[0] != pair.dataset_family or right[0] != pair.dataset_family:
            raise ValueError(f"Diagnostic pair crosses dataset families: {pair.pair_id}")
        union(left, right)

    return {
        pair.pair_id: ":".join(find(case_nodes[pair.left_case_id]))
        for pair in pairs
    }
