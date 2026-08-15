from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import orjson

from tcred.metrics.task_judge_models import TaskJudgeInput
from tcred.metrics.tcred_claims import decompose_claims
from tcred.metrics.tcred_models import TCredSemanticRecord


@dataclass(frozen=True)
class SemanticEvidence:
    kind: str
    evidence_id: str
    text: str


def semantic_evidence(input_row: TaskJudgeInput) -> list[SemanticEvidence]:
    """Return stable, non-concatenated evidence items for semantic link scoring."""

    output: list[SemanticEvidence] = []
    seen_textual: set[tuple[str, str]] = set()
    for kind, rows in (
        ("retrieved", input_row.retrieved_evidence),
        ("cited", input_row.cited_evidence),
    ):
        for row in rows:
            key = (kind, row.evidence_id)
            if key in seen_textual:
                continue
            seen_textual.add(key)
            output.append(SemanticEvidence(kind, row.evidence_id, row.text))
    for path in input_row.graph_paths:
        for index, edge in enumerate(path.edges):
            evidence_id = path_edge_evidence_id(path.path_id, index, edge.fact_id)
            text = edge.evidence_text.strip()
            if not text:
                text = (
                    f"{edge.source.label} {edge.relation_label or edge.relation} "
                    f"{edge.target.label}."
                )
            output.append(SemanticEvidence("path", evidence_id, text))
    return output


def path_edge_evidence_id(path_id: str, edge_index: int, fact_id: str) -> str:
    return f"{path_id}:edge:{edge_index}:{fact_id}"


def semantic_claims(input_row: TaskJudgeInput) -> tuple[list[str], list[str]]:
    """Decompose answers using only evidence visible in the public metric input.

    Some concise set-valued answers use commas that are also valid inside entity names and
    titles. The deterministic decomposer may resolve that ambiguity only when distinct displayed
    evidence items support the proposed members. Keeping this helper shared prevents the scorer,
    cache hash, and neural worker from silently using different claim boundaries.
    """

    decomposition_evidence = [
        (row.evidence_id, row.text) for row in semantic_evidence(input_row)
    ]
    return (
        decompose_claims(input_row.candidate_answer, evidence=decomposition_evidence),
        decompose_claims(input_row.reference_answer, evidence=decomposition_evidence),
    )


def semantic_input_payload(input_row: TaskJudgeInput) -> dict[str, object]:
    candidate_claims, reference_claims = semantic_claims(input_row)
    return {
        "metric_id": input_row.metric_id,
        "candidate_claims": candidate_claims,
        "reference_claims": reference_claims,
        "evidence": [row.__dict__ for row in semantic_evidence(input_row)],
    }


def semantic_input_hash(input_row: TaskJudgeInput) -> str:
    payload = orjson.dumps(semantic_input_payload(input_row), option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


def run_semantic_worker(
    *,
    inputs_path: Path,
    output_path: Path,
    cache_path: Path,
    model_cache_dir: Path,
    metric_python: Path,
    batch_size: int = 16,
) -> None:
    """Run pairwise AlignScore in its compatible 64-bit environment."""

    if batch_size <= 0:
        raise ValueError("Semantic batch size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME", "VIRTUAL_ENV"}
    }
    environment["PYTHONPATH"] = str(source_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if _alignscore_is_cached(model_cache_dir):
        environment["HF_HUB_OFFLINE"] = "1"
    command = [
        str(metric_python),
        "-m",
        "tcred.metrics.tcred_semantic_worker",
        "--inputs",
        str(inputs_path),
        "--output",
        str(output_path),
        "--cache",
        str(cache_path),
        "--model-cache-dir",
        str(model_cache_dir),
        "--batch-size",
        str(batch_size),
    ]
    completed = subprocess.run(
        command,
        check=False,
        env=environment,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "T-CRED semantic worker failed.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def read_semantic_records(path: Path) -> dict[str, TCredSemanticRecord]:
    output: dict[str, TCredSemanticRecord] = {}
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                record = TCredSemanticRecord.model_validate(orjson.loads(line))
                if record.metric_id in output:
                    raise ValueError(f"Duplicate semantic record: {record.metric_id}")
                output[record.metric_id] = record
    return output


def validate_semantic_records(
    inputs: Sequence[TaskJudgeInput],
    records: dict[str, TCredSemanticRecord],
) -> None:
    expected = {row.metric_id for row in inputs}
    if set(records) != expected:
        missing = sorted(expected - set(records))[:5]
        extra = sorted(set(records) - expected)[:5]
        raise ValueError(f"Semantic record mismatch: missing={missing}, extra={extra}")
    for row in inputs:
        if records[row.metric_id].input_sha256 != semantic_input_hash(row):
            raise ValueError(f"Stale semantic record: {row.metric_id}")


def _alignscore_is_cached(model_cache_dir: Path) -> bool:
    from tcred.metrics.config import (
        ALIGNSCORE_BACKBONE_REVISION,
        ALIGNSCORE_CHECKPOINT_REVISION,
    )

    backbone = (
        model_cache_dir
        / "models--roberta-base"
        / "snapshots"
        / ALIGNSCORE_BACKBONE_REVISION
    )
    checkpoint = (
        model_cache_dir
        / "models--yzha--AlignScore"
        / "snapshots"
        / ALIGNSCORE_CHECKPOINT_REVISION
    )
    return backbone.exists() and checkpoint.exists()
