from __future__ import annotations

import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

from tcred.trainable_metrics.artifacts import file_sha256
from tcred.trainable_metrics.calibration import apply_temperature
from tcred.trainable_metrics.formatting import (
    assert_no_prohibited_metadata,
    format_semantic_fields,
    formatted_text_hash,
)
from tcred.trainable_metrics.model import load_model_bundle
from tcred.trainable_metrics.preprocessing import truncate_preserving_terminal_special
from tcred.trainable_metrics.schema import EvidencePassage, GraphPathText, SemanticTask

CLASS_NAMES = {
    SemanticTask.SUPPORT: ("entailment", "neutral", "contradiction"),
    SemanticTask.TEMPORAL: ("support", "unknown", "contradiction"),
    SemanticTask.CITATION: ("appropriate", "incomplete", "inappropriate"),
}


class SemanticInferenceInput(BaseModel):
    """Source-blind model-visible input with no target or dataset identity."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    input_id: str
    task: SemanticTask
    question: str | None = None
    query_time_or_interval: str | None = None
    temporal_operator: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    candidate_or_claim: str
    evidence_passages: list[EvidencePassage] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    graph_paths: list[GraphPathText] = Field(default_factory=list)

    def formatted_text(self) -> str:
        return format_semantic_fields(
            task=self.task,
            question=self.question,
            query_time_or_interval=self.query_time_or_interval,
            temporal_operator=self.temporal_operator,
            reference_answers=self.reference_answers,
            candidate_or_claim=self.candidate_or_claim,
            evidence_passages=self.evidence_passages,
            citations=self.citations,
            graph_paths=self.graph_paths,
        )


class SemanticPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str
    task: SemanticTask
    values: dict[str, float]
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    formatted_text_sha256: str
    original_token_length: int = Field(ge=1)
    encoded_token_length: int = Field(ge=1)
    was_truncated: bool


class TCredSLInference:
    """Frozen, calibrated, batched inference for one T-CRED-SL model bundle."""

    def __init__(
        self,
        *,
        model_dir: Path,
        backbone_dir: Path,
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoTokenizer

        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        self.model_dir = model_dir.resolve()
        self.backbone_dir = backbone_dir.resolve()
        self.max_length = max_length
        self.model_config = _read_object(self.model_dir / "model_config.json")
        self.calibration = _read_object(self.model_dir / "calibration.json")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / "tokenizer",
            local_files_only=True,
            use_fast=True,
        )
        expected_tokenizer_size = int(self.model_config["tokenizer_size"])
        if len(self.tokenizer) != expected_tokenizer_size:
            raise ValueError(
                "Tokenizer size does not match the trained model: "
                f"{len(self.tokenizer)} != {expected_tokenizer_size}"
            )
        self.device = _resolve_device(device, torch=torch)
        self.model = load_model_bundle(
            bundle_dir=self.model_dir,
            backbone_dir=self.backbone_dir,
            tokenizer_size=len(self.tokenizer),
            dropout=float(self.model_config["dropout"]),
        )
        self.model.to(self.device)
        self.model.eval()
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.runtime = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(self.device),
            "max_length": self.max_length,
            "model_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "model_weight_sha256": file_sha256(self.model_dir / "model.safetensors"),
            "calibration_sha256": file_sha256(self.model_dir / "calibration.json"),
        }

    def predict(
        self,
        rows: list[SemanticInferenceInput],
        *,
        batch_size: int = 64,
    ) -> tuple[list[SemanticPrediction], dict[str, Any]]:
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not rows:
            return [], {"rows": 0, "batches": 0, "elapsed_seconds": 0.0}
        identities = [row.input_id for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError("Semantic inference input IDs must be unique")
        by_task: defaultdict[SemanticTask, list[tuple[int, SemanticInferenceInput]]] = defaultdict(
            list
        )
        for index, row in enumerate(rows):
            by_task[SemanticTask(row.task)].append((index, row))

        predictions: list[SemanticPrediction | None] = [None] * len(rows)
        started = time.perf_counter()
        batches = 0
        with torch.inference_mode():
            for task in SemanticTask:
                selected = by_task.get(task, [])
                for offset in range(0, len(selected), batch_size):
                    batch = selected[offset : offset + batch_size]
                    texts = [row.formatted_text() for _, row in batch]
                    assert_no_prohibited_metadata(texts)
                    encoded = self.tokenizer(
                        texts,
                        add_special_tokens=True,
                        padding=False,
                        truncation=False,
                        return_attention_mask=False,
                    )
                    original_ids = encoded["input_ids"]
                    input_ids = [
                        truncate_preserving_terminal_special(
                            list(token_ids),
                            max_length=self.max_length,
                            special_ids=self.special_ids,
                        )
                        for token_ids in original_ids
                    ]
                    tensors = self.tokenizer.pad(
                        {"input_ids": input_ids},
                        padding=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    outputs = self.model(
                        input_ids=tensors["input_ids"].to(self.device),
                        attention_mask=tensors["attention_mask"].to(self.device),
                        task=str(task),
                    )
                    outputs = apply_temperature(
                        outputs,
                        task=str(task),
                        calibration=self.calibration,
                    )
                    unpacked = _unpack_outputs(outputs, task=task, count=len(batch))
                    for batch_index, (source_index, row) in enumerate(batch):
                        values, classes = unpacked[batch_index]
                        predictions[source_index] = SemanticPrediction(
                            input_id=row.input_id,
                            task=task,
                            values=values,
                            class_probabilities=classes,
                            formatted_text_sha256=formatted_text_hash(texts[batch_index]),
                            original_token_length=len(original_ids[batch_index]),
                            encoded_token_length=len(input_ids[batch_index]),
                            was_truncated=len(original_ids[batch_index]) > self.max_length,
                        )
                    batches += 1
        result = [prediction for prediction in predictions if prediction is not None]
        if len(result) != len(rows):
            raise RuntimeError("Inference failed to produce one prediction per input")
        return result, {
            "rows": len(rows),
            "batches": batches,
            "batch_size": batch_size,
            "elapsed_seconds": time.perf_counter() - started,
            "truncated_rows": sum(row.was_truncated for row in result),
            "runtime": self.runtime,
        }


def _unpack_outputs(
    outputs: dict[str, Any],
    *,
    task: SemanticTask,
    count: int,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    def values(name: str) -> list[float]:
        return [float(value) for value in outputs[name].detach().float().cpu().tolist()]

    if task == SemanticTask.ANSWER:
        columns = {name: values(name) for name in ("u1", "u2", "equivalence", "score")}
        return [
            (
                {
                    "answer_u1": columns["u1"][index],
                    "answer_u2": columns["u2"][index],
                    "equivalence": columns["equivalence"][index],
                    "score": columns["score"][index],
                },
                {},
            )
            for index in range(count)
        ]
    if task in {SemanticTask.SUPPORT, SemanticTask.TEMPORAL}:
        supported = values("supported")
        probabilities = outputs["class_probabilities"].detach().float().cpu().tolist()
        names = CLASS_NAMES[task]
        return [
            (
                {"supported": supported[index]},
                {
                    name: float(probability)
                    for name, probability in zip(names, probabilities[index], strict=True)
                },
            )
            for index in range(count)
        ]
    if task == SemanticTask.RELEVANCE:
        relevance = values("relevance")
        return [({"relevance": relevance[index]}, {}) for index in range(count)]
    if task == SemanticTask.ANSWERABILITY:
        answerable = values("answerable")
        return [({"answerable": answerable[index]}, {}) for index in range(count)]
    if task == SemanticTask.CITATION:
        probabilities = outputs["class_probabilities"].detach().float().cpu().tolist()
        names = CLASS_NAMES[task]
        return [
            (
                {},
                {
                    name: float(probability)
                    for name, probability in zip(names, probabilities[index], strict=True)
                },
            )
            for index in range(count)
        ]
    raise ValueError(f"Unknown semantic task: {task}")


def _resolve_device(value: str, *, torch: Any) -> Any:
    normalized = value.strip().casefold()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
    return device


def _read_object(path: Path) -> dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value
