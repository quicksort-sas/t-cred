from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tcred.trainable_metrics.schema import CurriculumStage


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    name: str
    enabled: bool = True
    adapter: str
    curriculum_stage: CurriculumStage
    target_train_rows: int = Field(ge=0)
    dataset_id: str | None = None
    dataset_config: str | None = None
    revision: str
    loader: Literal["huggingface", "json", "jsonl", "csv", "project"]
    local_files: dict[str, str] = Field(default_factory=dict)
    split_map: dict[str, str | list[str]] = Field(default_factory=dict)
    license_id: str
    license_url: str
    source_url: str
    group_policy: str
    declared_language: Literal["en"] = "en"
    redistribution: Literal[
        "allowed_with_attribution",
        "allowed_share_alike",
        "noncommercial_research_only",
        "adapters_and_hashes_only",
    ]
    terms_required: bool = False
    terms_acceptance_env: str | None = None
    balance_labels: bool = False
    evaluation_only_splits: list[str] = Field(default_factory=list)
    streaming_shuffle_seed: int | None = None
    streaming_shuffle_buffer_rows: int | None = Field(default=None, ge=1_000)
    streaming_take_rows_per_split: int | None = Field(default=None, ge=1_000)
    notes: str = ""

    @model_validator(mode="after")
    def validate_locator(self) -> SourceConfig:
        if self.loader == "huggingface" and not self.dataset_id:
            raise ValueError("Hugging Face sources require dataset_id")
        if self.loader != "huggingface" and self.loader != "project" and not self.local_files:
            raise ValueError("file-based sources require local_files")
        if not self.revision.strip() or self.revision.lower() in {"main", "master", "latest"}:
            raise ValueError(f"source {self.name} must use an immutable revision")
        if self.terms_required and not self.terms_acceptance_env:
            raise ValueError(f"source {self.name} requires a terms-acceptance environment variable")
        stream_policy = (
            self.streaming_shuffle_seed,
            self.streaming_shuffle_buffer_rows,
            self.streaming_take_rows_per_split,
        )
        if any(value is not None for value in stream_policy) and (
            self.loader != "huggingface" or any(value is None for value in stream_policy)
        ):
            raise ValueError(
                f"source {self.name} must define the complete Hugging Face stream policy"
            )
        unknown_evaluation_splits = set(self.evaluation_only_splits) - set(self.split_map)
        if unknown_evaluation_splits:
            names = ", ".join(sorted(unknown_evaluation_splits))
            raise ValueError(f"source {self.name} has unmapped evaluation splits: {names}")
        return self


class DataBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["tcred-sl-data-build-v1"] = "tcred-sl-data-build-v1"
    seed: int = 20260816
    split_salt: str = "tcred-sl-splits-v1"
    development_fraction_without_upstream_dev: float = Field(default=0.08, ge=0, le=0.25)
    calibration_fraction_without_upstream_dev: float = Field(default=0.02, ge=0, le=0.10)
    calibration_fraction_of_upstream_dev: float = Field(default=0.25, gt=0, lt=1)
    near_duplicate_jaccard_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
    near_duplicate_candidate_threshold: float = Field(default=0.60, ge=0.1, le=1.0)
    near_duplicate_num_perm: int = Field(default=128, ge=32, le=1024)
    max_rows_per_group: int = Field(default=128, ge=1)
    sources: list[SourceConfig]

    @model_validator(mode="after")
    def validate_sources(self) -> DataBuildConfig:
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("source names must be unique")
        held_out = (
            self.development_fraction_without_upstream_dev
            + self.calibration_fraction_without_upstream_dev
        )
        if held_out >= 0.5:
            raise ValueError("fallback development and calibration fractions are too large")
        if self.near_duplicate_candidate_threshold > self.near_duplicate_jaccard_threshold:
            raise ValueError(
                "near-duplicate candidate threshold cannot exceed the exact Jaccard threshold"
            )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["tcred-sl-training-v1"] = "tcred-sl-training-v1"
    experiment_name: str = "tcred-sl-minilm-l12"
    backbone: str = "microsoft/MiniLM-L12-H384-uncased"
    backbone_revision: str
    seed: int = 42
    final_seeds: list[int] = Field(default_factory=lambda: [42, 31415, 271828])
    max_length: int = Field(default=256, ge=32, le=512)
    effective_batch_size: int = Field(default=128, ge=1)
    micro_batch_size: int = Field(default=64, ge=1)
    stage_a_epochs: int = Field(default=1, ge=0)
    stage_b_epochs: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=3e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    warmup_fraction: float = Field(default=0.06, ge=0, lt=1)
    dropout: float = Field(default=0.10, ge=0, lt=1)
    pair_margin: float = Field(default=0.20, gt=0)
    paired_loss_weight: float = Field(default=0.40, ge=0)
    invariance_loss_weight: float = Field(default=0.20, ge=0)
    calibration_loss_weight: float = Field(default=0.05, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    checkpoint_every_steps: int = Field(default=500, ge=1)
    evaluate_every_steps: int = Field(default=500, ge=1)
    monitor_rows_per_task: int = Field(default=2_000, ge=100)
    forecast_after_steps: int = Field(default=1000, ge=10)
    keep_checkpoints: int = Field(default=3, ge=1)
    early_stopping_patience: int = Field(default=4, ge=1)
    top_k_evidence: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_batching(self) -> TrainingConfig:
        if self.effective_batch_size % self.micro_batch_size != 0:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        if len(self.final_seeds) != len(set(self.final_seeds)):
            raise ValueError("final_seeds must be unique")
        if not self.backbone_revision.strip() or self.backbone_revision.lower() in {
            "main",
            "master",
            "latest",
        }:
            raise ValueError("backbone_revision must be immutable")
        return self

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.micro_batch_size


def load_yaml_model(path: Path, model: type[BaseModel]) -> BaseModel:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in the optional environment
        raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return model.model_validate(values)


def canonical_config_hash(config: BaseModel) -> str:
    payload: dict[str, Any] = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
