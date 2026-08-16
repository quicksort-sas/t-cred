from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from tcred.trainable_metrics.config import TrainingConfig, canonical_config_hash
from tcred.trainable_metrics.data import (
    HomogeneousBatchSampler,
    SemanticBatchCollator,
    load_tokenized_split,
)
from tcred.trainable_metrics.model import TCredSLModel, semantic_loss
from tcred.trainable_metrics.reproducibility import (
    configure_deterministic_runtime,
    deterministic_runtime_snapshot,
)
from tcred.trainable_metrics.source_io import file_sha256


def run_gpu_training_smoke(
    *,
    config: TrainingConfig,
    tokenized_dir: Path,
    backbone_dir: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one representative CUDA optimization step without retaining model weights."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"GPU smoke report already exists: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to report a GPU training smoke pass")
    if torch.cuda.device_count() != 1:
        device_count = torch.cuda.device_count()
        raise RuntimeError(
            f"The frozen protocol requires exactly one visible GPU, found {device_count}"
        )

    configure_deterministic_runtime(config.seed, torch=torch)
    autocast_dtype = _autocast_dtype(config.precision, torch=torch)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenized_dir / "tokenizer",
        local_files_only=True,
        use_fast=True,
    )
    dataset = load_tokenized_split(tokenized_dir, partition="train", stage="stage_a")
    sampler = HomogeneousBatchSampler(
        tasks=dataset["task"],
        sources=dataset["source_dataset"],
        pair_ids=dataset["pair_id"],
        batch_size=config.micro_batch_size,
        seed=config.seed,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=SemanticBatchCollator(tokenizer),
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(loader), None)
    if batch is None:
        raise RuntimeError("The Stage A training split produced no smoke-test batch")
    batch = {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }

    model = TCredSLModel.create(
        backbone_dir=backbone_dir,
        tokenizer_size=len(tokenizer),
        dropout=config.dropout,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()

    with torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            task=batch["task"],
        )
        loss = semantic_loss(
            outputs,
            batch,
            pair_margin=config.pair_margin,
            paired_loss_weight=config.paired_loss_weight,
            invariance_loss_weight=config.invariance_loss_weight,
            calibration_loss_weight=config.calibration_loss_weight,
        )
    if not torch.isfinite(loss.total):
        value = loss.total.detach().float().cpu()
        raise FloatingPointError(f"GPU smoke loss is non-finite: {value}")
    scaler.scale(loss.total).backward()
    scaler.unscale_(optimizer)
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise FloatingPointError(
            f"GPU smoke produced non-finite gradients: {nonfinite_gradients[:5]}"
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    properties = torch.cuda.get_device_properties(device)

    report = {
        "schema_version": "tcred-sl-gpu-training-smoke-v1",
        "status": "passed",
        "recorded_at": datetime.now(UTC),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": {
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "determinism": deterministic_runtime_snapshot(torch=torch),
        "device": {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "input_identity": {
            "training_config_hash": canonical_config_hash(config),
            "tokenized_manifest_sha256": file_sha256(tokenized_dir / "manifest.json"),
            "backbone_model_sha256": file_sha256(backbone_dir / "model.safetensors"),
        },
        "step": {
            "precision": config.precision,
            "gradient_scaling_enabled": scaler.is_enabled(),
            "declared_micro_batch_size": config.micro_batch_size,
            "observed_batch_size": int(batch["input_ids"].shape[0]),
            "sequence_length": int(batch["input_ids"].shape[1]),
            "task": str(batch["task"]),
            "source_dataset": str(batch["source_dataset"]),
            "loss": float(loss.total.detach().float().cpu()),
            "gradient_norm_before_clipping": float(gradient_norm.detach().float().cpu()),
            "elapsed_seconds": elapsed,
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _write_json(output_path, report)
    return report


def validate_gpu_training_smoke(
    *,
    config: TrainingConfig,
    tokenized_dir: Path,
    backbone_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Reject a stale smoke report before launching the full training process."""
    import torch

    if not report_path.is_file():
        raise FileNotFoundError(f"GPU smoke report does not exist: {report_path}")
    report = orjson.loads(report_path.read_bytes())
    if not isinstance(report, dict):
        raise ValueError("GPU smoke report root must be an object")
    expected = {
        "status": "passed",
        "training_config_hash": canonical_config_hash(config),
        "tokenized_manifest_sha256": file_sha256(tokenized_dir / "manifest.json"),
        "backbone_model_sha256": file_sha256(backbone_dir / "model.safetensors"),
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "precision": config.precision,
        "declared_micro_batch_size": config.micro_batch_size,
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms_enabled": True,
            "warn_only_enabled": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "flash_sdp_enabled": False,
            "memory_efficient_sdp_enabled": False,
            "math_sdp_enabled": True,
        },
    }
    identity = report.get("input_identity", {})
    step = report.get("step", {})
    torch_report = report.get("torch", {})
    actual = {
        "status": report.get("status"),
        "training_config_hash": identity.get("training_config_hash"),
        "tokenized_manifest_sha256": identity.get("tokenized_manifest_sha256"),
        "backbone_model_sha256": identity.get("backbone_model_sha256"),
        "torch_version": torch_report.get("version"),
        "cuda_build": torch_report.get("cuda_build"),
        "precision": step.get("precision"),
        "declared_micro_batch_size": step.get("declared_micro_batch_size"),
        "determinism": report.get("determinism"),
    }
    mismatches = {
        name: {"expected": expected_value, "actual": actual.get(name)}
        for name, expected_value in expected.items()
        if actual.get(name) != expected_value
    }
    if not torch.cuda.is_available():
        mismatches["cuda_available"] = {"expected": True, "actual": False}
    elif torch.cuda.device_count() != 1:
        mismatches["visible_gpu_count"] = {
            "expected": 1,
            "actual": torch.cuda.device_count(),
        }
    else:
        current_device = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
        reported_device = report.get("device", {})
        for field, expected_value in current_device.items():
            if reported_device.get(field) != expected_value:
                mismatches[f"device_{field}"] = {
                    "expected": expected_value,
                    "actual": reported_device.get(field),
                }
    if mismatches:
        raise RuntimeError(f"GPU smoke report is stale or incompatible: {mismatches}")
    return report


def _autocast_dtype(precision: str, *, torch: Any) -> Any | None:
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but the visible CUDA device does not support it")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
