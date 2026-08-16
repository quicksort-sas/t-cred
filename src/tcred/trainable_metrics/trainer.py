from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from tcred.trainable_metrics.calibration import apply_temperature, fit_model_calibration
from tcred.trainable_metrics.config import TrainingConfig, canonical_config_hash
from tcred.trainable_metrics.data import (
    HomogeneousBatchSampler,
    SemanticBatchCollator,
    load_tokenized_split,
)
from tcred.trainable_metrics.evaluation import (
    EvaluationAccumulator,
    summarize_source_macro,
)
from tcred.trainable_metrics.model import TCredSLModel, save_model_bundle, semantic_loss
from tcred.trainable_metrics.reproducibility import configure_deterministic_runtime
from tcred.trainable_metrics.source_io import file_sha256


@dataclass(frozen=True)
class StageSpec:
    name: str
    value: str
    epochs: int


def train_semantic_metric(
    *,
    config: TrainingConfig,
    tokenized_dir: Path,
    backbone_dir: Path,
    output_dir: Path,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    from accelerate import Accelerator
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    _validate_output_contract(output_dir=output_dir, resume_from=resume_from)
    _set_determinism(config.seed)
    mixed_precision = _mixed_precision(config.precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("The frozen v1 protocol currently supports one GPU process per run")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenized_dir / "tokenizer", local_files_only=True, use_fast=True
    )
    model = TCredSLModel.create(
        backbone_dir=backbone_dir,
        tokenizer_size=len(tokenizer),
        dropout=config.dropout,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    stages = [
        StageSpec("Stage A", "stage_a", config.stage_a_epochs),
        StageSpec("Stage B", "stage_b", config.stage_b_epochs),
    ]
    stages = [stage for stage in stages if stage.epochs > 0]
    datasets = {
        stage.value: load_tokenized_split(
            tokenized_dir,
            partition="train",
            stage=stage.value,
        )
        for stage in stages
    }
    total_optimizer_steps = sum(
        _optimizer_steps_for_stage(
            dataset,
            epochs=stage.epochs,
            batch_size=config.micro_batch_size,
            accumulation_steps=config.gradient_accumulation_steps,
            seed=config.seed,
        )
        for stage in stages
        for dataset in [datasets[stage.value]]
    )
    warmup_steps = round(total_optimizer_steps * config.warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    run_identity = {
        "training_config_hash": canonical_config_hash(config),
        "tokenized_manifest_sha256": file_sha256(tokenized_dir / "manifest.json"),
        "backbone_model_sha256": file_sha256(backbone_dir / "model.safetensors"),
        "total_planned_steps": total_optimizer_steps,
    }
    state = {
        "schema_version": "tcred-sl-trainer-state-v1",
        **run_identity,
        "stage_index": 0,
        "epoch": 0,
        "batch_index": 0,
        "global_step": 0,
        "best_objective": None,
        "bad_evaluations": 0,
    }
    resumed_rng_state: dict[str, Any] | None = None
    if resume_from is not None:
        resumed_state = _load_trainer_state(resume_from)
        _validate_resume_identity(resumed_state, expected=run_identity)
        state.update(resumed_state)
        accelerator.load_state(str(resume_from / "accelerate_state"))
        resumed_rng_state = _capture_rng_state()

    log_path = output_dir / "training_log.jsonl"
    run_started = time.monotonic()
    stop_training = False
    for stage_index, stage in enumerate(stages):
        if stage_index < int(state["stage_index"]):
            continue
        dataset = datasets[stage.value]
        monitor = _monitor_subset(dataset, rows_per_task=config.monitor_rows_per_task)
        monitor_loader = _data_loader(
            monitor,
            tokenizer=tokenizer,
            batch_size=config.micro_batch_size,
            seed=config.seed,
            shuffle=False,
            num_workers=0,
            prefetch_factor=config.prefetch_factor,
            DataLoader=DataLoader,
        )
        monitor_loader = accelerator.prepare(monitor_loader)
        start_epoch = int(state["epoch"]) if stage_index == int(state["stage_index"]) else 0
        for epoch in range(start_epoch, stage.epochs):
            sampler = _sampler_for(
                dataset,
                batch_size=config.micro_batch_size,
                seed=config.seed,
                shuffle=True,
            )
            sampler.set_epoch(epoch)
            loader = _data_loader(
                dataset,
                tokenizer=tokenizer,
                batch_size=config.micro_batch_size,
                seed=config.seed,
                shuffle=True,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor,
                DataLoader=DataLoader,
                sampler=sampler,
            )
            loader = accelerator.prepare(loader)
            skip_batches = (
                int(state["batch_index"])
                if stage_index == int(state["stage_index"]) and epoch == start_epoch
                else 0
            )
            if skip_batches:
                loader = accelerator.skip_first_batches(loader, skip_batches)
            model.train()
            loader_iterator = iter(loader)
            if resumed_rng_state is not None:
                # Accelerate's skipped-loader wrapper lazily creates its underlying iterator and
                # consumes Torch RNG on the first fetch. Prefetch the deterministic tokenized batch,
                # then restore checkpoint RNG before dropout or any optimizer computation.
                first_batch = next(loader_iterator, None)
                _restore_rng_state(resumed_rng_state)
                resumed_rng_state = None
                batch_iterator = chain((), loader_iterator) if first_batch is None else chain(
                    (first_batch,), loader_iterator
                )
            else:
                batch_iterator = loader_iterator
            for local_batch_index, batch in enumerate(batch_iterator, start=skip_batches):
                with accelerator.accumulate(model):
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
                    accelerator.backward(loss.total)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                state.update(
                    {
                        "stage_index": stage_index,
                        "epoch": epoch,
                        "batch_index": local_batch_index + 1,
                    }
                )
                if accelerator.sync_gradients:
                    state["global_step"] = int(state["global_step"]) + 1
                    _append_log(
                        log_path,
                        {
                            "event": "train_step",
                            "time": datetime.now(UTC),
                            "stage": stage.value,
                            "epoch": epoch,
                            "global_step": state["global_step"],
                            "task": batch["task"],
                            "source_dataset": batch["source_dataset"],
                            "loss": float(loss.total.detach().float().cpu()),
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                    )
                    if int(state["global_step"]) == config.forecast_after_steps:
                        _write_forecast(
                            output_dir / "runtime_forecast.json",
                            elapsed_seconds=time.monotonic() - run_started,
                            completed_steps=int(state["global_step"]),
                            total_steps=total_optimizer_steps,
                        )
                    if int(state["global_step"]) % config.evaluate_every_steps == 0:
                        metrics = _evaluate(
                            model,
                            monitor_loader,
                            accelerator=accelerator,
                            config=config,
                        )
                        _append_log(
                            log_path,
                            {
                                "event": "monitor_evaluation",
                                "time": datetime.now(UTC),
                                "stage": stage.value,
                                "epoch": epoch,
                                "global_step": state["global_step"],
                                "metrics": metrics,
                            },
                        )
                        if stage.value == "stage_b":
                            improved, stop_training = _update_early_stopping(
                                state,
                                objective=metrics["harmonic_normalized_objective"],
                                patience=config.early_stopping_patience,
                            )
                            if improved and accelerator.is_main_process:
                                save_model_bundle(
                                    accelerator.unwrap_model(model),
                                    output_dir / "best_monitor_model",
                                    metadata={
                                        "stage": stage.value,
                                        "global_step": int(state["global_step"]),
                                        "objective": metrics[
                                            "harmonic_normalized_objective"
                                        ],
                                    },
                                )
                        model.train()
                    if int(state["global_step"]) % config.checkpoint_every_steps == 0:
                        _save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            output_dir=output_dir,
                            state=state,
                            config=config,
                        )
                if stop_training:
                    break
            state["batch_index"] = 0
            state["epoch"] = epoch + 1
            if stop_training:
                break
        best_monitor_path = output_dir / "best_monitor_model" / "model.safetensors"
        if stage.value == "stage_b" and best_monitor_path.is_file():
            _restore_best_monitor_model(
                accelerator.unwrap_model(model),
                best_monitor_path,
            )
        full_dev = load_tokenized_split(
            tokenized_dir,
            partition="development",
            stage=stage.value,
        )
        full_dev_loader = _data_loader(
            full_dev,
            tokenizer=tokenizer,
            batch_size=config.micro_batch_size,
            seed=config.seed,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            DataLoader=DataLoader,
        )
        full_dev_loader = accelerator.prepare(full_dev_loader)
        stage_metrics = _evaluate(
            model,
            full_dev_loader,
            accelerator=accelerator,
            config=config,
        )
        if accelerator.is_main_process:
            _write_json(output_dir / f"development_{stage.value}.json", stage_metrics)
            unwrapped = accelerator.unwrap_model(model)
            save_model_bundle(
                unwrapped,
                output_dir / f"checkpoint_{stage.value}_final",
                metadata={
                    "training_config_hash": canonical_config_hash(config),
                    "stage": stage.value,
                    "global_step": int(state["global_step"]),
                },
            )
        accelerator.wait_for_everyone()
        state.update(
            {
                "stage_index": stage_index + 1,
                "epoch": 0,
                "batch_index": 0,
                "bad_evaluations": 0,
                "best_objective": None,
            }
        )
        stop_training = False

    if accelerator.is_main_process:
        final_dir = output_dir / "final_model"
        save_model_bundle(
            accelerator.unwrap_model(model),
            final_dir,
            metadata={
                "training_config_hash": canonical_config_hash(config),
                "global_step": int(state["global_step"]),
                "total_planned_steps": total_optimizer_steps,
            },
        )
        shutil.copytree(tokenized_dir / "tokenizer", final_dir / "tokenizer", dirs_exist_ok=True)
        calibration_sets = [
            load_tokenized_split(
                tokenized_dir,
                partition="calibration",
                stage=stage.value,
            )
            for stage in stages
        ]
        from datasets import concatenate_datasets

        calibration_dataset = concatenate_datasets(calibration_sets)
        calibration_loader = _data_loader(
            calibration_dataset,
            tokenizer=tokenizer,
            batch_size=config.micro_batch_size,
            seed=config.seed,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            DataLoader=DataLoader,
        )
        calibration_loader = accelerator.prepare(calibration_loader)
        calibration = fit_model_calibration(
            model=model,
            loader=calibration_loader,
            output_path=final_dir / "calibration.json",
        )
        calibrated_metrics = _evaluate(
            model,
            calibration_loader,
            accelerator=accelerator,
            config=config,
            calibration=calibration,
        )
        _write_json(output_dir / "calibration_evaluation.json", calibrated_metrics)
        _write_json(output_dir / "trainer_state.json", state)
        summary = {
            "schema_version": "tcred-sl-training-run-v1",
            "experiment_name": config.experiment_name,
            "training_config_hash": canonical_config_hash(config),
            "mixed_precision": mixed_precision,
            "total_planned_steps": total_optimizer_steps,
            "completed_steps": int(state["global_step"]),
            "warmup_steps": warmup_steps,
            "elapsed_seconds": time.monotonic() - run_started,
            "final_model": {
                path.relative_to(final_dir).as_posix(): {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(final_dir.rglob("*"))
                if path.is_file()
            },
        }
        _write_json(output_dir / "run_manifest.json", summary)
    accelerator.wait_for_everyone()
    return summary if accelerator.is_main_process else {}


def _data_loader(
    dataset: Any,
    *,
    tokenizer: Any,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    DataLoader: Any,
    sampler: HomogeneousBatchSampler | None = None,
) -> Any:
    sampler = sampler or _sampler_for(
        dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": SemanticBatchCollator(tokenizer),
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers:
        options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": prefetch_factor,
            }
        )
    return DataLoader(**options)


def _sampler_for(
    dataset: Any,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> HomogeneousBatchSampler:
    return HomogeneousBatchSampler(
        tasks=dataset["task"],
        sources=dataset["source_dataset"],
        pair_ids=dataset["pair_id"],
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )


def _optimizer_steps_for_stage(
    dataset: Any,
    *,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    seed: int,
) -> int:
    sampler = _sampler_for(
        dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    total = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total += math.ceil(len(sampler) / accumulation_steps)
    return total


def _monitor_subset(dataset: Any, *, rows_per_task: int) -> Any:
    packets: defaultdict[tuple[str, str], dict[str, list[int]]] = defaultdict(dict)
    for index, (task, source, pair_id) in enumerate(
        zip(
            dataset["task"],
            dataset["source_dataset"],
            dataset["pair_id"],
            strict=True,
        )
    ):
        packet_id = f"pair:{pair_id}" if pair_id else f"single:{index}"
        packets[(str(task), str(source))].setdefault(packet_id, []).append(index)

    chosen: set[int] = set()
    tasks = sorted({key[0] for key in packets})
    for task in tasks:
        sources = sorted(key[1] for key in packets if key[0] == task)
        source_packets = {
            source: list(packets[(task, source)].values()) for source in sources
        }
        cursors = {source: 0 for source in sources}
        selected_rows = 0
        while selected_rows < rows_per_task:
            progressed = False
            for source in sources:
                cursor = cursors[source]
                if cursor >= len(source_packets[source]):
                    continue
                packet = source_packets[source][cursor]
                cursors[source] += 1
                chosen.update(packet)
                selected_rows += len(packet)
                progressed = True
                if selected_rows >= rows_per_task:
                    break
            if not progressed:
                break
    return dataset.select(sorted(chosen))


def _evaluate(
    model: Any,
    loader: Any,
    *,
    accelerator: Any,
    config: TrainingConfig,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    accumulator = EvaluationAccumulator()
    source_accumulators: defaultdict[str, EvaluationAccumulator] = defaultdict(
        EvaluationAccumulator
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                task=batch["task"],
            )
            outputs = apply_temperature(
                outputs,
                task=batch["task"],
                calibration=calibration,
            )
            loss = semantic_loss(
                outputs,
                batch,
                pair_margin=config.pair_margin,
                paired_loss_weight=config.paired_loss_weight,
                invariance_loss_weight=config.invariance_loss_weight,
                calibration_loss_weight=config.calibration_loss_weight,
            )
            accumulator.add_batch(
                task=batch["task"],
                outputs=outputs,
                batch=batch,
                loss=float(loss.total.detach().float().cpu()),
            )
            source_accumulators[str(batch["source_dataset"])].add_batch(
                task=batch["task"],
                outputs=outputs,
                batch=batch,
                loss=float(loss.total.detach().float().cpu()),
            )
    result = accumulator.compute()
    source_reports = {
        source: value.compute() for source, value in sorted(source_accumulators.items())
    }
    source_macro = summarize_source_macro(source_reports)
    result["row_macro_harmonic_normalized_objective"] = result[
        "harmonic_normalized_objective"
    ]
    result["source_family_macro"] = source_macro
    result["source_family_reports"] = source_reports
    result["harmonic_normalized_objective"] = source_macro[
        "harmonic_normalized_objective"
    ]
    return result


def _update_early_stopping(
    state: dict[str, Any],
    *,
    objective: float | None,
    patience: int,
) -> tuple[bool, bool]:
    if objective is None:
        return False, False
    best = state.get("best_objective")
    if best is None or objective > float(best) + 1e-6:
        state["best_objective"] = objective
        state["bad_evaluations"] = 0
        return True, False
    state["bad_evaluations"] = int(state["bad_evaluations"]) + 1
    return False, int(state["bad_evaluations"]) >= patience


def _restore_best_monitor_model(model: Any, path: Path) -> None:
    from safetensors.torch import load_file

    if not path.is_file():
        raise FileNotFoundError(f"Early stopping requested but no best model exists: {path}")
    model.load_state_dict(load_file(str(path)), strict=True)


def _save_checkpoint(
    *,
    accelerator: Any,
    model: Any,
    output_dir: Path,
    state: dict[str, Any],
    config: TrainingConfig,
) -> None:
    checkpoint = output_dir / "checkpoints" / f"step_{int(state['global_step']):08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint / "accelerate_state"), safe_serialization=True)
    if accelerator.is_main_process:
        _write_json(checkpoint / "trainer_state.json", state)
        save_model_bundle(
            accelerator.unwrap_model(model),
            checkpoint / "model_bundle",
            metadata={"global_step": int(state["global_step"])},
        )
        _prune_checkpoints(output_dir / "checkpoints", keep=config.keep_checkpoints)
    accelerator.wait_for_everyone()


def _prune_checkpoints(root: Path, *, keep: int) -> None:
    checkpoints = sorted(
        (path for path in root.glob("step_*" ) if path.is_dir()),
        key=lambda path: path.name,
    )
    root_resolved = root.resolve()
    for path in checkpoints[:-keep]:
        resolved = path.resolve()
        if resolved.parent != root_resolved:
            raise RuntimeError(f"Refusing to prune checkpoint outside {root_resolved}: {resolved}")
        shutil.rmtree(resolved)


def _load_trainer_state(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "trainer_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint has no trainer state: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Invalid trainer state: {path}")
    return values


def _validate_output_contract(*, output_dir: Path, resume_from: Path | None) -> None:
    if resume_from is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Training output is not empty; use a new run directory: {output_dir}"
            )
        return
    expected_root = resume_from.resolve().parent.parent
    if output_dir.resolve() != expected_root:
        raise ValueError(
            "Resume checkpoint must belong to the selected output directory: "
            f"checkpoint_root={expected_root}, output_dir={output_dir.resolve()}"
        )


def _validate_resume_identity(
    state: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> None:
    if state.get("schema_version") != "tcred-sl-trainer-state-v1":
        raise ValueError("Resume checkpoint uses an unsupported trainer-state schema")
    mismatches = {
        name: {"checkpoint": state.get(name), "current": value}
        for name, value in expected.items()
        if state.get(name) != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={values['checkpoint']!r}->{values['current']!r}"
            for name, values in sorted(mismatches.items())
        )
        raise ValueError(f"Resume checkpoint identity mismatch: {details}")


def _capture_rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _mixed_precision(requested: str) -> str:
    import torch

    if requested != "auto":
        return "no" if requested == "fp32" else requested
    if not torch.cuda.is_available():
        return "no"
    major, _ = torch.cuda.get_device_capability()
    return "bf16" if major >= 8 else "fp16"


def _set_determinism(seed: int) -> None:
    import torch

    configure_deterministic_runtime(seed, torch=torch)


def _write_forecast(
    path: Path,
    *,
    elapsed_seconds: float,
    completed_steps: int,
    total_steps: int,
) -> None:
    projected = elapsed_seconds / completed_steps * total_steps * 1.20
    _write_json(
        path,
        {
            "schema_version": "tcred-sl-runtime-forecast-v1",
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "elapsed_seconds": elapsed_seconds,
            "projected_seconds_with_20_percent_contingency": projected,
            "projected_hours_with_20_percent_contingency": projected / 3600,
        },
    )


def _append_log(path: Path, value: dict[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
