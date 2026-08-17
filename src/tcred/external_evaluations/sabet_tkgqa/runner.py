from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import string
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    epochs: int
    learning_rate: float
    expected_test_examples: int
    checkpoint_file: str = "tcomplex.ckpt"


@dataclass(frozen=True)
class ModelSpec:
    upstream_model: str
    supervision: str
    temporal_hint_mode: Literal["hard", "disabled"]
    variant: Literal["standard", "hard"]
    evidence_tier: Literal["closest_public_code", "reconstructed"]


DATASETS = {
    "wikidata_big": DatasetSpec("CronQuestions", 20, 2e-4, 30_000),
    "wikidata_big_complex": DatasetSpec("Complex-CronQuestions", 20, 2e-4, 5_006),
    "MultiTQ": DatasetSpec("MultiTQ", 20, 2e-4, 54_584),
    "timequestions": DatasetSpec("TimeQuestions", 50, 6e-4, 3_237),
}

MODELS = {
    "sabet_hard": ModelSpec("sabet", "soft", "hard", "hard", "closest_public_code"),
    "tempo_qr_hard": ModelSpec(
        "tempo_qr", "soft", "hard", "hard", "closest_public_code"
    ),
    "sabet_standard": ModelSpec(
        "sabet", "soft", "disabled", "standard", "reconstructed"
    ),
    "tempo_qr_standard": ModelSpec(
        "tempo_qr", "soft1", "hard", "standard", "reconstructed"
    ),
}

_EXPECTED_PATCH_OUTPUTS = {
    "tkg_qa_models/train_qa_model.py": (
        "a6159a72ccad8251dc2943f8e36eebbf9626142ef5496f8cd286375ac0f4ac2b"
    ),
    "tkg_qa_models/qa_baselines.py": (
        "46d0c3299c4247bd05790c0369c1ea972238224c3c8fc5daad18de730a090dd8"
    ),
    "tkg_qa_models/qa_datasets.py": (
        "1ddcb49a7d071c49e3e069cb5b98e74b82dacb7b0dbcc752ff78d54e0cab76c8"
    ),
    "tkg_qa_models/utils.py": (
        "16899a23104890cd267a79d088c366fcb7aecbcf986354a2d43ff49a37c5bc6e"
    ),
    "tkg_qa_models/hard_supervision_functions.py": (
        "886ea9f1b009ea6ebbc55c56378a320b3b2b7d8cbf4ee44879766148d2891e4c"
    ),
}

_MULTITQ_CHECKPOINT_SHA256 = (
    "fa1ca83d084deb01fae98176903b8346230e75c7c34324b03baecd6b3cb3c997"
)

_DISTILBERT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
_DISTILBERT_FILES = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": (
        "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"
    ),
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def run_experiment(
    *,
    source_root: Path,
    data_root: Path,
    runs_root: Path,
    dataset: str,
    model: str,
    seed: int,
    purpose: Literal["confirmatory", "smoke", "sensitivity"] = "confirmatory",
    epochs_override: int | None = None,
    deterministic: bool = True,
    num_workers: int = 4,
    dry_run: bool = False,
) -> dict[str, object]:
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    runs_root = runs_root.resolve()
    dataset_spec = DATASETS[dataset]
    model_spec = MODELS[model]
    epochs = dataset_spec.epochs if epochs_override is None else epochs_override
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    source_identity = _verify_instrumented_source(source_root)
    input_identity = _verify_run_inputs(data_root, dataset, dataset_spec)
    language_model_identity = _verify_distilbert_cache()
    suffix = f"__epochs{epochs}" if epochs_override is not None else ""
    run_id = f"{dataset}__{model}__seed{seed}__{purpose}{suffix}"
    run_dir = runs_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")

    predictions_file = run_dir / "predictions.jsonl"
    command = _build_command(
        dataset=dataset,
        dataset_spec=dataset_spec,
        model_spec=model_spec,
        run_id=run_id,
        seed=seed,
        epochs=epochs,
        num_workers=num_workers,
        predictions_file=predictions_file,
        deterministic=deterministic,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "SABET_DATA_DIR": str(data_root),
            "SABET_OUTPUT_DIR": str(run_dir),
            "PYTHONHASHSEED": str(seed),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    if deterministic:
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    started = datetime.now(UTC)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "dry_run" if dry_run else "running",
        "run_id": run_id,
        "purpose": purpose,
        "started_utc": started.isoformat(),
        "source_root": str(source_root),
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "dataset": dataset,
        "dataset_spec": asdict(dataset_spec),
        "model": model,
        "model_spec": asdict(model_spec),
        "seed": seed,
        "epochs": epochs,
        "epochs_overridden": epochs_override is not None,
        "deterministic_algorithms": deterministic,
        "num_workers": num_workers,
        "command": command,
        "command_display": shlex.join(command),
        "source_identity": source_identity,
        "input_identity": input_identity,
        "language_model_identity": language_model_identity,
        "environment": _environment_snapshot(),
    }
    if dry_run:
        return manifest

    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    console_path = run_dir / "console.log"
    started_monotonic = time.monotonic()
    peak_gpu_memory_mib: int | None = None
    return_code = -1
    process: subprocess.Popen[str] | None = None
    try:
        with console_path.open("w", encoding="utf-8", newline="\n") as console:
            process = subprocess.Popen(
                command,
                cwd=source_root,
                env=environment,
                stdout=console,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                memory = _gpu_memory_used_mib()
                if memory is not None:
                    peak_gpu_memory_mib = max(peak_gpu_memory_mib or 0, memory)
                time.sleep(5)
            return_code = int(process.returncode)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        manifest["launcher_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        finished = datetime.now(UTC)
        manifest.update(
            {
                "status": "process_completed" if return_code == 0 else "failed",
                "return_code": return_code,
                "finished_utc": finished.isoformat(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "peak_gpu_memory_mib": peak_gpu_memory_mib,
                "outputs": _output_identity(run_dir, dataset, run_id),
            }
        )
        _write_json(manifest_path, manifest)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    try:
        _validate_completed_outputs(manifest)
        manifest["prediction_verification"] = _verify_predictions(
            predictions_file,
            expected_count=dataset_spec.expected_test_examples,
            expected_run_id=run_id,
            expected_dataset=dataset,
            expected_model=model_spec.upstream_model,
            expected_variant=model_spec.variant,
            expected_seed=seed,
        )
        manifest["output_verification"] = {
            "verified_utc": datetime.now(UTC).isoformat(),
            "verifier_file_sha256": _sha256(Path(__file__)),
            "mode": "integrated",
        }
    except (RuntimeError, ValueError):
        manifest["status"] = "invalid_outputs"
        _write_json(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    _write_json(manifest_path, manifest)
    return manifest


def _build_command(
    *,
    dataset: str,
    dataset_spec: DatasetSpec,
    model_spec: ModelSpec,
    run_id: str,
    seed: int,
    epochs: int,
    num_workers: int,
    predictions_file: Path,
    deterministic: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tkg_qa_models.train_qa_model",
        "--tkbc_model_file",
        dataset_spec.checkpoint_file,
        "--tkg_file",
        "full.txt",
        "--model",
        model_spec.upstream_model,
        "--dataset_name",
        dataset,
        "--supervision",
        model_spec.supervision,
        "--save_to",
        run_id,
        "--max_epochs",
        str(epochs),
        "--eval_k",
        "1",
        "--valid_freq",
        "1",
        "--batch_size",
        "150",
        "--valid_batch_size",
        "150",
        "--frozen",
        "1",
        "--lm_frozen",
        "1",
        "--lr",
        str(dataset_spec.learning_rate),
        "--mode",
        "train",
        "--eval_split",
        "valid",
        "--lm",
        "distill_bert",
        "--fuse",
        "add",
        "--corrupt_hard",
        "0.0",
        "--seed",
        str(seed),
        "--num_hops",
        "4",
        "--num_workers",
        str(num_workers),
        "--temporal_hint_mode",
        model_spec.temporal_hint_mode,
        "--variant",
        model_spec.variant,
        "--predictions_file",
        str(predictions_file),
        "--run_id",
        run_id,
    ]
    if deterministic:
        command.append("--deterministic")
    return command


def _verify_instrumented_source(source_root: Path) -> dict[str, object]:
    manifest_path = source_root / "CODEX_REPRODUCTION_PATCH_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("patch_name") != "sabet-tkgqa-reproduction-instrumentation-v5":
        raise ValueError("Source is not the frozen v5 instrumented copy")
    actual: dict[str, str] = {}
    for relative_path, expected_sha256 in _EXPECTED_PATCH_OUTPUTS.items():
        value = _sha256(source_root / relative_path)
        if value != expected_sha256:
            raise ValueError(f"Instrumented source mismatch: {relative_path}")
        actual[relative_path] = value
    return {
        "patch_manifest_sha256": _sha256(manifest_path),
        "changed_file_sha256": actual,
    }


def _verify_run_inputs(
    data_root: Path,
    dataset: str,
    dataset_spec: DatasetSpec,
) -> dict[str, object]:
    checkpoint = data_root / "models" / dataset / "kg_embeddings" / dataset_spec.checkpoint_file
    dataset_root = data_root / "data" / dataset
    if dataset == "MultiTQ":
        split_root = dataset_root / "questions" / "processed_questions"
        splits = {name: split_root / f"{name}.json" for name in ("train", "dev", "test")}
        kg_root = dataset_root / "kg"
        tkbc_root = kg_root / "tkbc_processed_data" / dataset
        dictionaries = {
            "entity_dictionary": kg_root / "entity2id.json",
            "relation_dictionary": kg_root / "relation2id.json",
            "tkbc_entity_ids": tkbc_root / "ent_id",
            "tkbc_relation_ids": tkbc_root / "rel_id",
            "tkbc_timestamp_ids": tkbc_root / "ts_id",
        }
    else:
        split_root = dataset_root / "questions"
        splits = {name: split_root / f"{name}.pickle" for name in ("train", "valid", "test")}
        kg_root = dataset_root / "kg"
        tkbc_root = kg_root / "tkbc_processed_data" / dataset
        dictionaries = {
            "entity_labels": kg_root / "wd_id2entity_text.txt",
            "relation_labels": kg_root / "wd_id2relation_text.txt",
            "tkbc_entity_ids": tkbc_root / "ent_id",
            "tkbc_relation_ids": tkbc_root / "rel_id",
            "tkbc_timestamp_ids": tkbc_root / "ts_id",
        }
    global_indices = {
        name: data_root / "saved_pkl" / f"{name}.pkl"
        for name in ("e2rt", "event2time", "e2tr")
    }
    required = {
        "checkpoint": checkpoint,
        **splits,
        "knowledge_graph": kg_root / "full.txt",
        **dictionaries,
        **global_indices,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run inputs: {missing}")
    checkpoint_sha256 = _sha256(checkpoint)
    if dataset == "MultiTQ" and checkpoint_sha256 != _MULTITQ_CHECKPOINT_SHA256:
        raise ValueError(
            "MultiTQ checkpoint does not match the released tensor-compatible artifact"
        )
    return {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": checkpoint_sha256 if name == "checkpoint" else _sha256(path),
        }
        for name, path in required.items()
    }


def _verify_distilbert_cache() -> dict[str, object]:
    hf_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ).expanduser()
    repository_root = hf_home / "hub" / "models--distilbert-base-uncased"
    reference = _verify_distilbert_reference(repository_root)
    snapshot = repository_root / "snapshots" / _DISTILBERT_REVISION
    files: dict[str, dict[str, object]] = {}
    for filename, expected_sha256 in _DISTILBERT_FILES.items():
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned DistilBERT artifact: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Pinned DistilBERT artifact mismatch: {filename}")
        files[filename] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    return {
        "repository": "distilbert-base-uncased",
        "revision": _DISTILBERT_REVISION,
        "reference": reference,
        "files": files,
    }


def _verify_distilbert_reference(repository_root: Path) -> dict[str, object]:
    reference_path = repository_root / "refs" / "main"
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Missing Hugging Face main reference for pinned DistilBERT: "
            f"{reference_path}"
        )
    resolved_revision = reference_path.read_text(encoding="utf-8").strip()
    if resolved_revision != _DISTILBERT_REVISION:
        raise ValueError(
            "Hugging Face main reference does not resolve to pinned DistilBERT revision: "
            f"{resolved_revision!r}"
        )
    return {
        "path": str(reference_path),
        "resolved_revision": resolved_revision,
        "sha256": _sha256(reference_path),
    }


def _environment_snapshot() -> dict[str, object]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": _module_version("torch"),
        "transformers": _module_version("transformers"),
        "numpy": _module_version("numpy"),
        "cuda": _command_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]
        ),
        "pip_freeze": _command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _gpu_memory_used_mib() -> int | None:
    output = _command_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    values = [int(value.strip()) for value in output.splitlines() if value.strip().isdigit()]
    return max(values) if values else None


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def _output_identity(run_dir: Path, dataset: str, run_id: str) -> dict[str, object]:
    paths = {
        "console": run_dir / "console.log",
        "training_log": run_dir / "results" / dataset / f"{run_id}.log",
        "checkpoint": run_dir / "qa_models" / dataset / f"{run_id}.ckpt",
        "predictions": run_dir / "predictions.jsonl",
    }
    output: dict[str, object] = {}
    for name, path in paths.items():
        if path.is_file():
            output[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                **({"line_count": _line_count(path)} if name == "predictions" else {}),
            }
        else:
            output[name] = None
    return output


def _validate_completed_outputs(manifest: dict[str, object]) -> None:
    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    missing = [name for name in ("training_log", "checkpoint", "predictions") if not outputs[name]]
    if missing:
        manifest["status"] = "invalid_outputs"
        raise RuntimeError(f"Successful process did not produce required outputs: {missing}")


def verify_existing_run(run_dir: Path) -> dict[str, object]:
    """Re-verify an immutable completed run with the current fail-closed contract."""

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"completed", "process_completed"}:
        raise ValueError(
            f"Run is not eligible for completed-output verification: {manifest.get('status')!r}"
        )
    dataset = str(manifest.get("dataset"))
    model = str(manifest.get("model"))
    if dataset not in DATASETS or model not in MODELS:
        raise ValueError("Run manifest names an unknown dataset or model")
    run_id = str(manifest.get("run_id"))
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Run manifest has an invalid seed")

    actual_outputs = _output_identity(run_dir, dataset, run_id)
    recorded_outputs = manifest.get("outputs")
    if isinstance(recorded_outputs, dict):
        for name, actual in actual_outputs.items():
            recorded = recorded_outputs.get(name)
            if isinstance(recorded, dict) and isinstance(actual, dict):
                if recorded.get("sha256") != actual.get("sha256"):
                    raise ValueError(f"Completed output changed after the run: {name}")
            elif recorded != actual:
                raise ValueError(f"Completed output presence changed after the run: {name}")
    previously_verified = "prediction_verification" in manifest
    manifest["outputs"] = actual_outputs
    _validate_completed_outputs(manifest)
    predictions = actual_outputs["predictions"]
    assert isinstance(predictions, dict)
    manifest["prediction_verification"] = _verify_predictions(
        Path(str(predictions["path"])),
        expected_count=DATASETS[dataset].expected_test_examples,
        expected_run_id=run_id,
        expected_dataset=dataset,
        expected_model=MODELS[model].upstream_model,
        expected_variant=MODELS[model].variant,
        expected_seed=seed,
    )
    manifest["output_verification"] = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "verifier_file_sha256": _sha256(Path(__file__)),
        "mode": "reverification" if previously_verified else "posthoc",
    }
    manifest["status"] = "completed"
    _write_json(manifest_path, manifest)
    return manifest


def _verify_predictions(
    path: Path,
    *,
    expected_count: int,
    expected_run_id: str,
    expected_dataset: str,
    expected_model: str,
    expected_variant: str,
    expected_seed: int,
    expected_rank_depth: int = 100,
) -> dict[str, object]:
    """Validate the complete prediction export and independently recompute native Hits."""

    if expected_count <= 0 or expected_rank_depth < 10:
        raise ValueError("Invalid prediction-verification contract")
    source_indices: set[int] = set()
    hit_at_1 = 0
    hit_at_10 = 0
    question_types: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    missing_reference_types: Counter[str] = Counter()
    entity_id_fallback_ranks: Counter[int] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed prediction JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Prediction line {line_number} is not an object")
            expected_metadata = {
                "schema_version": "1.1",
                "run_id": expected_run_id,
                "dataset": expected_dataset,
                "split": "test",
                "model": expected_model,
                "variant": expected_variant,
                "seed": expected_seed,
            }
            for field, expected in expected_metadata.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"Prediction line {line_number} has {field}={row.get(field)!r}; "
                        f"expected {expected!r}"
                    )

            source_index = row.get("source_index")
            if isinstance(source_index, bool) or not isinstance(source_index, int):
                raise ValueError(f"Prediction line {line_number} has invalid source_index")
            if source_index in source_indices:
                raise ValueError(f"Duplicate source_index {source_index}")
            source_indices.add(source_index)

            _require_nonempty_string(row, "qid", line_number)
            _require_nonempty_string(row, "question", line_number)
            question_type = _require_nonempty_string(row, "question_type", line_number)
            answer_type = _require_nonempty_string(row, "answer_type", line_number)
            if answer_type not in {"entity", "time"}:
                raise ValueError(
                    f"Prediction line {line_number} has unsupported answer_type {answer_type!r}"
                )
            question_types[question_type] += 1
            answer_types[answer_type] += 1

            gold_ids = _require_string_list(
                row, "gold_answer_ids", line_number, allow_empty=True
            )
            gold_labels = _require_string_list(
                row, "gold_answer_labels", line_number, allow_empty=True
            )
            predicted_ids = _require_string_list(row, "predicted_answer_ids", line_number)
            predicted_labels = _require_string_list(
                row, "predicted_answer_labels", line_number
            )
            _validate_namespaced_answer_ids(gold_ids, "gold_answer_ids", line_number)
            _validate_namespaced_answer_ids(
                predicted_ids, "predicted_answer_ids", line_number
            )
            scores = row.get("predicted_scores")
            if len(gold_ids) != len(gold_labels):
                raise ValueError(f"Gold IDs and labels are misaligned at line {line_number}")
            if not gold_ids:
                missing_reference_types[f"{question_type} | {answer_type}"] += 1
            if len(predicted_ids) != expected_rank_depth:
                raise ValueError(
                    f"Prediction line {line_number} exports {len(predicted_ids)} ranks; "
                    f"expected {expected_rank_depth}"
                )
            if len(predicted_labels) != expected_rank_depth:
                raise ValueError(f"Prediction labels are misaligned at line {line_number}")
            if len(set(predicted_ids)) != expected_rank_depth:
                raise ValueError(f"Prediction IDs are not unique at line {line_number}")
            if expected_dataset != "MultiTQ":
                for rank, (answer_id, label) in enumerate(
                    zip(predicted_ids, predicted_labels, strict=True)
                ):
                    namespace, raw_id = answer_id.split(":", 1)
                    if namespace == "entity" and label == raw_id:
                        entity_id_fallback_ranks[rank] += 1
            if not isinstance(scores, list) or len(scores) != expected_rank_depth:
                raise ValueError(f"Prediction scores are misaligned at line {line_number}")
            if any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                for score in scores
            ):
                raise ValueError(f"Prediction scores are non-finite at line {line_number}")
            if any(
                float(left) < float(right)
                for left, right in zip(scores, scores[1:], strict=False)
            ):
                raise ValueError(f"Prediction scores are not rank-sorted at line {line_number}")

            record_hash = _require_nonempty_string(
                row, "source_record_sha256", line_number
            )
            if len(record_hash) != 64 or any(
                character not in string.hexdigits for character in record_hash
            ):
                raise ValueError(f"Invalid source-record hash at line {line_number}")

            gold_id_set = set(gold_ids)
            hit_at_1 += int(predicted_ids[0] in gold_id_set)
            hit_at_10 += int(bool(set(predicted_ids[:10]) & gold_id_set))

    observed_count = len(source_indices)
    if observed_count != expected_count:
        raise ValueError(
            f"Prediction export has {observed_count} records; expected {expected_count}"
        )
    expected_indices = set(range(expected_count))
    if source_indices != expected_indices:
        missing = sorted(expected_indices - source_indices)[:10]
        unexpected = sorted(source_indices - expected_indices)[:10]
        raise ValueError(
            "Prediction source-index coverage is not contiguous: "
            f"missing={missing}, unexpected={unexpected}"
        )
    missing_reference_count = sum(missing_reference_types.values())
    return {
        "expected_record_count": expected_count,
        "observed_record_count": observed_count,
        "rank_depth": expected_rank_depth,
        "source_indices_contiguous": True,
        "scores_finite_and_nonincreasing": True,
        "native_hits_at_1_count": hit_at_1,
        "native_hits_at_10_count": hit_at_10,
        "native_hits_at_1": hit_at_1 / expected_count,
        "native_hits_at_10": hit_at_10 / expected_count,
        "reference_available_count": expected_count - missing_reference_count,
        "missing_reference_count": missing_reference_count,
        "missing_reference_rate": missing_reference_count / expected_count,
        "missing_reference_counts_by_type": dict(sorted(missing_reference_types.items())),
        "entity_id_fallback_label_count": sum(entity_id_fallback_ranks.values()),
        "entity_id_fallback_top_1_count": entity_id_fallback_ranks[0],
        "entity_id_fallback_top_10_count": sum(
            count for rank, count in entity_id_fallback_ranks.items() if rank < 10
        ),
        "entity_id_fallback_counts_by_rank": {
            str(rank): count for rank, count in sorted(entity_id_fallback_ranks.items())
        },
        "question_type_counts": dict(sorted(question_types.items())),
        "answer_type_counts": dict(sorted(answer_types.items())),
    }


def _require_nonempty_string(row: dict[str, object], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prediction line {line_number} has invalid {field}")
    return value


def _require_string_list(
    row: dict[str, object],
    field: str,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"Prediction line {line_number} has invalid {field}")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Prediction line {line_number} has empty/non-string {field}")
    return value


def _validate_namespaced_answer_ids(
    values: list[str], field: str, line_number: int
) -> None:
    for value in values:
        namespace, separator, raw_id = value.partition(":")
        if separator != ":" or namespace not in {"entity", "time"} or not raw_id.strip():
            raise ValueError(
                f"Prediction line {line_number} has invalid namespaced ID in {field}: {value!r}"
            )


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hash-checked SABET-QA experiment")
    parser.add_argument("--verify-existing-run", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--purpose",
        choices=("confirmatory", "smoke", "sensitivity"),
        default="confirmatory",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.verify_existing_run is not None:
        incompatible = any(
            value is not None
            for value in (
                args.source_root,
                args.data_root,
                args.runs_root,
                args.dataset,
                args.model,
                args.epochs,
            )
        ) or args.dry_run
        if incompatible:
            parser.error("--verify-existing-run cannot be combined with run arguments")
        print(json.dumps(verify_existing_run(args.verify_existing_run), indent=2, sort_keys=True))
        return
    required = {
        "--source-root": args.source_root,
        "--data-root": args.data_root,
        "--runs-root": args.runs_root,
        "--dataset": args.dataset,
        "--model": args.model,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"the following arguments are required for a run: {', '.join(missing)}")
    assert args.source_root is not None
    assert args.data_root is not None
    assert args.runs_root is not None
    assert args.dataset is not None
    assert args.model is not None
    manifest = run_experiment(
        source_root=args.source_root,
        data_root=args.data_root,
        runs_root=args.runs_root,
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        purpose=args.purpose,
        epochs_override=args.epochs,
        deterministic=not args.non_deterministic,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
