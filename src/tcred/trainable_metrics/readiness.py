from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcred.trainable_metrics.config import DataBuildConfig, TrainingConfig, canonical_config_hash
from tcred.trainable_metrics.formatting import SPECIAL_TOKENS
from tcred.trainable_metrics.source_io import file_sha256

_REQUIRED_TASKS = {"answer", "support", "relevance", "temporal", "answerability", "citation"}


def validate_gpu_readiness(
    *,
    data_config: DataBuildConfig,
    training_config: TrainingConfig,
    corpus_dir: Path,
    tokenized_dir: Path,
    backbone_dir: Path,
    near_duplicate_report: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "status": "passed" if condition else "failed",
                "detail": detail,
            }
        )

    corpus_manifest = _json(corpus_dir / "manifest.json")
    tokenized_manifest = _json(tokenized_dir / "manifest.json")
    duplicate_audit = _json(near_duplicate_report)
    backbone_manifest = _json(backbone_dir.parent / "manifest.json")
    corpus_manifest_path = corpus_dir / "manifest.json"
    tokenized_manifest_path = tokenized_dir / "manifest.json"
    backbone_manifest_path = backbone_dir.parent / "manifest.json"
    corpus_manifest_sha256 = file_sha256(corpus_manifest_path)
    audit_corpus_linked = (
        duplicate_audit.get("corpus_manifest_sha256") == corpus_manifest_sha256
    )

    check(
        "corpus_config_hash",
        corpus_manifest.get("config_hash") == canonical_config_hash(data_config),
        "Canonical corpus manifest must match the frozen data configuration.",
    )
    check(
        "exact_split_integrity",
        corpus_manifest.get("integrity", {}).get("cross_partition_groups") == 0
        and corpus_manifest.get("integrity", {}).get("duplicate_selected_inputs") == 0,
        "No source group or exact model-visible input may cross partitions.",
    )
    check(
        "near_duplicate_integrity",
        duplicate_audit.get("status") == "passed"
        and duplicate_audit.get("threshold") == data_config.near_duplicate_jaccard_threshold
        and duplicate_audit.get("candidate_threshold")
        == data_config.near_duplicate_candidate_threshold
        and int(duplicate_audit.get("num_perm", 0)) == data_config.near_duplicate_num_perm
        and int(duplicate_audit.get("seed", -1)) == data_config.seed + 1
        and audit_corpus_linked,
        (
            f"status={duplicate_audit.get('status')!r}, "
            f"threshold={duplicate_audit.get('threshold')!r}, "
            f"candidate_threshold={duplicate_audit.get('candidate_threshold')!r}, "
            f"num_perm={duplicate_audit.get('num_perm')!r}, "
            f"seed={duplicate_audit.get('seed')!r}, "
            f"corpus_linked={audit_corpus_linked}."
        ),
    )
    expected_tokenized = {
        name: artifact["rows"]
        for name, artifact in corpus_manifest.get("artifacts", {}).items()
    }
    actual_tokenized = {
        name: artifact["rows"]
        for name, artifact in tokenized_manifest.get("artifacts", {}).items()
    }
    check(
        "tokenized_row_parity",
        expected_tokenized == actual_tokenized,
        f"Canonical/tokenized row maps equal={expected_tokenized == actual_tokenized}.",
    )
    checksum_failures = _tokenized_checksum_failures(
        tokenized_dir=tokenized_dir,
        manifest=tokenized_manifest,
    )
    check(
        "tokenized_checksums",
        not checksum_failures,
        "All tokenized artifacts match their manifest."
        if not checksum_failures
        else f"Mismatches: {checksum_failures[:5]}",
    )
    linkage_failures = [
        name
        for name, artifact in tokenized_manifest.get("artifacts", {}).items()
        if artifact.get("source_jsonl_sha256")
        != corpus_manifest.get("artifacts", {}).get(name, {}).get("jsonl", {}).get("sha256")
    ]
    check(
        "canonical_tokenized_linkage",
        not linkage_failures,
        "Every tokenized partition is linked to the canonical JSONL checksum."
        if not linkage_failures
        else f"Unlinked partitions: {linkage_failures}",
    )
    check(
        "token_length_contract",
        tokenized_manifest.get("max_length") == training_config.max_length,
        "Tokenized maximum length must equal the training configuration.",
    )
    model_path = backbone_dir / "model.safetensors"
    check(
        "safe_backbone",
        model_path.is_file() and not (backbone_dir / "pytorch_model.bin").exists(),
        "GPU loading is restricted to the local safetensors checkpoint.",
    )
    safe_files = {
        _manifest_path_name(row["path"]): row["sha256"]
        for row in backbone_manifest.get("safe_checkpoint_files", [])
    }
    check(
        "backbone_checksum",
        model_path.is_file() and safe_files.get("model.safetensors") == file_sha256(model_path),
        "The safe backbone checksum must match its acquisition manifest.",
    )
    check(
        "backbone_identity",
        backbone_manifest.get("model_id") == training_config.backbone
        and backbone_manifest.get("revision") == training_config.backbone_revision,
        (
            f"manifest={backbone_manifest.get('model_id')}@{backbone_manifest.get('revision')}, "
            f"training={training_config.backbone}@{training_config.backbone_revision}."
        ),
    )
    required = {
        "train.stage_a",
        "development.stage_a",
        "calibration.stage_a",
        "train.stage_b",
        "development.stage_b",
        "calibration.stage_b",
    }
    check(
        "required_partitions",
        required.issubset(actual_tokenized),
        f"Missing tokenized partitions: {sorted(required - set(actual_tokenized))}",
    )
    check(
        "special_token_contract",
        set(tokenized_manifest.get("special_tokens", [])) == set(SPECIAL_TOKENS),
        "Pretokenized task/field tokens must exactly match the frozen formatter contract.",
    )
    heldout_coverage = _heldout_task_coverage(tokenized_manifest)
    insufficient = {
        partition: {
            task: heldout_coverage[partition].get(task, 0)
            for task in sorted(_REQUIRED_TASKS)
            if heldout_coverage[partition].get(task, 0) < 100
        }
        for partition in ("development", "calibration")
    }
    insufficient = {key: value for key, value in insufficient.items() if value}
    check(
        "heldout_task_coverage",
        not insufficient,
        "Development and calibration each contain at least 100 rows for every task."
        if not insufficient
        else f"Insufficient held-out task counts: {insufficient}",
    )
    status = "ready" if all(row["status"] == "passed" for row in checks) else "blocked"
    return {
        "schema_version": "tcred-sl-gpu-readiness-v2",
        "status": status,
        "data_config_hash": canonical_config_hash(data_config),
        "training_config_hash": canonical_config_hash(training_config),
        "input_artifacts": {
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "tokenized_manifest_sha256": file_sha256(tokenized_manifest_path),
            "near_duplicate_audit_sha256": file_sha256(near_duplicate_report),
            "backbone_manifest_sha256": file_sha256(backbone_manifest_path),
        },
        "checks": checks,
    }


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Readiness artifact is missing: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Readiness artifact root is not an object: {path}")
    return values


def _manifest_path_name(value: str) -> str:
    """Return a manifest basename independent of the producing operating system."""
    return value.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _tokenized_checksum_failures(
    *,
    tokenized_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for stem, artifact in manifest.get("artifacts", {}).items():
        path = tokenized_dir / f"{stem}.parquet"
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact.get("bytes", -1))
            or file_sha256(path) != artifact.get("sha256")
        ):
            failures.append(str(path))
    for name, artifact in manifest.get("tokenizer_files", {}).items():
        path = tokenized_dir / "tokenizer" / name
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact.get("bytes", -1))
            or file_sha256(path) != artifact.get("sha256")
        ):
            failures.append(str(path))
    return failures


def _heldout_task_coverage(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    coverage = {"development": {}, "calibration": {}}
    for stem, artifact in manifest.get("artifacts", {}).items():
        partition = stem.split(".", 1)[0]
        if partition not in coverage:
            continue
        for task, count in artifact.get("tasks", {}).items():
            coverage[partition][task] = coverage[partition].get(task, 0) + int(count)
    return coverage
