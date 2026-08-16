from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

from tcred.trainable_metrics.formatting import format_semantic_record
from tcred.trainable_metrics.near_duplicates import NearDuplicateIndex
from tcred.trainable_metrics.schema import SemanticRecord
from tcred.trainable_metrics.source_io import file_sha256


def audit_cross_partition_near_duplicates(
    *,
    corpus_dir: Path,
    output_path: Path,
    threshold: float = 0.90,
    candidate_threshold: float = 0.60,
    num_perm: int = 128,
    seed: int = 20260817,
) -> dict[str, Any]:
    records_dir = corpus_dir / "records"
    train_paths = sorted(records_dir.glob("train.*.jsonl"))
    heldout_paths = sorted(
        [*records_dir.glob("development.*.jsonl"), *records_dir.glob("calibration.*.jsonl")]
    )
    if not train_paths or not heldout_paths:
        raise FileNotFoundError("Near-duplicate audit requires train and held-out corpus files")
    index = NearDuplicateIndex(
        threshold=threshold,
        candidate_threshold=candidate_threshold,
        num_perm=num_perm,
        seed=seed,
    )
    metadata: dict[str, dict[str, Any]] = {}
    heldout_counts: Counter[str] = Counter()
    for path in heldout_paths:
        partition = path.name.split(".", 1)[0]
        for record_index, record in enumerate(_records(path), start=1):
            key = _lsh_key(partition=partition, path=path, record_index=record_index)
            index.add(key, format_semantic_record(record))
            metadata[key] = _metadata(record, partition=partition)
            heldout_counts[partition] += 1

    collisions: list[dict[str, Any]] = []
    checked = 0
    for path in train_paths:
        for record in _records(path):
            checked += 1
            for key, similarity in index.matches(format_semantic_record(record)):
                collisions.append(
                    {
                        "similarity": similarity,
                        "train": _metadata(record, partition="train"),
                        "heldout": metadata[key],
                    }
                )
    collisions.extend(
        _audit_development_calibration(
            heldout_paths,
            threshold=threshold,
            candidate_threshold=candidate_threshold,
            num_perm=num_perm,
            seed=seed,
        )
    )
    report = {
        "schema_version": "tcred-sl-near-duplicate-audit-v2",
        "corpus_manifest_sha256": file_sha256(corpus_dir / "manifest.json"),
        "method": (
            f"{num_perm}-permutation MinHash LSH at candidate threshold "
            f"{candidate_threshold:g}, plus exact word-trigram Jaccard verification"
        ),
        "threshold": threshold,
        "candidate_threshold": candidate_threshold,
        "num_perm": num_perm,
        "seed": seed,
        "train_rows_checked": checked,
        "heldout_rows_indexed": dict(sorted(heldout_counts.items())),
        "cross_partition_collisions": len(collisions),
        "status": "passed" if not collisions else "failed",
        "collisions": collisions,
    }
    _write_json(output_path, report)
    return report


def _audit_development_calibration(
    paths: list[Path],
    *,
    threshold: float,
    candidate_threshold: float,
    num_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    development = [path for path in paths if path.name.startswith("development.")]
    calibration = [path for path in paths if path.name.startswith("calibration.")]
    index = NearDuplicateIndex(
        threshold=threshold,
        candidate_threshold=candidate_threshold,
        num_perm=num_perm,
        seed=seed,
    )
    metadata: dict[str, dict[str, Any]] = {}
    for path in calibration:
        for record_index, record in enumerate(_records(path), start=1):
            key = _lsh_key(
                partition="calibration",
                path=path,
                record_index=record_index,
            )
            index.add(key, format_semantic_record(record))
            metadata[key] = _metadata(record, partition="calibration")
    collisions: list[dict[str, Any]] = []
    for path in development:
        for record in _records(path):
            for key, similarity in index.matches(format_semantic_record(record)):
                collisions.append(
                    {
                        "similarity": similarity,
                        "train": _metadata(record, partition="development"),
                        "heldout": metadata[key],
                    }
                )
    return collisions


def _records(path: Path) -> Iterator[SemanticRecord]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield SemanticRecord.model_validate(orjson.loads(line))
            except (orjson.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid semantic record at {path}:{line_number}") from exc


def _lsh_key(*, partition: str, path: Path, record_index: int) -> str:
    """Return an audit-local identity without assuming source IDs are globally unique."""

    return f"{partition}\x1f{path.name}\x1f{record_index}"


def _metadata(record: SemanticRecord, *, partition: str) -> dict[str, str]:
    return {
        "partition": partition,
        "unit_id": record.unit_id,
        "source_dataset": record.source_dataset,
        "source_group_id": record.source_group_id,
        "content_hash": record.content_hash,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
