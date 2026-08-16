from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from tcred.trainable_metrics.config import SourceConfig
from tcred.trainable_metrics.schema import SemanticRecord


@dataclass(frozen=True)
class RawExample:
    intended_partition: str
    native_split: str
    row: Mapping[str, Any]


def iter_raw_examples(source: SourceConfig, *, raw_root: Path) -> Iterator[RawExample]:
    assert_source_terms_accepted(source)
    if source.loader == "huggingface":
        yield from _iter_huggingface(source)
    elif source.name == "ragtruth":
        yield from _iter_ragtruth(source, raw_root=raw_root)
    elif source.name == "torque":
        yield from _iter_torque(source, raw_root=raw_root)
    elif source.loader == "jsonl":
        yield from _iter_jsonl(source, raw_root=raw_root)
    elif source.loader == "json":
        yield from _iter_json(source, raw_root=raw_root)
    elif source.loader == "csv":
        yield from _iter_delimited(source, raw_root=raw_root)
    elif source.loader == "project":
        raise ValueError("project sources contain common records; use iter_project_records")
    else:  # pragma: no cover - SourceConfig rejects unknown loader values
        raise ValueError(f"Unsupported source loader: {source.loader}")


def iter_project_records(source: SourceConfig, *, raw_root: Path) -> Iterator[SemanticRecord]:
    if source.loader != "project":
        raise ValueError(f"Source is not a project-record source: {source.name}")
    for path in _local_paths(source, raw_root=raw_root):
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield SemanticRecord.model_validate(orjson.loads(line))
                except Exception as exc:
                    raise ValueError(f"Invalid project record at {path}:{line_number}") from exc


def source_file_inventory(source: SourceConfig, *, raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if source.loader == "huggingface":
        return rows
    for path in _local_paths(source, raw_root=raw_root, include_auxiliary=True):
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assert_source_terms_accepted(source: SourceConfig) -> None:
    if not source.terms_required:
        return
    assert source.terms_acceptance_env is not None
    accepted = os.getenv(source.terms_acceptance_env, "").strip().lower()
    if accepted not in {"1", "true", "yes"}:
        raise PermissionError(
            f"{source.name} requires accepting its source terms. Read {source.license_url}, "
            f"then set {source.terms_acceptance_env}=1 only if you accept them."
        )


def _iter_huggingface(source: SourceConfig) -> Iterator[RawExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
    for intended_partition, upstream in source.split_map.items():
        for native_split in _as_list(upstream):
            dataset = load_dataset(
                source.dataset_id,
                source.dataset_config,
                revision=source.revision,
                split=native_split,
                streaming=True,
            )
            dataset = _apply_huggingface_stream_policy(dataset, source=source)
            for row in dataset:
                yield RawExample(
                    intended_partition=intended_partition,
                    native_split=native_split,
                    row=row,
                )


def _apply_huggingface_stream_policy(dataset: Any, *, source: SourceConfig) -> Any:
    if source.streaming_take_rows_per_split is None:
        return dataset
    assert source.streaming_shuffle_seed is not None
    assert source.streaming_shuffle_buffer_rows is not None
    shuffled = dataset.shuffle(
        seed=source.streaming_shuffle_seed,
        buffer_size=source.streaming_shuffle_buffer_rows,
    )
    return shuffled.take(source.streaming_take_rows_per_split)


def _iter_jsonl(source: SourceConfig, *, raw_root: Path) -> Iterator[RawExample]:
    for intended_partition, native in source.split_map.items():
        path = _path_for_partition(source, intended_partition, raw_root=raw_root)
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
                yield RawExample(intended_partition, _single_native_split(native), row)


def _iter_json(source: SourceConfig, *, raw_root: Path) -> Iterator[RawExample]:
    for intended_partition, native in source.split_map.items():
        path = _path_for_partition(source, intended_partition, raw_root=raw_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"JSON collection contains a non-object row: {path}")
            yield RawExample(intended_partition, _single_native_split(native), row)


def _iter_delimited(source: SourceConfig, *, raw_root: Path) -> Iterator[RawExample]:
    for intended_partition, native in source.split_map.items():
        path = _path_for_partition(source, intended_partition, raw_root=raw_root)
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield RawExample(intended_partition, _single_native_split(native), row)


def _iter_ragtruth(source: SourceConfig, *, raw_root: Path) -> Iterator[RawExample]:
    source_path = raw_root / source.local_files["source_info"]
    response_path = raw_root / source.local_files["responses"]
    source_by_id = {
        str(row["source_id"]): row for row in _read_jsonl_objects(source_path)
    }
    intended_by_native = {
        native_split: intended
        for intended, value in source.split_map.items()
        for native_split in _as_list(value)
    }
    for response in _read_jsonl_objects(response_path):
        native_split = str(response.get("split") or "")
        intended = intended_by_native.get(native_split)
        if intended is None:
            continue
        source_id = str(response.get("source_id") or "")
        source_row = source_by_id.get(source_id)
        if source_row is None:
            raise ValueError(f"RAGTruth response references missing source_id={source_id}")
        merged = dict(response)
        merged.update(
            {
                "source_info": source_row.get("source_info"),
                "source_family": source_row.get("source"),
                "task_type": source_row.get("task_type"),
                "prompt": source_row.get("prompt"),
            }
        )
        yield RawExample(intended, native_split, merged)


def _iter_torque(source: SourceConfig, *, raw_root: Path) -> Iterator[RawExample]:
    for intended_partition, native in source.split_map.items():
        path = _path_for_partition(source, intended_partition, raw_root=raw_root)
        assignments = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(assignments, list):
            raise ValueError(f"TORQUE root must be a list: {path}")
        for assignment in assignments:
            for passage in assignment.get("passages", []):
                text = passage.get("passage")
                for qa in passage.get("question_answer_pairs", []):
                    yield RawExample(
                        intended_partition,
                        _single_native_split(native),
                        {
                            "id": qa.get("question_id"),
                            "passageID": qa.get("passageID"),
                            "passage": text,
                            "question": qa.get("question"),
                            "answer": qa.get("answer"),
                            "isAnswered": qa.get("isAnswered"),
                            "is_default_question": qa.get("is_default_question"),
                            "derived_from": qa.get("derived_from"),
                        },
                    )


def _read_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


def _local_paths(
    source: SourceConfig,
    *,
    raw_root: Path,
    include_auxiliary: bool = False,
) -> list[Path]:
    if include_auxiliary:
        values = source.local_files.values()
    else:
        values = (
            source.local_files[key]
            for key in source.split_map
            if key in source.local_files
        )
    return list(dict.fromkeys(raw_root / value for value in values))


def _path_for_partition(source: SourceConfig, partition: str, *, raw_root: Path) -> Path:
    try:
        relative = source.local_files[partition]
    except KeyError as exc:
        raise ValueError(f"{source.name} has no local file for partition {partition}") from exc
    path = raw_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file for {source.name}: {path}")
    return path


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def _single_native_split(value: str | list[str]) -> str:
    values = _as_list(value)
    if len(values) != 1:
        raise ValueError("local-file partitions must map to one native split")
    return values[0]
