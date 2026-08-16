from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from tcred.trainable_metrics.formatting import (
    add_special_tokens,
    assert_no_prohibited_metadata,
    format_semantic_record,
    formatted_text_hash,
)
from tcred.trainable_metrics.schema import SemanticRecord, SemanticTask
from tcred.trainable_metrics.source_io import file_sha256

CLASS_ORDERS = {
    SemanticTask.SUPPORT: ("entailment", "neutral", "contradiction"),
    SemanticTask.TEMPORAL: ("support", "unknown", "contradiction"),
    SemanticTask.CITATION: ("appropriate", "incomplete", "inappropriate"),
}
SCALAR_TARGETS = (
    "answer_u1",
    "answer_u2",
    "equivalence",
    "supported",
    "relevance",
    "answerable",
    "scalar_rating",
)


def pretokenize_corpus(
    *,
    corpus_dir: Path,
    backbone_dir: Path,
    output_dir: Path,
    max_length: int = 256,
    batch_size: int = 512,
    overwrite: bool = False,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
    if max_length < 32:
        raise ValueError("max_length must be at least 32")
    _prepare_output_dir(output_dir, overwrite=overwrite)
    tokenizer = AutoTokenizer.from_pretrained(backbone_dir, local_files_only=True, use_fast=True)
    added_tokens = add_special_tokens(tokenizer)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    files = sorted((corpus_dir / "records").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No canonical corpus JSONL files found under {corpus_dir}")
    artifacts: dict[str, Any] = {}
    total_tasks: Counter[str] = Counter()
    total_sources: Counter[str] = Counter()
    for source_path in files:
        destination = output_dir / f"{source_path.stem}.parquet"
        stats = _pretokenize_file(
            source_path=source_path,
            destination=destination,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            pa=pa,
            pq=pq,
        )
        total_tasks.update(stats["tasks"])
        total_sources.update(stats["sources"])
        artifacts[source_path.stem] = stats
    manifest = {
        "schema_version": "tcred-sl-tokenized-corpus-v1",
        "max_length": max_length,
        "special_tokens_added": added_tokens,
        "tokenizer_size": len(tokenizer),
        "special_tokens": list(tokenizer.additional_special_tokens),
        "tasks": dict(sorted(total_tasks.items())),
        "sources": dict(sorted(total_sources.items())),
        "artifacts": artifacts,
        "tokenizer_files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(tokenizer_dir.glob("*"))
            if path.is_file()
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def flatten_record(record: SemanticRecord, *, text: str) -> dict[str, Any]:
    target = record.target
    class_order = CLASS_ORDERS.get(SemanticTask(record.task))
    class_target = [-1.0, -1.0, -1.0]
    if class_order and target.class_distribution:
        unknown = set(target.class_distribution) - set(class_order)
        if unknown:
            raise ValueError(f"Unexpected classes for {record.task}: {sorted(unknown)}")
        class_target = [float(target.class_distribution.get(label, 0.0)) for label in class_order]
    values = {
        "unit_id": record.unit_id,
        "source_dataset": record.source_dataset,
        "source_group_id": record.source_group_id,
        "task": str(record.task),
        "content_hash": record.content_hash,
        "formatted_text_hash": formatted_text_hash(text),
        "class_target": class_target,
        "pair_id": target.pair_id or "",
        "pair_role": target.pair_role or "",
        "invariance_group_id": target.invariance_group_id or "",
    }
    values.update(
        {
            name: -1.0 if getattr(target, name) is None else float(getattr(target, name))
            for name in SCALAR_TARGETS
        }
    )
    return values


def _pretokenize_file(
    *,
    source_path: Path,
    destination: Path,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    pa: Any,
    pq: Any,
) -> dict[str, Any]:
    schema = pa.schema(
        [
            ("unit_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_group_id", pa.string()),
            ("task", pa.string()),
            ("content_hash", pa.string()),
            ("formatted_text_hash", pa.string()),
            ("input_ids", pa.list_(pa.int32())),
            ("attention_mask", pa.list_(pa.int8())),
            ("token_length", pa.int16()),
            ("was_truncated", pa.bool_()),
            ("class_target", pa.list_(pa.float32(), 3)),
            *((name, pa.float32()) for name in SCALAR_TARGETS),
            ("pair_id", pa.string()),
            ("pair_role", pa.string()),
            ("invariance_group_id", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(destination, schema=schema, compression="zstd")
    tasks: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    lengths: list[int] = []
    truncated = 0
    rows = 0
    pending_records: list[SemanticRecord] = []
    pending_texts: list[str] = []
    special_ids = set(tokenizer.all_special_ids)

    def flush() -> None:
        nonlocal rows, truncated
        if not pending_records:
            return
        assert_no_prohibited_metadata(pending_texts)
        encoded = tokenizer(
            pending_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        batch: list[dict[str, Any]] = []
        for record, text, ids in zip(
            pending_records,
            pending_texts,
            encoded["input_ids"],
            strict=True,
        ):
            original_length = len(ids)
            was_truncated = original_length > max_length
            token_ids = truncate_preserving_terminal_special(
                ids,
                max_length=max_length,
                special_ids=special_ids,
            )
            row = flatten_record(record, text=text)
            row.update(
                {
                    "input_ids": token_ids,
                    "attention_mask": [1] * len(token_ids),
                    "token_length": len(token_ids),
                    "was_truncated": was_truncated,
                }
            )
            batch.append(row)
            rows += 1
            truncated += int(was_truncated)
            lengths.append(len(token_ids))
            tasks[str(record.task)] += 1
            sources[record.source_dataset] += 1
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        pending_records.clear()
        pending_texts.clear()

    try:
        with source_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = SemanticRecord.model_validate(orjson.loads(line))
                except (orjson.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid semantic record at {source_path}:{line_number}"
                    ) from exc
                pending_records.append(record)
                pending_texts.append(format_semantic_record(record))
                if len(pending_records) >= batch_size:
                    flush()
        flush()
    finally:
        writer.close()
    if not rows:
        raise RuntimeError(f"No rows were tokenized from {source_path}")
    return {
        "rows": rows,
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
        "source_jsonl_sha256": file_sha256(source_path),
        "truncated_rows": truncated,
        "truncated_fraction": truncated / rows,
        "token_length": {
            "mean": sum(lengths) / rows,
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "tasks": dict(tasks),
        "sources": dict(sources),
    }


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def truncate_preserving_terminal_special(
    input_ids: list[int],
    *,
    max_length: int,
    special_ids: set[int],
) -> list[int]:
    if len(input_ids) <= max_length:
        return input_ids
    truncated = input_ids[:max_length]
    if input_ids[-1] in special_ids:
        truncated[-1] = input_ids[-1]
    return truncated


# Backward-compatible private alias for callers created before inference reused this contract.
_truncate_preserving_terminal_special = truncate_preserving_terminal_special


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Pretokenized output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    (output_dir / "manifest.json").unlink(missing_ok=True)
    for path in output_dir.glob("*.parquet"):
        path.unlink()
    tokenizer_dir = output_dir / "tokenizer"
    if not tokenizer_dir.exists():
        return
    for path in sorted(tokenizer_dir.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    tokenizer_dir.rmdir()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)
