from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import shutil
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from tcred.metrics.models import MetricInput

_SCHEMA_VERSION = "1.0"
_SELECTION_SALT = "batch-calibration-v1"


class _LongestHeapItem:
    """Make the least-preferred retained long input the min-heap root."""

    __slots__ = ("length", "row")

    def __init__(self, row: MetricInput) -> None:
        self.row = row
        self.length = _text_length(row)

    def __lt__(self, other: _LongestHeapItem) -> bool:
        if self.length != other.length:
            return self.length < other.length
        return self.row.metric_id > other.row.metric_id


class _HashHeapItem:
    """Make the least-preferred retained salted-hash input the min-heap root."""

    __slots__ = ("rank", "row")

    def __init__(self, row: MetricInput) -> None:
        self.row = row
        self.rank = _salted_rank(row.metric_id)

    def __lt__(self, other: _HashHeapItem) -> bool:
        return (self.rank, self.row.metric_id) > (
            other.rank,
            other.row.metric_id,
        )


def prepare_batch_calibration_sample(
    *,
    input_path: Path,
    output_dir: Path,
    sample_size: int = 4_096,
    long_input_count: int = 256,
) -> dict[str, Path]:
    """Select the preregistered, outcome-blind neural batch-calibration sample."""

    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Neural input does not exist: {input_path}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0 <= long_input_count <= sample_size:
        raise ValueError("long_input_count must be between zero and sample_size")
    if output_dir.exists():
        raise FileExistsError(f"Calibration output already exists: {output_dir}")

    longest_heap: list[_LongestHeapItem] = []
    hash_heap: list[_HashHeapItem] = []
    metric_ids: set[str] = set()
    source_row_count = 0
    for line_number, line in _iter_nonempty_lines(input_path):
        try:
            row = MetricInput.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(
                f"Invalid neural input at {input_path}:{line_number}"
            ) from error
        if row.metric_id in metric_ids:
            raise ValueError(f"Duplicate neural input metric_id: {row.metric_id}")
        metric_ids.add(row.metric_id)
        source_row_count += 1
        _retain_long_input(longest_heap, row, limit=long_input_count)
        _retain_hash_input(hash_heap, row, limit=sample_size)
    if source_row_count < sample_size:
        raise ValueError(
            f"Calibration requires {sample_size} rows, but input contains {source_row_count}"
        )

    longest = sorted(
        (item.row for item in longest_heap),
        key=lambda row: (-_text_length(row), row.metric_id),
    )
    longest_ids = {row.metric_id for row in longest}
    typical = sorted(
        (item.row for item in hash_heap if item.row.metric_id not in longest_ids),
        key=lambda row: (_salted_rank(row.metric_id), row.metric_id),
    )[: sample_size - long_input_count]
    selected = longest + typical
    if len({row.metric_id for row in selected}) != sample_size:
        raise RuntimeError("Calibration selection is not unique and complete")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        sample_path = staging / "calibration_inputs.jsonl"
        sample_path.write_text(
            "".join(row.model_dump_json() + "\n" for row in selected),
            encoding="utf-8",
            newline="\n",
        )
        sample_artifact = _file_identity(sample_path)
        sample_artifact["path"] = str(
            (output_dir / sample_path.name).resolve()
        )
        manifest_path = staging / "calibration_sample_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "selection_algorithm": (
                    "longest-combined-text-then-salted-sha256-remainder-v1"
                ),
                "selection_salt": _SELECTION_SALT,
                "ordering": (
                    "long-input stratum by descending combined character length and "
                    "metric_id, followed by typical stratum by salted SHA-256 and metric_id"
                ),
                "outcome_fields_used": [],
                "source": _file_identity(input_path),
                "source_row_count": source_row_count,
                "sample_size": sample_size,
                "long_input_count": len(longest),
                "typical_input_count": len(typical),
                "retained_metric_input_object_bound": sample_size + long_input_count,
                "combined_character_length": {
                    "long_input": _length_summary(longest),
                    "typical_input": _length_summary(typical),
                    "complete_sample": _length_summary(selected),
                },
                "selected_metric_id_set_sha256": hashlib.sha256(
                    "".join(
                        f"{metric_id}\n"
                        for metric_id in sorted(row.metric_id for row in selected)
                    ).encode("utf-8")
                ).hexdigest(),
                "artifact": sample_artifact,
            },
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "manifest": output_dir / "calibration_sample_manifest.json",
        "inputs": output_dir / "calibration_inputs.jsonl",
    }


def _text_length(row: MetricInput) -> int:
    return len(row.question) + len(row.reference_answer) + len(row.candidate_answer)


def _retain_long_input(
    heap: list[_LongestHeapItem], row: MetricInput, *, limit: int
) -> None:
    if limit == 0:
        return
    item = _LongestHeapItem(row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    current = heap[0]
    if item.length > current.length or (
        item.length == current.length
        and item.row.metric_id < current.row.metric_id
    ):
        heapq.heapreplace(heap, item)


def _retain_hash_input(heap: list[_HashHeapItem], row: MetricInput, *, limit: int) -> None:
    item = _HashHeapItem(row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    current = heap[0]
    if (item.rank, item.row.metric_id) < (current.rank, current.row.metric_id):
        heapq.heapreplace(heap, item)


def _salted_rank(metric_id: str) -> bytes:
    return hashlib.sha256(f"{_SELECTION_SALT}:{metric_id}".encode()).digest()


def _length_summary(rows: list[MetricInput]) -> dict[str, float | int | None]:
    if not rows:
        return {"minimum": None, "maximum": None, "mean": None}
    values = [_text_length(row) for row in rows]
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _iter_nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                yield line_number, line


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the outcome-blind SABET neural batch-calibration sample"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=4_096)
    parser.add_argument("--long-input-count", type=int, default=256)
    args = parser.parse_args()
    outputs = prepare_batch_calibration_sample(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        long_input_count=args.long_input_count,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
