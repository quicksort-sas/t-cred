from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from pydantic import ValidationError

from tcred.metrics.models import MetricInput

_SHARD_SCHEMA_VERSION = "1.0"
_SHARD_ALGORITHM = "sha256(metric_id)-first-8-bytes-mod-n-v1"


def shard_neural_inputs(
    *, input_path: Path, output_dir: Path, shard_count: int
) -> dict[str, Path]:
    """Partition validated metric inputs without changing their JSON payloads."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Neural input does not exist: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"Neural shard output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    staging.mkdir()
    paths = [
        staging / f"part-{index:03d}-of-{shard_count:03d}.jsonl"
        for index in range(shard_count)
    ]
    streams: list[TextIO] = []
    counts = [0] * shard_count
    metric_ids: set[str] = set()
    try:
        streams = [path.open("w", encoding="utf-8", newline="\n") for path in paths]
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
            shard_index = _shard_index(row.metric_id, shard_count)
            streams[shard_index].write(line.rstrip("\r\n") + "\n")
            counts[shard_index] += 1
        for stream in streams:
            stream.close()
        streams.clear()
        shard_records = []
        for index, path in enumerate(paths):
            artifact = _file_identity(path)
            artifact["path"] = str((output_dir / path.name).resolve())
            shard_records.append(
                {
                    "index": index,
                    "row_count": counts[index],
                    "artifact": artifact,
                }
            )
        manifest_path = staging / "shard_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": _SHARD_SCHEMA_VERSION,
                "algorithm": _SHARD_ALGORITHM,
                "source": _file_identity(input_path),
                "source_row_count": sum(counts),
                "unique_metric_id_count": len(metric_ids),
                "metric_id_set_sha256": hashlib.sha256(
                    "".join(f"{value}\n" for value in sorted(metric_ids)).encode("utf-8")
                ).hexdigest(),
                "shard_count": shard_count,
                "shards": shard_records,
            },
        )
        staging.replace(output_dir)
    except Exception:
        for stream in streams:
            stream.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "manifest": output_dir / "shard_manifest.json",
        **{
            f"shard_{index}": output_dir / path.name
            for index, path in enumerate(paths)
        },
    }


def _shard_index(metric_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(metric_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shard_count


def _iter_nonempty_lines(path: Path) -> Iterator[tuple[int, str]]:
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
        description="Deterministically shard SABET neural metric inputs"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    outputs = shard_neural_inputs(
        input_path=args.input,
        output_dir=args.output_dir,
        shard_count=args.shards,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
