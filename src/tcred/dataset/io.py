from __future__ import annotations

import uuid
from pathlib import Path

import orjson
from pydantic import BaseModel

from tcred.dataset.models import (
    AnswerVariant,
    ContextPack,
    DatasetBundle,
    Entity,
    Fact,
    GraphPath,
    Question,
    Scenario,
    Snapshot,
)


def load_bundle(dataset_dir: Path) -> DatasetBundle:
    return DatasetBundle(
        scenarios=_read_model_jsonl(dataset_dir / "scenarios.jsonl", Scenario),
        entities=_read_model_jsonl(dataset_dir / "entities.jsonl", Entity),
        facts=_read_model_jsonl(dataset_dir / "facts.jsonl", Fact),
        snapshots=_read_model_jsonl(dataset_dir / "snapshots.jsonl", Snapshot),
        questions=_read_model_jsonl(dataset_dir / "questions.jsonl", Question),
        graph_paths=_read_model_jsonl(dataset_dir / "graph_paths.jsonl", GraphPath),
        context_packs=_read_model_jsonl(dataset_dir / "context_packs.jsonl", ContextPack),
        answer_variants=_read_model_jsonl(
            dataset_dir / "answer_variants.jsonl",
            AnswerVariant,
        ),
        splits=orjson.loads((dataset_dir / "splits.json").read_bytes()),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL file is missing: {path}")
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = _temporary_sibling(path)
    try:
        write_jsonl(temporary, rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(
    path: Path,
    value: dict[str, object],
    *,
    sort_keys: bool = False,
) -> None:
    option = orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    temporary = _temporary_sibling(path)
    try:
        temporary.write_bytes(orjson.dumps(value, option=option))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _read_model_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    return [model.model_validate(row) for row in read_jsonl(path)]
