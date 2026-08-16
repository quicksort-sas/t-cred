from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

_QID = re.compile(r"(?<![A-Za-z0-9])Q\d+(?!\d)", flags=re.IGNORECASE)
_SOURCE_SERIES = re.compile(r"wikidata:(Q\d+):(P\d+)", flags=re.IGNORECASE)
_RELEVANT_KEYS = {
    "aliases",
    "answer_source_id",
    "context_source_id",
    "source_id",
    "source_record_id",
    "source_record_ids",
    "statement_id",
}


@dataclass(frozen=True)
class SourceExclusionLedger:
    entity_ids: frozenset[str]
    source_ids: frozenset[str]
    files: tuple[dict[str, Any], ...]
    observed_key_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tcred-sl-source-exclusion-ledger-v1",
            "entity_ids": sorted(self.entity_ids),
            "source_ids": sorted(self.source_ids),
            "counts": {
                "entity_ids": len(self.entity_ids),
                "source_ids": len(self.source_ids),
            },
            "observed_key_counts": dict(sorted(self.observed_key_counts.items())),
            "files": list(self.files),
        }


def build_source_exclusion_ledger(
    *,
    data_root: Path,
    include_roots: Iterable[Path] | None = None,
) -> SourceExclusionLedger:
    """Collect Wikidata identities already exposed by project evaluation artifacts.

    The scanner deliberately reads only semantically relevant JSON fields. It does
    not treat a Q-number that happens to occur in ordinary prose as an entity ID.
    """

    roots = tuple(include_roots or _default_roots(data_root))
    entities: set[str] = set()
    sources: set[str] = set()
    key_counts: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    for path in _iter_candidate_files(roots):
        before_entities = len(entities)
        before_sources = len(sources)
        row_count = 0
        for row in _iter_json_records(path):
            row_count += 1
            _collect_ids(
                row,
                entities=entities,
                sources=sources,
                key_counts=key_counts,
            )
        if row_count:
            files.append(
                {
                    "path": _display_path(path, data_root=data_root),
                    "sha256": _sha256(path),
                    "rows": row_count,
                    "new_entity_ids": len(entities) - before_entities,
                    "new_source_ids": len(sources) - before_sources,
                }
            )
    if not entities or not sources:
        raise RuntimeError(
            "Source exclusion scan found no Wikidata entities or source series; "
            "the protected artifact roots are likely wrong"
        )
    return SourceExclusionLedger(
        entity_ids=frozenset(sorted(entities)),
        source_ids=frozenset(sorted(sources)),
        files=tuple(files),
        observed_key_counts=dict(key_counts),
    )


def source_disjointness_violations(
    sources_to_check: Iterable[Any],
    *,
    ledger: SourceExclusionLedger,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for source in sources_to_check:
        source_id = str(source.source_id)
        entity_ids = {
            str(source.context_source_id),
            *(str(claim.answer_source_id) for claim in source.claims),
        }
        source_overlap = source_id in ledger.source_ids
        entity_overlap = sorted(entity_ids & ledger.entity_ids)
        if source_overlap or entity_overlap:
            violations.append(
                {
                    "source_id": source_id,
                    "source_id_overlap": source_overlap,
                    "entity_id_overlaps": entity_overlap,
                }
            )
    return violations


def _default_roots(data_root: Path) -> Iterator[Path]:
    for relative in ("generated", "human_eval", "validation", "external/wikidata"):
        path = data_root / relative
        if path.exists():
            yield path


def _iter_candidate_files(roots: Iterable[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _iter_json_records(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        return
    try:
        yield orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def _collect_ids(
    value: Any,
    *,
    entities: set[str],
    sources: set[str],
    key_counts: Counter[str],
    parent_key: str | None = None,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key)
            _collect_ids(
                child,
                entities=entities,
                sources=sources,
                key_counts=key_counts,
                parent_key=normalized_key,
            )
        return
    if isinstance(value, list):
        for child in value:
            _collect_ids(
                child,
                entities=entities,
                sources=sources,
                key_counts=key_counts,
                parent_key=parent_key,
            )
        return
    if parent_key not in _RELEVANT_KEYS or not isinstance(value, str):
        return
    source_match = _SOURCE_SERIES.fullmatch(value.strip())
    if source_match:
        context_id = source_match.group(1).upper()
        property_id = source_match.group(2).upper()
        sources.add(f"wikidata:{context_id}:{property_id}")
        entities.add(context_id)
        key_counts[parent_key] += 1
        return
    matches = {match.upper() for match in _QID.findall(value)}
    if matches:
        entities.update(matches)
        key_counts[parent_key] += len(matches)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, *, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
