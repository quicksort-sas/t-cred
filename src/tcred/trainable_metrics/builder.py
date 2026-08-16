from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from tcred.trainable_metrics.adapters import adapt_row
from tcred.trainable_metrics.config import DataBuildConfig, SourceConfig, canonical_config_hash
from tcred.trainable_metrics.formatting import format_semantic_record
from tcred.trainable_metrics.near_duplicates import NearDuplicateIndex
from tcred.trainable_metrics.schema import DatasetPartition, SemanticRecord, stable_group_bucket
from tcred.trainable_metrics.source_io import (
    assert_source_terms_accepted,
    file_sha256,
    iter_project_records,
    iter_raw_examples,
    source_file_inventory,
)

_SQLITE_BATCH_SIZE = 1000
_OUTPUT_BATCH_SIZE = 5000
_SOURCE_INGEST_ATTEMPTS = 5
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupCandidate:
    group_id: str
    row_ids: tuple[int, ...]
    strata: Counter[str]
    priority: str
    model_texts: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    @property
    def dominant_stratum(self) -> str:
        if not self.strata:
            return "unlabelled"
        return min(self.strata, key=lambda key: (-self.strata[key], key))


def build_semantic_corpus(
    *,
    config: DataBuildConfig,
    raw_root: Path,
    output_dir: Path,
    source_names: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    selected_sources = [
        source
        for source in config.sources
        if source.enabled and (source_names is None or source.name in source_names)
    ]
    if not selected_sources:
        raise ValueError("No enabled sources were selected")
    _preflight_sources(selected_sources, raw_root=raw_root)
    _prepare_output_dir(output_dir, overwrite=overwrite)
    staging_path = output_dir / "staging.sqlite3"

    build_started = datetime.now(UTC)
    source_stats: dict[str, dict[str, Any]] = {}
    with closing(sqlite3.connect(staging_path)) as connection:
        _initialize_database(connection)
        for source_order, source in enumerate(selected_sources):
            source_stats[source.name] = _ingest_source_with_retries(
                connection,
                source=source,
                source_order=source_order,
                raw_root=raw_root,
            )
        conflict_stats = _quarantine_conflicts_and_duplicates(connection)
        selection_stats = _select_all_sources(
            connection,
            config=config,
            sources=selected_sources,
        )
        artifact_stats = _write_selected_artifacts(connection, output_dir=output_dir)
        integrity_stats = _selected_integrity_stats(connection)

    ledger = _license_ledger(selected_sources, raw_root=raw_root)
    ledger_path = output_dir / "license_ledger.json"
    _write_json(ledger_path, ledger)

    manifest = {
        "schema_version": "tcred-sl-corpus-manifest-v1",
        "build_started": build_started,
        "build_finished": datetime.now(UTC),
        "config_hash": canonical_config_hash(config),
        "seed": config.seed,
        "split_salt": config.split_salt,
        "split_integrity_policy": {
            "selection_order": ["calibration", "development", "train"],
            "model_text_near_duplicate_definition": "exact word-trigram Jaccard",
            "exact_jaccard_threshold": config.near_duplicate_jaccard_threshold,
            "minhash_candidate_threshold": config.near_duplicate_candidate_threshold,
            "minhash_num_perm": config.near_duplicate_num_perm,
            "minhash_seed": config.seed,
            "oversized_group_policy": "exclude complete group; never truncate",
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sources": source_stats,
        "conflicts_and_duplicates": conflict_stats,
        "selection": selection_stats,
        "integrity": integrity_stats,
        "artifacts": artifact_stats,
        "license_ledger": {
            "path": ledger_path.name,
            "sha256": file_sha256(ledger_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    _remove_staging_database(staging_path)
    return manifest


def _preflight_sources(sources: list[SourceConfig], *, raw_root: Path) -> None:
    missing: list[str] = []
    for source in sources:
        assert_source_terms_accepted(source)
        missing.extend(
            str(row["path"])
            for row in source_file_inventory(source, raw_root=raw_root)
            if not row["exists"]
        )
    if missing:
        raise FileNotFoundError("Required local source files are missing: " + ", ".join(missing))


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for name in (
        "staging.sqlite3",
        "staging.sqlite3-shm",
        "staging.sqlite3-wal",
        "manifest.json",
        "license_ledger.json",
        "near_duplicate_audit.json",
    ):
        (output_dir / name).unlink(missing_ok=True)
    records_dir = output_dir / "records"
    if not records_dir.exists():
        return
    for path in records_dir.iterdir():
        if not path.is_file() or path.suffix not in {".jsonl", ".parquet"}:
            raise ValueError(f"Refusing to remove an unknown corpus artifact: {path}")
        path.unlink()
    records_dir.rmdir()


def _remove_staging_database(staging_path: Path) -> None:
    for path in (
        staging_path,
        Path(f"{staging_path}-wal"),
        Path(f"{staging_path}-shm"),
    ):
        path.unlink(missing_ok=True)


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE candidates (
            row_id INTEGER PRIMARY KEY,
            source_order INTEGER NOT NULL,
            source_name TEXT NOT NULL,
            stage TEXT NOT NULL,
            task TEXT NOT NULL,
            group_id TEXT NOT NULL,
            intended_partition TEXT NOT NULL,
            stratum TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            record_hash TEXT NOT NULL,
            priority TEXT NOT NULL,
            model_text TEXT NOT NULL,
            record_json BLOB NOT NULL,
            eligible INTEGER NOT NULL DEFAULT 1,
            exclusion_reason TEXT,
            selected_partition TEXT
        );
        CREATE INDEX idx_candidates_source ON candidates(source_name, eligible);
        CREATE INDEX idx_candidates_content ON candidates(content_hash);
        CREATE INDEX idx_candidates_group ON candidates(source_name, group_id);
        CREATE INDEX idx_candidates_selected ON candidates(selected_partition, stage);
        """
    )


def _ingest_source(
    connection: sqlite3.Connection,
    *,
    source: SourceConfig,
    source_order: int,
    raw_root: Path,
) -> dict[str, Any]:
    raw_rows = 0
    admitted_rows = 0
    exclusions: Counter[str] = Counter()
    pending: list[tuple[Any, ...]] = []

    if source.loader == "project":
        records: Iterable[tuple[str, SemanticRecord]] = (
            ("train", record) for record in iter_project_records(source, raw_root=raw_root)
        )
        for intended_partition, record in records:
            raw_rows += 1
            _validate_project_record(record, source=source)
            pending.append(
                _candidate_tuple(
                    record,
                    source=source,
                    source_order=source_order,
                    intended_partition=intended_partition,
                )
            )
            admitted_rows += 1
            if len(pending) >= _SQLITE_BATCH_SIZE:
                _insert_candidates(connection, pending)
                pending.clear()
    else:
        for example in iter_raw_examples(source, raw_root=raw_root):
            raw_rows += 1
            result = adapt_row(
                example.row,
                source=source,
                native_split=example.native_split,
            )
            exclusions.update(result.exclusions)
            for record in result.records:
                pending.append(
                    _candidate_tuple(
                        record,
                        source=source,
                        source_order=source_order,
                        intended_partition=example.intended_partition,
                    )
                )
                admitted_rows += 1
                if len(pending) >= _SQLITE_BATCH_SIZE:
                    _insert_candidates(connection, pending)
                    pending.clear()
    if pending:
        _insert_candidates(connection, pending)
    connection.commit()
    return {
        "raw_rows": raw_rows,
        "adapted_records": admitted_rows,
        "adapter_exclusions": dict(exclusions.most_common()),
        "local_files": source_file_inventory(source, raw_root=raw_root),
    }


def _ingest_source_with_retries(
    connection: sqlite3.Connection,
    *,
    source: SourceConfig,
    source_order: int,
    raw_root: Path,
    attempts: int = _SOURCE_INGEST_ATTEMPTS,
) -> dict[str, Any]:
    """Retry only transient network failures at a complete source boundary."""

    if attempts < 1:
        raise ValueError("Source ingest attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return _ingest_source(
                connection,
                source=source,
                source_order=source_order,
                raw_root=raw_root,
            )
        except Exception as exc:
            connection.rollback()
            if attempt == attempts or not _is_transient_network_error(exc):
                raise
            delay = min(30.0, 2.0**attempt)
            LOGGER.warning(
                "Transient network failure while ingesting %s; retrying complete source "
                "(%d/%d) in %.0fs: %s",
                source.name,
                attempt + 1,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _is_transient_network_error(exc: BaseException) -> bool:
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(
            current,
            (ConnectionError, TimeoutError, RequestsConnectionError, RequestsTimeout),
        ):
            return True
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code == 429 or isinstance(status_code, int) and status_code >= 500:
            return True
        current = current.__cause__ or current.__context__
    return False


def _candidate_tuple(
    record: SemanticRecord,
    *,
    source: SourceConfig,
    source_order: int,
    intended_partition: str,
) -> tuple[Any, ...]:
    dumped = record.model_dump(mode="json")
    target_json = json.dumps(dumped["target"], sort_keys=True, separators=(",", ":"))
    target_hash = hashlib.sha256(target_json.encode()).hexdigest()
    priority = hashlib.sha256(
        f"{source.name}\x1f{record.source_group_id}\x1f{record.unit_id}".encode()
    ).hexdigest()
    return (
        source_order,
        source.name,
        str(record.curriculum_stage),
        str(record.task),
        record.source_group_id,
        intended_partition,
        _target_stratum(record),
        record.content_hash,
        target_hash,
        record.record_hash,
        priority,
        format_semantic_record(record),
        orjson.dumps(dumped),
    )


def _insert_candidates(connection: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO candidates (
            source_order, source_name, stage, task, group_id, intended_partition,
            stratum, content_hash, target_hash, record_hash, priority, model_text, record_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _quarantine_conflicts_and_duplicates(connection: sqlite3.Connection) -> dict[str, int]:
    conflicting_hashes = [
        row[0]
        for row in connection.execute(
            """
            SELECT content_hash
            FROM candidates
            GROUP BY content_hash
            HAVING COUNT(DISTINCT target_hash) > 1
            """
        )
    ]
    _update_hash_batches(
        connection,
        hashes=conflicting_hashes,
        reason="same model input has conflicting supervision",
    )

    duplicate_ids = [
        row[0]
        for row in connection.execute(
            """
            WITH ranked AS (
                SELECT row_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY content_hash, target_hash
                           ORDER BY source_order, priority, row_id
                       ) AS duplicate_rank
                FROM candidates
                WHERE eligible = 1
            )
            SELECT row_id FROM ranked WHERE duplicate_rank > 1
            """
        )
    ]
    for start in range(0, len(duplicate_ids), _SQLITE_BATCH_SIZE):
        batch = duplicate_ids[start : start + _SQLITE_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        connection.execute(
            f"""
            UPDATE candidates
            SET eligible = 0, exclusion_reason = 'exact duplicate with identical supervision'
            WHERE row_id IN ({placeholders})
            """,
            batch,
        )
    connection.commit()
    return {
        "conflicting_content_hashes": len(conflicting_hashes),
        "conflicting_rows_quarantined": _count_reason(
            connection, "same model input has conflicting supervision"
        ),
        "exact_duplicate_rows_removed": len(duplicate_ids),
    }


def _update_hash_batches(
    connection: sqlite3.Connection,
    *,
    hashes: list[str],
    reason: str,
) -> None:
    for start in range(0, len(hashes), _SQLITE_BATCH_SIZE):
        batch = hashes[start : start + _SQLITE_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        connection.execute(
            f"""
            UPDATE candidates
            SET eligible = 0, exclusion_reason = ?
            WHERE content_hash IN ({placeholders})
            """,
            [reason, *batch],
        )


def _select_all_sources(
    connection: sqlite3.Connection,
    *,
    config: DataBuildConfig,
    sources: list[SourceConfig],
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    for source in sources:
        groups = _load_groups(
            connection,
            source=source,
            max_rows_per_group=config.max_rows_per_group,
            include_model_text=False,
        )
        if not groups:
            raise RuntimeError(f"No eligible records remain for source {source.name}")
        oversized = _oversized_group_stats(
            connection,
            source=source,
            max_rows_per_group=config.max_rows_per_group,
        )
        targets = _partition_targets(source=source, config=config)
        states[source.name] = {
            "eligible_groups": len(groups),
            "eligible_rows": sum(group.row_count for group in groups),
            "oversized_groups_excluded": oversized["groups"],
            "oversized_rows_excluded": oversized["rows"],
            "targets": {str(partition): target for partition, target in targets.items()},
            "selected": {},
        }

    used_disjoint_groups: defaultdict[str, set[str]] = defaultdict(set)
    completed_partitions: list[DatasetPartition] = []
    for partition in (
        DatasetPartition.CALIBRATION,
        DatasetPartition.DEVELOPMENT,
        DatasetPartition.TRAIN,
    ):
        guard = (
            _build_selected_near_duplicate_index(
                connection,
                partitions=tuple(completed_partitions),
                threshold=config.near_duplicate_jaccard_threshold,
                candidate_threshold=config.near_duplicate_candidate_threshold,
                num_perm=config.near_duplicate_num_perm,
                seed=config.seed,
            )
            if completed_partitions
            else None
        )
        for source in sources:
            groups = _load_groups(
                connection,
                source=source,
                max_rows_per_group=config.max_rows_per_group,
                include_model_text=guard is not None,
            )
            disjoint_family = _disjoint_family(source)
            if disjoint_family:
                groups = [
                    group
                    for group in groups
                    if group.group_id not in used_disjoint_groups[disjoint_family]
                ]
            target = int(states[source.name]["targets"][str(partition)])
            selected, rejected = _select_partition_groups(
                groups,
                source=source,
                partition=partition,
                target=target,
                config=config,
                guard=guard,
            )
            selected_rows = sum(group.row_count for group in selected)
            if selected_rows < target:
                requested = sum(int(value) for value in states[source.name]["targets"].values())
                raise RuntimeError(
                    f"{source.name} has {sum(group.row_count for group in groups)} remaining "
                    f"group-safe rows but cannot satisfy the {partition} target of {target} "
                    f"after rejecting {rejected['groups']} near-duplicate groups "
                    f"({rejected['rows']} rows; requested total={requested})"
                )
            _mark_selected(connection, selected, partition=partition)
            if disjoint_family:
                used_disjoint_groups[disjoint_family].update(
                    group.group_id for group in selected
                )
            states[source.name]["selected"][str(partition)] = {
                "groups": len(selected),
                "rows": selected_rows,
                "strata": dict(
                    sum((group.strata for group in selected), start=Counter()).most_common()
                ),
                "near_duplicate_groups_rejected": rejected["groups"],
                "near_duplicate_rows_rejected": rejected["rows"],
            }
        connection.commit()
        completed_partitions.append(partition)
    return states


def _partition_targets(
    *, source: SourceConfig, config: DataBuildConfig
) -> dict[DatasetPartition, int]:
    train_target = source.target_train_rows
    train_fraction = (
        1.0
        - config.development_fraction_without_upstream_dev
        - config.calibration_fraction_without_upstream_dev
    )
    return {
        DatasetPartition.TRAIN: train_target,
        DatasetPartition.DEVELOPMENT: round(
            train_target * config.development_fraction_without_upstream_dev / train_fraction
        ),
        DatasetPartition.CALIBRATION: round(
            train_target * config.calibration_fraction_without_upstream_dev / train_fraction
        ),
    }


def _select_partition_groups(
    groups: list[GroupCandidate],
    *,
    source: SourceConfig,
    partition: DatasetPartition,
    target: int,
    config: DataBuildConfig,
    guard: NearDuplicateIndex | None,
) -> tuple[list[GroupCandidate], dict[str, int]]:
    rejected = {"groups": 0, "rows": 0}

    def accept(group: GroupCandidate) -> bool:
        if guard is None:
            return True
        if not group.model_texts:
            raise RuntimeError("Near-duplicate selection requires model-visible group text")
        if not any(guard.has_match(text) for text in group.model_texts):
            return True
        rejected["groups"] += 1
        rejected["rows"] += group.row_count
        return False

    preferred = [
        group
        for group in groups
        if _preferred_partition(group.group_id, config=config) == partition
    ]
    selected = _balanced_group_pick(
        preferred,
        target=target,
        balance=source.balance_labels,
        accept=accept,
    )
    selected_rows = sum(group.row_count for group in selected)
    if selected_rows >= target:
        return selected, rejected
    preferred_ids = {group.group_id for group in preferred}
    fallback = [group for group in groups if group.group_id not in preferred_ids]
    selected.extend(
        _balanced_group_pick(
            fallback,
            target=target - selected_rows,
            balance=source.balance_labels,
            accept=accept,
        )
    )
    return selected, rejected


def _build_selected_near_duplicate_index(
    connection: sqlite3.Connection,
    *,
    partitions: tuple[DatasetPartition, ...],
    threshold: float,
    candidate_threshold: float,
    num_perm: int,
    seed: int,
) -> NearDuplicateIndex:
    if not partitions:
        raise ValueError("At least one selected partition is required")
    index = NearDuplicateIndex(
        threshold=threshold,
        candidate_threshold=candidate_threshold,
        num_perm=num_perm,
        seed=seed,
    )
    placeholders = ",".join("?" for _ in partitions)
    for row_id, model_text in connection.execute(
        f"""
        SELECT row_id, model_text
        FROM candidates
        WHERE selected_partition IN ({placeholders})
        ORDER BY row_id
        """,
        [str(partition) for partition in partitions],
    ):
        index.add(f"selected:{row_id}", str(model_text))
    return index


def _load_groups(
    connection: sqlite3.Connection,
    *,
    source: SourceConfig,
    max_rows_per_group: int,
    include_model_text: bool,
) -> list[GroupCandidate]:
    grouped: defaultdict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    model_text_column = "model_text" if include_model_text else "''"
    for row_id, group_id, stratum, priority, model_text in connection.execute(
        f"""
        SELECT row_id, group_id, stratum, priority, {model_text_column}
        FROM candidates
        WHERE source_name = ? AND eligible = 1 AND intended_partition = 'train'
              AND selected_partition IS NULL
        ORDER BY group_id, priority
        """,
        (source.name,),
    ):
        grouped[str(group_id)].append(
            (int(row_id), str(stratum), str(priority), str(model_text))
        )
    result: list[GroupCandidate] = []
    for group_id, rows in grouped.items():
        if len(rows) > max_rows_per_group:
            continue
        result.append(
            GroupCandidate(
                group_id=group_id,
                row_ids=tuple(row_id for row_id, _, _, _ in rows),
                strata=Counter(stratum for _, stratum, _, _ in rows),
                priority=hashlib.sha256(f"{source.name}\x1f{group_id}".encode()).hexdigest(),
                model_texts=(
                    tuple(model_text for _, _, _, model_text in rows)
                    if include_model_text
                    else ()
                ),
            )
        )
    return sorted(result, key=lambda group: (group.priority, group.group_id))


def _oversized_group_stats(
    connection: sqlite3.Connection,
    *,
    source: SourceConfig,
    max_rows_per_group: int,
) -> dict[str, int]:
    row = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(row_count), 0)
        FROM (
            SELECT COUNT(*) AS row_count
            FROM candidates
            WHERE source_name = ? AND eligible = 1 AND intended_partition = 'train'
            GROUP BY group_id
            HAVING COUNT(*) > ?
        )
        """,
        (source.name, max_rows_per_group),
    ).fetchone()
    return {"groups": int(row[0]), "rows": int(row[1])}


def _balanced_group_pick(
    groups: list[GroupCandidate],
    *,
    target: int,
    balance: bool,
    accept: Callable[[GroupCandidate], bool] | None = None,
) -> list[GroupCandidate]:
    if target <= 0 or not groups:
        return []
    if not balance:
        return _take_whole_groups(groups, target=target, accept=accept)
    by_stratum: defaultdict[str, list[GroupCandidate]] = defaultdict(list)
    for group in groups:
        by_stratum[group.dominant_stratum].append(group)
    strata = sorted(by_stratum)
    selected: list[GroupCandidate] = []
    selected_rows = 0
    cursors = {stratum: 0 for stratum in strata}
    while selected_rows < target:
        progressed = False
        for stratum in strata:
            cursor = cursors[stratum]
            pool = by_stratum[stratum]
            if cursor >= len(pool):
                continue
            group = pool[cursor]
            cursors[stratum] += 1
            progressed = True
            if accept is not None and not accept(group):
                continue
            selected.append(group)
            selected_rows += group.row_count
            if selected_rows >= target:
                break
        if not progressed:
            break
    return selected


def _take_whole_groups(
    groups: list[GroupCandidate],
    *,
    target: int,
    accept: Callable[[GroupCandidate], bool] | None = None,
) -> list[GroupCandidate]:
    selected: list[GroupCandidate] = []
    count = 0
    for group in groups:
        if accept is not None and not accept(group):
            continue
        selected.append(group)
        count += group.row_count
        if count >= target:
            break
    return selected


def _mark_selected(
    connection: sqlite3.Connection,
    groups: list[GroupCandidate],
    *,
    partition: DatasetPartition,
) -> None:
    row_ids = [row_id for group in groups for row_id in group.row_ids]
    for start in range(0, len(row_ids), _SQLITE_BATCH_SIZE):
        batch = row_ids[start : start + _SQLITE_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        connection.execute(
            f"UPDATE candidates SET selected_partition = ? WHERE row_id IN ({placeholders})",
            [str(partition), *batch],
        )


def _preferred_partition(group_id: str, *, config: DataBuildConfig) -> DatasetPartition:
    bucket = stable_group_bucket(group_id, salt=config.split_salt)
    train_ceiling = round(
        10_000
        * (
            1.0
            - config.development_fraction_without_upstream_dev
            - config.calibration_fraction_without_upstream_dev
        )
    )
    development_ceiling = round(
        10_000 * (1.0 - config.calibration_fraction_without_upstream_dev)
    )
    if bucket < train_ceiling:
        return DatasetPartition.TRAIN
    if bucket < development_ceiling:
        return DatasetPartition.DEVELOPMENT
    return DatasetPartition.CALIBRATION


def _write_selected_artifacts(
    connection: sqlite3.Connection,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifact_stats: dict[str, Any] = {}
    keys = list(
        connection.execute(
            """
            SELECT DISTINCT selected_partition, stage
            FROM candidates
            WHERE selected_partition IS NOT NULL
            ORDER BY selected_partition, stage
            """
        )
    )
    for partition, stage in keys:
        stem = f"{partition}.{stage}"
        jsonl_path = records_dir / f"{stem}.jsonl"
        parquet_path = records_dir / f"{stem}.parquet"
        count = _write_partition(
            connection,
            partition=str(partition),
            stage=str(stage),
            jsonl_path=jsonl_path,
            parquet_path=parquet_path,
        )
        artifact_stats[stem] = {
            "rows": count,
            "jsonl": _artifact_entry(jsonl_path),
            "parquet": _artifact_entry(parquet_path),
        }
    return artifact_stats


def _write_partition(
    connection: sqlite3.Connection,
    *,
    partition: str,
    stage: str,
    jsonl_path: Path,
    parquet_path: Path,
) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
    schema = pa.schema(
        [
            ("unit_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_group_id", pa.string()),
            ("task", pa.string()),
            ("curriculum_stage", pa.string()),
            ("partition", pa.string()),
            ("stratum", pa.string()),
            ("content_hash", pa.string()),
            ("record_hash", pa.string()),
            ("record_json", pa.large_string()),
        ]
    )
    writer = pq.ParquetWriter(parquet_path, schema=schema, compression="zstd")
    count = 0
    batch: list[dict[str, str]] = []
    with jsonl_path.open("wb") as jsonl:
        cursor = connection.execute(
            """
            SELECT stratum, record_json
            FROM candidates
            WHERE selected_partition = ? AND stage = ?
            ORDER BY source_name, task, priority, row_id
            """,
            (partition, stage),
        )
        try:
            for stratum, payload in cursor:
                values = orjson.loads(payload)
                values["partition"] = partition
                canonical = orjson.dumps(values, option=orjson.OPT_SORT_KEYS)
                jsonl.write(canonical + b"\n")
                batch.append(
                    {
                        "unit_id": values["unit_id"],
                        "source_dataset": values["source_dataset"],
                        "source_group_id": values["source_group_id"],
                        "task": values["task"],
                        "curriculum_stage": values["curriculum_stage"],
                        "partition": partition,
                        "stratum": str(stratum),
                        "content_hash": values["content_hash"],
                        "record_hash": values["record_hash"],
                        "record_json": canonical.decode("utf-8"),
                    }
                )
                count += 1
                if len(batch) >= _OUTPUT_BATCH_SIZE:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        finally:
            writer.close()
    return count


def _selected_integrity_stats(connection: sqlite3.Connection) -> dict[str, Any]:
    selected_rows = int(
        connection.execute(
            "SELECT COUNT(*) FROM candidates WHERE selected_partition IS NOT NULL"
        ).fetchone()[0]
    )
    selected_hashes = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT content_hash)
            FROM candidates WHERE selected_partition IS NOT NULL
            """
        ).fetchone()[0]
    )
    cross_partition_groups = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT source_name, group_id
                FROM candidates
                WHERE selected_partition IS NOT NULL
                GROUP BY source_name, group_id
                HAVING COUNT(DISTINCT selected_partition) > 1
            )
            """
        ).fetchone()[0]
    )
    duplicate_selected_inputs = selected_rows - selected_hashes
    if cross_partition_groups or duplicate_selected_inputs:
        raise RuntimeError(
            "Selected corpus failed exact leakage checks: "
            f"cross_partition_groups={cross_partition_groups}, "
            f"duplicate_inputs={duplicate_selected_inputs}"
        )
    return {
        "selected_rows": selected_rows,
        "unique_model_input_hashes": selected_hashes,
        "cross_partition_groups": cross_partition_groups,
        "duplicate_selected_inputs": duplicate_selected_inputs,
    }


def _target_stratum(record: SemanticRecord) -> str:
    target = record.target
    if target.class_distribution:
        return min(
            target.class_distribution,
            key=lambda label: (-target.class_distribution[label], label),
        )
    for name in ("equivalence", "supported", "relevance", "answerable"):
        value = getattr(target, name)
        if value is not None:
            return f"{name}:{int(value >= 0.5)}"
    if target.scalar_rating is not None:
        bucket = min(4, math.floor(target.scalar_rating * 5))
        return f"rating_bin:{bucket}"
    if target.answer_u2 is not None:
        return f"answer_u2:{int(target.answer_u2 >= 0.5)}"
    return "other"


def _validate_project_record(record: SemanticRecord, *, source: SourceConfig) -> None:
    if record.source_dataset != source.name:
        raise ValueError(
            f"Project record source_dataset={record.source_dataset} does not match {source.name}"
        )
    if record.source_version != source.revision:
        raise ValueError(
            f"Project record source_version={record.source_version} does not match "
            f"{source.revision}"
        )
    if str(record.curriculum_stage) != str(source.curriculum_stage):
        raise ValueError("Project record curriculum stage does not match its source registry")


def _disjoint_family(source: SourceConfig) -> str | None:
    if source.name in {"vitaminc_broad", "vitaminc_task"}:
        return "vitaminc"
    return None


def _license_ledger(sources: list[SourceConfig], *, raw_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "tcred-sl-license-ledger-v1",
        "generated_at": datetime.now(UTC),
        "warning": (
            "This ledger records source-declared terms for research reproducibility; "
            "it is not legal advice. Source text is not redistributed by default."
        ),
        "sources": [
            {
                "name": source.name,
                "source_url": source.source_url,
                "revision": source.revision,
                "license_id": source.license_id,
                "license_url": source.license_url,
                "redistribution": source.redistribution,
                "terms_required": source.terms_required,
                "terms_acceptance_env": source.terms_acceptance_env,
                "declared_language": source.declared_language,
                "files": source_file_inventory(source, raw_root=raw_root),
                "notes": source.notes,
            }
            for source in sources
        ],
    }


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _count_reason(connection: sqlite3.Connection, reason: str) -> int:
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM candidates WHERE exclusion_reason = ?", (reason,)
        ).fetchone()[0]
    )


def _write_json(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps(values, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    temporary.replace(path)
