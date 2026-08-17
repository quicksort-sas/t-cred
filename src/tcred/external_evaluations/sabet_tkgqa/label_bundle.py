from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tcred.external_evaluations.sabet_tkgqa.schema import SabetPredictionRecord

_BUNDLE_SCHEMA_VERSION = "1.1"
_SUPPORTED_BUNDLE_SCHEMA_VERSIONS = {"1.0", _BUNDLE_SCHEMA_VERSION}
_SNAPSHOT_SCHEMA_VERSION = "1.1"
_SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS = {"1.0", _SNAPSHOT_SCHEMA_VERSION}
_SNAPSHOT_PROGRESS_SCHEMA_VERSION = "1.0"
_OFFICIAL_SPLIT_COUNTS = {"train": 9_708, "dev": 3_236, "test": 3_237}
_OFFICIAL_DECLARED_SPLITS = {
    "train": "train",
    "dev": "development",
    "test": "test",
}
_OFFICIAL_URLS = {
    split: f"https://exaqt.mpi-inf.mpg.de/data/{split}.json"
    for split in _OFFICIAL_SPLIT_COUNTS
}
_WIKIDATA_ENDPOINT = "https://www.wikidata.org/w/api.php"
_WIKIDATA_REQUEST_CONTRACT = {
    "action": "wbgetentities",
    "props": "labels|info",
    "languages": "en",
    "format": "json",
    "formatversion": "2",
}
_WIKIDATA_QID = re.compile(r"Q\d+")
_TIMEQUESTIONS_PREFIXED_QID = re.compile(r"\+\s+(Q\d+)")
_WIKIDATA_RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
_WIKIDATA_BATCH_SIZE = 50


@dataclass(frozen=True)
class SupplementalLabel:
    text: str
    source: Literal[
        "official_timequestions_test",
        "official_timequestions_global",
        "wikidata_api_snapshot",
    ]
    wikidata_lastrevid: int | None = None
    wikidata_canonical_qid: str | None = None


class TimeQuestionsLabelResolver:
    """Resolve opaque TimeQuestions QIDs from a provenance-bound label bundle."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read TimeQuestions label bundle: {path}") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in _SUPPORTED_BUNDLE_SCHEMA_VERSIONS
        ):
            raise ValueError("Unsupported TimeQuestions label-bundle schema")
        if payload.get("dataset") != "timequestions":
            raise ValueError("Label bundle is not for the timequestions dataset")
        rows = payload.get("test_rows")
        labels = payload.get("global_labels")
        if not isinstance(rows, list) or not isinstance(labels, dict):
            raise ValueError("Malformed TimeQuestions label bundle")
        self._rows_by_index: dict[int, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Malformed TimeQuestions test-row label record")
            source_index = row.get("source_index")
            qid = row.get("qid")
            question = row.get("question")
            entity_answers = row.get("entity_answers")
            value_answers = row.get("value_answers", {})
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 0
                or not isinstance(qid, str)
                or not qid
                or not isinstance(question, str)
                or not question
                or not isinstance(entity_answers, dict)
                or not isinstance(value_answers, dict)
            ):
                raise ValueError("Malformed TimeQuestions test-row label record")
            if source_index in self._rows_by_index:
                raise ValueError(f"Duplicate TimeQuestions source index: {source_index}")
            _validate_label_mapping(entity_answers, context=f"test row {source_index}")
            _validate_value_mapping(value_answers, context=f"test row {source_index}")
            self._rows_by_index[source_index] = row
        if set(self._rows_by_index) != set(range(_OFFICIAL_SPLIT_COUNTS["test"])):
            raise ValueError("TimeQuestions label bundle does not cover the complete test split")
        self._global_labels: dict[str, dict[str, object]] = {}
        for qid, entry in labels.items():
            if not isinstance(qid, str) or _WIKIDATA_QID.fullmatch(qid) is None:
                raise ValueError(f"Invalid global label QID: {qid!r}")
            if not isinstance(entry, dict):
                raise ValueError(f"Malformed global label entry: {qid}")
            text = entry.get("label")
            source = entry.get("source")
            revision = entry.get("wikidata_lastrevid")
            canonical_qid = entry.get("wikidata_canonical_qid")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty global label: {qid}")
            if source not in {"official_timequestions_global", "wikidata_api_snapshot"}:
                raise ValueError(f"Invalid global label source: {qid}")
            if revision is not None and (
                isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0
            ):
                raise ValueError(f"Invalid Wikidata revision: {qid}")
            if source == "wikidata_api_snapshot" and revision is None:
                raise ValueError(f"Wikidata label has no revision: {qid}")
            if canonical_qid is not None and (
                not isinstance(canonical_qid, str)
                or _WIKIDATA_QID.fullmatch(canonical_qid) is None
            ):
                raise ValueError(f"Invalid canonical Wikidata QID: {qid}")
            self._global_labels[qid] = entry
        self.identity = {
            **_file_identity(self.path),
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "summary": payload.get("summary"),
            "sources": payload.get("sources"),
        }

    def validate_record(self, record: SabetPredictionRecord) -> None:
        if record.dataset != "timequestions":
            return
        row = self._rows_by_index.get(record.source_index)
        if row is None:
            raise ValueError(
                f"TimeQuestions source index is absent from label bundle: {record.source_index}"
            )
        if str(row["qid"]) != record.qid or row["question"] != record.question:
            raise ValueError(
                "TimeQuestions prediction does not align with the official label row: "
                f"source_index={record.source_index}"
            )
        expected = set(row["entity_answers"])
        observed = {
            normalized_qid
            for answer_id in record.gold_answer_ids
            if answer_id.startswith("entity:")
            and (
                normalized_qid := _processed_timequestions_qid(
                    answer_id.partition(":")[2]
                )
            )
            is not None
        }
        if observed != expected:
            raise ValueError(
                "TimeQuestions gold entity IDs differ from the official release: "
                f"source_index={record.source_index}, expected={sorted(expected)}, "
                f"observed={sorted(observed)}"
            )

    def resolve(
        self,
        record: SabetPredictionRecord,
        *,
        answer_id: str,
        role: Literal["candidate", "reference"],
    ) -> SupplementalLabel | None:
        if record.dataset != "timequestions":
            return None
        self.validate_record(record)
        namespace, separator, raw_id = answer_id.partition(":")
        if not separator:
            return None
        row = self._rows_by_index[record.source_index]
        value_answers = row.get("value_answers", {})
        assert isinstance(value_answers, dict)
        value_key = raw_id[1:].strip() if raw_id.startswith("+") else raw_id.strip()
        row_value = value_answers.get(value_key)
        if isinstance(row_value, str) and row_value.strip():
            return SupplementalLabel(
                text=row_value.strip(),
                source="official_timequestions_test",
            )
        if namespace != "entity":
            return None
        normalized_qid = _processed_timequestions_qid(raw_id)
        if normalized_qid is None:
            return None
        row_labels = row["entity_answers"]
        assert isinstance(row_labels, dict)
        row_label = row_labels.get(normalized_qid)
        if isinstance(row_label, str) and row_label.strip():
            return SupplementalLabel(
                text=row_label.strip(),
                source="official_timequestions_test",
            )
        entry = self._global_labels.get(normalized_qid)
        if entry is None:
            return None
        source = entry["source"]
        assert source in {"official_timequestions_global", "wikidata_api_snapshot"}
        revision = entry.get("wikidata_lastrevid")
        canonical_qid = entry.get("wikidata_canonical_qid")
        return SupplementalLabel(
            text=str(entry["label"]),
            source=source,
            wikidata_lastrevid=int(revision) if revision is not None else None,
            wikidata_canonical_qid=(
                str(canonical_qid) if canonical_qid is not None else None
            ),
        )


def build_timequestions_label_bundle(
    *,
    official_paths: dict[str, Path],
    wikidata_snapshot_indexes: Sequence[Path],
    output_path: Path,
) -> Path:
    """Build a self-contained label bundle from immutable, hash-recorded source files."""

    if set(official_paths) != set(_OFFICIAL_SPLIT_COUNTS):
        raise ValueError("Official paths must contain exactly train, dev, and test")
    official_rows: dict[str, list[dict[str, object]]] = {}
    official_sources: dict[str, dict[str, object]] = {}
    label_occurrences: defaultdict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for split in ("train", "dev", "test"):
        path = official_paths[split].resolve()
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read official TimeQuestions {split} split") from error
        if not isinstance(rows, list) or len(rows) != _OFFICIAL_SPLIT_COUNTS[split]:
            raise ValueError(f"Unexpected official TimeQuestions {split} split size")
        validated_rows = []
        seen_qids: set[str] = set()
        for source_index, row in enumerate(rows):
            validated = _validate_official_row(row, split=split, source_index=source_index)
            qid = str(validated["Id"])
            if qid in seen_qids:
                raise ValueError(f"Duplicate official TimeQuestions question ID: {qid}")
            seen_qids.add(qid)
            validated_rows.append(validated)
            for answer_id, label in _entity_answer_labels(validated).items():
                label_occurrences[answer_id].append((split, source_index, label))
        official_rows[split] = validated_rows
        official_sources[split] = {
            "url": _OFFICIAL_URLS[split],
            "artifact": _file_identity(path),
        }

    official_unique: dict[str, dict[str, object]] = {}
    official_conflicts: dict[str, list[str]] = {}
    for qid, occurrences in sorted(label_occurrences.items()):
        labels = sorted({label for _split, _index, label in occurrences})
        if len(labels) == 1:
            official_unique[qid] = {
                "label": labels[0],
                "source": "official_timequestions_global",
                "official_splits": sorted({split for split, _index, _label in occurrences}),
                "wikidata_lastrevid": None,
            }
        else:
            official_conflicts[qid] = labels

    wikidata_labels: dict[str, dict[str, object]] = {}
    snapshot_sources = []
    for index_path in wikidata_snapshot_indexes:
        labels, source = _load_wikidata_snapshot(index_path.resolve())
        snapshot_sources.append(source)
        for qid, entry in labels.items():
            previous = wikidata_labels.get(qid)
            if previous is not None and previous != entry:
                raise ValueError(f"Conflicting cached Wikidata labels for {qid}")
            wikidata_labels[qid] = entry

    global_labels = dict(official_unique)
    wikidata_added = 0
    for qid, entry in sorted(wikidata_labels.items()):
        if qid not in official_unique:
            global_labels[qid] = entry
            wikidata_added += 1

    test_rows = [
        {
            "source_index": source_index,
            "qid": str(row["Id"]),
            "question": row["Question"],
            "entity_answers": _entity_answer_labels(row),
            "value_answers": _value_answer_labels(row),
        }
        for source_index, row in enumerate(official_rows["test"])
    ]
    payload = {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "dataset": "timequestions",
        "policy": {
            "reference_priority": "row-specific official TimeQuestions test label",
            "candidate_priority": [
                "row-specific official TimeQuestions test label",
                "unique label in the official TimeQuestions train/dev/test release",
                "English label in a pinned cached Wikidata API response",
            ],
            "unresolved": "text metrics are not applicable",
            "prediction_mutation": False,
        },
        "sources": {
            "official_timequestions": official_sources,
            "wikidata_api_snapshots": snapshot_sources,
        },
        "test_rows": test_rows,
        "global_labels": dict(sorted(global_labels.items())),
        "official_label_conflicts": official_conflicts,
        "summary": {
            "test_row_count": len(test_rows),
            "row_specific_entity_label_count": sum(
                len(row["entity_answers"]) for row in test_rows
            ),
            "row_specific_value_label_count": sum(
                len(row["value_answers"]) for row in test_rows
            ),
            "official_unique_global_label_count": len(official_unique),
            "official_conflicting_qid_count": len(official_conflicts),
            "wikidata_snapshot_label_count": len(wikidata_labels),
            "wikidata_labels_added_after_official_release": wikidata_added,
            "global_label_count": len(global_labels),
        },
    }
    _atomic_write_json(output_path, payload)
    return output_path


def collect_required_wikidata_ids(
    *,
    prediction_paths: Sequence[Path],
    official_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    """List opaque top-1 QIDs not uniquely labeled in the official release."""

    occurrences: defaultdict[str, set[str]] = defaultdict(set)
    for split, path in official_paths.items():
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != _OFFICIAL_SPLIT_COUNTS.get(split):
            raise ValueError(f"Unexpected official TimeQuestions {split} split")
        for source_index, row in enumerate(rows):
            validated = _validate_official_row(row, split=split, source_index=source_index)
            for qid, label in _entity_answer_labels(validated).items():
                occurrences[qid].add(label)
    uniquely_labeled = {qid for qid, labels in occurrences.items() if len(labels) == 1}
    opaque: set[str] = set()
    input_files = []
    for path in sorted(item.resolve() for item in prediction_paths):
        input_files.append(_file_identity(path))
        for record in _read_predictions(path):
            answer_id = record.predicted_answer_ids[0]
            label = record.predicted_answer_labels[0].strip()
            namespace, separator, raw_id = answer_id.partition(":")
            if (
                record.dataset == "timequestions"
                and separator
                and namespace == "entity"
                and _WIKIDATA_QID.fullmatch(raw_id) is not None
                and label == raw_id
            ):
                opaque.add(raw_id)
    required = sorted(opaque - uniquely_labeled)
    payload = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "selection": (
            "unique opaque TimeQuestions top-1 entity IDs without a unique official "
            "TimeQuestions train/dev/test label"
        ),
        "prediction_files": input_files,
        "official_files": {
            split: _file_identity(path.resolve()) for split, path in sorted(official_paths.items())
        },
        "opaque_candidate_qid_count": len(opaque),
        "required_qid_count": len(required),
        "required_qids": required,
    }
    _atomic_write_json(output_path, payload)
    return output_path


def fetch_wikidata_label_snapshot(
    *,
    required_ids_path: Path,
    output_dir: Path,
    pause_seconds: float = 1.0,
    max_attempts: int = 6,
    retry_base_seconds: float = 5.0,
    timeout_seconds: float = 180.0,
) -> Path:
    """Fetch and cache English labels; metric scoring never performs live requests.

    Completed batches are content-validated and reused after interruption. Responses,
    headers, and the progress manifest are written atomically so a partial write cannot
    silently enter the final snapshot.
    """

    required = json.loads(required_ids_path.read_text(encoding="utf-8"))
    ids = required.get("required_qids") if isinstance(required, dict) else None
    if (
        not isinstance(ids, list)
        or ids != sorted(set(ids))
        or any(not isinstance(qid, str) or _WIKIDATA_QID.fullmatch(qid) is None for qid in ids)
    ):
        raise ValueError("Malformed required Wikidata ID manifest")
    if pause_seconds < 0 or retry_base_seconds < 0 or timeout_seconds <= 0:
        raise ValueError("Wikidata request timing values must be non-negative")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    output_dir.mkdir(parents=True, exist_ok=True)
    required_identity = _file_identity(required_ids_path.resolve())
    index_path = output_dir / "index.json"
    if index_path.exists():
        _validate_completed_snapshot(
            index_path=index_path,
            requested_ids=ids,
            required_identity=required_identity,
        )
        return index_path
    _reject_unknown_snapshot_files(output_dir)

    progress_path = output_dir / "progress.json"
    progress = _load_or_initialize_progress(
        progress_path=progress_path,
        requested_ids=ids,
        required_identity=required_identity,
    )
    batches: list[dict[str, object]] = []
    network_request_count = 0
    resumed_batch_count = 0
    started_utc = str(progress["started_utc"])
    for batch_index, start in enumerate(range(0, len(ids), _WIKIDATA_BATCH_SIZE)):
        requested_ids = ids[start : start + _WIKIDATA_BATCH_SIZE]
        cached = _load_cached_wikidata_batch(
            output_dir=output_dir,
            batch_index=batch_index,
            requested_ids=requested_ids,
        )
        if cached is not None:
            batches.append(cached)
            resumed_batch_count += 1
            continue
        params = {**_WIKIDATA_REQUEST_CONTRACT, "ids": "|".join(requested_ids)}
        url = f"{_WIKIDATA_ENDPOINT}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "T-CRED-reproduction/1.0 "
                    "(research artifact; contact via project repository)"
                )
            },
        )
        body, headers, attempt_count = _request_wikidata_batch(
            request=request,
            max_attempts=max_attempts,
            retry_base_seconds=retry_base_seconds,
            timeout_seconds=timeout_seconds,
        )
        _validate_wikidata_response(body, requested_ids=requested_ids, batch_index=batch_index)
        network_request_count += attempt_count
        response_path = output_dir / f"batch-{batch_index:02d}.json"
        headers_path = output_dir / f"batch-{batch_index:02d}.headers.json"
        _atomic_write_bytes(response_path, body)
        _atomic_write_json(headers_path, headers)
        batch = _snapshot_batch_record(
            batch_index=batch_index,
            requested_ids=requested_ids,
            response_path=response_path,
            headers_path=headers_path,
            attempt_count=attempt_count,
            resumed=False,
        )
        batches.append(batch)
        progress = {
            "schema_version": _SNAPSHOT_PROGRESS_SCHEMA_VERSION,
            "started_utc": started_utc,
            "updated_utc": datetime.now(UTC).isoformat(),
            "endpoint": _WIKIDATA_ENDPOINT,
            "request_contract": _WIKIDATA_REQUEST_CONTRACT,
            "required_ids_manifest": required_identity,
            "requested_ids": ids,
            "completed_batches": batches,
        }
        _atomic_write_json(progress_path, progress)
        if start + _WIKIDATA_BATCH_SIZE < len(ids):
            time.sleep(pause_seconds)
    if [qid for batch in batches for qid in batch["requested_ids"]] != ids:
        raise ValueError("Completed Wikidata batches do not match the required ID manifest")
    completed_utc = datetime.now(UTC).isoformat()
    progress = {
        "schema_version": _SNAPSHOT_PROGRESS_SCHEMA_VERSION,
        "started_utc": started_utc,
        "updated_utc": completed_utc,
        "endpoint": _WIKIDATA_ENDPOINT,
        "request_contract": _WIKIDATA_REQUEST_CONTRACT,
        "required_ids_manifest": required_identity,
        "requested_ids": ids,
        "completed_batches": batches,
    }
    _atomic_write_json(progress_path, progress)
    index = {
        "schema_version": _SNAPSHOT_SCHEMA_VERSION,
        "started_utc": started_utc,
        "retrieved_utc": completed_utc,
        "endpoint": _WIKIDATA_ENDPOINT,
        "request_contract": _WIKIDATA_REQUEST_CONTRACT,
        "required_ids_manifest": required_identity,
        "progress_manifest": _file_identity(progress_path),
        "requested_ids": ids,
        "batches": batches,
        "acquisition": {
            "batch_size": _WIKIDATA_BATCH_SIZE,
            "max_attempts_per_batch": max_attempts,
            "retry_base_seconds": retry_base_seconds,
            "inter_batch_pause_seconds": pause_seconds,
            "network_attempt_count_this_invocation": network_request_count,
            "resumed_batch_count_this_invocation": resumed_batch_count,
        },
    }
    _atomic_write_json(index_path, index)
    return index_path


def _request_wikidata_batch(
    *,
    request: urllib.request.Request,
    max_attempts: int,
    retry_base_seconds: float,
    timeout_seconds: float,
) -> tuple[bytes, dict[str, str], int]:
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return response.read(), dict(response.headers.items()), attempt
        except urllib.error.HTTPError as error:
            if error.code not in _WIKIDATA_RETRYABLE_HTTP_STATUS or attempt == max_attempts:
                raise
            delay = _retry_delay_seconds(
                retry_after=error.headers.get("Retry-After") if error.headers else None,
                attempt=attempt,
                retry_base_seconds=retry_base_seconds,
            )
        except (TimeoutError, urllib.error.URLError):
            if attempt == max_attempts:
                raise
            delay = retry_base_seconds * (2 ** (attempt - 1))
        time.sleep(delay)
    raise AssertionError("Wikidata retry loop exited unexpectedly")


def _retry_delay_seconds(
    *,
    retry_after: str | None,
    attempt: int,
    retry_base_seconds: float,
) -> float:
    fallback = retry_base_seconds * (2 ** (attempt - 1))
    if retry_after is None:
        return fallback
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _load_or_initialize_progress(
    *,
    progress_path: Path,
    requested_ids: list[str],
    required_identity: dict[str, object],
) -> dict[str, object]:
    if not progress_path.exists():
        progress = {
            "schema_version": _SNAPSHOT_PROGRESS_SCHEMA_VERSION,
            "started_utc": datetime.now(UTC).isoformat(),
            "updated_utc": datetime.now(UTC).isoformat(),
            "endpoint": _WIKIDATA_ENDPOINT,
            "request_contract": _WIKIDATA_REQUEST_CONTRACT,
            "required_ids_manifest": required_identity,
            "requested_ids": requested_ids,
            "completed_batches": [],
        }
        _atomic_write_json(progress_path, progress)
        return progress
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot resume malformed Wikidata progress manifest") from error
    expected = {
        "schema_version": _SNAPSHOT_PROGRESS_SCHEMA_VERSION,
        "endpoint": _WIKIDATA_ENDPOINT,
        "request_contract": _WIKIDATA_REQUEST_CONTRACT,
        "required_ids_manifest": required_identity,
        "requested_ids": requested_ids,
    }
    if not isinstance(progress, dict) or any(
        progress.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Wikidata progress manifest does not match this acquisition")
    if not isinstance(progress.get("started_utc"), str):
        raise ValueError("Wikidata progress manifest has no valid start time")
    return progress


def _load_cached_wikidata_batch(
    *,
    output_dir: Path,
    batch_index: int,
    requested_ids: list[str],
) -> dict[str, object] | None:
    response_path = output_dir / f"batch-{batch_index:02d}.json"
    headers_path = output_dir / f"batch-{batch_index:02d}.headers.json"
    if not response_path.exists() and not headers_path.exists():
        return None
    if not response_path.exists():
        raise ValueError(f"Wikidata batch {batch_index} has headers but no response")
    body = response_path.read_bytes()
    _validate_wikidata_response(body, requested_ids=requested_ids, batch_index=batch_index)
    if headers_path.exists():
        try:
            headers = json.loads(headers_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Malformed Wikidata headers for batch {batch_index}") from error
        if not isinstance(headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ValueError(f"Malformed Wikidata headers for batch {batch_index}")
    return _snapshot_batch_record(
        batch_index=batch_index,
        requested_ids=requested_ids,
        response_path=response_path,
        headers_path=headers_path if headers_path.exists() else None,
        attempt_count=None,
        resumed=True,
    )


def _snapshot_batch_record(
    *,
    batch_index: int,
    requested_ids: list[str],
    response_path: Path,
    headers_path: Path | None,
    attempt_count: int | None,
    resumed: bool,
) -> dict[str, object]:
    identity = _file_identity(response_path)
    return {
        "batch_index": batch_index,
        "requested_ids": requested_ids,
        "response_file": response_path.name,
        "response_success": 1,
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
        "headers_file": headers_path.name if headers_path is not None else None,
        "headers_identity": _file_identity(headers_path) if headers_path is not None else None,
        "network_attempt_count": attempt_count,
        "resumed_from_validated_cache": resumed,
    }


def _validate_wikidata_response(
    body: bytes,
    *,
    requested_ids: list[str],
    batch_index: int,
) -> None:
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed Wikidata response for batch {batch_index}") from error
    entities = parsed.get("entities") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or parsed.get("success") != 1 or not isinstance(entities, dict):
        raise ValueError(f"Wikidata request failed for batch {batch_index}")
    if set(entities) != set(requested_ids):
        raise ValueError(f"Wikidata response coverage mismatch for batch {batch_index}")
    for qid, entity in entities.items():
        _canonical_wikidata_qid(
            entity,
            requested_qid=qid,
            context=f"batch {batch_index}",
        )


def _validate_completed_snapshot(
    *,
    index_path: Path,
    requested_ids: list[str],
    required_identity: dict[str, object],
) -> None:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read completed Wikidata snapshot index") from error
    if (
        index.get("requested_ids") != requested_ids
        or index.get("required_ids_manifest") != required_identity
    ):
        raise ValueError("Completed Wikidata snapshot does not match this acquisition")
    _load_wikidata_snapshot(index_path)


def _reject_unknown_snapshot_files(output_dir: Path) -> None:
    allowed = re.compile(r"(?:progress\.json|batch-\d+\.(?:json|headers\.json)|\..+\.tmp)")
    unknown = sorted(
        path.name
        for path in output_dir.iterdir()
        if allowed.fullmatch(path.name) is None
    )
    if unknown:
        raise ValueError(f"Unexpected files in Wikidata snapshot directory: {unknown}")


def _validate_official_row(
    value: object,
    *,
    split: str,
    source_index: int,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Malformed official TimeQuestions row: {split}/{source_index}")
    qid = value.get("Id")
    question = value.get("Question")
    answers = value.get("Answer")
    declared_split = value.get("Data set")
    if (
        isinstance(qid, bool)
        or not isinstance(qid, (int, str))
        or not str(qid).strip()
        or not isinstance(question, str)
        or not question
        or not isinstance(answers, list)
        or declared_split != _OFFICIAL_DECLARED_SPLITS[split]
    ):
        raise ValueError(f"Malformed official TimeQuestions row: {split}/{source_index}")
    for answer in answers:
        if not isinstance(answer, dict) or not isinstance(answer.get("AnswerType"), str):
            raise ValueError(f"Malformed official TimeQuestions answer: {split}/{source_index}")
    return value


def _entity_answer_labels(row: dict[str, object]) -> dict[str, str]:
    output: dict[str, str] = {}
    answers = row["Answer"]
    assert isinstance(answers, list)
    for answer in answers:
        assert isinstance(answer, dict)
        qid = answer.get("WikidataQid")
        label = answer.get("WikidataLabel")
        if qid is None:
            continue
        if not isinstance(qid, str) or _WIKIDATA_QID.fullmatch(qid.strip()) is None:
            raise ValueError(f"Invalid official TimeQuestions Wikidata ID: {qid!r}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Missing official TimeQuestions label for {qid.strip()}")
        normalized_qid = qid.strip()
        normalized_label = label.strip()
        previous = output.get(normalized_qid)
        if previous is not None and previous != normalized_label:
            raise ValueError(f"Conflicting labels within one TimeQuestions row: {normalized_qid}")
        output[normalized_qid] = normalized_label
    return output


def _value_answer_labels(row: dict[str, object]) -> dict[str, str]:
    output: dict[str, str] = {}
    answers = row["Answer"]
    assert isinstance(answers, list)
    for answer in answers:
        assert isinstance(answer, dict)
        if answer.get("WikidataQid") is not None:
            continue
        argument = answer.get("AnswerArgument")
        if not isinstance(argument, str) or not argument.strip():
            raise ValueError("Missing official TimeQuestions value answer")
        normalized = argument.strip()
        output[normalized] = normalized
    return output


def _load_wikidata_snapshot(
    index_path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Wikidata label snapshot: {index_path}") from error
    if index.get("schema_version") not in _SUPPORTED_SNAPSHOT_SCHEMA_VERSIONS:
        raise ValueError("Wikidata snapshot contract mismatch: schema_version")
    expected = {
        "endpoint": _WIKIDATA_ENDPOINT,
        "request_contract": _WIKIDATA_REQUEST_CONTRACT,
    }
    for name, value in expected.items():
        if index.get(name) != value:
            raise ValueError(f"Wikidata snapshot contract mismatch: {name}")
    requested_ids = index.get("requested_ids")
    batches = index.get("batches")
    if (
        not isinstance(requested_ids, list)
        or requested_ids != sorted(set(requested_ids))
        or not isinstance(batches, list)
    ):
        raise ValueError("Malformed Wikidata snapshot index")
    observed_ids: list[str] = []
    labels: dict[str, dict[str, object]] = {}
    response_sources = []
    for expected_index, batch in enumerate(batches):
        if not isinstance(batch, dict) or batch.get("batch_index") != expected_index:
            raise ValueError("Malformed Wikidata snapshot batch")
        batch_ids = batch.get("requested_ids")
        filename = batch.get("response_file")
        if not isinstance(batch_ids, list) or not isinstance(filename, str):
            raise ValueError("Malformed Wikidata snapshot batch")
        observed_ids.extend(batch_ids)
        response_path = index_path.parent / filename
        identity = _file_identity(response_path)
        if (
            identity["sha256"] != batch.get("sha256")
            or identity["size_bytes"] != batch.get("size_bytes")
        ):
            raise ValueError(f"Wikidata snapshot response hash mismatch: {filename}")
        if index.get("schema_version") == _SNAPSHOT_SCHEMA_VERSION:
            headers_filename = batch.get("headers_file")
            headers_identity = batch.get("headers_identity")
            if headers_filename is not None:
                if not isinstance(headers_filename, str) or not isinstance(headers_identity, dict):
                    raise ValueError(f"Malformed Wikidata snapshot headers: {filename}")
                observed_headers_identity = _file_identity(index_path.parent / headers_filename)
                if any(
                    observed_headers_identity.get(name) != headers_identity.get(name)
                    for name in ("sha256", "size_bytes", "line_count")
                ):
                    raise ValueError(f"Wikidata snapshot headers hash mismatch: {filename}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("success") != 1:
            raise ValueError(f"Wikidata snapshot response failed: {filename}")
        entities = response.get("entities")
        if not isinstance(entities, dict) or set(entities) != set(batch_ids):
            raise ValueError(f"Wikidata snapshot entity coverage mismatch: {filename}")
        for qid, entity in entities.items():
            canonical_qid = _canonical_wikidata_qid(
                entity,
                requested_qid=qid,
                context=f"snapshot response {filename}",
            )
            assert isinstance(entity, dict)
            label = (entity.get("labels") or {}).get("en", {}).get("value")
            if not isinstance(label, str) or not label.strip():
                continue
            revision = entity.get("lastrevid")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
                raise ValueError(f"Wikidata label has no valid revision: {qid}")
            labels[qid] = {
                "label": label.strip(),
                "source": "wikidata_api_snapshot",
                "official_splits": [],
                "wikidata_lastrevid": revision,
                "wikidata_canonical_qid": canonical_qid,
            }
        response_sources.append(identity)
    if observed_ids != requested_ids:
        raise ValueError("Wikidata snapshot batches do not match requested IDs")
    source = {
        "index": _file_identity(index_path),
        "retrieved_utc": index.get("retrieved_utc"),
        "endpoint": _WIKIDATA_ENDPOINT,
        "request_contract": _WIKIDATA_REQUEST_CONTRACT,
        "requested_qid_count": len(requested_ids),
        "resolved_english_label_count": len(labels),
        "responses": response_sources,
    }
    return labels, source


def _canonical_wikidata_qid(
    entity: object,
    *,
    requested_qid: str,
    context: str,
) -> str:
    if not isinstance(entity, dict):
        raise ValueError(f"Malformed Wikidata entity in {context}: {requested_qid}")
    canonical_qid = entity.get("id")
    if canonical_qid == requested_qid:
        return requested_qid
    redirects = entity.get("redirects")
    if (
        isinstance(canonical_qid, str)
        and _WIKIDATA_QID.fullmatch(canonical_qid) is not None
        and isinstance(redirects, dict)
        and redirects.get("from") == requested_qid
        and redirects.get("to") == canonical_qid
    ):
        return canonical_qid
    raise ValueError(f"Malformed Wikidata entity in {context}: {requested_qid}")


def _validate_label_mapping(value: dict[object, object], *, context: str) -> None:
    for qid, label in value.items():
        if (
            not isinstance(qid, str)
            or _WIKIDATA_QID.fullmatch(qid) is None
            or not isinstance(label, str)
            or not label.strip()
        ):
            raise ValueError(f"Malformed label mapping in {context}")


def _validate_value_mapping(value: dict[object, object], *, context: str) -> None:
    for key, label in value.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(label, str)
            or not label.strip()
        ):
            raise ValueError(f"Malformed value mapping in {context}")


def _processed_timequestions_qid(value: str) -> str | None:
    if _WIKIDATA_QID.fullmatch(value) is not None:
        return value
    match = _TIMEQUESTIONS_PREFIXED_QID.fullmatch(value)
    return match.group(1) if match is not None else None


def _read_predictions(path: Path) -> Iterator[SabetPredictionRecord]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield SabetPredictionRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid prediction at {path}:{line_number}") from error


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    line_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            line_count += block.count(b"\n")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "line_count": line_count,
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _official_paths(values: Iterable[str]) -> dict[str, Path]:
    paths = list(values)
    if len(paths) != 3:
        raise ValueError("Expected official paths in train, dev, test order")
    return dict(zip(("train", "dev", "test"), map(Path, paths), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pinned TimeQuestions label artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    required = subparsers.add_parser("required")
    required.add_argument("--predictions", nargs="+", required=True)
    required.add_argument("--official", nargs=3, required=True, metavar=("TRAIN", "DEV", "TEST"))
    required.add_argument("--output", type=Path, required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--required-ids", type=Path, required=True)
    fetch.add_argument("--output-dir", type=Path, required=True)
    fetch.add_argument("--pause-seconds", type=float, default=1.0)
    fetch.add_argument("--max-attempts", type=int, default=6)
    fetch.add_argument("--retry-base-seconds", type=float, default=5.0)
    build = subparsers.add_parser("build")
    build.add_argument("--official", nargs=3, required=True, metavar=("TRAIN", "DEV", "TEST"))
    build.add_argument("--wikidata-snapshot-index", nargs="*", default=[])
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "required":
        output = collect_required_wikidata_ids(
            prediction_paths=[Path(value) for value in args.predictions],
            official_paths=_official_paths(args.official),
            output_path=args.output,
        )
    elif args.command == "fetch":
        if args.pause_seconds < 0:
            raise ValueError("pause-seconds cannot be negative")
        output = fetch_wikidata_label_snapshot(
            required_ids_path=args.required_ids,
            output_dir=args.output_dir,
            pause_seconds=args.pause_seconds,
            max_attempts=args.max_attempts,
            retry_base_seconds=args.retry_base_seconds,
        )
    else:
        output = build_timequestions_label_bundle(
            official_paths=_official_paths(args.official),
            wikidata_snapshot_indexes=[Path(value) for value in args.wikidata_snapshot_index],
            output_path=args.output,
        )
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
