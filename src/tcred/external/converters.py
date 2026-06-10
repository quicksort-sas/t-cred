from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import orjson

from tcred.dataset.writer import DatasetWriter
from tcred.external.certified import (
    build_hoh_bundle,
    build_pat_bundle,
    pat_row_is_convertible,
)

PAT_SNAPSHOT_DATES = {
    "Dec2021": date(2021, 12, 30),
    "Dec2023": date(2023, 12, 30),
    "March2024": date(2024, 3, 30),
}


def convert_pat_dataset(
    *,
    pat_data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    rows = _load_pat_rows(pat_data_dir, limit=limit)
    bundle = build_pat_bundle(rows)
    return DatasetWriter(output_dir, overwrite=overwrite).write_bundle(bundle)


def convert_hoh_dataset(
    *,
    input_path: Path | None,
    output_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    rows = (
        fetch_hoh_rows(limit=limit or 200)
        if input_path is None
        else _load_generic_rows(input_path, limit=limit)
    )
    bundle = build_hoh_bundle(rows)
    return DatasetWriter(output_dir, overwrite=overwrite).write_bundle(bundle)


def fetch_hoh_rows(*, limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        return []
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = min(100, limit)
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {"User-Agent": "T-CRED dataset converter/2.1"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        while len(rows) < limit:
            params = {
                "dataset": "russwest404/HoH-QAs",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": min(page_size, limit - len(rows)),
            }
            response = client.get(
                "https://datasets-server.huggingface.co/rows",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            page = payload.get("rows", [])
            rows.extend(item["row"] for item in page)
            if not page or payload.get("partial"):
                break
            offset += len(page)
    return rows[:limit]


def _load_pat_rows(pat_data_dir: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not pat_data_dir.is_dir():
        raise FileNotFoundError(f"PAT source directory does not exist: {pat_data_dir}")
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for snapshot_dir in sorted(path for path in pat_data_dir.iterdir() if path.is_dir()):
        snapshot_name = snapshot_dir.name
        if snapshot_name not in PAT_SNAPSHOT_DATES:
            raise ValueError(
                f"PAT snapshot directory has no registered observation date: {snapshot_name}"
            )
        for file_name, hop_type in (
            ("PAT-singlehop.json", "singlehop"),
            ("PAT-multihop.json", "multihop"),
        ):
            path = snapshot_dir / file_name
            if not path.exists():
                continue
            payload = orjson.loads(path.read_bytes())
            values = payload.values() if isinstance(payload, dict) else payload
            stratum = (snapshot_name, hop_type)
            by_stratum.setdefault(stratum, [])
            for source_row in values:
                row = dict(source_row)
                if not pat_row_is_convertible(row):
                    continue
                row["_snapshot_name"] = snapshot_name
                row["_snapshot_date"] = PAT_SNAPSHOT_DATES[snapshot_name]
                row["_hop_type"] = hop_type
                by_stratum[stratum].append(row)

    for stratum, values in by_stratum.items():
        by_stratum[stratum] = sorted(values, key=_source_row_key)
    if limit is None:
        return [row for key in sorted(by_stratum) for row in by_stratum[key]]

    selected: list[dict[str, Any]] = []
    keys = sorted(by_stratum)
    positions = dict.fromkeys(keys, 0)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            position = positions[key]
            if position >= len(by_stratum[key]) or len(selected) >= limit:
                continue
            selected.append(by_stratum[key][position])
            positions[key] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _load_generic_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Source dataset file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    else:
        payload = orjson.loads(path.read_bytes())
        rows = payload if isinstance(payload, list) else list(payload.values())
    return rows[:limit] if limit is not None else rows


def _source_row_key(row: dict[str, Any]) -> str:
    stable = orjson.dumps(
        {
            "question": row.get("question"),
            "subject": row.get("subject"),
            "relations": row.get("relations"),
            "uniq_id": row.get("uniq_id"),
        },
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(stable).hexdigest()
