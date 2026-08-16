from __future__ import annotations

import json
import shutil
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import orjson

from tcred.trainable_metrics.source_io import file_sha256

FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
FEVER_WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"
FEVER_TRAIN_SHA256 = "eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6"
FEVER_WIKI_SHA256 = "4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2"
MOCHA_DATA_COMMIT = "18c74cd7f8c2ffccf64bb97b371bee10cd10a98a"
MOCHA_ARCHIVE_SHA256 = "036a14c1ad2eb554b77d32051940cb2808f20557f5edcc24b658df703d3c4302"
MOCHA_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/anthonywchen/MOCHA/"
    f"{MOCHA_DATA_COMMIT}/data/mocha.tar.gz"
)
ATTRIBUTION_BENCH_COMMIT = "62569e644f4186606f54f742178a4517431b42e1"
ATTRIBUTION_BENCH_TRAIN_FILE = "train_all_subset_balanced.jsonl"
ATTRIBUTION_BENCH_TRAIN_SHA256 = (
    "c7d6232048298b8f06afc556214c3739b5432f10f2641db2f73715fd126b1a63"
)
ATTRIBUTION_BENCH_TRAIN_URL = (
    "https://huggingface.co/datasets/osunlp/AttributionBench/resolve/"
    f"{ATTRIBUTION_BENCH_COMMIT}/{ATTRIBUTION_BENCH_TRAIN_FILE}"
)


def download_file(
    *,
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    attempts: int = 5,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        digest = file_sha256(destination)
        if expected_sha256 is None or digest == expected_sha256:
            return _file_entry(destination) | {"url": url, "reused": True}
        raise ValueError(
            f"Existing file has the wrong checksum: {destination}; "
            f"expected={expected_sha256}, actual={digest}"
        )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, attempts + 1):
        try:
            with (
                httpx.stream(
                    "GET",
                    url,
                    timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
                    follow_redirects=True,
                    headers={"User-Agent": "T-CRED-research-data-preparation/1.0"},
                ) as response,
                temporary.open("wb") as handle,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
            digest = file_sha256(temporary)
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError(
                    f"Downloaded checksum mismatch for {url}; "
                    f"expected={expected_sha256}, actual={digest}"
                )
            temporary.replace(destination)
            return _file_entry(destination) | {"url": url, "reused": False}
        except (httpx.HTTPError, OSError):
            if attempt == attempts:
                raise
            time.sleep(min(30.0, 2.0**attempt))
    raise AssertionError("unreachable")


def prepare_mocha(*, raw_root: Path) -> dict[str, Any]:
    output_dir = raw_root / "mocha"
    archive_path = output_dir / "mocha.tar.gz"
    archive = download_file(
        url=MOCHA_ARCHIVE_URL,
        destination=archive_path,
        expected_sha256=MOCHA_ARCHIVE_SHA256,
    )
    output_path = output_dir / "train.jsonl"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    row_count = 0
    dataset_counts: Counter[str] = Counter()
    with tarfile.open(archive_path, mode="r:gz") as tar:
        member = tar.getmember("mocha/train.json")
        if not member.isfile():
            raise ValueError("MOCHA archive train member is not a regular file")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError("MOCHA train member could not be read")
        payload = json.load(extracted)
    if not isinstance(payload, dict):
        raise ValueError("MOCHA train root must be a dataset mapping")
    with temporary.open("wb") as handle:
        for constituent_dataset in sorted(payload):
            samples = payload[constituent_dataset]
            if not isinstance(samples, dict):
                raise ValueError("MOCHA constituent dataset must map IDs to samples")
            for native_id in sorted(samples):
                sample = dict(samples[native_id])
                sample["id"] = native_id
                sample["constituent_dataset"] = constituent_dataset
                handle.write(orjson.dumps(sample, option=orjson.OPT_APPEND_NEWLINE))
                row_count += 1
                dataset_counts[constituent_dataset] += 1
    temporary.replace(output_path)
    manifest = {
        "schema_version": "tcred-sl-mocha-acquisition-v1",
        "created_at": datetime.now(UTC),
        "data_commit": MOCHA_DATA_COMMIT,
        "archive": archive,
        "train": _file_entry(output_path) | {"rows": row_count},
        "constituent_datasets": dict(dataset_counts),
    }
    _write_manifest(output_dir / "acquisition_manifest.json", manifest)
    return manifest


def prepare_attribution_bench(*, raw_root: Path) -> dict[str, Any]:
    output_dir = raw_root / "attribution_bench"
    output_path = output_dir / ATTRIBUTION_BENCH_TRAIN_FILE
    download = download_file(
        url=ATTRIBUTION_BENCH_TRAIN_URL,
        destination=output_path,
        expected_sha256=ATTRIBUTION_BENCH_TRAIN_SHA256,
    )
    rows = 0
    labels: Counter[str] = Counter()
    with output_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"AttributionBench row is not an object at {output_path}:{line_number}"
                )
            if not row.get("id") or not row.get("claim") or not row.get("references"):
                raise ValueError(
                    f"AttributionBench row lacks required fields at {output_path}:{line_number}"
                )
            label = str(row.get("attribution_label") or "").strip().lower()
            if label not in {"attributable", "not attributable", "not_attributable"}:
                raise ValueError(
                    f"AttributionBench row has an unsupported label at {output_path}:{line_number}"
                )
            labels[label] += 1
            rows += 1
    if rows != 13_322:
        raise ValueError(f"Expected 13,322 AttributionBench rows, found {rows}")
    manifest = {
        "schema_version": "tcred-sl-attribution-bench-acquisition-v1",
        "created_at": datetime.now(UTC),
        "dataset_revision": ATTRIBUTION_BENCH_COMMIT,
        "download": download,
        "train": _file_entry(output_path) | {"rows": rows},
        "labels": dict(sorted(labels.items())),
        "loader_decision": (
            "Pinned raw JSONL is used because generic Hugging Face streaming casts a field "
            "declared as null to later string values and aborts before yielding the split."
        ),
    }
    _write_manifest(output_dir / "acquisition_manifest.json", manifest)
    return manifest


def prepare_fever(*, raw_root: Path) -> dict[str, Any]:
    output_dir = raw_root / "fever"
    train_path = output_dir / "train.jsonl"
    wiki_path = output_dir / "wiki-pages.zip"
    train = download_file(
        url=FEVER_TRAIN_URL,
        destination=train_path,
        expected_sha256=FEVER_TRAIN_SHA256,
    )
    wiki = download_file(
        url=FEVER_WIKI_URL,
        destination=wiki_path,
        expected_sha256=FEVER_WIKI_SHA256,
    )

    claims, needed_sentences = _load_fever_claims(train_path)
    sentence_text, legacy_unicode_rows = _read_fever_wiki_sentences(
        wiki_path,
        needed_sentences=needed_sentences,
    )
    output_path = output_dir / "fever_train_enriched.jsonl"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    admitted = Counter()
    excluded = Counter()
    with temporary.open("wb") as handle:
        for claim in claims:
            evidence_set = _first_complete_evidence_set(claim["evidence_sets"], sentence_text)
            if evidence_set is None:
                excluded["no_complete_released_evidence_set"] += 1
                continue
            evidence_texts = [
                sentence_text[(page, sentence_id)] for page, sentence_id in evidence_set
            ]
            row = {
                "id": claim["id"],
                "label": claim["label"],
                "claim": claim["claim"],
                "evidence_texts": evidence_texts,
                "evidence_source_ids": [
                    f"{page}:sentence:{sentence_id}" for page, sentence_id in evidence_set
                ],
            }
            handle.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
            admitted[str(claim["label"])] += 1
    temporary.replace(output_path)
    manifest = {
        "schema_version": "tcred-sl-fever-acquisition-v1",
        "created_at": datetime.now(UTC),
        "train_download": train,
        "wiki_download": wiki,
        "eligible_claims_before_join": len(claims),
        "requested_wiki_sentences": len(needed_sentences),
        "resolved_wiki_sentences": len(sentence_text),
        "legacy_unicode_rows_repaired": legacy_unicode_rows,
        "admitted": dict(admitted),
        "excluded": dict(excluded),
        "output": _file_entry(output_path) | {"rows": sum(admitted.values())},
        "evidence_policy": (
            "For each SUPPORTS/REFUTES claim, choose the shortest lexicographically stable "
            "released evidence set for which every referenced Wikipedia sentence resolves."
        ),
    }
    _write_manifest(output_dir / "acquisition_manifest.json", manifest)
    return manifest


def prepare_backbone(
    *,
    model_id: str,
    revision: str,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
    snapshot_dir = output_dir / "upstream_snapshot"
    safe_dir = output_dir / "safe_checkpoint"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    required_files = (
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "README.md",
    )
    for filename in required_files:
        download_file(
            url=f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}",
            destination=snapshot_dir / filename,
        )
    model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    safe_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(safe_dir, safe_serialization=True)
    tokenizer.save_pretrained(safe_dir)
    del model

    source_files = [_file_entry(path) for path in sorted(snapshot_dir.glob("*")) if path.is_file()]
    safe_files = [_file_entry(path) for path in sorted(safe_dir.glob("*")) if path.is_file()]
    if not any(row["path"].endswith("model.safetensors") for row in safe_files):
        raise RuntimeError("Safe backbone conversion did not produce model.safetensors")
    manifest = {
        "schema_version": "tcred-sl-backbone-manifest-v1",
        "created_at": datetime.now(UTC),
        "model_id": model_id,
        "revision": revision,
        "source_files": source_files,
        "safe_checkpoint_files": safe_files,
        "load_policy": "GPU training loads only safe_checkpoint with local_files_only=True",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_dir / "manifest.json", manifest)
    return manifest


def _load_fever_claims(
    path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    claims: list[dict[str, Any]] = []
    needed: set[tuple[str, int]] = set()
    for row in _read_jsonl(path):
        if row.get("label") not in {"SUPPORTS", "REFUTES"}:
            continue
        evidence_sets: list[tuple[tuple[str, int], ...]] = []
        for evidence_set in row.get("evidence", []):
            normalized: list[tuple[str, int]] = []
            for item in evidence_set:
                if not isinstance(item, list) or len(item) < 4:
                    normalized = []
                    break
                page, sentence_id = item[2], item[3]
                if not isinstance(page, str) or not isinstance(sentence_id, int):
                    normalized = []
                    break
                normalized.append((page, sentence_id))
            if normalized:
                candidate = tuple(dict.fromkeys(normalized))
                evidence_sets.append(candidate)
                needed.update(candidate)
        if not evidence_sets:
            continue
        evidence_sets.sort(key=lambda values: (len(values), values))
        claims.append(
            {
                "id": row["id"],
                "label": row["label"],
                "claim": row["claim"],
                "evidence_sets": evidence_sets,
            }
        )
    return claims, needed


def _read_fever_wiki_sentences(
    path: Path,
    *,
    needed_sentences: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], str], int]:
    needed_by_page: defaultdict[str, set[int]] = defaultdict(set)
    for page, sentence_id in needed_sentences:
        needed_by_page[page].add(sentence_id)
    resolved: dict[tuple[str, int], str] = {}
    legacy_unicode_rows = 0
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if (
                not member.filename.endswith(".jsonl")
                or member.filename.startswith("__MACOSX/")
                or Path(member.filename).name.startswith("._")
            ):
                continue
            with archive.open(member) as handle:
                for line in handle:
                    try:
                        row = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        row = json.loads(line)
                        legacy_unicode_rows += 1
                    page = _replace_invalid_surrogates(str(row.get("id") or ""))
                    sentence_ids = needed_by_page.get(page)
                    if not sentence_ids:
                        continue
                    line_map = _parse_fever_lines(
                        _replace_invalid_surrogates(str(row.get("lines") or ""))
                    )
                    for sentence_id in sentence_ids:
                        text = line_map.get(sentence_id)
                        if text:
                            resolved[(page, sentence_id)] = text
    return resolved, legacy_unicode_rows


def _parse_fever_lines(value: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in value.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        try:
            sentence_id = int(fields[0])
        except ValueError:
            continue
        text = " ".join(fields[1].split())
        if text:
            result[sentence_id] = text
    return result


def _replace_invalid_surrogates(value: str) -> str:
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _first_complete_evidence_set(
    evidence_sets: list[tuple[tuple[str, int], ...]],
    sentence_text: dict[tuple[str, int], str],
) -> tuple[tuple[str, int], ...] | None:
    return next(
        (
            evidence_set
            for evidence_set in evidence_sets
            if all(item in sentence_text for item in evidence_set)
        ),
        None,
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_bytes().splitlines():
        if line.strip():
            row = orjson.loads(line)
            if isinstance(row, dict):
                yield row


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    temporary.replace(path)


def clear_partial_download(path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()


def copy_manifest(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
