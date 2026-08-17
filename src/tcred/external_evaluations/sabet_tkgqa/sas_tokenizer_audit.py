from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TOKENIZER_FILES = (
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_ENCODING_FIELDS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "special_tokens_mask",
)


def normalize_bpe_merge_schema(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Convert pair-array BPE merges to the string schema accepted by older tokenizers."""
    normalized = copy.deepcopy(payload)
    model = normalized.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("Expected a tokenizer JSON payload with a BPE model")
    merges = model.get("merges")
    if not isinstance(merges, list) or not merges:
        raise ValueError("Expected a non-empty BPE merge list")

    converted: list[str] = []
    pair_count = 0
    for index, merge in enumerate(merges):
        if isinstance(merge, str):
            converted.append(merge)
            continue
        if (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(token, str) and token for token in merge)
        ):
            converted.append(f"{merge[0]} {merge[1]}")
            pair_count += 1
            continue
        raise ValueError(f"Unsupported BPE merge at index {index}: {merge!r}")
    model["merges"] = converted
    return normalized, pair_count


def audit_sas_tokenizer_compatibility(
    *,
    snapshot_path: Path,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compare slow RoBERTa encodings with the normalized intended fast-tokenizer schema."""
    from transformers import AutoTokenizer

    snapshot_path = snapshot_path.resolve()
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite tokenizer audit: {output_path}")
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"SAS snapshot does not exist: {snapshot_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Metric input does not exist: {input_path}")

    tokenizer_payload = json.loads(
        (snapshot_path / "tokenizer.json").read_text(encoding="utf-8")
    )
    normalized_payload, converted_merge_count = normalize_bpe_merge_schema(
        tokenizer_payload
    )
    if converted_merge_count == 0:
        raise ValueError("The pinned SAS tokenizer did not contain pair-array BPE merges")

    slow = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_fast=False,
    )
    mismatch_examples: list[dict[str, Any]] = []
    mismatch_count = 0
    slow_digest = hashlib.sha256()
    fast_digest = hashlib.sha256()
    seen_metric_ids: set[str] = set()
    row_count = 0

    with tempfile.TemporaryDirectory(prefix="sas-tokenizer-audit-") as temporary:
        compatibility_snapshot = Path(temporary)
        for name in _TOKENIZER_FILES:
            source = snapshot_path / name
            if not source.is_file():
                raise FileNotFoundError(f"Required SAS tokenizer file is missing: {source}")
            if name != "tokenizer.json":
                shutil.copy2(source, compatibility_snapshot / name)
        (compatibility_snapshot / "tokenizer.json").write_text(
            json.dumps(normalized_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        fast = AutoTokenizer.from_pretrained(
            compatibility_snapshot,
            local_files_only=True,
            use_fast=True,
        )
        if not getattr(fast, "is_fast", False):
            raise RuntimeError("Schema-normalized tokenizer did not load as a fast tokenizer")

        maximum_length = min(slow.model_max_length, fast.model_max_length)
        with input_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                metric_id = row.get("metric_id")
                reference = row.get("reference_answer")
                candidate = row.get("candidate_answer")
                if not isinstance(metric_id, str) or not metric_id:
                    raise ValueError(f"Missing metric_id at input line {line_number}")
                if metric_id in seen_metric_ids:
                    raise ValueError(f"Duplicate metric_id in tokenizer audit: {metric_id}")
                if not isinstance(reference, str) or not isinstance(candidate, str):
                    raise ValueError(f"Non-text answer at input line {line_number}")
                seen_metric_ids.add(metric_id)

                kwargs = {
                    "max_length": maximum_length,
                    "truncation": True,
                    "padding": False,
                    "return_attention_mask": True,
                    "return_token_type_ids": True,
                    "return_special_tokens_mask": True,
                }
                slow_encoding = _canonical_encoding(slow(reference, candidate, **kwargs))
                fast_encoding = _canonical_encoding(fast(reference, candidate, **kwargs))
                slow_digest.update(_encoding_bytes(metric_id, slow_encoding))
                fast_digest.update(_encoding_bytes(metric_id, fast_encoding))
                if slow_encoding != fast_encoding:
                    mismatch_count += 1
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append(
                            {
                                "metric_id": metric_id,
                                "slow": slow_encoding,
                                "normalized_fast": fast_encoding,
                            }
                        )
                row_count += 1

    if row_count == 0:
        raise ValueError("Tokenizer audit input is empty")
    slow_sha256 = slow_digest.hexdigest()
    fast_sha256 = fast_digest.hexdigest()
    report = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "passed": mismatch_count == 0 and slow_sha256 == fast_sha256,
        "source_input": _file_identity(input_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_files": {
            name: _file_identity(snapshot_path / name) for name in _TOKENIZER_FILES
        },
        "schema_normalization": {
            "source_merge_representation": "two-element arrays",
            "comparison_merge_representation": "space-delimited strings",
            "converted_merge_count": converted_merge_count,
        },
        "comparison": {
            "row_count": row_count,
            "unique_metric_id_count": len(seen_metric_ids),
            "encoding_fields": list(_ENCODING_FIELDS),
            "maximum_length": maximum_length,
            "slow_tokenizer_class": type(slow).__name__,
            "normalized_fast_tokenizer_class": type(fast).__name__,
            "slow_encoding_sha256": slow_sha256,
            "normalized_fast_encoding_sha256": fast_sha256,
            "mismatch_count": mismatch_count,
            "retained_mismatch_example_count": len(mismatch_examples),
            "mismatch_examples": mismatch_examples,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_output = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_output, output_path)
    return report


def _canonical_encoding(encoding: Any) -> dict[str, list[int]]:
    canonical: dict[str, list[int]] = {}
    for name in _ENCODING_FIELDS:
        value = encoding.get(name)
        if value is None:
            canonical[name] = []
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            canonical[name] = value
        else:
            raise ValueError(f"Unexpected tokenizer encoding field {name}: {type(value)}")
    return canonical


def _encoding_bytes(metric_id: str, encoding: dict[str, list[int]]) -> bytes:
    return (
        json.dumps(
            {"metric_id": metric_id, "encoding": encoding},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen slow SAS tokenizer against its normalized fast schema"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_sas_tokenizer_compatibility(
        snapshot_path=args.snapshot,
        input_path=args.input,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
