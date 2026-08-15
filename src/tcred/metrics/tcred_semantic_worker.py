from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from tcred.metrics.config import (
    ALIGNSCORE_BACKBONE,
    ALIGNSCORE_BACKBONE_REVISION,
    ALIGNSCORE_CHECKPOINT_FILENAME,
    ALIGNSCORE_CHECKPOINT_REPO,
    ALIGNSCORE_CHECKPOINT_REVISION,
)
from tcred.metrics.tcred_claims import decompose_claims

_MODEL_ID = (
    f"AlignScore-base@{ALIGNSCORE_CHECKPOINT_REVISION}:"
    f"{ALIGNSCORE_BACKBONE}@{ALIGNSCORE_BACKBONE_REVISION}:pairwise-nli-v1"
)
_CLASS_MAPPING = {"entailment": 0, "neutral": 1, "contradiction": 2}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score T-CRED claim-evidence links")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    rows = _read_jsonl(args.inputs)
    cache = {str(row["metric_id"]): row for row in _read_jsonl(args.cache)}
    valid = {
        str(row["metric_id"]): cache[str(row["metric_id"])]
        for row in rows
        if str(row["metric_id"]) in cache
        and cache[str(row["metric_id"])].get("input_sha256") == _semantic_input_hash(row)
        and cache[str(row["metric_id"])].get("model") == _MODEL_ID
    }
    pending = [row for row in rows if str(row["metric_id"]) not in valid]
    if pending:
        valid.update(
            _score_rows(
                pending,
                model_cache_dir=args.model_cache_dir,
                batch_size=args.batch_size,
            )
        )
        _write_jsonl(args.cache, [valid[str(row["metric_id"])] for row in rows])
    _write_jsonl(args.output, [valid[str(row["metric_id"])] for row in rows])
    return 0


def _score_rows(
    rows: list[dict[str, object]],
    *,
    model_cache_dir: Path,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer, RobertaModel

    offline = os.getenv("HF_HUB_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        ALIGNSCORE_BACKBONE,
        revision=ALIGNSCORE_BACKBONE_REVISION,
        cache_dir=model_cache_dir,
        local_files_only=offline,
    )
    config = AutoConfig.from_pretrained(
        ALIGNSCORE_BACKBONE,
        revision=ALIGNSCORE_BACKBONE_REVISION,
        cache_dir=model_cache_dir,
        local_files_only=offline,
    )
    base_model = RobertaModel(config, add_pooling_layer=True)
    tri_layer = torch.nn.Linear(base_model.config.hidden_size, 3)
    checkpoint_path = hf_hub_download(
        repo_id=ALIGNSCORE_CHECKPOINT_REPO,
        filename=ALIGNSCORE_CHECKPOINT_FILENAME,
        revision=ALIGNSCORE_CHECKPOINT_REVISION,
        cache_dir=model_cache_dir,
        local_files_only=offline,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = checkpoint["state_dict"]
    base_state = {
        name.removeprefix("base_model."): value
        for name, value in state.items()
        if name.startswith("base_model.")
    }
    missing, unexpected = base_model.load_state_dict(base_state, strict=False)
    allowed = {"embeddings.position_ids"}
    if set(missing) - allowed or set(unexpected) - allowed:
        raise RuntimeError(
            f"AlignScore checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    tri_layer.load_state_dict(
        {
            name.removeprefix("tri_layer."): value
            for name, value in state.items()
            if name.startswith("tri_layer.")
        }
    )
    base_model.eval()
    tri_layer.eval()
    torch.set_grad_enabled(False)

    jobs: list[tuple[str, str, int, str, str, str]] = []
    unique_pairs: list[tuple[str, str]] = []
    pair_index: dict[tuple[str, str], int] = {}
    layouts: list[list[int]] = []
    row_claims: dict[str, tuple[list[str], list[str]]] = {}
    evidence_by_metric: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        metric_id = str(row["metric_id"])
        evidence_rows = _semantic_evidence(row)
        decomposition_evidence = [
            (evidence["evidence_id"], evidence["text"]) for evidence in evidence_rows
        ]
        candidate = decompose_claims(
            str(row["candidate_answer"]), evidence=decomposition_evidence
        )
        reference = decompose_claims(
            str(row["reference_answer"]), evidence=decomposition_evidence
        )
        row_claims[metric_id] = (candidate, reference)
        evidence_by_metric[metric_id] = evidence_rows
        for claim_source, claims in (("candidate", candidate), ("reference", reference)):
            for claim_index, claim in enumerate(claims):
                for evidence in evidence_rows:
                    job_index = len(jobs)
                    jobs.append(
                        (
                            metric_id,
                            claim_source,
                            claim_index,
                            claim,
                            evidence["kind"],
                            evidence["evidence_id"],
                        )
                    )
                    key = (evidence["text"], claim)
                    existing = pair_index.get(key)
                    if existing is None:
                        pair_index[key] = len(unique_pairs)
                        unique_pairs.append(key)
                        layouts.append([job_index])
                    else:
                        layouts[existing].append(job_index)

    probabilities: list[tuple[float, float, float]] = []
    for start in range(0, len(unique_pairs), batch_size):
        batch = unique_pairs[start : start + batch_size]
        encoded = tokenizer(
            [premise for premise, _claim in batch],
            [claim for _premise, claim in batch],
            truncation="only_first",
            padding=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        pooled = base_model(**encoded).pooler_output
        values = torch.softmax(tri_layer(pooled), dim=-1).cpu().tolist()
        probabilities.extend((float(row[0]), float(row[1]), float(row[2])) for row in values)

    by_job: dict[int, tuple[float, float, float]] = {}
    for indices, value in zip(layouts, probabilities, strict=True):
        for job_index in indices:
            by_job[job_index] = value
    by_metric: dict[str, list[dict[str, object]]] = {
        str(row["metric_id"]): [] for row in rows
    }
    evidence_texts = {
        metric_id: {
            (item["kind"], item["evidence_id"]): item["text"] for item in evidence_rows
        }
        for metric_id, evidence_rows in evidence_by_metric.items()
    }
    for index, job in enumerate(jobs):
        metric_id, source, claim_index, claim, kind, evidence_id = job
        entailment, neutral, contradiction = by_job[index]
        text = evidence_texts[metric_id][(kind, evidence_id)]
        by_metric[metric_id].append(
            {
                "claim_source": source,
                "claim_index": claim_index,
                "claim": claim,
                "evidence_kind": kind,
                "evidence_id": evidence_id,
                "evidence_text_sha256": _sha256_text(text),
                "entailment": entailment,
                "neutral": neutral,
                "contradiction": contradiction,
            }
        )

    output = {
        str(row["metric_id"]): {
            "schema_version": "1.0",
            "metric_id": str(row["metric_id"]),
            "input_sha256": _semantic_input_hash(row),
            "model": _MODEL_ID,
            "class_mapping": _CLASS_MAPPING,
            "candidate_claims": row_claims[str(row["metric_id"])][0],
            "reference_claims": row_claims[str(row["metric_id"])][1],
            "pairs": by_metric[str(row["metric_id"])],
        }
        for row in rows
    }
    del checkpoint, state, base_state, base_model, tri_layer
    gc.collect()
    return output


def _semantic_evidence(row: dict[str, object]) -> list[dict[str, str]]:
    output = []
    seen: set[tuple[str, str]] = set()
    for kind, field in (("retrieved", "retrieved_evidence"), ("cited", "cited_evidence")):
        for evidence in _list_of_dicts(row.get(field)):
            evidence_id = str(evidence.get("evidence_id", ""))
            key = (kind, evidence_id)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "kind": kind,
                    "evidence_id": evidence_id,
                    "text": str(evidence.get("text", "")),
                }
            )
    for path in _list_of_dicts(row.get("graph_paths")):
        path_id = str(path.get("path_id", ""))
        for index, edge in enumerate(_list_of_dicts(path.get("edges"))):
            fact_id = str(edge.get("fact_id", ""))
            evidence_id = f"{path_id}:edge:{index}:{fact_id}"
            text = str(edge.get("evidence_text", "")).strip()
            if not text:
                source = _dict(edge.get("source"))
                target = _dict(edge.get("target"))
                relation = str(
                    edge.get("relation_label") or edge.get("relation") or "related to"
                )
                text = f"{source.get('label', '')} {relation} {target.get('label', '')}."
            output.append({"kind": "path", "evidence_id": evidence_id, "text": text})
    return output


def _semantic_input_hash(row: dict[str, object]) -> str:
    evidence = _semantic_evidence(row)
    decomposition_evidence = [
        (item["evidence_id"], item["text"]) for item in evidence
    ]
    payload = {
        "metric_id": str(row["metric_id"]),
        "candidate_claims": decompose_claims(
            str(row["candidate_answer"]), evidence=decomposition_evidence
        ),
        "reference_claims": decompose_claims(
            str(row["reference_answer"]), evidence=decomposition_evidence
        ),
        "evidence": evidence,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                output.append(row)
    metric_ids = [str(row.get("metric_id", "")) for row in output]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"Duplicate metric IDs in {path}")
    return output


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
