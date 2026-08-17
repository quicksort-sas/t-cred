from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import urllib.request
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tcred.metrics.config import (
    ALIGNSCORE_BACKBONE,
    ALIGNSCORE_BACKBONE_REVISION,
    ALIGNSCORE_BATCH_SIZE,
    ALIGNSCORE_CHECKPOINT_FILENAME,
    ALIGNSCORE_CHECKPOINT_REPO,
    ALIGNSCORE_CHECKPOINT_REVISION,
    ALIGNSCORE_IMPLEMENTATION_VERSION,
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_REVISION,
    MINICHECK_MODEL,
    MINICHECK_REVISION,
    NLTK_PUNKT_TAB_SHA256,
    PEDANTS_REVISION,
    SAS_MODEL,
    SAS_REVISION,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_NEURAL_INPUT_VERSION = "tcred-neural-metrics-v2"
_NEURAL_OUTPUT_VERSION = "tcred-neural-output-v1"
_NEURAL_RUNTIME_VERSION = "tcred-neural-runtime-v1"
_CACHE_CHECKPOINT_VERSION = "content-addressed-jsonl-parts-v1"
_BERTSCORE_CHECKPOINT_ROWS = 256
_MINICHECK_CHECKPOINT_ROWS = 64
_SAS_CHECKPOINT_ROWS = 256
_PEDANTS_CHECKPOINT_ROWS = 512
_ALIGNSCORE_CHECKPOINT_ROWS = 64
_SAS_TOKENIZER_BACKEND = "slow-roberta-vocab-merges-v1"
_TRANSFORMERS_RUNTIME_FILES = (
    "config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_SAS_RUNTIME_FILES = (
    "config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
_PEDANTS_ASSETS = {
    "lr_classifier.pkl": "4c132d47ea40c352831c158671d7b11a445769b9402cf13f4698ea10b6f59ab1",
    "rule_classifier.pkl": "ff5761ea5a7f84bc9911bf9f4ca5248b18a4125298c79f8122b834c04d323b63",
    "type_classifier.pkl": "e9a823f3f4417d7f68a6e47f78d7ad9fff9266d21f65350e4233fb429d15ee1a",
    "tf-idf_vectorizer.pkl": "4d869e2f0b121789920e9da416549c3b96da9548cb603d73c5d0c314c9a10920",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pinned local non-LLM metrics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--metrics", default="bertscore,minicheck,sas,pedants,alignscore")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bertscore-model", default=BERTSCORE_MODEL)
    parser.add_argument("--minicheck-model", default=MINICHECK_MODEL)
    parser.add_argument("--sas-model", default=SAS_MODEL)
    parser.add_argument("--alignscore-model", default=ALIGNSCORE_BACKBONE)
    parser.add_argument(
        "--minicheck-scope",
        choices=("human_gold", "all"),
        default="human_gold",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--minicheck-batch-size", type=int, default=8)
    parser.add_argument("--sas-batch-size", type=int, default=8)
    parser.add_argument("--alignscore-batch-size", type=int, default=ALIGNSCORE_BATCH_SIZE)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Neural input and output paths must differ")
    if args.manifest is not None and args.manifest.resolve() in {
        args.input.resolve(),
        args.output.resolve(),
    }:
        raise ValueError("Neural manifest must not overwrite the input or score output")
    for name in (
        "bertscore_batch_size",
        "minicheck_batch_size",
        "sas_batch_size",
        "alignscore_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    args.device = _resolve_device(args.device)
    os.environ.setdefault("HF_HOME", str(args.model_cache_dir.resolve()))
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((args.model_cache_dir / "matplotlib").resolve()),
    )

    requested = {value.strip() for value in args.metrics.split(",") if value.strip()}
    if not requested:
        raise ValueError("At least one neural metric must be requested")
    unknown = requested - {"bertscore", "minicheck", "sas", "pedants", "alignscore"}
    if unknown:
        raise ValueError(f"Unknown neural metric(s): {sorted(unknown)}")
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    rows = _read_jsonl(args.input)
    _validate_input_rows(rows)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    runtime = _neural_runtime_contract(device=args.device, requested=requested)
    runtime_sha256 = _canonical_sha256(runtime)

    records: dict[str, dict[str, Any]] = {
        row["metric_id"]: {
            "schema_version": _NEURAL_OUTPUT_VERSION,
            "metric_id": row["metric_id"],
            "input_sha256": neural_input_hash(row),
            "scores": {},
            "metadata": {"runtime_contract_sha256": runtime_sha256},
        }
        for row in rows
    }
    if "bertscore" in requested:
        _compute_bertscore(rows, records, args)
    if "minicheck" in requested:
        minicheck_rows = (
            [row for row in rows if row["population"] == "human_gold"]
            if args.minicheck_scope == "human_gold"
            else rows
        )
        _compute_minicheck(minicheck_rows, records, args)
    if "sas" in requested:
        _compute_sas(rows, records, args)
    if "pedants" in requested:
        _compute_pedants(rows, records, args)
    if "alignscore" in requested:
        _compute_alignscore(rows, records, args)
    _write_jsonl(args.output, [records[row["metric_id"]] for row in rows])
    if args.manifest is not None:
        completed_utc = datetime.now(UTC)
        manifest = {
            "schema_version": "1.0",
            "worker_contract": _NEURAL_OUTPUT_VERSION,
            "worker_implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "started_utc": started_utc.isoformat(),
            "completed_utc": completed_utc.isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "requested_metrics": sorted(requested),
            "row_count": len(rows),
            "input": _file_identity(args.input.resolve()),
            "output": _file_identity(args.output.resolve()),
            "runtime_contract": runtime,
            "runtime_contract_sha256": runtime_sha256,
            "metric_configuration": _metric_configuration(requested, args),
            "cache_checkpoint_contract": {
                "version": _CACHE_CHECKPOINT_VERSION,
                "part_identity": "SHA-256 of canonical JSONL bytes",
                "part_publication": "fsync then atomic replace",
                "restart_policy": "merge valid consolidated cache and immutable parts",
                "conflict_policy": "reject differing valid records for one metric_id",
                "completion_policy": "atomic ordered consolidation then remove parts",
            },
            "invocation": {
                "device": args.device,
                "minicheck_scope": args.minicheck_scope,
                "bertscore_batch_size": args.bertscore_batch_size,
                "minicheck_batch_size": args.minicheck_batch_size,
                "sas_batch_size": args.sas_batch_size,
                "alignscore_batch_size": args.alignscore_batch_size,
                "cache_dir": str(args.cache_dir.resolve()),
                "model_cache_dir": str(args.model_cache_dir.resolve()),
            },
        }
        _write_json(args.manifest, manifest)


def _compute_bertscore(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    from bert_score import BERTScorer
    from huggingface_hub import snapshot_download

    offline = os.getenv("HF_HUB_OFFLINE") == "1"
    snapshot_path = snapshot_download(
        repo_id=args.bertscore_model,
        revision=BERTSCORE_REVISION,
        cache_dir=args.model_cache_dir / "hub",
        local_files_only=offline,
        allow_patterns=list(_TRANSFORMERS_RUNTIME_FILES),
    )
    identity = (
        f"{args.bertscore_model}@{BERTSCORE_REVISION}:layer{BERTSCORE_NUM_LAYERS}:idf0:rescale0"
    )
    cache_path = args.cache_dir / "bertscore.jsonl"
    cached = _valid_cache(cache_path, rows, model=identity)
    pending = [row for row in rows if row["metric_id"] not in cached]
    if pending:
        print(f"BERTScore: scoring {len(pending)} rows ({len(cached)} cached)", flush=True)
        scorer = BERTScorer(
            model_type=snapshot_path,
            num_layers=BERTSCORE_NUM_LAYERS,
            batch_size=args.bertscore_batch_size,
            device=args.device,
            idf=False,
            rescale_with_baseline=False,
        )
        for start in range(0, len(pending), _BERTSCORE_CHECKPOINT_ROWS):
            chunk = pending[start : start + _BERTSCORE_CHECKPOINT_ROWS]
            precision, recall, f1 = scorer.score(
                [row["candidate_answer"] for row in chunk],
                [row["reference_answer"] for row in chunk],
            )
            for row, p_score, r_score, f_score in zip(
                chunk,
                precision,
                recall,
                f1,
                strict=True,
            ):
                cached[row["metric_id"]] = {
                    "metric_id": row["metric_id"],
                    "input_sha256": neural_input_hash(row),
                    "model": identity,
                    "scores": {
                        "bertscore_precision": float(p_score),
                        "bertscore_recall": float(r_score),
                        "bertscore_f1": float(f_score),
                    },
                }
            _write_cache_checkpoint(
                cache_path,
                [cached[row["metric_id"]] for row in chunk],
            )
            print(
                f"BERTScore: {min(start + len(chunk), len(pending))}/{len(pending)} new rows",
                flush=True,
            )
        del scorer
        gc.collect()
    _consolidate_cache(cache_path, cached, rows)
    selected_ids = {row["metric_id"] for row in rows}
    for metric_id, record in cached.items():
        if metric_id not in selected_ids:
            continue
        records[metric_id]["scores"].update(record["scores"])
        records[metric_id]["metadata"]["bertscore"] = {
            "model": args.bertscore_model,
            "revision": BERTSCORE_REVISION,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "idf": False,
            "rescale_with_baseline": False,
            "runtime_device": args.device,
        }


def _compute_minicheck(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    offline = os.getenv("HF_HUB_OFFLINE") == "1"
    identity = f"{args.minicheck_model}@{MINICHECK_REVISION}:tcred-sentence-max-mean-v1"
    cache_path = args.cache_dir / "minicheck.jsonl"
    cached = _valid_cache(cache_path, rows, model=identity)
    pending = [row for row in rows if row["metric_id"] not in cached]
    if pending:
        print(
            f"MiniCheck: loading model for {len(pending)} rows ({len(cached)} cached)", flush=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.minicheck_model,
            revision=MINICHECK_REVISION,
            cache_dir=args.model_cache_dir,
            use_fast=True,
            local_files_only=offline,
            fix_mistral_regex=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.minicheck_model,
            revision=MINICHECK_REVISION,
            cache_dir=args.model_cache_dir,
            use_safetensors=False,
            local_files_only=offline,
        )
        model.to(args.device)
        model.eval()
        torch.set_grad_enabled(False)
        for row_start in range(0, len(pending), _MINICHECK_CHECKPOINT_ROWS):
            chunk_rows = pending[row_start : row_start + _MINICHECK_CHECKPOINT_ROWS]
            pairs, layouts = _minicheck_pairs(chunk_rows, tokenizer)
            probabilities: list[float] = []
            for start in range(0, len(pairs), args.minicheck_batch_size):
                batch = pairs[start : start + args.minicheck_batch_size]
                encoded = tokenizer(
                    [f"{document}{tokenizer.eos_token}{claim}" for document, claim in batch],
                    max_length=2048,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: value.to(args.device) for name, value in encoded.items()}
                logits = model(**encoded).logits
                probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

            grouped: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
            for keys, probability in zip(layouts, probabilities, strict=True):
                for metric_id, evidence_kind, sentence_index, _chunk_index in keys:
                    grouped[(metric_id, evidence_kind, sentence_index)].append(probability)

            for row in chunk_rows:
                scores: dict[str, float | None] = {}
                sentence_count = len(_claim_sentences(row["candidate_answer"]))
                for evidence_kind in ("retrieved", "cited"):
                    per_sentence = [
                        max(
                            grouped[(row["metric_id"], evidence_kind, index)],
                            default=float("nan"),
                        )
                        for index in range(sentence_count)
                    ]
                    valid = [value for value in per_sentence if not math.isnan(value)]
                    scores[f"minicheck_{evidence_kind}_mean"] = (
                        sum(valid) / len(valid) if valid else None
                    )
                    scores[f"minicheck_{evidence_kind}_strict"] = min(valid) if valid else None
                cached[row["metric_id"]] = {
                    "metric_id": row["metric_id"],
                    "input_sha256": neural_input_hash(row),
                    "model": identity,
                    "scores": scores,
                }
            _write_cache_checkpoint(
                cache_path,
                [cached[row["metric_id"]] for row in chunk_rows],
            )
            print(
                f"MiniCheck: {min(row_start + len(chunk_rows), len(pending))}/"
                f"{len(pending)} new rows ({len(pairs)} unique pairs in checkpoint)",
                flush=True,
            )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    _consolidate_cache(cache_path, cached, rows)
    for metric_id, record in cached.items():
        records[metric_id]["scores"].update(record["scores"])
        records[metric_id]["metadata"]["minicheck"] = {
            "model": args.minicheck_model,
            "revision": MINICHECK_REVISION,
            "runtime_device": args.device,
        }


def _compute_sas(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Compute the paper's English SAS cross-encoder score."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    identity = f"{args.sas_model}@{SAS_REVISION}:{_SAS_TOKENIZER_BACKEND}"
    cache_path = args.cache_dir / "sas.jsonl"
    cached = _valid_cache(cache_path, rows, model=identity)
    pending = [row for row in rows if row["metric_id"] not in cached]
    if pending:
        offline = os.getenv("HF_HUB_OFFLINE") == "1"
        print(f"SAS: loading model for {len(pending)} rows ({len(cached)} cached)", flush=True)
        snapshot_path = snapshot_download(
            repo_id=args.sas_model,
            revision=SAS_REVISION,
            cache_dir=args.model_cache_dir / "hub",
            local_files_only=offline,
            allow_patterns=list(_SAS_RUNTIME_FILES),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        model.to(args.device)
        model.eval()
        torch.set_grad_enabled(False)
        for row_start in range(0, len(pending), _SAS_CHECKPOINT_ROWS):
            chunk_rows = pending[row_start : row_start + _SAS_CHECKPOINT_ROWS]
            scores: list[float] = []
            for start in range(0, len(chunk_rows), args.sas_batch_size):
                batch = chunk_rows[start : start + args.sas_batch_size]
                encoded = tokenizer(
                    [row["reference_answer"] for row in batch],
                    [row["candidate_answer"] for row in batch],
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: value.to(args.device) for name, value in encoded.items()}
                logits = model(**encoded).logits.reshape(-1)
                scores.extend(torch.sigmoid(logits).cpu().tolist())
            for row, score in zip(chunk_rows, scores, strict=True):
                cached[row["metric_id"]] = {
                    "metric_id": row["metric_id"],
                    "input_sha256": neural_input_hash(row),
                    "model": identity,
                    "scores": {"sas_cross_encoder": float(score)},
                }
            _write_cache_checkpoint(
                cache_path,
                [cached[row["metric_id"]] for row in chunk_rows],
            )
            print(
                f"SAS: {min(row_start + len(chunk_rows), len(pending))}/{len(pending)} new rows",
                flush=True,
            )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    _consolidate_cache(cache_path, cached, rows)
    for metric_id, record in cached.items():
        records[metric_id]["scores"].update(record["scores"])
        records[metric_id]["metadata"]["sas"] = {
            "model": args.sas_model,
            "revision": SAS_REVISION,
            "activation": "sigmoid",
            "tokenizer_backend": _SAS_TOKENIZER_BACKEND,
            "allowed_snapshot_files": list(_SAS_RUNTIME_FILES),
            "runtime_device": args.device,
        }


def _compute_pedants(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Compute PEDANTS with the authors' pinned logistic-regression artifacts."""
    import joblib
    import numpy as np
    import sklearn
    from scipy.sparse import hstack

    identity = f"pedants@{PEDANTS_REVISION}"
    cache_path = args.cache_dir / "pedants.jsonl"
    cached = _valid_cache(
        cache_path,
        rows,
        model=identity,
        input_hash=_pedants_input_hash,
    )
    pending = [row for row in rows if row["metric_id"] not in cached]
    compatibility_shim_required = _pedants_sklearn_compatibility_required()
    compatibility_shim_installed = False
    if pending:
        assets = _ensure_pedants_assets(args.model_cache_dir)
        compatibility_shim_installed = _install_pedants_sklearn_compatibility()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = joblib.load(assets["lr_classifier.pkl"])
            rule_classifier = joblib.load(assets["rule_classifier.pkl"])
            type_classifier = joblib.load(assets["type_classifier.pkl"])
            vectorizer = joblib.load(assets["tf-idf_vectorizer.pkl"])
        classes = list(classifier.classes_)
        if classes != ["correct", "incorrect"]:
            raise RuntimeError(f"Unexpected PEDANTS class order: {classes}")
        print(f"PEDANTS: scoring {len(pending)} rows ({len(cached)} cached)", flush=True)
        for row_start in range(0, len(pending), _PEDANTS_CHECKPOINT_ROWS):
            chunk_rows = pending[row_start : row_start + _PEDANTS_CHECKPOINT_ROWS]
            scored: dict[str, tuple[float, float]] = {}
            model_rows: list[tuple[dict[str, Any], str, str, str]] = []
            for row in chunk_rows:
                reference = str(row["reference_answer"])
                candidate = str(row["candidate_answer"])
                if not reference or not candidate:
                    scored[row["metric_id"]] = (0.0, 0.0)
                    continue
                normalized_reference = _pedants_normalize(reference)
                normalized_candidate = _pedants_normalize(candidate)
                normalized_question = _pedants_normalize(str(row["question"]))
                if normalized_reference in normalized_candidate:
                    scored[row["metric_id"]] = (1.0, 1.0)
                    continue
                model_rows.append(
                    (row, normalized_reference, normalized_candidate, normalized_question)
                )

            if model_rows:
                main_text = [
                    f"[CLS] {question} [SEP] {reference} [SEP] {candidate} [SEP]"
                    for _row, reference, candidate, question in model_rows
                ]
                type_text = [
                    f"[CLS] {question} [SEP] {reference} [SEP]"
                    for _row, reference, _candidate, question in model_rows
                ]
                overlap = np.asarray(
                    [
                        _pedants_overlap(reference, candidate)
                        for _, reference, candidate, _ in model_rows
                    ],
                    dtype=float,
                )
                text_features = vectorizer.transform(main_text)
                rule_features = rule_classifier.predict_proba(
                    hstack([text_features, overlap[:, 0:1], overlap[:, 1:2], overlap[:, 2:3]])
                )
                type_features = type_classifier.predict_proba(vectorizer.transform(type_text))
                features = hstack(
                    [
                        overlap[:, 0:1],
                        overlap[:, 1:2],
                        overlap[:, 2:3],
                        rule_features,
                        type_features,
                        text_features,
                    ]
                )
                probabilities = classifier.predict_proba(features)[:, 0]
                predictions = classifier.predict(features)
                for (row, _reference, _candidate, _question), probability, prediction in zip(
                    model_rows,
                    probabilities,
                    predictions,
                    strict=True,
                ):
                    scored[row["metric_id"]] = (
                        float(probability),
                        float(prediction == "correct"),
                    )

            for row in chunk_rows:
                probability, match = scored[row["metric_id"]]
                cached[row["metric_id"]] = {
                    "metric_id": row["metric_id"],
                    "input_sha256": _pedants_input_hash(row),
                    "model": identity,
                    "scores": {
                        "pedants_probability": probability,
                        "pedants_match": match,
                    },
                }
            _write_cache_checkpoint(
                cache_path,
                [cached[row["metric_id"]] for row in chunk_rows],
            )
            print(
                f"PEDANTS: {min(row_start + len(chunk_rows), len(pending))}/"
                f"{len(pending)} new rows",
                flush=True,
            )
    _consolidate_cache(cache_path, cached, rows)
    for metric_id, record in cached.items():
        records[metric_id]["scores"].update(record["scores"])
        records[metric_id]["metadata"]["pedants"] = {
            "artifact_revision": PEDANTS_REVISION,
            "artifact_sha256": dict(_PEDANTS_ASSETS),
            "scikit_learn_runtime": sklearn.__version__,
            "legacy_loss_unpickle_shim_required": compatibility_shim_required,
            "legacy_loss_unpickle_shim_installed_during_run": compatibility_shim_installed,
        }


def _compute_alignscore(
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Compute official AlignScore-base NLI-SP aggregation over displayed evidence."""
    import nltk
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer, RobertaModel

    identity = (
        f"AlignScore-base@{ALIGNSCORE_CHECKPOINT_REVISION}:"
        f"{args.alignscore_model}@{ALIGNSCORE_BACKBONE_REVISION}:"
        f"{ALIGNSCORE_IMPLEMENTATION_VERSION}"
    )
    cache_path = args.cache_dir / "alignscore.jsonl"
    cached = _valid_cache(cache_path, rows, model=identity)
    pending = [row for row in rows if row["metric_id"] not in cached]
    nltk_dir = args.model_cache_dir / "nltk"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(nltk_dir.resolve()))
    offline = os.getenv("HF_HUB_OFFLINE") == "1"
    _ensure_nltk_punkt(nltk, nltk_dir, offline=offline)
    punkt_tab_sha256 = _directory_sha256(nltk_dir / "tokenizers" / "punkt_tab")
    if punkt_tab_sha256 != NLTK_PUNKT_TAB_SHA256:
        raise RuntimeError(
            "NLTK punkt_tab resource checksum mismatch: "
            f"expected {NLTK_PUNKT_TAB_SHA256}, found {punkt_tab_sha256}"
        )
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(
            args.alignscore_model,
            revision=ALIGNSCORE_BACKBONE_REVISION,
            cache_dir=args.model_cache_dir,
            local_files_only=offline,
        )
        config = AutoConfig.from_pretrained(
            args.alignscore_model,
            revision=ALIGNSCORE_BACKBONE_REVISION,
            cache_dir=args.model_cache_dir,
            local_files_only=offline,
        )
        base_model = RobertaModel(config, add_pooling_layer=True)
        tri_layer = torch.nn.Linear(base_model.config.hidden_size, 3)
        checkpoint_path = hf_hub_download(
            repo_id=ALIGNSCORE_CHECKPOINT_REPO,
            filename=ALIGNSCORE_CHECKPOINT_FILENAME,
            revision=ALIGNSCORE_CHECKPOINT_REVISION,
            cache_dir=args.model_cache_dir,
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
        del checkpoint, state, base_state
        base_model.eval()
        tri_layer.eval()
        base_model.to(args.device)
        tri_layer.to(args.device)
        torch.set_grad_enabled(False)
        print(
            f"AlignScore-base: scoring {len(pending)} rows ({len(cached)} cached)",
            flush=True,
        )
        for row_start in range(0, len(pending), _ALIGNSCORE_CHECKPOINT_ROWS):
            chunk_rows = pending[row_start : row_start + _ALIGNSCORE_CHECKPOINT_ROWS]
            pairs, layouts = _alignscore_pairs(chunk_rows, nltk.sent_tokenize)
            entailment: list[float] = []
            for start in range(0, len(pairs), args.alignscore_batch_size):
                batch = pairs[start : start + args.alignscore_batch_size]
                encoded = tokenizer(
                    [premise for premise, _hypothesis in batch],
                    [hypothesis for _premise, hypothesis in batch],
                    truncation="only_first",
                    padding=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                encoded = {name: value.to(args.device) for name, value in encoded.items()}
                pooled = base_model(**encoded).pooler_output
                logits = tri_layer(pooled)
                entailment.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())

            grouped: defaultdict[tuple[str, str, int], list[float]] = defaultdict(list)
            for keys, score in zip(layouts, entailment, strict=True):
                for metric_id, evidence_kind, sentence_index, _chunk_index in keys:
                    grouped[(metric_id, evidence_kind, sentence_index)].append(score)

            for row in chunk_rows:
                candidate_sentences = _alignscore_sentences(
                    str(row["candidate_answer"]),
                    nltk.sent_tokenize,
                )
                scores: dict[str, float | None] = {}
                for evidence_kind, field_name in (
                    ("retrieved", "retrieved_evidence"),
                    ("cited", "cited_evidence"),
                ):
                    if not row[field_name] or not candidate_sentences:
                        scores[f"alignscore_{evidence_kind}"] = None
                        continue
                    sentence_scores = [
                        max(grouped[(row["metric_id"], evidence_kind, index)], default=0.0)
                        for index in range(len(candidate_sentences))
                    ]
                    scores[f"alignscore_{evidence_kind}"] = sum(sentence_scores) / len(
                        sentence_scores
                    )
                cached[row["metric_id"]] = {
                    "metric_id": row["metric_id"],
                    "input_sha256": neural_input_hash(row),
                    "model": identity,
                    "scores": scores,
                }
            _write_cache_checkpoint(
                cache_path,
                [cached[row["metric_id"]] for row in chunk_rows],
            )
            print(
                f"AlignScore-base: {min(row_start + len(chunk_rows), len(pending))}/"
                f"{len(pending)} new rows ({len(pairs)} unique pairs in checkpoint)",
                flush=True,
            )
        del base_model, tri_layer
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    _consolidate_cache(cache_path, cached, rows)
    for metric_id, record in cached.items():
        records[metric_id]["scores"].update(record["scores"])
        records[metric_id]["metadata"]["alignscore"] = {
            "checkpoint": f"{ALIGNSCORE_CHECKPOINT_REPO}/{ALIGNSCORE_CHECKPOINT_FILENAME}",
            "checkpoint_revision": ALIGNSCORE_CHECKPOINT_REVISION,
            "backbone": args.alignscore_model,
            "backbone_revision": ALIGNSCORE_BACKBONE_REVISION,
            "evaluation_mode": "nli_sp",
            "implementation_version": ALIGNSCORE_IMPLEMENTATION_VERSION,
            "sentence_tokenizer": "nltk.sent_tokenize",
            "nltk_version": nltk.__version__,
            "punkt_tab_sha256": punkt_tab_sha256,
            "runtime_device": args.device,
        }


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for neural metrics but is unavailable")
    return requested


def _minicheck_pairs(
    rows: list[dict[str, Any]],
    tokenizer: object,
) -> tuple[list[tuple[str, str]], list[list[tuple[str, str, int, int]]]]:
    pairs: list[tuple[str, str]] = []
    layouts: list[list[tuple[str, str, int, int]]] = []
    pair_indices: dict[tuple[str, str], int] = {}
    for row in rows:
        claims = _claim_sentences(row["candidate_answer"])
        for evidence_kind, field_name in (
            ("retrieved", "retrieved_evidence"),
            ("cited", "cited_evidence"),
        ):
            evidence = [item["text"] for item in row[field_name]]
            chunks = _document_chunks(evidence, tokenizer, max_tokens=400)
            for sentence_index, claim in enumerate(claims):
                for chunk_index, chunk in enumerate(chunks):
                    pair = (chunk, claim)
                    key = (row["metric_id"], evidence_kind, sentence_index, chunk_index)
                    pair_index = pair_indices.get(pair)
                    if pair_index is None:
                        pair_indices[pair] = len(pairs)
                        pairs.append(pair)
                        layouts.append([key])
                    else:
                        layouts[pair_index].append(key)
    return pairs, layouts


def _document_chunks(evidence: list[str], tokenizer: object, *, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for text in evidence:
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(token_ids) > max_tokens:
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            for start in range(0, len(token_ids), max_tokens):
                chunks.append(tokenizer.decode(token_ids[start : start + max_tokens]))
            continue
        if current and current_tokens + len(token_ids) > max_tokens:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += len(token_ids)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _claim_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["The response is empty."]
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(stripped) if part.strip()]


def _ensure_pedants_assets(model_cache_dir: Path) -> dict[str, Path]:
    asset_dir = model_cache_dir / "pedants" / PEDANTS_REVISION
    asset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, expected_sha256 in _PEDANTS_ASSETS.items():
        path = asset_dir / filename
        if path.is_file() and _file_sha256(path) == expected_sha256:
            paths[filename] = path
            continue
        url = (
            "https://raw.githubusercontent.com/zli12321/pedant_models/"
            f"{PEDANTS_REVISION}/{filename}"
        )
        temporary = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "tcred-metric-audit/1.0"})
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        actual_sha256 = _file_sha256(temporary)
        if actual_sha256 != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"PEDANTS asset checksum mismatch for {filename}: {actual_sha256}")
        temporary.replace(path)
        paths[filename] = path
    return paths


def _install_pedants_sklearn_compatibility() -> bool:
    """Allow audited sklearn-1.3.2 SGD pickles to load on current sklearn."""
    import sklearn.linear_model._sgd_fast as sgd_fast

    if hasattr(sgd_fast, "Log"):
        return False
    legacy_log = type("Log", (), {})
    legacy_log.__module__ = sgd_fast.__name__
    sgd_fast.Log = legacy_log
    return True


def _pedants_sklearn_compatibility_required() -> bool:
    import sklearn.linear_model._sgd_fast as sgd_fast

    return not hasattr(sgd_fast, "Log")


def _pedants_normalize(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if not unicodedata.category(character).startswith("P")
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split()).strip()


def _pedants_overlap(reference: str, candidate: str) -> tuple[float, float, float]:
    reference_words = set(reference.split())
    candidate_words = set(candidate.split())
    overlap = len(reference_words & candidate_words)
    precision = overlap / len(reference_words) if reference_words else 0.0
    recall = overlap / len(candidate_words) if candidate_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def _pedants_input_hash(row: dict[str, Any]) -> str:
    content = {
        "implementation_version": "tcred-pedants-v1",
        "question": row["question"],
        "candidate_answer": row["candidate_answer"],
        "reference_answer": row["reference_answer"],
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _ensure_nltk_punkt(nltk: Any, nltk_dir: Path, *, offline: bool) -> None:
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for package, resource in resources.items():
        try:
            nltk.data.find(resource, paths=[str(nltk_dir.resolve())])
            continue
        except LookupError:
            if offline:
                raise RuntimeError(
                    f"AlignScore requires local NLTK resource {package}, but the worker is offline"
                ) from None
        if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
            raise RuntimeError(f"Could not download NLTK resource: {package}")
        try:
            nltk.data.find(resource, paths=[str(nltk_dir.resolve())])
        except LookupError as error:
            raise RuntimeError(
                f"NLTK reported a successful download but {resource} is unavailable"
            ) from error
    nltk.sent_tokenize("A sentence. Another sentence.")


def _alignscore_sentences(
    text: str,
    sentence_splitter: Callable[[str], list[str]],
) -> list[str]:
    if not text.strip():
        return []
    return [sentence.strip() for sentence in sentence_splitter(text) if sentence.strip()]


def _alignscore_chunks(
    text: str,
    sentence_splitter: Callable[[str], list[str]],
) -> list[str]:
    sentences = _alignscore_sentences(text, sentence_splitter) or [""]
    target_chunks = len(text.strip().split()) // 350 + 1
    sentences_per_chunk = max(len(sentences) // target_chunks, 1)
    return [
        " ".join(sentences[start : start + sentences_per_chunk])
        for start in range(0, len(sentences), sentences_per_chunk)
    ]


def _alignscore_pairs(
    rows: list[dict[str, Any]],
    sentence_splitter: Callable[[str], list[str]],
) -> tuple[list[tuple[str, str]], list[list[tuple[str, str, int, int]]]]:
    pairs: list[tuple[str, str]] = []
    layouts: list[list[tuple[str, str, int, int]]] = []
    pair_indices: dict[tuple[str, str], int] = {}
    for row in rows:
        candidate_sentences = _alignscore_sentences(
            str(row["candidate_answer"]),
            sentence_splitter,
        )
        for evidence_kind, field_name in (
            ("retrieved", "retrieved_evidence"),
            ("cited", "cited_evidence"),
        ):
            evidence_text = " ".join(str(item["text"]) for item in row[field_name])
            if not evidence_text.strip() or not candidate_sentences:
                continue
            chunks = _alignscore_chunks(evidence_text, sentence_splitter)
            for sentence_index, candidate_sentence in enumerate(candidate_sentences):
                for chunk_index, chunk in enumerate(chunks):
                    pair = (chunk, candidate_sentence)
                    key = (row["metric_id"], evidence_kind, sentence_index, chunk_index)
                    pair_index = pair_indices.get(pair)
                    if pair_index is None:
                        pair_indices[pair] = len(pairs)
                        pairs.append(pair)
                        layouts.append([key])
                    else:
                        layouts[pair_index].append(key)
    return pairs, layouts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot hash missing directory: {path}")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _valid_cache(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    model: str,
    input_hash: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, dict[str, Any]]:
    hash_function = input_hash or neural_input_hash
    expected = {row["metric_id"]: hash_function(row) for row in rows}
    cached: dict[str, dict[str, Any]] = {}
    for cache_file in _cache_files(path):
        for record in _read_jsonl(cache_file):
            metric_id = record.get("metric_id")
            if not (
                metric_id in expected
                and record.get("input_sha256") == expected[metric_id]
                and record.get("model") == model
                and isinstance(record.get("scores"), dict)
            ):
                continue
            previous = cached.get(metric_id)
            if previous is not None and previous != record:
                raise ValueError(
                    f"Conflicting valid cache records for metric_id {metric_id!r}"
                )
            cached[metric_id] = record
    return cached


def _ordered_cache(
    cached: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [cached[row["metric_id"]] for row in rows if row["metric_id"] in cached]


def _cache_parts_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}.parts")


def _cache_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else []
    parts_dir = _cache_parts_dir(path)
    if parts_dir.is_dir():
        files.extend(sorted(item for item in parts_dir.glob("*.jsonl") if item.is_file()))
    return files


def _write_cache_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    payload = b"".join(_jsonl_record(row) for row in rows)
    digest = hashlib.sha256(payload).hexdigest()
    parts_dir = _cache_parts_dir(path)
    parts_dir.mkdir(parents=True, exist_ok=True)
    target = parts_dir / f"part-{digest}.jsonl"
    if target.is_file():
        if _file_sha256(target) != digest:
            raise ValueError(f"Cache checkpoint hash mismatch: {target}")
        return
    _write_bytes_atomic(target, payload)


def _consolidate_cache(
    path: Path,
    cached: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    _write_jsonl(path, _ordered_cache(cached, rows))
    parts_dir = _cache_parts_dir(path)
    if parts_dir.is_dir():
        shutil.rmtree(parts_dir)


def neural_input_hash(row: dict[str, Any]) -> str:
    content = {
        "implementation_version": _NEURAL_INPUT_VERSION,
        "candidate_answer": row["candidate_answer"],
        "reference_answer": row["reference_answer"],
        "retrieved_evidence": row["retrieved_evidence"],
        "cited_evidence": row["cited_evidence"],
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _validate_input_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    required = {
        "metric_id",
        "population",
        "question",
        "reference_answer",
        "candidate_answer",
        "retrieved_evidence",
        "cited_evidence",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Neural input row {index} is missing fields: {sorted(missing)}")
        metric_id = row["metric_id"]
        if not isinstance(metric_id, str) or not metric_id:
            raise ValueError(f"Neural input row {index} has an invalid metric_id")
        if metric_id in seen:
            raise ValueError(f"Duplicate neural input metric_id: {metric_id}")
        seen.add(metric_id)
        for field in ("question", "reference_answer", "candidate_answer"):
            if not isinstance(row[field], str):
                raise ValueError(f"Neural input row {index} has non-string {field}")
        for field in ("retrieved_evidence", "cited_evidence"):
            if not isinstance(row[field], list):
                raise ValueError(f"Neural input row {index} has non-list {field}")
            if any(
                not isinstance(item, dict) or not isinstance(item.get("text"), str)
                for item in row[field]
            ):
                raise ValueError(f"Neural input row {index} has malformed {field}")


def _neural_runtime_contract(*, device: str, requested: set[str]) -> dict[str, Any]:
    import torch

    distributions = {
        name: _distribution_version(name)
        for name in (
            "bert-score",
            "huggingface-hub",
            "joblib",
            "nltk",
            "numpy",
            "pydantic",
            "pydantic-core",
            "scikit-learn",
            "scipy",
            "torch",
            "transformers",
        )
    }
    modules = {
        name: _module_origin(name)
        for name in (
            "bert_score",
            "huggingface_hub",
            "joblib",
            "nltk",
            "numpy",
            "pydantic",
            "pydantic_core",
            "scipy",
            "sklearn",
            "torch",
            "transformers",
        )
    }
    cuda: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda["selected_device"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "contract_version": _NEURAL_RUNTIME_VERSION,
        "input_contract_version": _NEURAL_INPUT_VERSION,
        "requested_metrics": sorted(requested),
        "resolved_device": device,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": platform.platform(),
        "distributions": distributions,
        "module_origins": modules,
        "cuda": cuda,
    }


def _metric_configuration(
    requested: set[str], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    configuration: dict[str, dict[str, Any]] = {}
    if "bertscore" in requested:
        configuration["bertscore"] = {
            "model": args.bertscore_model,
            "revision": BERTSCORE_REVISION,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "idf": False,
            "rescale_with_baseline": False,
            "allowed_snapshot_files": list(_TRANSFORMERS_RUNTIME_FILES),
        }
    if "minicheck" in requested:
        configuration["minicheck"] = {
            "model": args.minicheck_model,
            "revision": MINICHECK_REVISION,
            "scope": args.minicheck_scope,
        }
    if "sas" in requested:
        configuration["sas"] = {
            "model": args.sas_model,
            "revision": SAS_REVISION,
            "activation": "sigmoid",
            "tokenizer_backend": _SAS_TOKENIZER_BACKEND,
            "allowed_snapshot_files": list(_SAS_RUNTIME_FILES),
        }
    if "pedants" in requested:
        configuration["pedants"] = {
            "artifact_revision": PEDANTS_REVISION,
            "artifact_sha256": dict(_PEDANTS_ASSETS),
        }
    if "alignscore" in requested:
        configuration["alignscore"] = {
            "checkpoint_repo": ALIGNSCORE_CHECKPOINT_REPO,
            "checkpoint_filename": ALIGNSCORE_CHECKPOINT_FILENAME,
            "checkpoint_revision": ALIGNSCORE_CHECKPOINT_REVISION,
            "backbone": args.alignscore_model,
            "backbone_revision": ALIGNSCORE_BACKBONE_REVISION,
            "implementation_version": ALIGNSCORE_IMPLEMENTATION_VERSION,
            "punkt_tab_sha256": NLTK_PUNKT_TAB_SHA256,
        }
    return configuration


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return None if spec is None or spec.origin is None else str(Path(spec.origin).resolve())


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    line_count = 0
    digest = hashlib.sha256()
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            for row in rows:
                stream.write(_jsonl_record(row))
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _jsonl_record(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


if __name__ == "__main__":
    main()
