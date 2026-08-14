from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import orjson

from tcred.metrics.analysis import analyze_metric_scores
from tcred.metrics.config import (
    ALIGNSCORE_BACKBONE,
    ALIGNSCORE_BACKBONE_REVISION,
    ALIGNSCORE_BATCH_SIZE,
    ALIGNSCORE_CHECKPOINT_REVISION,
    ALIGNSCORE_IMPLEMENTATION_VERSION,
    BERTSCORE_MODEL,
    BERTSCORE_REVISION,
    MINICHECK_MODEL,
    MINICHECK_REVISION,
    PEDANTS_REVISION,
    SAS_MODEL,
    SAS_REVISION,
)
from tcred.metrics.deterministic import claim_judge_scores, reference_answer_scores
from tcred.metrics.inputs import load_metric_inputs, write_metric_inputs
from tcred.metrics.judge import JudgeProvider, score_with_claim_judge
from tcred.metrics.judge_batch import score_with_batch_judge
from tcred.metrics.models import JudgeCacheRecord, MetricInput, MetricScoreRecord
from tcred.metrics.reporting import render_metric_report

DEFAULT_JUDGE_PROVIDER: JudgeProvider = "mistral"
DEFAULT_JUDGE_MODEL = "mistral-large-2512"


def run_current_metrics(
    *,
    gold_dir: Path,
    dataset_root: Path,
    system_output_root: Path,
    output_dir: Path,
    metric_python: Path,
    judge_provider: JudgeProvider = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 12,
    judge_requests_per_second: float | None = 0.06,
    judge_transport: Literal["direct", "batch"] = "direct",
    judge_scope: Literal["human_gold", "all"] = "human_gold",
    judge_stability_sample: int = 40,
    minicheck_scope: Literal["human_gold", "all"] = "human_gold",
    alignscore_batch_size: int = ALIGNSCORE_BATCH_SIZE,
    bootstrap_samples: int = 2000,
    skip_judge: bool = False,
    skip_neural: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path("data/cache/metrics")
    rows = load_metric_inputs(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
    )
    inputs_path = output_dir / "metric_inputs.jsonl"
    write_metric_inputs(rows, inputs_path)

    judge_records: dict[str, JudgeCacheRecord] = {}
    stability: dict[str, object] = {"sample_size": 0, "status": "skipped"}
    if not skip_judge:
        judge_rows = (
            [row for row in rows if row.population == "human_gold"]
            if judge_scope == "human_gold"
            else rows
        )
        judge_cache_dir = (
            cache_root / "claim_judge" / _safe_name(judge_provider) / _safe_name(judge_model)
        )
        if judge_transport == "batch":
            if judge_provider not in {"mistral", "groq"}:
                raise ValueError("Batch metric judging requires judge_provider=mistral or groq")
            judge_records = asyncio.run(
                score_with_batch_judge(
                    judge_rows,
                    cache_dir=judge_cache_dir,
                    batch_dir=output_dir / "judge_batch",
                    provider=judge_provider,
                    model=judge_model,
                )
            )
        else:
            judge_records = asyncio.run(
                score_with_claim_judge(
                    judge_rows,
                    cache_dir=judge_cache_dir,
                    provider=judge_provider,
                    model=judge_model,
                    concurrency=judge_concurrency,
                    requests_per_second=judge_requests_per_second,
                )
            )
        stability = _judge_stability(
            judge_rows,
            primary=judge_records,
            cache_dir=(
                cache_root
                / "claim_judge_stability"
                / _safe_name(judge_provider)
                / _safe_name(judge_model)
            ),
            batch_dir=output_dir / "judge_stability_batch",
            provider=judge_provider,
            model=judge_model,
            concurrency=judge_concurrency,
            requests_per_second=judge_requests_per_second,
            transport=judge_transport,
            sample_size=judge_stability_sample,
        )
        _write_jsonl(
            output_dir / "claim_judgments.jsonl",
            [judge_records[row.metric_id].model_dump(mode="json") for row in judge_rows],
        )

    neural_records: dict[str, dict[str, object]] = {}
    neural_path = output_dir / "neural_scores.jsonl"
    if not skip_neural:
        _run_neural_worker(
            inputs_path=inputs_path,
            output_path=neural_path,
            cache_dir=cache_root / "neural_scores",
            model_cache_dir=cache_root / "huggingface",
            metric_python=metric_python,
            minicheck_scope=minicheck_scope,
            alignscore_batch_size=alignscore_batch_size,
        )
        neural_records = {str(record["metric_id"]): record for record in _read_jsonl(neural_path)}
        missing = {row.metric_id for row in rows} - set(neural_records)
        if missing:
            raise RuntimeError(f"Neural worker omitted metric rows: {sorted(missing)[:5]}")

    score_records = [
        _score_row(
            row,
            judge=judge_records.get(row.metric_id),
            neural=neural_records.get(row.metric_id),
        )
        for row in rows
    ]
    _validate_score_records(score_records)
    scores_path = output_dir / "metric_scores.jsonl"
    _write_jsonl(scores_path, [record.model_dump(mode="json") for record in score_records])
    analysis = analyze_metric_scores(
        score_records,
        bootstrap_samples=bootstrap_samples,
    )
    analysis["human_full_answer_overlap"] = _human_full_answer_overlap(rows)
    analysis["judge_stability"] = stability
    analysis_path = output_dir / "metric_analysis.json"
    _write_json(analysis_path, analysis)

    report_path = output_dir / "metric_report.md"
    report_path.write_text(render_metric_report(analysis), encoding="utf-8", newline="\n")
    manifest_path = output_dir / "metric_manifest.json"
    manifest = {
        "schema_version": "1.2",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "inputs": {
            "gold_dir": str(gold_dir),
            "dataset_root": str(dataset_root),
            "system_output_root": str(system_output_root),
            "human_gold_units": sum(row.population == "human_gold" for row in rows),
            "full_system_outputs": sum(row.population == "system_full" for row in rows),
        },
        "configuration": {
            "judge_enabled": not skip_judge,
            "judge_provider": judge_provider if not skip_judge else None,
            "judge_model": judge_model if not skip_judge else None,
            "judge_concurrency": judge_concurrency,
            "judge_requests_per_second": judge_requests_per_second,
            "judge_transport": judge_transport,
            "judge_scope": judge_scope,
            "judge_scored_units": len(judge_records),
            "judge_stability_sample": judge_stability_sample,
            "neural_enabled": not skip_neural,
            "bertscore_model": (
                f"{BERTSCORE_MODEL}@{BERTSCORE_REVISION}" if not skip_neural else None
            ),
            "minicheck_model": (
                f"{MINICHECK_MODEL}@{MINICHECK_REVISION}" if not skip_neural else None
            ),
            "minicheck_scope": minicheck_scope if not skip_neural else None,
            "sas_model": (f"{SAS_MODEL}@{SAS_REVISION}" if not skip_neural else None),
            "pedants_artifact": (
                f"zli12321/pedant_models@{PEDANTS_REVISION}" if not skip_neural else None
            ),
            "alignscore_model": (
                f"yzha/AlignScore@{ALIGNSCORE_CHECKPOINT_REVISION}:"
                f"{ALIGNSCORE_BACKBONE}@{ALIGNSCORE_BACKBONE_REVISION}:"
                f"{ALIGNSCORE_IMPLEMENTATION_VERSION}"
                if not skip_neural
                else None
            ),
            "alignscore_scope": "all" if not skip_neural else None,
            "alignscore_batch_size": alignscore_batch_size if not skip_neural else None,
            "bootstrap_samples": bootstrap_samples,
        },
        "metric_suite": _metric_suite_manifest(
            skip_judge=skip_judge,
            skip_neural=skip_neural,
            minicheck_scope=minicheck_scope,
        ),
        "implementation": _implementation_provenance(metric_python),
        "judge_usage": _judge_usage(judge_records),
        "artifacts": [],
        "limitations": _manifest_limitations(
            skip_judge=skip_judge,
            skip_neural=skip_neural,
            minicheck_scope=minicheck_scope,
        ),
    }
    artifact_paths = [inputs_path, scores_path, analysis_path, report_path]
    if not skip_judge:
        artifact_paths.append(output_dir / "claim_judgments.jsonl")
    if not skip_neural:
        artifact_paths.append(neural_path)
    manifest["artifacts"] = [_file_record(path, relative_to=output_dir) for path in artifact_paths]
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "scores": scores_path,
        "analysis": analysis_path,
        "report": report_path,
    }


def _score_row(
    row: MetricInput,
    *,
    judge: JudgeCacheRecord | None,
    neural: dict[str, object] | None,
) -> MetricScoreRecord:
    scores: dict[str, float | None] = reference_answer_scores(
        row.candidate_answer,
        row.reference_answer,
    )
    scores.update(row.retrieval_metrics)
    scores.update(row.citation_metrics)
    scores["unresolved_citation_indicator"] = float(row.unresolved_citation_count > 0)
    scores["citation_presence"] = float(bool(row.cited_evidence))
    metadata: dict[str, object] = {}
    if judge is not None:
        scores.update(
            claim_judge_scores(
                judge.result,
                retrieved_count=len(row.retrieved_evidence),
                cited_count=len(row.cited_evidence),
            )
        )
        metadata["claim_judge"] = {
            "model": judge.model,
            "rationale": judge.result.rationale,
            "candidate_claim_count": len(judge.result.candidate_claims),
            "reference_claim_count": len(judge.result.reference_claims),
        }
    if neural is not None:
        neural_scores = neural.get("scores")
        if not isinstance(neural_scores, dict):
            raise ValueError(f"Malformed neural score row: {row.metric_id}")
        scores.update(
            {
                str(name): float(value) if value is not None else None
                for name, value in neural_scores.items()
            }
        )
        metadata["neural"] = neural.get("metadata", {})
    return MetricScoreRecord(
        metric_id=row.metric_id,
        population=row.population,
        dataset_family=row.dataset_family,
        source_kind=row.source_kind,
        system_name=row.system_name,
        unit_id=row.unit_id,
        qid=row.qid,
        scenario_id=row.scenario_id,
        gold_labels=row.gold_labels,
        gold_provenance=row.gold_provenance,
        scores=scores,
        metric_metadata=metadata,
    )


def score_metric_input(
    row: MetricInput,
    *,
    judge: JudgeCacheRecord | None = None,
    neural: dict[str, object] | None = None,
) -> MetricScoreRecord:
    """Score one canonical input using the same implementation as the main metric run."""
    return _score_row(row, judge=judge, neural=neural)


def run_neural_metric_worker(
    *,
    inputs_path: Path,
    output_path: Path,
    cache_dir: Path,
    model_cache_dir: Path,
    metric_python: Path,
    minicheck_scope: Literal["human_gold", "all"] = "all",
    alignscore_batch_size: int = ALIGNSCORE_BATCH_SIZE,
    metrics: tuple[str, ...] | None = None,
) -> None:
    """Run the pinned local neural metric worker for an externally built input table."""
    _run_neural_worker(
        inputs_path=inputs_path,
        output_path=output_path,
        cache_dir=cache_dir,
        model_cache_dir=model_cache_dir,
        metric_python=metric_python,
        minicheck_scope=minicheck_scope,
        alignscore_batch_size=alignscore_batch_size,
        metrics=metrics,
    )


def _validate_score_records(records: list[MetricScoreRecord]) -> None:
    """Reject non-finite or out-of-contract scores before analysis."""
    tolerance = 1e-6
    for record in records:
        for name, value in record.scores.items():
            if value is None:
                continue
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"Non-finite metric score for {record.metric_id}: {name}")
            lower_bound = -1.0 if name.startswith("bertscore_") else 0.0
            if score < lower_bound - tolerance or score > 1.0 + tolerance:
                raise ValueError(
                    f"Metric score outside [{lower_bound}, 1] for "
                    f"{record.metric_id}: {name}={score}"
                )


def _judge_stability(
    rows: list[MetricInput],
    *,
    primary: dict[str, JudgeCacheRecord],
    cache_dir: Path,
    batch_dir: Path,
    provider: JudgeProvider,
    model: str,
    concurrency: int,
    requests_per_second: float | None,
    transport: Literal["direct", "batch"],
    sample_size: int,
) -> dict[str, object]:
    if sample_size <= 0:
        return {"sample_size": 0, "status": "skipped"}
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"judge-stability-v1:{row.metric_id}".encode()).hexdigest(),
    )[: min(sample_size, len(rows))]
    repeated = [
        row.model_copy(update={"metric_id": f"stability:{row.metric_id}"}) for row in ordered
    ]
    if transport == "batch":
        secondary = asyncio.run(
            score_with_batch_judge(
                repeated,
                cache_dir=cache_dir,
                batch_dir=batch_dir,
                provider=provider,
                model=model,
            )
        )
    else:
        secondary = asyncio.run(
            score_with_claim_judge(
                repeated,
                cache_dir=cache_dir,
                provider=provider,
                model=model,
                concurrency=concurrency,
                requests_per_second=requests_per_second,
            )
        )
    score_names = (
        "g_eval_answer_correctness",
        "g_eval_answer_relevance",
        "ragchecker_f1",
        "ragchecker_faithfulness",
    )
    absolute_differences: dict[str, list[float]] = {name: [] for name in score_names}
    exact_structured = 0
    for original, repeated_row in zip(ordered, repeated, strict=True):
        first = primary[original.metric_id]
        second = secondary[repeated_row.metric_id]
        if first.result == second.result:
            exact_structured += 1
        first_scores = claim_judge_scores(
            first.result,
            retrieved_count=len(original.retrieved_evidence),
            cited_count=len(original.cited_evidence),
        )
        second_scores = claim_judge_scores(
            second.result,
            retrieved_count=len(original.retrieved_evidence),
            cited_count=len(original.cited_evidence),
        )
        for name in score_names:
            left, right = first_scores[name], second_scores[name]
            if left is not None and right is not None:
                absolute_differences[name].append(abs(left - right))
    return {
        "status": "complete",
        "sample_size": len(ordered),
        "exact_structured_result_rate": exact_structured / len(ordered),
        "mean_absolute_score_difference": {
            name: sum(values) / len(values) if values else None
            for name, values in absolute_differences.items()
        },
    }


def _run_neural_worker(
    *,
    inputs_path: Path,
    output_path: Path,
    cache_dir: Path,
    model_cache_dir: Path,
    metric_python: Path,
    minicheck_scope: Literal["human_gold", "all"],
    alignscore_batch_size: int,
    metrics: tuple[str, ...] | None = None,
) -> None:
    if not metric_python.is_file():
        raise FileNotFoundError(
            f"64-bit neural metric Python not found: {metric_python}. See the metrics README."
        )
    worker = Path(__file__).with_name("neural_worker.py")
    environment = _neural_worker_environment(model_cache_dir)
    command = [
        str(metric_python.resolve()),
        str(worker),
        "--input",
        str(inputs_path),
        "--output",
        str(output_path),
        "--cache-dir",
        str(cache_dir),
        "--model-cache-dir",
        str(model_cache_dir),
        "--bertscore-model",
        BERTSCORE_MODEL,
        "--minicheck-model",
        MINICHECK_MODEL,
        "--sas-model",
        SAS_MODEL,
        "--alignscore-model",
        ALIGNSCORE_BACKBONE,
        "--alignscore-batch-size",
        str(alignscore_batch_size),
        "--minicheck-scope",
        minicheck_scope,
    ]
    if metrics is not None:
        if not metrics:
            raise ValueError("metrics cannot be empty when provided")
        command.extend(["--metrics", ",".join(metrics)])
    subprocess.run(
        command,
        check=True,
        env=environment,
    )


def _neural_worker_environment(model_cache_dir: Path) -> dict[str, str]:
    """Build an isolated worker environment with an explicit project import root."""
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["HF_HOME"] = str(model_cache_dir.resolve())
    environment["MPLCONFIGDIR"] = str((model_cache_dir / "matplotlib").resolve())
    if _neural_models_are_cached(model_cache_dir):
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _judge_usage(records: dict[str, JudgeCacheRecord]) -> dict[str, int]:
    names = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    return {name: sum(record.usage.get(name, 0) for record in records.values()) for name in names}


def _human_full_answer_overlap(rows: list[MetricInput]) -> dict[str, object]:
    full_answers = {
        (row.dataset_family, row.qid, row.system_name): row.candidate_answer
        for row in rows
        if row.population == "system_full"
    }
    totals = {"total": 0, "exact": 0, "different": 0, "missing": 0}
    by_system: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.population != "human_gold" or row.source_kind != "system_output":
            continue
        system = row.system_name or "unknown"
        system_counts = by_system.setdefault(
            system,
            {"total": 0, "exact": 0, "different": 0, "missing": 0},
        )
        key = (row.dataset_family, row.qid, row.system_name)
        if key not in full_answers:
            status = "missing"
        elif full_answers[key] == row.candidate_answer:
            status = "exact"
        else:
            status = "different"
        totals["total"] += 1
        totals[status] += 1
        system_counts["total"] += 1
        system_counts[status] += 1
    return {"all": totals, "by_system": dict(sorted(by_system.items()))}


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in value
    )


def _metric_suite_manifest(
    *,
    skip_judge: bool,
    skip_neural: bool,
    minicheck_scope: Literal["human_gold", "all"],
) -> dict[str, object]:
    deterministic = {
        "scope": "all",
        "scores": [
            "exact_match",
            "token_precision",
            "token_recall",
            "token_f1",
            "rouge_1",
            "rouge_2",
            "rouge_l",
            "required_citation_precision",
            "required_citation_recall",
            "citation_resolution_rate",
            "citation_presence",
            "unresolved_citation_indicator",
            "retrieval_precision_at_10",
            "retrieval_recall_at_10",
            "retrieval_hit_at_10",
            "retrieval_average_precision_at_10",
            "retrieval_mrr",
            "retrieval_r_precision",
            "retrieval_ndcg_at_10",
        ],
    }
    local_models: list[dict[str, object]] = []
    if not skip_neural:
        local_models = [
            {
                "method": "BERTScore",
                "scope": "all",
                "scores": [
                    "bertscore_precision",
                    "bertscore_recall",
                    "bertscore_f1",
                ],
            },
            {
                "method": "SAS cross-encoder",
                "scope": "all",
                "scores": ["sas_cross_encoder"],
            },
            {
                "method": "PEDANTS",
                "scope": "all",
                "scores": ["pedants_probability", "pedants_match"],
            },
            {
                "method": "MiniCheck",
                "scope": minicheck_scope,
                "scores": [
                    "minicheck_retrieved_mean",
                    "minicheck_retrieved_strict",
                    "minicheck_cited_mean",
                    "minicheck_cited_strict",
                ],
            },
            {
                "method": "AlignScore-base NLI-SP",
                "scope": "all",
                "scores": ["alignscore_retrieved", "alignscore_cited"],
            },
        ]
    llm_judge: dict[str, object] = {"enabled": not skip_judge, "scores": []}
    if not skip_judge:
        llm_judge["scores"] = [
            "g_eval_answer_correctness",
            "g_eval_answer_relevance",
            "ragchecker_precision",
            "ragchecker_recall",
            "ragchecker_f1",
            "ragchecker_claim_recall",
            "ragchecker_context_precision",
            "ragchecker_faithfulness",
            "ragchecker_hallucination",
            "ragchecker_non_hallucination",
            "ragchecker_self_knowledge",
            "ragchecker_relevant_noise_sensitivity",
            "ragchecker_irrelevant_noise_sensitivity",
            "ragchecker_context_utilization",
            "alce_citation_completeness",
            "alce_citation_precision",
        ]
    return {
        "deterministic": deterministic,
        "local_models": local_models,
        "llm_judge": llm_judge,
    }


def _manifest_limitations(
    *,
    skip_judge: bool,
    skip_neural: bool,
    minicheck_scope: Literal["human_gold", "all"],
) -> list[str]:
    limitations = [
        "The human gold source is final only for its frozen interim annotation snapshot.",
        "Adjudicated labels include decisions by one AI adjudicator, as declared by the gold "
        "manifest.",
        "Full-run automatic metric means have narrower sampling intervals but no additional "
        "human labels.",
    ]
    if not skip_judge:
        limitations.append(
            "RAGChecker-style scores use the published formulas with a pinned LLM claim "
            "extractor/checker, not the original Llama-3-70B implementation."
        )
    if not skip_neural:
        limitations.extend(
            [
                "MiniCheck uses sentence-level aggregation over displayed evidence and a "
                "CPU-compatible DeBERTa checkpoint.",
                "SAS and PEDANTS assess reference-answer equivalence; neither verifies evidence "
                "or temporal validity.",
                "AlignScore-base evaluates textual factual alignment with displayed evidence; "
                "its NLI-SP score does not validate evidence intervals or graph paths.",
            ]
        )
        if minicheck_scope == "human_gold":
            limitations.append(
                "MiniCheck is restricted to human-gold units; BERTScore, SAS, PEDANTS, "
                "AlignScore, and deterministic metrics cover the complete system run."
            )
    limitations.extend(
        [
            "No selected off-the-shelf metric directly measures graph-path sufficiency or "
            "response-decision appropriateness.",
            "Temporal correlations of ordinary answer, grounding, and citation metrics are "
            "diagnostic associations, not scores from a dedicated temporal metric.",
        ]
    )
    return limitations


def _implementation_provenance(metric_python: Path) -> dict[str, object]:
    module_dir = Path(__file__).parent
    source_paths = [
        module_dir / name
        for name in (
            "analysis.py",
            "config.py",
            "deterministic.py",
            "inputs.py",
            "models.py",
            "neural_worker.py",
            "reporting.py",
            "runner.py",
            "statistics.py",
        )
    ]
    dependency_paths = [
        path for path in (Path("pyproject.toml"), Path("uv.lock")) if path.is_file()
    ]
    provenance: dict[str, object] = {
        "metric_python": str(metric_python),
        "source_sha256": {
            path.relative_to(Path.cwd()).as_posix(): _sha256(path) for path in source_paths
        },
        "dependency_sha256": {path.as_posix(): _sha256(path) for path in dependency_paths},
    }
    if metric_python.is_file():
        provenance["neural_runtime"] = _neural_runtime_provenance(metric_python)
    return provenance


def _neural_runtime_provenance(metric_python: Path) -> dict[str, object]:
    packages = (
        "bert-score",
        "huggingface-hub",
        "joblib",
        "nltk",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    )
    script = (
        "import importlib.metadata as m, json, platform, struct, sys; "
        f"names={packages!r}; "
        "versions={name:m.version(name) for name in names}; "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'pointer_bits':struct.calcsize('P')*8,'platform':platform.platform(),"
        "'packages':versions},sort_keys=True))"
    )
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    completed = subprocess.run(
        [str(metric_python.resolve()), "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("Neural runtime provenance did not return an object")
    return value


def _neural_models_are_cached(model_cache_dir: Path) -> bool:
    snapshots = (
        model_cache_dir / "hub" / "models--roberta-large" / "snapshots" / BERTSCORE_REVISION,
        model_cache_dir
        / "models--lytang--MiniCheck-DeBERTa-v3-Large"
        / "snapshots"
        / MINICHECK_REVISION,
        model_cache_dir / "models--cross-encoder--stsb-roberta-large" / "snapshots" / SAS_REVISION,
        model_cache_dir / "models--roberta-base" / "snapshots" / ALIGNSCORE_BACKBONE_REVISION,
        model_cache_dir / "models--yzha--AlignScore" / "snapshots" / ALIGNSCORE_CHECKPOINT_REVISION,
    )
    pedants = model_cache_dir / "pedants" / PEDANTS_REVISION
    nltk = model_cache_dir / "nltk" / "tokenizers" / "punkt_tab"
    return (
        all(snapshot.is_dir() and any(snapshot.iterdir()) for snapshot in snapshots)
        and pedants.is_dir()
        and nltk.is_dir()
    )


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
