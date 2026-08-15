from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import orjson

from tcred.metrics.diagnostic_analysis import analyze_diagnostic_suite, write_analysis
from tcred.metrics.diagnostic_builder import (
    DEFAULT_DIAGNOSTIC_SEED,
    DEFAULT_PAIR_CAP,
    build_diagnostic_suite,
)
from tcred.metrics.diagnostic_reporting import render_diagnostic_report
from tcred.metrics.judge import JudgeProvider, score_with_claim_judge
from tcred.metrics.judge_batch import score_with_batch_judge
from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.runner import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    run_neural_metric_worker,
    score_metric_input,
)
from tcred.metrics.statistics import LABEL_SCORE
from tcred.metrics.task_judge import (
    CONTRACT_VERSION,
    DEFAULT_REQUESTS_PER_SECOND,
    score_with_task_judge,
    write_task_judgments,
)
from tcred.metrics.task_judge import (
    DEFAULT_MODEL as DEFAULT_TASK_JUDGE_MODEL,
)
from tcred.metrics.task_judge_analysis import FIELD_METRICS
from tcred.metrics.task_judge_batch import score_with_task_judge_batch
from tcred.metrics.task_judge_models import PromptVariant

DEFAULT_DIAGNOSTIC_OUTPUT = Path("data/metrics/diagnostic_meta_evaluation/2026-08-15")


def run_diagnostic_meta_evaluation(
    *,
    dataset_root: Path,
    gold_dir: Path,
    output_dir: Path = DEFAULT_DIAGNOSTIC_OUTPUT,
    metric_python: Path,
    seed: int = DEFAULT_DIAGNOSTIC_SEED,
    pair_cap_per_phenomenon: int = DEFAULT_PAIR_CAP,
    source_split: str = "test_auto",
    bootstrap_samples: int = 2000,
    judge_provider: JudgeProvider = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    task_judge_model: str = DEFAULT_TASK_JUDGE_MODEL,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    concurrency: int = 8,
    judge_transport: Literal["direct", "batch"] = "batch",
    prompt_variant: PromptVariant = "contrastive_few_shot",
    skip_claim_judge: bool = False,
    skip_task_judge: bool = False,
    skip_neural: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = build_diagnostic_suite(
        dataset_root,
        seed=seed,
        pair_cap_per_phenomenon=pair_cap_per_phenomenon,
        source_split=source_split,
    )
    paths = _write_suite(suite, output_dir=output_dir)
    metric_rows = [case.metric_input for case in suite.cases]
    task_rows = [case.task_judge_input for case in suite.cases]

    claim_records = {}
    if not skip_claim_judge:
        claim_cache = (
            Path("data/cache/metrics/diagnostic_meta_evaluation/claim_judge")
            / judge_provider
            / _safe_name(judge_model)
        )
        if judge_transport == "batch":
            if judge_provider != "mistral":
                raise ValueError("Diagnostic batch judging currently requires provider=mistral")
            claim_records = asyncio.run(
                score_with_batch_judge(
                    metric_rows,
                    cache_dir=claim_cache,
                    batch_dir=output_dir / "claim_judge_batch",
                    provider="mistral",
                    model=judge_model,
                )
            )
        else:
            claim_records = asyncio.run(
                score_with_claim_judge(
                    metric_rows,
                    cache_dir=claim_cache,
                    provider=judge_provider,
                    model=judge_model,
                    concurrency=concurrency,
                    requests_per_second=requests_per_second,
                )
            )
        _write_jsonl(
            output_dir / "claim_judgments.jsonl",
            [claim_records[row.metric_id].model_dump(mode="json") for row in metric_rows],
        )

    neural_records: dict[str, dict[str, object]] = {}
    neural_path = output_dir / "neural_scores.jsonl"
    if not skip_neural:
        run_neural_metric_worker(
            inputs_path=paths["metric_inputs"],
            output_path=neural_path,
            cache_dir=Path("data/cache/metrics/diagnostic_meta_evaluation/neural"),
            model_cache_dir=Path("data/cache/metrics/huggingface"),
            metric_python=metric_python,
            minicheck_scope="all",
        )
        neural_records = {str(row["metric_id"]): row for row in _read_jsonl(neural_path)}

    score_records = [
        score_metric_input(
            row,
            judge=claim_records.get(row.metric_id),
            neural=neural_records.get(row.metric_id),
        )
        for row in metric_rows
    ]

    task_records = {}
    if not skip_task_judge:
        task_cache = (
            Path("data/cache/metrics/diagnostic_meta_evaluation/task_judge")
            / _safe_name(task_judge_model)
            / CONTRACT_VERSION
        )
        if judge_transport == "batch":
            task_records = asyncio.run(
                score_with_task_judge_batch(
                    task_rows,
                    cache_dir=task_cache,
                    batch_dir=output_dir / "task_judge_batch",
                    prompt_variant=prompt_variant,
                    model=task_judge_model,
                    concurrency=concurrency,
                    requests_per_second=requests_per_second,
                    random_seed=seed,
                )
            )
        else:
            task_records = asyncio.run(
                score_with_task_judge(
                    task_rows,
                    cache_dir=task_cache,
                    prompt_variant=prompt_variant,
                    model=task_judge_model,
                    concurrency=concurrency,
                    requests_per_second=requests_per_second,
                    random_seed=seed,
                )
            )
        write_task_judgments(task_records, output_dir / "task_judgments.jsonl")
        score_records = _add_task_judge_scores(score_records, task_records)

    scores_path = output_dir / "metric_scores.jsonl"
    _write_jsonl(scores_path, [row.model_dump(mode="json") for row in score_records])
    analysis = analyze_diagnostic_suite(
        suite,
        score_records,
        dataset_root=dataset_root,
        gold_dir=gold_dir,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    analysis_path = output_dir / "diagnostic_analysis.json"
    write_analysis(analysis, analysis_path)
    report_path = output_dir / "diagnostic_report.md"
    report_path.write_text(
        render_diagnostic_report(analysis),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output_dir / "manifest.json"
    artifact_paths = [
        *paths.values(),
        scores_path,
        analysis_path,
        report_path,
    ]
    if not skip_claim_judge:
        artifact_paths.append(output_dir / "claim_judgments.jsonl")
    if not skip_task_judge:
        artifact_paths.append(output_dir / "task_judgments.jsonl")
    if not skip_neural:
        artifact_paths.append(neural_path)
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "complete",
            "method": (
                "BUMP/DEMETR/ACES contrastive consistency and ROC-AUC + CheckList "
                "directional/invariance tests + MetricEval concurrent/construct validity + "
                "connected-source-scenario bootstrap/permutation inference"
            ),
            "configuration": {
                "seed": seed,
                "source_split": suite.source_split,
                "pair_cap_per_phenomenon": pair_cap_per_phenomenon,
                "bootstrap_samples": bootstrap_samples,
                "claim_judge_enabled": not skip_claim_judge,
                "judge_transport": judge_transport,
                "claim_judge_provider": judge_provider if not skip_claim_judge else None,
                "claim_judge_model": judge_model if not skip_claim_judge else None,
                "task_judge_enabled": not skip_task_judge,
                "task_judge_model": task_judge_model if not skip_task_judge else None,
                "task_judge_contract": CONTRACT_VERSION if not skip_task_judge else None,
                "task_judge_prompt_variant": prompt_variant if not skip_task_judge else None,
                "neural_metrics_enabled": not skip_neural,
                "requests_per_minute": requests_per_second * 60,
            },
            "suite_audit": suite.audit,
            "dataset_content_hashes": suite.dataset_content_hashes,
            "artifacts": [_file_record(path, relative_to=output_dir) for path in artifact_paths],
            "limitations": [
                "Formal diagnostic validity is construct-bounded and does not replace human "
                "concurrent validity.",
                f"The challenge suite is generated from T-CRED {suite.source_split} cases and "
                "is not an external benchmark.",
                "Phenomenon-level samples are diagnostic; uncertainty intervals must accompany "
                "ranks.",
                "LLM-judge rows share one provider/model and are not an independent model-family "
                "replication.",
            ],
        },
    )
    return {
        "manifest": manifest_path,
        "scores": scores_path,
        "analysis": analysis_path,
        "report": report_path,
    }


def _add_task_judge_scores(
    records: list[MetricScoreRecord],
    judgments: dict[str, object],
) -> list[MetricScoreRecord]:
    output = []
    for record in records:
        judgment = judgments[record.metric_id]
        scores = dict(record.scores)
        fields: dict[str, object] = {}
        for field, metric_name in FIELD_METRICS.items():
            field_judgment = judgment.result.field(field)
            scores[metric_name] = LABEL_SCORE.get(field_judgment.label)
            fields[field] = field_judgment.model_dump(mode="json")
        metadata = dict(record.metric_metadata)
        metadata["tcred_task_judge"] = {
            "provider": judgment.provider,
            "model": judgment.model,
            "prompt_variant": judgment.prompt_variant,
            "fields": fields,
        }
        output.append(record.model_copy(update={"scores": scores, "metric_metadata": metadata}))
    return output


def _write_suite(suite, *, output_dir: Path) -> dict[str, Path]:
    cases_path = output_dir / "diagnostic_cases.jsonl"
    pairs_path = output_dir / "diagnostic_pairs.jsonl"
    metric_inputs_path = output_dir / "metric_inputs.jsonl"
    task_inputs_path = output_dir / "task_judge_inputs.jsonl"
    _write_jsonl(cases_path, [case.model_dump(mode="json") for case in suite.cases])
    _write_jsonl(pairs_path, [pair.model_dump(mode="json") for pair in suite.pairs])
    _write_jsonl(
        metric_inputs_path,
        [case.metric_input.model_dump(mode="json") for case in suite.cases],
    )
    _write_jsonl(
        task_inputs_path,
        [case.task_judge_input.model_dump(mode="json") for case in suite.cases],
    )
    return {
        "cases": cases_path,
        "pairs": pairs_path,
        "metric_inputs": metric_inputs_path,
        "task_inputs": task_inputs_path,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as stream:
        return [orjson.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in value
    )
