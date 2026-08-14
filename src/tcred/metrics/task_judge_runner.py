from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import orjson

from tcred.metrics.judge import JudgeProvider
from tcred.metrics.reporting import render_metric_report
from tcred.metrics.task_judge import (
    CONTRACT_VERSION,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REQUESTS_PER_SECOND,
    MINIMUM_ACCEPTABLE_RPM,
    prompt_contract,
    score_with_task_judge,
    write_task_judgments,
)
from tcred.metrics.task_judge_analysis import (
    classification_analysis,
    combined_task_judge_usage,
    complete_comparison_analysis,
    full_judgment_distribution_analysis,
    merge_with_baseline_scores,
    select_prompt_variant,
    stability_analysis,
    support_pointer_analysis,
    task_judge_usage,
    write_disagreement_records,
    write_score_records,
)
from tcred.metrics.task_judge_inputs import (
    load_or_create_split,
    load_task_judge_inputs,
    task_input_hash,
    write_task_judge_inputs,
)
from tcred.metrics.task_judge_models import (
    PromptSelection,
    PromptVariant,
    TaskJudgeInput,
    TaskJudgeRecord,
)
from tcred.metrics.task_judge_reporting import render_task_judge_report

PROMPT_VARIANTS: tuple[PromptVariant, ...] = ("rubric_only", "contrastive_few_shot")
DEFAULT_OUTPUT_DIR = Path("data/metrics/tcred_task_judge/2026-08-14")
DEFAULT_NON_LLM_SCORES = Path("data/metrics/non_llm_expansion/2026-08-14/metric_scores.jsonl")
DEFAULT_LEGACY_LLM_SCORES = Path("data/metrics/current_sota/2026-08-13/metric_scores.jsonl")


def run_task_judge_experiment(
    *,
    gold_dir: Path,
    dataset_root: Path,
    system_output_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    non_llm_scores_path: Path = DEFAULT_NON_LLM_SCORES,
    legacy_llm_scores_path: Path = DEFAULT_LEGACY_LLM_SCORES,
    provider: JudgeProvider = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    random_seed: int = DEFAULT_RANDOM_SEED,
    stability_sample: int = 40,
    bootstrap_samples: int = 2000,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _verify_metric_score_artifact(non_llm_scores_path)
    _verify_metric_score_artifact(legacy_llm_scores_path)
    rows = load_task_judge_inputs(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
    )
    inputs_path = output_dir / "task_judge_inputs.jsonl"
    write_task_judge_inputs(rows, inputs_path)
    split_path = output_dir / "split_manifest.json"
    split = load_or_create_split(rows, split_path)
    calibration_ids = set(split.calibration_metric_ids)
    calibration_rows = [row for row in rows if row.metric_id in calibration_ids]
    cache_root = (
        Path("data/cache/metrics/task_judge") / provider / _safe_name(model) / CONTRACT_VERSION
    )

    candidate_records: dict[PromptVariant, dict[str, TaskJudgeRecord]] = {}
    for index, variant in enumerate(PROMPT_VARIANTS):
        if index:
            time.sleep(max(1 / requests_per_second, 1.0))
        records = asyncio.run(
            score_with_task_judge(
                calibration_rows,
                cache_dir=cache_root / f"seed_{random_seed}",
                prompt_variant=variant,
                provider=provider,
                model=model,
                concurrency=concurrency,
                requests_per_second=requests_per_second,
                random_seed=random_seed,
            )
        )
        candidate_records[variant] = records
        write_task_judgments(records, output_dir / "calibration" / f"{variant}.jsonl")

    selection_path = output_dir / "prompt_selection.json"
    selection, calibration_detail = select_prompt_variant(
        rows,
        candidate_records,
        calibration_ids=calibration_ids,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed,
        contract_version=CONTRACT_VERSION,
    )
    _freeze_or_validate_selection(selection_path, selection)
    _write_json(output_dir / "calibration_analysis.json", calibration_detail)

    time.sleep(max(1 / requests_per_second, 1.0))
    selected_records = asyncio.run(
        score_with_task_judge(
            rows,
            cache_dir=cache_root / f"seed_{random_seed}",
            prompt_variant=selection.selected_variant,
            provider=provider,
            model=model,
            concurrency=concurrency,
            requests_per_second=requests_per_second,
            random_seed=random_seed,
        )
    )
    judgments_path = output_dir / "task_judgments.jsonl"
    write_task_judgments(selected_records, judgments_path)

    stability_rows = _stability_rows(rows, split.held_out_metric_ids, stability_sample)
    time.sleep(max(1 / requests_per_second, 1.0))
    repeated_records = asyncio.run(
        score_with_task_judge(
            stability_rows,
            cache_dir=cache_root / f"seed_{random_seed + 1}",
            prompt_variant=selection.selected_variant,
            provider=provider,
            model=model,
            concurrency=concurrency,
            requests_per_second=requests_per_second,
            random_seed=random_seed + 1,
        )
    )
    stability_path = output_dir / "stability_judgments.jsonl"
    write_task_judgments(repeated_records, stability_path)

    merged_scores = merge_with_baseline_scores(
        rows,
        selected_records,
        non_llm_scores_path=non_llm_scores_path,
        legacy_llm_scores_path=legacy_llm_scores_path,
    )
    scores_path = output_dir / "metric_scores.jsonl"
    write_score_records(merged_scores, scores_path)

    held_out_ids = set(split.held_out_metric_ids)
    human_rows = [row for row in rows if row.population == "human_gold"]
    held_out_rows = [row for row in human_rows if row.metric_id in held_out_ids]
    disagreement_path = output_dir / "held_out_disagreements.jsonl"
    disagreement_count = write_disagreement_records(
        held_out_rows,
        selected_records,
        disagreement_path,
    )
    human_system_rows = [row for row in human_rows if row.source_kind == "system_output"]
    calibration_classification = classification_analysis(
        calibration_rows,
        selected_records,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 10,
    )
    held_out_classification = classification_analysis(
        held_out_rows,
        selected_records,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 20,
    )
    all_gold_classification = classification_analysis(
        human_rows,
        selected_records,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 30,
    )
    human_system_classification = classification_analysis(
        human_system_rows,
        selected_records,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 40,
    )
    complete_analysis = complete_comparison_analysis(
        merged_scores,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 50,
    )
    held_out_score_records = [
        record for record in merged_scores if record.metric_id in held_out_ids
    ]
    held_out_metric_analysis = complete_comparison_analysis(
        held_out_score_records,
        bootstrap_samples=bootstrap_samples,
        seed=random_seed + 60,
    )
    stability = stability_analysis(stability_rows, selected_records, repeated_records)
    final_usage = task_judge_usage(selected_records)
    calibration_usage = {
        variant: task_judge_usage(records) for variant, records in candidate_records.items()
    }
    stability_usage = task_judge_usage(repeated_records)
    experiment_usage = combined_task_judge_usage(
        [*candidate_records.values(), selected_records, repeated_records]
    )
    rate_limit_observation = _observed_rate_limits(
        [*candidate_records.values(), selected_records, repeated_records]
    )
    pointer_audit = {
        "final_selected": support_pointer_analysis(selected_records),
        "stability_repeat": support_pointer_analysis(repeated_records),
    }
    judgment_distributions = full_judgment_distribution_analysis(rows, selected_records)
    analysis = {
        "schema_version": "1.1",
        "split": split.model_dump(mode="json"),
        "prompt_selection": selection.model_dump(mode="json"),
        "classification": {
            "calibration": calibration_classification,
            "held_out": held_out_classification,
            "all_human_gold": all_gold_classification,
            "human_system_outputs": human_system_classification,
        },
        "held_out_metric_analysis": held_out_metric_analysis,
        "complete_metric_analysis": complete_analysis,
        "stability": stability,
        "support_pointer_audit": pointer_audit,
        "full_judgment_distributions": judgment_distributions,
        "held_out_disagreement_count": disagreement_count,
        "usage": {
            "final_selected_logical": final_usage,
            "calibration_logical": calibration_usage,
            "stability_logical": stability_usage,
            "unique_accepted_stage_records": experiment_usage,
        },
    }
    analysis_path = output_dir / "task_judge_analysis.json"
    _write_json(analysis_path, analysis)
    standard_report_path = output_dir / "complete_metric_report.md"
    standard_report_path.write_text(
        render_metric_report(complete_analysis), encoding="utf-8", newline="\n"
    )
    report_path = output_dir / "task_judge_report.md"
    report_path.write_text(
        render_task_judge_report(
            split=split.model_dump(mode="json"),
            selection=selection,
            calibration_classification=calibration_classification,
            held_out_classification=held_out_classification,
            all_gold_classification=all_gold_classification,
            human_system_classification=human_system_classification,
            held_out_metric_analysis=held_out_metric_analysis,
            complete_metric_analysis=complete_analysis,
            stability=stability,
            pointer_audit=pointer_audit,
            judgment_distributions=judgment_distributions,
            usage={
                "final_selected_logical": final_usage,
                "calibration_logical": calibration_usage,
                "stability_logical": stability_usage,
                "unique_accepted_stage_records": experiment_usage,
            },
            provider=provider,
            model=model,
            requests_per_second=requests_per_second,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output_dir / "manifest.json"
    manifest = _manifest(
        rows=rows,
        split=split.model_dump(mode="json"),
        selection=selection,
        provider=provider,
        model=model,
        concurrency=concurrency,
        requests_per_second=requests_per_second,
        random_seed=random_seed,
        stability_sample=stability_sample,
        bootstrap_samples=bootstrap_samples,
        usage={
            "final_selected_logical": final_usage,
            "calibration_logical": calibration_usage,
            "stability_logical": stability_usage,
            "unique_accepted_stage_records": experiment_usage,
        },
        rate_limit_observation=rate_limit_observation,
        pointer_audit=pointer_audit,
        source_paths={
            "gold_dir": gold_dir,
            "dataset_root": dataset_root,
            "system_output_root": system_output_root,
            "non_llm_scores": non_llm_scores_path,
            "legacy_llm_scores": legacy_llm_scores_path,
        },
        artifact_paths=[
            inputs_path,
            split_path,
            selection_path,
            judgments_path,
            stability_path,
            scores_path,
            analysis_path,
            standard_report_path,
            report_path,
            disagreement_path,
            output_dir / "calibration_analysis.json",
            *[output_dir / "calibration" / f"{variant}.jsonl" for variant in PROMPT_VARIANTS],
        ],
        relative_to=output_dir,
    )
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "analysis": analysis_path,
        "report": report_path,
        "scores": scores_path,
        "judgments": judgments_path,
    }


def _freeze_or_validate_selection(path: Path, candidate: PromptSelection) -> None:
    if path.is_file():
        existing = PromptSelection.model_validate_json(path.read_bytes())
        if existing != candidate:
            raise ValueError(
                "Frozen prompt selection differs from recomputed calibration result; use a new "
                "output directory for a different experiment"
            )
        return
    _write_json(path, candidate.model_dump(mode="json"))


def _stability_rows(
    rows: list[TaskJudgeInput], held_out_ids: list[str], sample_size: int
) -> list[TaskJudgeInput]:
    held_out = set(held_out_ids)
    candidates = [row for row in rows if row.metric_id in held_out]
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"tcred-task-judge-stability-v1:{row.metric_id}".encode()
        ).hexdigest(),
    )[: min(sample_size, len(candidates))]


def _manifest(
    *,
    rows: list[TaskJudgeInput],
    split: dict[str, object],
    selection: PromptSelection,
    provider: JudgeProvider,
    model: str,
    concurrency: int,
    requests_per_second: float,
    random_seed: int,
    stability_sample: int,
    bootstrap_samples: int,
    usage: dict[str, object],
    rate_limit_observation: dict[str, object],
    pointer_audit: dict[str, object],
    source_paths: dict[str, Path],
    artifact_paths: list[Path],
    relative_to: Path,
) -> dict[str, object]:
    prompts = {
        f"{stage}:{variant}": {
            "sha256": prompt_contract(stage, variant)[1],
        }
        for stage in ("evidence", "answer")
        for variant in PROMPT_VARIANTS
    }
    return {
        "schema_version": "1.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "input_rows": len(rows),
        "input_sha256": task_input_hash(rows),
        "population_counts": {
            "human_gold": sum(row.population == "human_gold" for row in rows),
            "system_full": sum(row.population == "system_full" for row in rows),
        },
        "configuration": {
            "provider": provider,
            "model": model,
            "concurrency": concurrency,
            "requests_per_second": requests_per_second,
            "random_seed": random_seed,
            "stability_sample": stability_sample,
            "bootstrap_samples": bootstrap_samples,
            "prompt_variants": list(PROMPT_VARIANTS),
            "selected_prompt_variant": selection.selected_variant,
            "contract_version": CONTRACT_VERSION,
            "two_stage_reference_isolation": True,
            "system_identity_blinded": True,
        },
        "observed_api_limits": rate_limit_observation,
        "split": split,
        "prompts": prompts,
        "usage": usage,
        "support_pointer_audit": pointer_audit,
        "sources": {name: _path_record(path) for name, path in source_paths.items()},
        "artifacts": [_path_record(path, relative_to=relative_to) for path in artifact_paths],
        "limitations": [
            "Only one model/provider family is included in this run.",
            "Rare graph and response-decision labels limit field-specific power.",
            "Full-system scores have no additional human labels.",
            "Self-reported confidence is diagnostic until calibrated on held-out gold.",
            "The judge is an automatic metric and does not replace the human gold authority.",
        ],
    }


def _observed_rate_limits(
    record_groups: list[dict[str, TaskJudgeRecord]],
) -> dict[str, object]:
    """Summarize deduplicated provider limits from accepted response headers."""

    seen: set[tuple[str, str, int]] = set()
    request_limits: list[int] = []
    token_limits: list[int] = []
    accepted_stage_records = 0
    for records in record_groups:
        for record in records.values():
            for stage in (record.answer_stage, record.evidence_stage):
                if stage is None:
                    continue
                identity = (stage.judgment_id, stage.prompt_variant, stage.random_seed)
                if identity in seen:
                    continue
                seen.add(identity)
                accepted_stage_records += 1
                request_limit = _integer_rate_header(
                    stage.rate_limit_headers,
                    "x-ratelimit-limit-req-minute",
                )
                if request_limit is not None:
                    request_limits.append(request_limit)
                token_limit = _integer_rate_header(
                    stage.rate_limit_headers,
                    "x-ratelimit-limit-tokens-minute",
                )
                if token_limit is not None:
                    token_limits.append(token_limit)

    if not request_limits:
        raise ValueError("No accepted task-judge record contains a request-rate limit header")
    minimum_request_limit = min(request_limits)
    if minimum_request_limit < MINIMUM_ACCEPTABLE_RPM:
        raise ValueError(
            "Accepted task-judge records advertise a request limit below the experiment gate: "
            f"{minimum_request_limit} RPM < {MINIMUM_ACCEPTABLE_RPM} RPM"
        )
    return {
        "source": "deduplicated accepted provider-response headers",
        "minimum_acceptable_rpm": MINIMUM_ACCEPTABLE_RPM,
        "gate_passed": True,
        "accepted_stage_records": accepted_stage_records,
        "records_with_request_limit_header": len(request_limits),
        "request_limit_rpm": {
            "minimum": minimum_request_limit,
            "maximum": max(request_limits),
        },
        "records_with_token_limit_header": len(token_limits),
        "token_limit_per_minute": (
            {"minimum": min(token_limits), "maximum": max(token_limits)} if token_limits else None
        ),
    }


def _integer_rate_header(headers: dict[str, str], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError(f"Invalid cached rate-limit header {name}: {raw!r}") from exc
    if value < 0:
        raise ValueError(f"Negative cached rate-limit header {name}: {value}")
    return value


def _path_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    record: dict[str, object] = {
        "path": str(resolved.relative_to(relative_to.resolve())) if relative_to else str(path),
    }
    if path.is_file():
        record.update(bytes=path.stat().st_size, sha256=_sha256(path))
    return record


def _verify_metric_score_artifact(path: Path) -> None:
    """Reject silently modified baseline scores before any provider requests are sent."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing baseline score artifact: {path}")
    manifest_path = path.with_name("metric_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing baseline metric manifest: {manifest_path}")
    manifest = orjson.loads(manifest_path.read_bytes())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Malformed baseline metric manifest: {manifest_path}")
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    for raw in artifacts:
        if not isinstance(raw, Mapping) or Path(str(raw.get("path", ""))).name != path.name:
            continue
        expected_sha256 = str(raw.get("sha256") or "") or None
        expected_bytes = int(raw["bytes"]) if raw.get("bytes") is not None else None
        break
    if expected_sha256 is None:
        raise ValueError(f"Baseline manifest does not bind {path.name}: {manifest_path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Baseline score byte count does not match its manifest: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Baseline score SHA-256 does not match its manifest: {path}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


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
