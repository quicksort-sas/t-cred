from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

import orjson

from tcred.metrics.analysis import analyze_metric_scores
from tcred.metrics.inputs import load_metric_inputs
from tcred.metrics.models import MetricInput, MetricScoreRecord
from tcred.metrics.reporting import render_metric_report
from tcred.metrics.task_judge_inputs import (
    load_task_judge_inputs,
    task_input_hash,
    write_task_judge_inputs,
)
from tcred.metrics.task_judge_models import TaskJudgeInput
from tcred.metrics.tcred_diagnostic_runner import TCRED_SUITE_VERSION
from tcred.metrics.tcred_models import TCredMetricResult
from tcred.metrics.tcred_semantic import (
    read_semantic_records,
    run_semantic_worker,
    validate_semantic_records,
)
from tcred.metrics.tcred_suite import score_tcred_suite
from tcred.metrics.tcred_validation import (
    PRIMARY_FIELD_COMPONENT,
    analyze_human_calibration,
    analyze_update_stability,
)

DEFAULT_COMPARATOR_SCORE_PATHS = (
    Path("data/metrics/current_sota/2026-08-13/metric_scores.jsonl"),
    Path("data/metrics/non_llm_expansion/2026-08-14/metric_scores.jsonl"),
    Path("data/metrics/tcred_task_judge/2026-08-14/metric_scores.jsonl"),
)


def run_tcred_population_evaluation(
    *,
    gold_dir: Path,
    dataset_root: Path,
    system_output_root: Path,
    output_dir: Path,
    metric_python: Path,
    comparator_score_paths: tuple[Path, ...] = DEFAULT_COMPARATOR_SCORE_PATHS,
    bootstrap_samples: int = 2000,
    semantic_batch_size: int = 16,
) -> dict[str, Path]:
    """Evaluate T-CRED on human gold and every available QA-system output."""

    if bootstrap_samples < 100:
        raise ValueError("Population evaluation requires at least 100 bootstrap replicates")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = load_task_judge_inputs(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
    )
    source_metric_rows = load_metric_inputs(
        gold_dir=gold_dir,
        dataset_root=dataset_root,
        system_output_root=system_output_root,
    )
    _validate_population_comparator_artifacts(
        comparator_score_paths,
        source_contract_sha256=_metric_input_contract_hash(source_metric_rows),
        task_input_sha256=task_input_hash(task_rows),
    )
    expected_ids = [row.metric_id for row in task_rows]
    baseline = _merge_comparator_records(comparator_score_paths, expected_ids=expected_ids)
    baseline_by_id = {row.metric_id: row for row in baseline}

    task_inputs_path = output_dir / "task_judge_inputs.jsonl"
    write_task_judge_inputs(task_rows, task_inputs_path)
    semantic_path = output_dir / "semantic_pair_scores.jsonl"
    run_semantic_worker(
        inputs_path=task_inputs_path,
        output_path=semantic_path,
        cache_path=Path("data/cache/metrics/tcred_semantic/population/alignscore_pairwise.jsonl"),
        model_cache_dir=Path("data/cache/metrics/huggingface"),
        metric_python=metric_python,
        batch_size=semantic_batch_size,
    )
    semantics = read_semantic_records(semantic_path)
    validate_semantic_records(task_rows, semantics)
    components = [
        score_tcred_suite(
            row,
            semantics[row.metric_id],
            baseline_scores=baseline_by_id[row.metric_id].scores,
            baseline_input_aligned=not row.presentation_changed_fields,
        )
        for row in task_rows
    ]
    component_path = output_dir / "tcred_component_results.jsonl"
    _write_jsonl(component_path, [row.model_dump(mode="json") for row in components])
    merged = _merge_tcred_components(baseline, components)
    merged_path = output_dir / "metric_scores_with_tcred.jsonl"
    _write_jsonl(merged_path, [row.model_dump(mode="json") for row in merged])

    aligned_ids = {row.metric_id for row in task_rows if not row.presentation_changed_fields}
    aligned_records = [row for row in merged if row.metric_id in aligned_ids]
    analysis = analyze_metric_scores(
        merged,
        bootstrap_samples=bootstrap_samples,
        seed=20260815,
    )
    aligned_analysis = analyze_metric_scores(
        aligned_records,
        bootstrap_samples=bootstrap_samples,
        seed=20260816,
    )
    analysis["tcred_presentation_aligned_comparator_analysis"] = aligned_analysis
    analysis["tcred_human_calibration_and_selective_risk"] = analyze_human_calibration(merged)
    update = analyze_update_stability(
        merged,
        dataset_root=dataset_root,
        candidate_by_metric_id={row.metric_id: row.candidate_answer for row in task_rows},
    )
    update_observations = list(update.pop("observations"))
    analysis["tcred_update_stability"] = update
    analysis["tcred_population_evaluation"] = {
        "suite_version": TCRED_SUITE_VERSION,
        "information_boundary": "automatic",
        "input_rows": len(task_rows),
        "human_gold_rows": sum(row.population == "human_gold" for row in task_rows),
        "system_full_rows": sum(row.population == "system_full" for row in task_rows),
        "presentation_aligned_rows": len(aligned_records),
        "presentation_aligned_system_full_rows": sum(
            row.population == "system_full" and not row.presentation_changed_fields
            for row in task_rows
        ),
        "presentation_drift_rows": len(task_rows) - len(aligned_records),
        "presentation_drift_by_field": {
            field: sum(field in row.presentation_changed_fields for row in task_rows)
            for field in ("question", "reference_answer", "candidate_answer")
        },
        "primary_comparator_scope": (
            "presentation-aligned rows only; all-row scores are descriptive because legacy "
            "comparators and T-CRED may otherwise see different rendered text"
        ),
        "task_input_sha256": task_input_hash(task_rows),
    }
    analysis_path = output_dir / "tcred_population_analysis.json"
    _write_json(analysis_path, analysis)
    update_path = output_dir / "update_pair_observations.jsonl"
    _write_jsonl(update_path, update_observations)
    report_path = output_dir / "tcred_population_report.md"
    report_path.write_text(
        render_metric_report(analysis).rstrip()
        + "\n\n"
        + _render_tcred_population_appendix(analysis),
        encoding="utf-8",
        newline="\n",
    )
    aligned_report_path = output_dir / "tcred_presentation_aligned_comparator_report.md"
    aligned_report_path.write_text(
        "# Presentation-Aligned Metric Comparison\n\n"
        "This is the primary metric-to-metric population comparison because every retained "
        "metric scored the same rendered question, reference, and candidate text. Rows with "
        "annotation-presentation repairs are excluded here and remain available in the all-row "
        "descriptive report.\n\n" + render_metric_report(aligned_analysis),
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = output_dir / "manifest.json"
    artifacts = [
        task_inputs_path,
        semantic_path,
        component_path,
        merged_path,
        analysis_path,
        update_path,
        report_path,
        aligned_report_path,
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "status": "complete",
            "generated_at": datetime.now(UTC).isoformat(),
            "suite_version": TCRED_SUITE_VERSION,
            "information_boundary": "automatic",
            "configuration": {
                "bootstrap_samples": bootstrap_samples,
                "semantic_batch_size": semantic_batch_size,
                "comparator_score_paths": [str(path) for path in comparator_score_paths],
            },
            "input": {
                "rows": len(task_rows),
                "task_input_sha256": task_input_hash(task_rows),
                "comparator_files": [
                    {"path": str(path), "sha256": _sha256(path)} for path in comparator_score_paths
                ],
            },
            "artifacts": [_file_record(path, relative_to=output_dir) for path in artifacts],
            "interpretation_limits": [
                "Human concurrent validity is based on the fixed, limited gold subset.",
                "Full-system means improve descriptive precision but add no human labels.",
                "Calibration outputs are diagnostic because component scores are not fitted "
                "probabilities.",
                "Update flip accuracy is restricted to disjoint gold-answer changes; overlapping "
                "updates are reported as adaptation rather than identifiable semantic flips.",
                "Metric-to-metric population comparisons are primary only on exact-presentation "
                "rows. Legacy scores for presentation-repaired rows were computed on different "
                "rendered text and are retained only for descriptive provenance.",
            ],
        },
    )
    return {
        "manifest": manifest_path,
        "scores": merged_path,
        "components": component_path,
        "analysis": analysis_path,
        "update_pairs": update_path,
        "report": report_path,
        "aligned_comparator_report": aligned_report_path,
    }


def _merge_comparator_records(
    paths: tuple[Path, ...],
    *,
    expected_ids: list[str],
) -> list[MetricScoreRecord]:
    if not paths:
        raise ValueError("At least one comparator score file is required")
    merged: dict[str, MetricScoreRecord] = {}
    expected = set(expected_ids)
    for path in paths:
        rows = _read_score_records(path)
        ids = {row.metric_id for row in rows}
        if ids != expected or len(rows) != len(expected_ids):
            raise ValueError(f"Comparator score file does not match population inputs: {path}")
        for row in rows:
            previous = merged.get(row.metric_id)
            if previous is None:
                merged[row.metric_id] = row
                continue
            _validate_record_identity(previous, row, path=path)
            scores = dict(previous.scores)
            for name, value in row.scores.items():
                old = scores.get(name)
                if (
                    old is not None
                    and value is not None
                    and not math.isclose(
                        float(old),
                        float(value),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(
                        f"Conflicting comparator score for {row.metric_id}/{name}: "
                        f"{old} versus {value} in {path}"
                    )
                if old is None:
                    scores[name] = value
            metadata = dict(previous.metric_metadata)
            metadata.update(row.metric_metadata)
            merged[row.metric_id] = previous.model_copy(
                update={"scores": scores, "metric_metadata": metadata}
            )
    return [merged[metric_id] for metric_id in expected_ids]


def _validate_population_comparator_artifacts(
    score_paths: tuple[Path, ...],
    *,
    source_contract_sha256: str,
    task_input_sha256: str,
) -> None:
    for score_path in score_paths:
        manifest_path = score_path.parent / "metric_manifest.json"
        task_manifest_path = score_path.parent / "manifest.json"
        if manifest_path.is_file():
            manifest = orjson.loads(manifest_path.read_bytes())
            _validate_manifest_status(manifest, manifest_path)
            _validate_declared_artifact(manifest, manifest_path, score_path)
            input_path = score_path.parent / "metric_inputs.jsonl"
            _validate_declared_artifact(manifest, manifest_path, input_path)
            if _metric_input_contract_hash_from_path(input_path) != source_contract_sha256:
                raise ValueError(f"Comparator metric-input contract is stale: {manifest_path}")
            continue
        if task_manifest_path.is_file():
            manifest = orjson.loads(task_manifest_path.read_bytes())
            _validate_manifest_status(manifest, task_manifest_path)
            _validate_declared_artifact(manifest, task_manifest_path, score_path)
            input_path = score_path.parent / "task_judge_inputs.jsonl"
            _validate_declared_artifact(manifest, task_manifest_path, input_path)
            artifact_input_sha256 = _task_input_hash_from_path(input_path)
            if manifest.get("input_sha256") != artifact_input_sha256:
                raise ValueError(
                    f"Comparator task-input hash does not match its artifact: {task_manifest_path}"
                )
            if artifact_input_sha256 != task_input_sha256:
                raise ValueError(f"Comparator task-input contract is stale: {task_manifest_path}")
            continue
        raise ValueError(f"Comparator score file has no checksum manifest: {score_path}")


def _validate_manifest_status(manifest: object, manifest_path: Path) -> None:
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError(f"Comparator manifest is not complete: {manifest_path}")


def _validate_declared_artifact(
    manifest: dict[str, object],
    manifest_path: Path,
    artifact_path: Path,
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Comparator manifest has no artifact ledger: {manifest_path}")
    matches = [
        row
        for row in artifacts
        if isinstance(row, dict) and Path(str(row.get("path", ""))).name == artifact_path.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Comparator manifest must declare one {artifact_path.name}: {manifest_path}"
        )
    if not artifact_path.is_file():
        raise ValueError(f"Comparator artifact is missing: {artifact_path}")
    if matches[0].get("sha256") != _sha256(artifact_path):
        raise ValueError(f"Comparator artifact hash is corrupt: {artifact_path}")


def _metric_input_contract_hash(rows: list[MetricInput]) -> str:
    payload = []
    for row in sorted(rows, key=lambda item: item.metric_id):
        value = row.model_dump(mode="json")
        value.pop("retrieval_metrics", None)
        value.pop("citation_metrics", None)
        payload.append(value)
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _metric_input_contract_hash_from_path(path: Path) -> str:
    rows = [
        MetricInput.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    return _metric_input_contract_hash(rows)


def _task_input_hash_from_path(path: Path) -> str:
    rows = [
        TaskJudgeInput.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    return task_input_hash(rows)


def _validate_record_identity(
    left: MetricScoreRecord,
    right: MetricScoreRecord,
    *,
    path: Path,
) -> None:
    fields = (
        "population",
        "dataset_family",
        "source_kind",
        "system_name",
        "unit_id",
        "qid",
        "scenario_id",
        "gold_labels",
        "gold_provenance",
    )
    for field in fields:
        if getattr(left, field) != getattr(right, field):
            raise ValueError(f"Comparator identity conflict for {left.metric_id}/{field} in {path}")


def _merge_tcred_components(
    baseline: list[MetricScoreRecord],
    components: list[TCredMetricResult],
) -> list[MetricScoreRecord]:
    by_id = {row.metric_id: row for row in components}
    if set(by_id) != {row.metric_id for row in baseline}:
        raise ValueError("T-CRED component IDs do not match comparator records")
    output = []
    for row in baseline:
        component = by_id[row.metric_id]
        metadata = dict(row.metric_metadata)
        metadata["tcred"] = {
            "suite_version": TCRED_SUITE_VERSION,
            "mode": component.mode,
            "query": component.query.model_dump(mode="json"),
            "coverage": component.coverage,
            "audit": component.audit,
        }
        output.append(
            row.model_copy(
                update={
                    "scores": {**row.scores, **component.scores},
                    "metric_metadata": metadata,
                }
            )
        )
    return output


def _render_tcred_population_appendix(analysis: dict[str, object]) -> str:
    population = analysis["tcred_population_evaluation"]
    calibration = analysis["tcred_human_calibration_and_selective_risk"]
    updates = analysis["tcred_update_stability"]
    primary_human = analysis["human_gold_all"]["human_correlations"]
    anomaly_human = analysis["human_gold_preexisting_anomaly_sensitivity"]["human_correlations"]
    lines = [
        "# T-CRED Population Validation Appendix",
        "",
        "## Scope",
        "",
        f"- Suite version: `{population['suite_version']}`",
        f"- Human-gold cards: **{population['human_gold_rows']}**",
        f"- Complete QA outputs: **{population['system_full_rows']}**",
        f"- Exact-presentation comparison rows: **{population['presentation_aligned_rows']}** "
        f"({population['presentation_aligned_system_full_rows']} complete QA outputs)",
        f"- Presentation-drift rows: **{population['presentation_drift_rows']}**",
        "- Human correlations and calibration are concurrent-validity evidence; complete-run "
        "means are descriptive only.",
        "- Metric-to-metric population comparisons are primary only in "
        "`tcred_presentation_aligned_comparator_report.md`; legacy metrics on repaired rows did "
        "not score the same rendered text.",
        "",
        "## Human Calibration and Selective Risk",
        "",
        "| Human field | T-CRED component | n | Brier | ECE | AURC | Risk@50% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, field_row in calibration["fields"].items():
        metric = field_row["primary_tcred_component"]
        values = field_row["metrics"].get(metric, {}) if metric else {}
        if not values:
            continue
        lines.append(
            f"| `{field}` | `{metric}` | {values['n']} | {values['brier']:.3f} | "
            f"{values['ece_equal_count']:.3f} | {values['aurc']:.3f} | "
            f"{values['selective_risk_at_50pct']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These values are not post-hoc calibrated probabilities. AURC uses tie-aware expected "
            "risk within equal-score blocks; partial human labels remain at 0.5.",
            "",
            "## Pre-existing Gold-Anomaly Sensitivity",
            "",
            "The primary gold is unchanged. This field-level sensitivity excludes only anomalies "
            "documented independently on 14 August 2026, before T-CRED development: evidence and "
            "graph labels for `heu_766c85bb0a247b84dc64`, and the evidence label for "
            "`heu_98f2dd90833923f2512e`.",
            "",
            "| Human field | T-CRED component | Primary n | Primary rho | Sensitivity n | "
            "Sensitivity rho |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for field, metric in PRIMARY_FIELD_COMPONENT.items():
        primary = primary_human.get(field, {}).get(metric, {})
        sensitivity = anomaly_human.get(field, {}).get(metric, {})
        if not primary:
            continue
        lines.append(
            f"| `{field}` | `{metric}` | {primary['n']} | {_format(primary['spearman'])} | "
            f"{sensitivity.get('n', 0)} | {_format(sensitivity.get('spearman'))} |"
        )
    lines.extend(
        [
            "",
            "## Paired Update Stability",
            "",
            "| QA system | Metric | Stable n | Stable score | Pass@.50/.75 | Affected n | "
            "Adaptation score | Pass@.50/.75 | Disjoint n | Flip score | Pass@.50/.75 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system, metric_rows in updates["by_system"].items():
        for metric in (
            "tcred_answer_equivalence",
            "tcred_temporal_correctness",
            "tcred_response_decision",
        ):
            values = metric_rows.get(metric)
            if not values:
                continue
            lines.append(
                f"| {system} | `{metric}` | {values['stable_pairs']} | "
                f"{_format(values['update_stable_score'])} | "
                f"{_format(values['update_stable_pass_at_0_50'])}/"
                f"{_format(values['update_stable_pass_at_0_75'])} | "
                f"{values['affected_pairs']} | {_format(values['update_adaptation_score'])} | "
                f"{_format(values['update_adaptation_pass_at_0_50'])}/"
                f"{_format(values['update_adaptation_pass_at_0_75'])} | "
                f"{values['disjoint_flip_pairs']} | {_format(values['update_flip_score'])} | "
                f"{_format(values['update_flip_pass_at_0_50'])}/"
                f"{_format(values['update_flip_pass_at_0_75'])} |"
            )
    lines.extend(
        [
            "",
            "Continuous scores are primary. Pass@.50 and Pass@.75 are sensitivity analyses at "
            "two bounded-score operating points, not calibrated accuracy claims. Flip results "
            "are limited to disjoint gold entity sets, where local correctness at both snapshots "
            "establishes semantic answer change without a lexical candidate-to-candidate judge.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_score_records(path: Path) -> list[MetricScoreRecord]:
    with path.open("rb") as stream:
        return [
            MetricScoreRecord.model_validate(orjson.loads(line)) for line in stream if line.strip()
        ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            stream.write(b"\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    temporary.replace(path)


def _file_record(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}"
