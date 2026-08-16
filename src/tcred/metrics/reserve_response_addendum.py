from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tcred.metrics.deterministic import claim_judge_scores
from tcred.metrics.diagnostic_analysis import analyze_diagnostic_suite
from tcred.metrics.diagnostic_models import DiagnosticCase, DiagnosticPair, DiagnosticSuite
from tcred.metrics.judge import score_with_claim_judge
from tcred.metrics.judge_batch import score_with_batch_judge
from tcred.metrics.models import JudgeCacheRecord, MetricInput, MetricScoreRecord
from tcred.metrics.source_disjoint_io import (
    file_record,
    mapping,
    read_json,
    read_jsonl,
    resolve_path,
    sha256,
    write_json,
    write_jsonl,
)
from tcred.metrics.tcred_diagnostic_runner import summarize_matched_baseline_dominance

DEFAULT_BASELINE_DIR = Path("data/metrics/diagnostic_meta_evaluation/reserve-v1.4")
DEFAULT_TCRED_DIR = Path("data/metrics/tcred_suite/reserve-v1.4")
DEFAULT_OUTPUT_DIR = Path(
    "data/metrics/tcred_suite/reserve-v1.4-response-addendum-2026-08-16"
)
DEFAULT_PROTOCOL = Path(
    "docs/protocols/tcred-v1.4-source-disjoint-validation-v1.json"
)
_CLAIM_SCORE_NAMES = (
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
)


def run_reserve_response_addendum(
    *,
    repository_root: Path,
    gold_dir: Path,
    dataset_root: Path = Path("data/generated/tcred_release"),
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    tcred_dir: Path = DEFAULT_TCRED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    protocol_path: Path = DEFAULT_PROTOCOL,
    judge_transport: Literal["direct", "batch"] = "batch",
    concurrency: int = 8,
    requests_per_second: float = 0.24,
) -> dict[str, Path]:
    """Add the missing response comparator to an immutable copy of the opened reserve."""

    repository_root = repository_root.resolve()
    baseline_dir = resolve_path(repository_root, baseline_dir)
    tcred_dir = resolve_path(repository_root, tcred_dir)
    output_dir = resolve_path(repository_root, output_dir)
    gold_dir = resolve_path(repository_root, gold_dir)
    dataset_root = resolve_path(repository_root, dataset_root)
    protocol_path = resolve_path(repository_root, protocol_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest = read_json(baseline_dir / "manifest.json")
    protocol = read_json(protocol_path)
    comparator = mapping(protocol, "comparator_execution")
    provider = str(comparator["ragchecker_claim_judge_provider"])
    model = str(comparator["ragchecker_claim_judge_model"])
    if provider != "mistral":
        raise ValueError("The locked response comparator requires provider=mistral")
    suite = _load_reserve_suite(baseline_dir, baseline_manifest)
    source_tcred_scores = tcred_dir / "metric_scores_with_tcred.jsonl"
    _verify_manifest_artifacts(baseline_dir, baseline_manifest)
    tcred_manifest = read_json(tcred_dir / "manifest.json")
    _verify_named_artifact(
        tcred_dir,
        tcred_manifest,
        "metric_scores_with_tcred.jsonl",
    )

    response_case_ids = {
        case_id
        for pair in suite.pairs
        if pair.target_construct == "response_decision"
        for case_id in (pair.left_case_id, pair.right_case_id)
    }
    response_rows = [
        case.metric_input for case in suite.cases if case.case_id in response_case_ids
    ]
    response_suite = DiagnosticSuite(
        seed=suite.seed,
        source_split=suite.source_split,
        pair_cap_per_phenomenon=suite.pair_cap_per_phenomenon,
        dataset_content_hashes=suite.dataset_content_hashes,
        cases=[case for case in suite.cases if case.case_id in response_case_ids],
        pairs=[pair for pair in suite.pairs if pair.target_construct == "response_decision"],
        audit={
            "scope": "opened_reserve_response_decision_only",
            "case_count": len(response_case_ids),
            "pair_count": sum(
                pair.target_construct == "response_decision" for pair in suite.pairs
            ),
            "parent_suite_audit": suite.audit,
        },
    )
    cache_dir = (
        repository_root
        / "data/cache/metrics/reserve_response_addendum/claim_judge"
        / provider
        / _safe_name(model)
    )
    if judge_transport == "batch":
        claims = asyncio.run(
            score_with_batch_judge(
                response_rows,
                cache_dir=cache_dir,
                batch_dir=output_dir / "claim_judge_batch",
                provider="mistral",
                model=model,
                fallback_concurrency=concurrency,
                fallback_requests_per_second=requests_per_second,
            )
        )
    else:
        claims = asyncio.run(
            score_with_claim_judge(
                response_rows,
                cache_dir=cache_dir,
                provider="mistral",
                model=model,
                concurrency=concurrency,
                requests_per_second=requests_per_second,
            )
        )
    _validate_claim_records(
        claims,
        expected_ids=response_case_ids,
        expected_model=model,
    )

    claims_path = output_dir / "claim_judgments_response_only.jsonl"
    write_jsonl(
        claims_path,
        [claims[row.metric_id].model_dump(mode="json") for row in response_rows],
    )
    source_records = [
        MetricScoreRecord.model_validate(row) for row in read_jsonl(source_tcred_scores)
    ]
    augmented = _augment_scores(source_records, response_rows=response_rows, claims=claims)
    scores_path = output_dir / "metric_scores_with_response_comparator.jsonl"
    write_jsonl(scores_path, [row.model_dump(mode="json") for row in augmented])

    analysis = analyze_diagnostic_suite(
        response_suite,
        [record for record in augmented if record.metric_id in response_case_ids],
        dataset_root=dataset_root,
        gold_dir=gold_dir,
        bootstrap_samples=10_000,
        seed=int(mapping(protocol, "inference")["seed"]),
    )
    matched = summarize_matched_baseline_dominance(analysis)
    response_result = _matched_construct(matched, "response_decision")
    result = {
        "schema_version": "1.0",
        "status": "posthoc_descriptive_nonconfirmatory",
        "reason": (
            "The original reserve was already opened before this predeclared comparator became "
            "available. This addendum cannot repair or replace the original confirmatory result."
        ),
        "response_decision": response_result,
        "response_construct_analysis": mapping(
            mapping(analysis, "constructs"), "response_decision"
        ),
        "matched_summary_context": matched,
    }
    analysis_path = output_dir / "response_addendum_analysis.json"
    write_json(analysis_path, result)
    report_path = output_dir / "response_addendum_report.md"
    report_path.write_text(
        _render_report(
            response_result=response_result,
            response_case_count=len(response_rows),
            response_pair_count=sum(
                pair.target_construct == "response_decision" for pair in suite.pairs
            ),
            model=model,
        ),
        encoding="utf-8",
        newline="\n",
    )

    prompt_hashes = sorted({record.prompt_sha256 for record in claims.values()})
    if len(prompt_hashes) != 1:
        raise ValueError("Reserve response judgments do not share one prompt hash")
    manifest_path = output_dir / "manifest.json"
    artifacts = [claims_path, scores_path, analysis_path, report_path]
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "status": "complete_posthoc_descriptive_nonconfirmatory",
            "generated_at": datetime.now(UTC).isoformat(),
            "protocol_id": protocol["protocol_id"],
            "source_artifacts": {
                "baseline_manifest": {
                    "path": baseline_dir.as_posix(),
                    "sha256": sha256(baseline_dir / "manifest.json"),
                },
                "frozen_tcred_scores": {
                    "path": source_tcred_scores.as_posix(),
                    "sha256": sha256(source_tcred_scores),
                },
                "frozen_tcred_manifest_sha256": sha256(tcred_dir / "manifest.json"),
            },
            "configuration": {
                "source_split": suite.source_split,
                "response_case_count": len(response_rows),
                "response_pair_count": sum(
                    pair.target_construct == "response_decision" for pair in suite.pairs
                ),
                "provider": provider,
                "model": model,
                "transport": judge_transport,
                "prompt_sha256": prompt_hashes[0],
                "contract_version": comparator["ragchecker_contract_version"],
                "temperature": comparator["ragchecker_temperature"],
                "random_seed": comparator["ragchecker_random_seed"],
                "bootstrap_replicates": 10_000,
                "task_llm_judge_enabled": False,
            },
            "usage": _judge_usage(claims),
            "artifacts": [file_record(path, relative_to=output_dir) for path in artifacts],
            "interpretation": (
                "RAGChecker-style response-decision evidence on an already opened reserve; "
                "descriptive only and excluded from the source-disjoint confirmatory claim."
            ),
            "limitations": [
                "The reserve was already inspected before this addendum was run.",
                "The hosted claim judge is pinned but not assumed bitwise deterministic.",
                "The published RAGChecker formulas are reproduced with the frozen project "
                "claim schema/prompt, not a bit-identical official package execution.",
            ],
        },
    )
    return {
        "manifest": manifest_path,
        "claim_judgments": claims_path,
        "scores": scores_path,
        "analysis": analysis_path,
        "report": report_path,
    }


def _load_reserve_suite(
    baseline_dir: Path,
    manifest: dict[str, object],
) -> DiagnosticSuite:
    configuration = mapping(manifest, "configuration")
    return DiagnosticSuite(
        seed=int(configuration["seed"]),
        source_split=str(configuration["source_split"]),
        pair_cap_per_phenomenon=int(configuration["pair_cap_per_phenomenon"]),
        dataset_content_hashes={
            str(key): str(value)
            for key, value in mapping(manifest, "dataset_content_hashes").items()
        },
        cases=[
            DiagnosticCase.model_validate(row)
            for row in read_jsonl(baseline_dir / "diagnostic_cases.jsonl")
        ],
        pairs=[
            DiagnosticPair.model_validate(row)
            for row in read_jsonl(baseline_dir / "diagnostic_pairs.jsonl")
        ],
        audit=mapping(manifest, "suite_audit"),
    )


def _verify_manifest_artifacts(
    baseline_dir: Path,
    manifest: dict[str, object],
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Reserve manifest has no artifact ledger")
    required = {
        "diagnostic_cases.jsonl",
        "diagnostic_pairs.jsonl",
        "metric_inputs.jsonl",
        "task_judge_inputs.jsonl",
        "metric_scores.jsonl",
    }
    observed: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise ValueError("Malformed reserve artifact ledger")
        path_value = str(raw.get("path", ""))
        if path_value not in required:
            continue
        path = baseline_dir / path_value
        if not path.is_file() or sha256(path) != str(raw.get("sha256", "")):
            raise ValueError(f"Opened reserve artifact differs from its manifest: {path_value}")
        observed.add(path_value)
    if observed != required:
        raise ValueError(f"Reserve manifest is missing required artifacts: {required - observed}")


def _verify_named_artifact(
    directory: Path,
    manifest: dict[str, object],
    name: str,
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Manifest has no artifact ledger: {directory}")
    entries = [
        row for row in artifacts if isinstance(row, dict) and row.get("path") == name
    ]
    if len(entries) != 1:
        raise ValueError(f"Manifest must contain exactly one {name} entry")
    path = directory / name
    if not path.is_file() or sha256(path) != str(entries[0].get("sha256", "")):
        raise ValueError(f"Frozen artifact differs from its manifest: {path}")


def _augment_scores(
    source_records: list[MetricScoreRecord],
    *,
    response_rows: list[MetricInput],
    claims: dict[str, JudgeCacheRecord],
) -> list[MetricScoreRecord]:
    input_by_id = {row.metric_id: row for row in response_rows}
    if set(input_by_id) != set(claims):
        raise ValueError("Response inputs and claim judgments do not align")
    output = []
    for record in source_records:
        judgment = claims.get(record.metric_id)
        if judgment is None:
            output.append(record)
            continue
        row = input_by_id[record.metric_id]
        claim_scores = claim_judge_scores(
            judgment.result,
            retrieved_count=len(row.retrieved_evidence),
            cited_count=len(row.cited_evidence),
        )
        scores = dict(record.scores)
        for name in _CLAIM_SCORE_NAMES:
            scores[name] = claim_scores[name]
        metadata = dict(record.metric_metadata)
        metadata["posthoc_response_comparator"] = {
            "provider": judgment.provider,
            "model": judgment.model,
            "prompt_sha256": judgment.prompt_sha256,
            "candidate_claim_count": len(judgment.result.candidate_claims),
            "reference_claim_count": len(judgment.result.reference_claims),
        }
        output.append(record.model_copy(update={"scores": scores, "metric_metadata": metadata}))
    if {record.metric_id for record in output} != {record.metric_id for record in source_records}:
        raise ValueError("Score augmentation changed reserve case identity")
    return output


def _judge_usage(records: dict[str, JudgeCacheRecord]) -> dict[str, int]:
    names = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    return {name: sum(record.usage.get(name, 0) for record in records.values()) for name in names}


def _validate_claim_records(
    records: dict[str, JudgeCacheRecord],
    *,
    expected_ids: set[str],
    expected_model: str,
) -> None:
    if set(records) != expected_ids:
        raise ValueError("Reserve response claim-judge IDs do not match the expected cases")
    if any(record.model != expected_model for record in records.values()):
        raise ValueError("Reserve response claim judgments contain a different model")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )


def _matched_construct(
    matched: dict[str, object],
    construct: str,
) -> dict[str, object]:
    comparisons = matched.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("Matched comparison summary has no comparisons table")
    rows = [
        row
        for row in comparisons
        if isinstance(row, dict) and row.get("construct") == construct
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one matched comparison for {construct}, found {len(rows)}")
    return rows[0]


def _render_report(
    *,
    response_result: dict[str, object],
    response_case_count: int,
    response_pair_count: int,
    model: str,
) -> str:
    return "\n".join(
        [
            "# Opened-Reserve Response Comparator Addendum",
            "",
            "> **Status: post-hoc descriptive, non-confirmatory.** The reserve was opened before",
            "> this comparator became available. This result cannot repair the original reserve",
            "> confirmation and is not pooled with the source-disjoint validation.",
            "",
            f"- response-decision cases judged: **{response_case_count}**",
            f"- response-decision pairs: **{response_pair_count}**",
            f"- claim model: `{model}`",
            "- task-level LLM judge: **disabled**",
            "- resampling: **10,000** connected-source-cluster replicates",
            "",
            "## Matched result",
            "",
            "```json",
            _pretty_json(response_result),
            "```",
            "",
            "## Interpretation",
            "",
            "This addendum answers only the historical missing-comparator question. The formal",
            "source-disjoint study remains the sole eligible test of the newly preregistered",
            "eight-construct matched protocol. Hosted judge output is not treated as human truth.",
            "",
        ]
    )


def _pretty_json(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
