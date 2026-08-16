from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tcred.metrics.judge import score_with_claim_judge
from tcred.metrics.judge_batch import score_with_batch_judge
from tcred.metrics.models import JudgeCacheRecord
from tcred.metrics.runner import (
    _validate_score_records,
    run_neural_metric_worker,
    score_metric_input,
)
from tcred.metrics.source_disjoint_io import (
    DEFAULT_STUDY_ROOT,
    challenge_hashes,
    file_record,
    load_prepared_suite,
    mapping,
    read_jsonl,
    resolve_path,
    write_json,
    write_jsonl,
)


def run_source_disjoint_comparators(
    *,
    repository_root: Path,
    metric_python: Path,
    study_root: Path = DEFAULT_STUDY_ROOT,
    judge_transport: Literal["direct", "batch"] = "batch",
    concurrency: int = 8,
    requests_per_second: float = 0.24,
    alignscore_batch_size: int = 32,
) -> dict[str, Path]:
    """Score the locked challenge with local baselines and scoped RAGChecker-style F1."""

    repository_root = repository_root.resolve()
    study_root = resolve_path(repository_root, study_root)
    metric_python = resolve_path(repository_root, metric_python)
    suite, protocol = load_prepared_suite(
        repository_root=repository_root,
        study_root=study_root,
    )
    comparator_contract = mapping(protocol, "comparator_execution")
    provider = str(comparator_contract["ragchecker_claim_judge_provider"])
    model = str(comparator_contract["ragchecker_claim_judge_model"])
    if provider != "mistral":
        raise ValueError("The locked comparator protocol requires provider=mistral")
    if not metric_python.is_file():
        raise FileNotFoundError(f"Pinned neural metric Python is missing: {metric_python}")

    output_dir = study_root / "comparators"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_inputs_path = study_root / "challenge" / "metric_inputs.jsonl"
    neural_path = output_dir / "neural_scores.jsonl"
    run_neural_metric_worker(
        inputs_path=metric_inputs_path,
        output_path=neural_path,
        cache_dir=repository_root / "data/cache/metrics/source_disjoint_validation/neural",
        model_cache_dir=repository_root / "data/cache/metrics/huggingface",
        metric_python=metric_python,
        minicheck_scope="all",
        alignscore_batch_size=alignscore_batch_size,
        metrics=("pedants", "alignscore"),
    )
    neural = {str(row["metric_id"]): row for row in read_jsonl(neural_path)}
    expected_ids = {case.case_id for case in suite.cases}
    if set(neural) != expected_ids:
        raise ValueError("Neural comparator IDs do not match the locked challenge")

    response_case_ids = {
        case_id
        for pair in suite.pairs
        if pair.target_construct == "response_decision"
        for case_id in (pair.left_case_id, pair.right_case_id)
    }
    response_rows = [
        case.metric_input for case in suite.cases if case.case_id in response_case_ids
    ]
    claim_cache = (
        repository_root
        / "data/cache/metrics/source_disjoint_validation/claim_judge"
        / provider
        / _safe_name(model)
    )
    if judge_transport == "batch":
        claim_records = asyncio.run(
            score_with_batch_judge(
                response_rows,
                cache_dir=claim_cache,
                batch_dir=output_dir / "claim_judge_batch",
                provider="mistral",
                model=model,
                fallback_concurrency=concurrency,
                fallback_requests_per_second=requests_per_second,
            )
        )
    else:
        claim_records = asyncio.run(
            score_with_claim_judge(
                response_rows,
                cache_dir=claim_cache,
                provider="mistral",
                model=model,
                concurrency=concurrency,
                requests_per_second=requests_per_second,
            )
        )
    _validate_claim_records(
        claim_records,
        expected_ids=response_case_ids,
        expected_model=model,
    )
    claims_path = output_dir / "claim_judgments_response_only.jsonl"
    write_jsonl(
        claims_path,
        [claim_records[row.metric_id].model_dump(mode="json") for row in response_rows],
    )

    records = [
        score_metric_input(
            case.metric_input,
            judge=claim_records.get(case.case_id),
            neural=neural[case.case_id],
        )
        for case in suite.cases
    ]
    _validate_score_records(records)
    scores_path = output_dir / "metric_scores.jsonl"
    write_jsonl(scores_path, [row.model_dump(mode="json") for row in records])
    manifest_path = output_dir / "manifest.json"
    artifacts = [neural_path, claims_path, scores_path]
    prompt_hashes = sorted({record.prompt_sha256 for record in claim_records.values()})
    if len(prompt_hashes) != 1:
        raise ValueError("Response claim judgments do not share one frozen prompt hash")
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_content_hashes": suite.dataset_content_hashes,
            "challenge_artifact_sha256": challenge_hashes(study_root),
            "configuration": {
                "source_split": suite.source_split,
                "case_count": len(suite.cases),
                "pair_count": len(suite.pairs),
                "neural_metrics_enabled": True,
                "neural_metric_scope": ["pedants", "alignscore"],
                "alignscore_batch_size": alignscore_batch_size,
                "claim_judge_scope": "response_decision_pair_cases_only",
                "claim_judge_case_count": len(response_rows),
                "claim_judge_provider": provider,
                "claim_judge_model": model,
                "claim_judge_transport": judge_transport,
                "claim_judge_prompt_sha256": prompt_hashes[0],
                "claim_judge_contract_version": comparator_contract[
                    "ragchecker_contract_version"
                ],
                "claim_judge_temperature": comparator_contract[
                    "ragchecker_temperature"
                ],
                "claim_judge_random_seed": comparator_contract[
                    "ragchecker_random_seed"
                ],
                "task_llm_judge_enabled": False,
            },
            "ragchecker_status": (
                "The published claim-level precision/recall/F1 formulas are reproduced. Claim "
                "decomposition and support judgments use the frozen project schema/prompt and "
                "are therefore reported as RAGChecker-style rather than bit-identical official "
                "package output."
            ),
            "artifacts": [file_record(path, relative_to=output_dir) for path in artifacts],
            "limitations": [
                "The hosted claim judge is pinned by provider, model name, prompt hash, schema, "
                "temperature, and seed, but hosted inference is not assumed bitwise deterministic.",
                "RAGChecker-style scores are intentionally absent outside response-decision "
                "cases; this is preregistered missingness, not a failed computation.",
                "No task-level LLM judge is included in this comparator pool.",
            ],
        },
    )
    return {
        "manifest": manifest_path,
        "scores": scores_path,
        "neural": neural_path,
        "claim_judgments": claims_path,
    }


def _validate_claim_records(
    records: dict[str, JudgeCacheRecord],
    *,
    expected_ids: set[str],
    expected_model: str,
) -> None:
    if set(records) != expected_ids:
        raise ValueError("Scoped response claim-judge IDs do not match the locked response cases")
    if any(record.model != expected_model for record in records.values()):
        raise ValueError("Scoped response claim judgments contain a different model")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )
