from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from tcred.metrics.models import MetricScoreRecord
from tcred.metrics.task_judge_models import JudgeEvidence, JudgeGraphPath, TaskJudgeInput
from tcred.metrics.tcred_claims import is_refusal
from tcred.metrics.tcred_models import TCredMetricResult
from tcred.trainable_metrics.inference import (
    SemanticInferenceInput,
    SemanticPrediction,
    TCredSLInference,
)
from tcred.trainable_metrics.schema import EvidencePassage, GraphPathText, SemanticTask

_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass
class _CasePlan:
    row: TaskJudgeInput
    exact: TCredMetricResult
    claims: list[str]
    evidence: list[JudgeEvidence]
    query_text: str | None
    answer_ids: list[tuple[str | None, str]] = field(default_factory=list)
    support_ids: dict[tuple[int, str], str] = field(default_factory=dict)
    temporal_ids: dict[tuple[int, str], str] = field(default_factory=dict)
    relevance_ids: dict[str, str] = field(default_factory=dict)
    answerability_ids: list[tuple[str | None, str]] = field(default_factory=list)
    graph_ids: dict[tuple[int, str], str] = field(default_factory=dict)
    citation_ids: dict[tuple[int, str], str] = field(default_factory=dict)


def score_metric_cases(
    *,
    engine: TCredSLInference,
    rows: list[TaskJudgeInput],
    exact_results: list[TCredMetricResult],
    batch_size: int = 64,
    max_evidence: int = 8,
    max_answer_evidence: int = 5,
    max_graph_paths: int = 8,
    support_threshold: float = 0.5,
) -> tuple[list[MetricScoreRecord], dict[str, Any]]:
    """Score evaluator cards with learned semantics plus frozen deterministic rules."""

    if not 0.0 < support_threshold < 1.0:
        raise ValueError("support_threshold must lie strictly between zero and one")
    if min(batch_size, max_evidence, max_answer_evidence, max_graph_paths) < 1:
        raise ValueError("Batch and bounded-input limits must be positive")
    exact_by_id = {result.metric_id: result for result in exact_results}
    if len(exact_by_id) != len(exact_results):
        raise ValueError("Exact T-CRED component results contain duplicate metric IDs")
    row_ids = {row.metric_id for row in rows}
    if row_ids != set(exact_by_id):
        missing = sorted(row_ids - set(exact_by_id))[:10]
        extra = sorted(set(exact_by_id) - row_ids)[:10]
        raise ValueError(
            f"Task inputs and exact results disagree; missing={missing}, extra={extra}"
        )

    started = time.perf_counter()
    plans: list[_CasePlan] = []
    primary_inputs: list[SemanticInferenceInput] = []
    for row in rows:
        plan, model_inputs = _build_primary_inputs(
            row,
            exact_by_id[row.metric_id],
            max_evidence=max_evidence,
            max_answer_evidence=max_answer_evidence,
            max_graph_paths=max_graph_paths,
        )
        plans.append(plan)
        primary_inputs.extend(model_inputs)
    primary_predictions, primary_runtime = engine.predict(primary_inputs, batch_size=batch_size)
    prediction_by_id = {prediction.input_id: prediction for prediction in primary_predictions}

    citation_inputs: list[SemanticInferenceInput] = []
    for plan in plans:
        citation_inputs.extend(_build_citation_inputs(plan, prediction_by_id))
    citation_predictions, citation_runtime = engine.predict(citation_inputs, batch_size=batch_size)
    prediction_by_id.update(
        {prediction.input_id: prediction for prediction in citation_predictions}
    )

    outputs = [
        _assemble_score_record(
            plan,
            prediction_by_id,
            support_threshold=support_threshold,
            engine=engine,
        )
        for plan in plans
    ]
    return outputs, {
        "schema_version": "tcred-sl-scoring-run-v1",
        "cases": len(rows),
        "semantic_inputs": len(primary_inputs) + len(citation_inputs),
        "primary_pass": primary_runtime,
        "citation_pass": citation_runtime,
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": {
            "batch_size": batch_size,
            "max_evidence": max_evidence,
            "max_answer_evidence": max_answer_evidence,
            "max_graph_paths": max_graph_paths,
            "support_threshold": support_threshold,
            "claim_weight_cap_tokens": 32,
        },
    }


def _build_primary_inputs(
    row: TaskJudgeInput,
    exact: TCredMetricResult,
    *,
    max_evidence: int,
    max_answer_evidence: int,
    max_graph_paths: int,
) -> tuple[_CasePlan, list[SemanticInferenceInput]]:
    if exact.metric_id != row.metric_id:
        raise ValueError(f"Exact result ID mismatch: {row.metric_id}")
    claims = list(exact.candidate_claims)
    if not claims and row.candidate_answer.strip() and not is_refusal(row.candidate_answer):
        claims = [row.candidate_answer]
    evidence = _displayed_evidence(row)[:max_evidence]
    query_text = _query_text(exact)
    plan = _CasePlan(
        row=row,
        exact=exact,
        claims=claims,
        evidence=evidence,
        query_text=query_text,
    )
    inputs: list[SemanticInferenceInput] = []

    answer_evidence = evidence[:max_answer_evidence]
    if not answer_evidence:
        answer_id = f"{row.metric_id}::answer::none"
        plan.answer_ids.append((None, answer_id))
        inputs.append(_semantic_input(plan, answer_id, SemanticTask.ANSWER))
    else:
        for item in answer_evidence:
            input_id = f"{row.metric_id}::answer::{item.evidence_id}"
            plan.answer_ids.append((item.evidence_id, input_id))
            inputs.append(
                _semantic_input(
                    plan,
                    input_id,
                    SemanticTask.ANSWER,
                    evidence=[item],
                )
            )

    for item in evidence:
        input_id = f"{row.metric_id}::relevance::{item.evidence_id}"
        plan.relevance_ids[item.evidence_id] = input_id
        inputs.append(
            _semantic_input(
                plan,
                input_id,
                SemanticTask.RELEVANCE,
                candidate_or_claim=item.text,
                evidence=[item],
            )
        )
    for claim_index, claim in enumerate(claims):
        for item in evidence:
            support_id = f"{row.metric_id}::support::{claim_index}::{item.evidence_id}"
            plan.support_ids[(claim_index, item.evidence_id)] = support_id
            inputs.append(
                _semantic_input(
                    plan,
                    support_id,
                    SemanticTask.SUPPORT,
                    candidate_or_claim=claim,
                    evidence=[item],
                )
            )
            if _temporal_applicable(exact):
                temporal_id = f"{row.metric_id}::temporal::{claim_index}::{item.evidence_id}"
                plan.temporal_ids[(claim_index, item.evidence_id)] = temporal_id
                inputs.append(
                    _semantic_input(
                        plan,
                        temporal_id,
                        SemanticTask.TEMPORAL,
                        candidate_or_claim=claim,
                        evidence=[item],
                    )
                )

    if not evidence:
        input_id = f"{row.metric_id}::answerability::none"
        plan.answerability_ids.append((None, input_id))
        inputs.append(_semantic_input(plan, input_id, SemanticTask.ANSWERABILITY))
    else:
        for item in evidence:
            input_id = f"{row.metric_id}::answerability::{item.evidence_id}"
            plan.answerability_ids.append((item.evidence_id, input_id))
            inputs.append(
                _semantic_input(
                    plan,
                    input_id,
                    SemanticTask.ANSWERABILITY,
                    evidence=[item],
                )
            )

    paths = row.graph_paths[:max_graph_paths]
    path_assessments = {path.path_id: path for path in exact.path_assessments}
    missing_path_results = [path.path_id for path in paths if path.path_id not in path_assessments]
    if missing_path_results:
        raise ValueError(
            f"Graph paths have no deterministic assessment for {row.metric_id}: "
            f"{missing_path_results}"
        )
    for claim_index, claim in enumerate(claims):
        for path in paths:
            input_id = f"{row.metric_id}::graph::{claim_index}::{path.path_id}"
            plan.graph_ids[(claim_index, path.path_id)] = input_id
            path_text = _serialize_graph_path(path)
            evidence_text = " ".join(
                edge.evidence_text.strip() for edge in path.edges if edge.evidence_text.strip()
            ) or path_text
            inputs.append(
                _semantic_input(
                    plan,
                    input_id,
                    SemanticTask.SUPPORT,
                    candidate_or_claim=claim,
                    evidence=[
                        JudgeEvidence(
                            evidence_id=f"path:{path.path_id}",
                            text=evidence_text,
                        )
                    ],
                    paths=[GraphPathText(path_id=path.path_id, text=path_text)],
                )
            )
    return plan, inputs


def _build_citation_inputs(
    plan: _CasePlan,
    predictions: dict[str, SemanticPrediction],
) -> list[SemanticInferenceInput]:
    cited = {item.evidence_id: item for item in plan.row.cited_evidence}
    if not cited or not plan.claims:
        return []
    cited_ids = set(cited)
    inputs: list[SemanticInferenceInput] = []
    for claim_index, claim in enumerate(plan.claims):
        alternatives = [item for item in plan.evidence if item.evidence_id not in cited_ids]
        alternatives.sort(
            key=lambda item: _support_entailment(
                predictions[plan.support_ids[(claim_index, item.evidence_id)]]
            ),
            reverse=True,
        )
        best_other = alternatives[0] if alternatives else None
        for evidence_id, item in cited.items():
            input_id = f"{plan.row.metric_id}::citation::{claim_index}::{evidence_id}"
            plan.citation_ids[(claim_index, evidence_id)] = input_id
            visible = [item, *([best_other] if best_other is not None else [])]
            inputs.append(
                _semantic_input(
                    plan,
                    input_id,
                    SemanticTask.CITATION,
                    candidate_or_claim=claim,
                    evidence=visible,
                    citations=[evidence_id],
                )
            )
    return inputs


def _assemble_score_record(
    plan: _CasePlan,
    predictions: dict[str, SemanticPrediction],
    *,
    support_threshold: float,
    engine: TCredSLInference,
) -> MetricScoreRecord:
    answer_score, answer_auxiliary = _answer_score(plan, predictions)
    claim_weights = _claim_weights(plan.claims)
    support_values: list[float] = []
    support_binary_values: list[float] = []
    contradiction_values: list[float] = []
    temporal_values: list[float] = []
    temporal_contradictions: list[float] = []
    fallback_temporal_links = 0
    exact_temporal_links = 0
    time_by_id = {row.evidence_id: row for row in plan.exact.evidence_time}

    for claim_index, _claim in enumerate(plan.claims):
        entailments: list[float] = []
        binary_support: list[float] = []
        contradictions: list[float] = []
        temporal_support: list[float] = []
        temporal_conflict: list[float] = []
        for item in plan.evidence:
            support = predictions[plan.support_ids[(claim_index, item.evidence_id)]]
            entailment = _support_entailment(support)
            contradiction = support.class_probabilities["contradiction"]
            entailments.append(entailment)
            binary_support.append(support.values["supported"])
            contradictions.append(contradiction)
            temporal_id = plan.temporal_ids.get((claim_index, item.evidence_id))
            if temporal_id is None:
                continue
            temporal_prediction = predictions[temporal_id]
            exact_value, used_exact = _temporal_value(
                plan,
                time_by_id.get(item.evidence_id),
                temporal_prediction,
            )
            exact_temporal_links += int(used_exact)
            fallback_temporal_links += int(not used_exact)
            temporal_support.append(entailment * exact_value)
            temporal_conflict.append(
                max(
                    contradiction,
                    entailment * (1.0 - exact_value),
                    temporal_prediction.class_probabilities["contradiction"],
                )
            )
        support_values.append(max(entailments, default=0.0))
        support_binary_values.append(max(binary_support, default=0.0))
        contradiction_values.append(max(contradictions, default=0.0))
        if temporal_support:
            temporal_values.append(max(temporal_support))
            temporal_contradictions.append(max(temporal_conflict))

    evidence_support = _weighted(support_values, claim_weights)
    evidence_support_binary = _weighted(support_binary_values, claim_weights)
    contradiction_rate = _weighted(contradiction_values, claim_weights)
    unsupported_rate = (
        _weighted(
            [float(value < support_threshold) for value in support_values],
            claim_weights,
        )
        if support_values
        else None
    )
    temporal_attribution = _weighted(temporal_values, claim_weights)
    temporal_contradiction = _weighted(temporal_contradictions, claim_weights)
    explicit_validity = plan.exact.scores.get("tcred_explicit_claim_query_validity")
    temporal_correctness = (
        min(answer_score, float(explicit_validity))
        if explicit_validity is not None
        else answer_score
    )
    grounded_temporal = (
        min(answer_score, temporal_attribution)
        if temporal_attribution is not None
        else None
    )

    relevance = [
        predictions[input_id].values["relevance"]
        for input_id in plan.relevance_ids.values()
    ]
    retrieval_relevance = _discounted_mean(relevance)
    answerability_raw = [
        predictions[input_id].values["answerable"]
        for _evidence_id, input_id in plan.answerability_ids
    ]
    answerability_model = max(answerability_raw, default=0.0)
    evidence_sufficiency = answerability_model if plan.evidence else 0.0
    response_decision = (
        1.0 - evidence_sufficiency
        if is_refusal(plan.row.candidate_answer)
        else evidence_sufficiency
    )

    citation = _citation_scores(plan, predictions, time_by_id, support_threshold)
    graph = _graph_scores(plan, predictions)
    scores: dict[str, float | None] = {
        "tcred_sl_answer_equivalence_semantic": answer_score,
        "tcred_sl_answer_equivalence_auxiliary": answer_auxiliary,
        "tcred_sl_evidence_support": evidence_support,
        "tcred_sl_evidence_support_binary": evidence_support_binary,
        "tcred_sl_contradiction_rate": contradiction_rate,
        "tcred_sl_unsupported_claim_rate": unsupported_rate,
        "tcred_sl_retrieval_relevance": retrieval_relevance,
        "tcred_sl_temporal_correctness": temporal_correctness,
        "tcred_sl_temporal_attribution": temporal_attribution,
        "tcred_sl_grounded_temporal_correctness": grounded_temporal,
        "tcred_sl_temporal_contradiction": temporal_contradiction,
        "tcred_sl_evidence_sufficiency": evidence_sufficiency,
        "tcred_sl_answerability_model_raw": answerability_model,
        "tcred_sl_response_decision": response_decision,
        **citation,
        **graph,
    }
    _validate_scores(scores, metric_id=plan.row.metric_id)
    relevant_predictions = [
        predictions[input_id]
        for input_id in [
            *(identifier for _evidence_id, identifier in plan.answer_ids),
            *plan.support_ids.values(),
            *plan.temporal_ids.values(),
            *plan.relevance_ids.values(),
            *(identifier for _evidence_id, identifier in plan.answerability_ids),
            *plan.graph_ids.values(),
            *plan.citation_ids.values(),
        ]
    ]
    return MetricScoreRecord(
        metric_id=plan.row.metric_id,
        population=plan.row.population,
        dataset_family=plan.row.dataset_family,
        source_kind=plan.row.source_kind,
        system_name=plan.row.system_name,
        unit_id=plan.row.unit_id,
        qid=plan.row.qid,
        scenario_id=plan.row.scenario_id,
        gold_labels=plan.row.gold_labels,
        gold_provenance=plan.row.gold_provenance,
        scores=scores,
        metric_metadata={
            "tcred_sl": {
                "suite_version": "tcred-sl-hybrid-v1",
                "model_weight_sha256": engine.runtime["model_weight_sha256"],
                "calibration_sha256": engine.runtime["calibration_sha256"],
                "semantic_input_count": len(relevant_predictions),
                "truncated_input_count": sum(item.was_truncated for item in relevant_predictions),
                "candidate_claim_count": len(plan.claims),
                "visible_evidence_count": len(plan.evidence),
                "graph_path_count": len(plan.row.graph_paths),
                "exact_temporal_links": exact_temporal_links,
                "learned_temporal_fallback_links": fallback_temporal_links,
                "deterministic_precedence": (
                    "Exact interval, path continuity/direction/time/provenance, citation "
                    "resolution, and explicit missingness values cannot be overridden by the model."
                ),
                "global_scalar": "not_defined",
            }
        },
    )


def _answer_score(
    plan: _CasePlan,
    predictions: dict[str, SemanticPrediction],
) -> tuple[float, float]:
    values = [predictions[input_id] for _evidence_id, input_id in plan.answer_ids]
    if len(values) == 1 and plan.answer_ids[0][0] is None:
        return values[0].values["score"], values[0].values["equivalence"]
    weights = [
        predictions[plan.relevance_ids[str(evidence_id)]].values["relevance"]
        for evidence_id, _input_id in plan.answer_ids
    ]
    normalized = _normalize_weights(weights)
    return (
        sum(
            weight * prediction.values["score"]
            for weight, prediction in zip(normalized, values, strict=True)
        ),
        sum(
            weight * prediction.values["equivalence"]
            for weight, prediction in zip(normalized, values, strict=True)
        ),
    )


def _citation_scores(
    plan: _CasePlan,
    predictions: dict[str, SemanticPrediction],
    time_by_id: dict[str, Any],
    support_threshold: float,
) -> dict[str, float | None]:
    emitted_ids = list(dict.fromkeys(plan.row.cited_evidence_ids))
    if not emitted_ids:
        return {
            "tcred_sl_citation_precision": None,
            "tcred_sl_citation_recall": None,
            "tcred_sl_citation_context": None,
            "tcred_sl_citation_quality": None,
            "tcred_sl_unresolved_citation_rate": None,
        }
    cited = {item.evidence_id: item for item in plan.row.cited_evidence}
    emitted_quality: list[float] = []
    emitted_context: list[float] = []
    claim_support: list[float] = []
    for claim_index, _claim in enumerate(plan.claims):
        values: list[float] = []
        for evidence_id in emitted_ids:
            if evidence_id not in cited:
                continue
            support_id = plan.support_ids.get((claim_index, evidence_id))
            if support_id is None:
                continue
            support = _support_entailment(predictions[support_id])
            temporal_id = plan.temporal_ids.get((claim_index, evidence_id))
            temporal = 1.0
            if temporal_id is not None:
                temporal, _used_exact = _temporal_value(
                    plan,
                    time_by_id.get(evidence_id),
                    predictions[temporal_id],
                )
            values.append(support * temporal)
        claim_support.append(max(values, default=0.0))
    for evidence_id in emitted_ids:
        if evidence_id not in cited:
            emitted_quality.append(0.0)
            emitted_context.append(0.0)
            continue
        per_claim = []
        per_context = []
        for claim_index, _claim in enumerate(plan.claims):
            support_id = plan.support_ids.get((claim_index, evidence_id))
            citation_id = plan.citation_ids.get((claim_index, evidence_id))
            if support_id is None or citation_id is None:
                continue
            support = _support_entailment(predictions[support_id])
            temporal_id = plan.temporal_ids.get((claim_index, evidence_id))
            temporal = 1.0
            if temporal_id is not None:
                temporal, _used_exact = _temporal_value(
                    plan,
                    time_by_id.get(evidence_id),
                    predictions[temporal_id],
                )
            per_claim.append(support * temporal)
            per_context.append(predictions[citation_id].class_probabilities["appropriate"])
        emitted_quality.append(max(per_claim, default=0.0))
        emitted_context.append(max(per_context, default=0.0))
    precision = mean(emitted_quality)
    recall = (
        _weighted(
            [float(value >= support_threshold) for value in claim_support],
            _claim_weights(plan.claims),
        )
        if plan.claims
        else None
    )
    return {
        "tcred_sl_citation_precision": precision,
        "tcred_sl_citation_recall": recall,
        "tcred_sl_citation_context": mean(emitted_context),
        "tcred_sl_citation_quality": _harmonic(precision, recall),
        "tcred_sl_unresolved_citation_rate": (
            sum(evidence_id not in cited for evidence_id in emitted_ids) / len(emitted_ids)
        ),
    }


def _graph_scores(
    plan: _CasePlan,
    predictions: dict[str, SemanticPrediction],
) -> dict[str, float | None]:
    if not plan.row.graph_paths or not plan.claims:
        return {
            "tcred_sl_graph_semantic_support": None,
            "tcred_sl_graph_sufficiency": None,
        }
    exact_paths = {path.path_id: path for path in plan.exact.path_assessments}
    semantic_values: list[float] = []
    claim_coverage: list[float] = []
    for claim_index, claim in enumerate(plan.claims):
        coherences: list[float] = []
        endpoint_matched: list[float] = []
        for path in plan.row.graph_paths:
            prediction = predictions.get(plan.graph_ids.get((claim_index, path.path_id), ""))
            if prediction is None:
                continue
            semantic = prediction.values["supported"]
            semantic_values.append(semantic)
            exact = exact_paths[path.path_id]
            if not exact.structural_valid:
                coherence = 0.0
            elif exact.path_time is None:
                continue
            else:
                coherence = semantic * exact.path_time * exact.path_provenance
            coherences.append(coherence)
            labels = exact.answer_endpoint_labels or [exact.answer_endpoint_label]
            if any(_mentions(claim, label) for label in labels if label):
                endpoint_matched.append(coherence)
        claim_coverage.append(max(endpoint_matched, default=max(coherences, default=0.0)))
    return {
        "tcred_sl_graph_semantic_support": max(semantic_values, default=0.0),
        "tcred_sl_graph_sufficiency": _weighted(claim_coverage, _claim_weights(plan.claims)),
    }


def _semantic_input(
    plan: _CasePlan,
    input_id: str,
    task: SemanticTask,
    *,
    candidate_or_claim: str | None = None,
    evidence: list[JudgeEvidence] | None = None,
    citations: list[str] | None = None,
    paths: list[GraphPathText] | None = None,
) -> SemanticInferenceInput:
    evidence = evidence or []
    return SemanticInferenceInput(
        input_id=input_id,
        task=task,
        question=plan.row.question,
        query_time_or_interval=plan.query_text,
        temporal_operator=(
            plan.exact.query.operator if plan.exact.query.operator != "not_temporal" else None
        ),
        reference_answers=[plan.row.reference_answer],
        candidate_or_claim=candidate_or_claim or plan.row.candidate_answer,
        evidence_passages=[
            EvidencePassage(
                evidence_id=item.evidence_id,
                text=item.text,
                rank=index,
            )
            for index, item in enumerate(evidence, start=1)
        ],
        citations=citations or [],
        graph_paths=paths or [],
    )


def _displayed_evidence(row: TaskJudgeInput) -> list[JudgeEvidence]:
    output: list[JudgeEvidence] = []
    seen: set[str] = set()
    for item in [*row.retrieved_evidence, *row.cited_evidence]:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        output.append(item)
    return output


def _temporal_value(
    plan: _CasePlan,
    exact: Any | None,
    learned: SemanticPrediction,
) -> tuple[float, bool]:
    if not _temporal_applicable(plan.exact):
        return 1.0, True
    if exact is not None:
        if exact.status in {"stale", "future_invalid"}:
            return 0.0, True
        if exact.exact_validity is not None:
            return float(exact.compatibility or 0.0), True
        if exact.temporal_source != "none" and exact.compatibility is not None:
            return float(exact.compatibility), True
    return float(learned.class_probabilities["support"]), False


def _temporal_applicable(exact: TCredMetricResult) -> bool:
    return exact.query.basis != "not_temporal" and exact.query.status in {"exact", "partial"}


def _query_text(exact: TCredMetricResult) -> str | None:
    query = exact.query
    if query.query_start is None and query.query_end is None:
        return None
    start = query.query_start.isoformat() if query.query_start else "unknown"
    end = query.query_end.isoformat() if query.query_end else start
    return start if start == end else f"{start} to {end}"


def _serialize_graph_path(path: JudgeGraphPath) -> str:
    if not path.edges:
        return f"{path.path_id}: empty path"
    parts = []
    for edge in path.edges:
        relation = edge.relation_label or edge.relation or "related to"
        interval = _interval_text(edge.valid_time.start, edge.valid_time.end)
        direction = "reverse traversal" if edge.traversal_direction == "reverse" else "forward"
        parts.append(
            f"{edge.source.label} --[{relation}; {direction}; {interval}]--> {edge.target.label}"
        )
    return " | ".join(parts)


def _interval_text(start: str | None, end: str | None) -> str:
    if start and end:
        return f"valid {start} to {end}"
    if start:
        return f"valid since {start}"
    if end:
        return f"valid until {end}"
    return "valid time unknown"


def _support_entailment(prediction: SemanticPrediction) -> float:
    return float(prediction.class_probabilities["entailment"])


def _claim_weights(claims: list[str]) -> list[float]:
    raw = [float(min(32, max(1, len(_WORD.findall(claim))))) for claim in claims]
    return _normalize_weights(raw)


def _normalize_weights(values: list[float]) -> list[float]:
    if not values:
        return []
    total = sum(max(0.0, value) for value in values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [max(0.0, value) / total for value in values]


def _weighted(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    if len(values) != len(weights):
        raise ValueError("Values and claim weights are misaligned")
    return sum(value * weight for value, weight in zip(values, weights, strict=True))


def _discounted_mean(values: list[float]) -> float | None:
    if not values:
        return None
    discounts = [1.0 / math.log2(rank + 1) for rank in range(1, len(values) + 1)]
    return sum(value * discount for value, discount in zip(values, discounts, strict=True)) / sum(
        discounts
    )


def _harmonic(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if left + right == 0:
        return 0.0
    return 2.0 * left * right / (left + right)


def _mentions(text: str, phrase: str) -> bool:
    normalized_text = " ".join(_WORD.findall(text.casefold()))
    normalized_phrase = " ".join(_WORD.findall(phrase.casefold()))
    return bool(normalized_phrase and normalized_phrase in normalized_text)


def _validate_scores(scores: dict[str, float | None], *, metric_id: str) -> None:
    invalid = {
        name: value
        for name, value in scores.items()
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
    }
    if invalid:
        raise ValueError(f"T-CRED-SL scores outside [0,1] for {metric_id}: {invalid}")
