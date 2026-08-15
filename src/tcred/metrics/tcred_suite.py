from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from statistics import mean

from tcred.metrics.deterministic import reference_answer_scores
from tcred.metrics.task_judge_models import (
    JudgeEvidence,
    JudgeGraphPath,
    JudgePathNode,
    TaskJudgeInput,
    VisibleInterval,
)
from tcred.metrics.tcred_claims import (
    is_refusal,
    split_top_level_answer_items,
)
from tcred.metrics.tcred_models import (
    ClaimAssessment,
    ClaimEvidenceAssessment,
    EvidenceTemporalAssessment,
    NormalizedTemporalQuery,
    PathAssessment,
    SemanticPairScore,
    TCredMetricResult,
    TCredSemanticRecord,
)
from tcred.metrics.tcred_semantic import (
    path_edge_evidence_id,
    semantic_claims,
    semantic_evidence,
    semantic_input_hash,
)
from tcred.metrics.tcred_temporal import (
    ClaimTimeConstraint,
    assess_evidence_times,
    claim_evidence_time_compatibility,
    claim_query_time_compatibility,
    normalize_temporal_query,
    parse_claim_time,
    temporal_status_rates,
)

_SUPPORT_THRESHOLD = 0.5
_CONTRADICTION_THRESHOLD = 0.5
_ORDINAL_OPERATORS = {
    "after",
    "before",
    "expired",
    "first",
    "latest",
    "last",
    "previous",
    "next",
}
_TOKEN = re.compile(r"[a-z0-9]+")
_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "whom",
    "whose",
}
_DISCOURSE_SUPPORT_TOKENS = {
    "according",
    "answer",
    "evidence",
    "recorded",
    "revision",
    "snapshot",
    "states",
}


def score_tcred_suite(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    *,
    baseline_scores: Mapping[str, float | None] | None = None,
    baseline_input_aligned: bool = False,
    provenance: Mapping[str, float] | None = None,
    mode: str = "automatic",
) -> TCredMetricResult:
    """Compute the preregistered T-CRED component vector for one response card."""

    if semantic.metric_id != input_row.metric_id:
        raise ValueError("Semantic record and metric input IDs differ")
    if semantic.input_sha256 != semantic_input_hash(input_row):
        raise ValueError(f"Stale semantic record for {input_row.metric_id}")
    candidate_claims, reference_claims = semantic_claims(input_row)
    if semantic.candidate_claims != candidate_claims:
        raise ValueError("Semantic record candidate claims do not match the evaluator input")
    if semantic.reference_claims != reference_claims:
        raise ValueError("Semantic record reference claims do not match the evaluator input")
    if semantic.class_mapping != {"entailment": 0, "neutral": 1, "contradiction": 2}:
        raise ValueError("Unexpected AlignScore class mapping")
    if mode not in {"automatic", "oracle"}:
        raise ValueError(f"Unsupported T-CRED mode: {mode}")
    if mode == "oracle":
        raise NotImplementedError(
            "Oracle mode requires a separately hashed oracle-claim and relevance input contract; "
            "automatic claims must not be relabeled as oracle results."
        )

    query = normalize_temporal_query(input_row.question)
    evidence = _unique_textual_evidence(input_row)
    pair_index = _pair_index(semantic.pairs)
    _validate_semantic_pair_matrix(input_row, semantic, pair_index)
    relevance = _reference_relevance_by_evidence(
        semantic,
        evidence,
        question=input_row.question,
        pair_index=pair_index,
    )
    evidence_time = assess_evidence_times(
        query,
        evidence,
        semantic_relevance=relevance,
    )
    time_by_id = {row.evidence_id: row for row in evidence_time}
    provenance_by_id = {
        row.evidence_id: _provenance_value(row.evidence_id, provenance) for row in evidence
    }

    claim_assessments = _claim_assessments(
        semantic,
        evidence,
        question=input_row.question,
        query=query,
        pair_index=pair_index,
        time_by_id=time_by_id,
        provenance_by_id=provenance_by_id,
    )
    semantic_attribution = _mean_or_none([row.semantic_attribution for row in claim_assessments])
    temporal_attribution = _mean_or_none([row.temporal_attribution for row in claim_assessments])
    conflict_exposure = _mean_or_none([row.conflict_exposure for row in claim_assessments])
    global_conflict_sensitive_attribution = _mean_or_none(
        [row.global_conflict_sensitive_attribution for row in claim_assessments]
    )
    explicit_claim_time_validity = _explicit_claim_time_validity(claim_assessments)
    explicit_claim_query_validity = _explicit_claim_query_validity(claim_assessments)
    no_time_attribution = _claim_ablation_mean(claim_assessments, kind="no_time")
    no_contradiction_attribution = _claim_ablation_mean(
        claim_assessments,
        kind="no_contradiction",
    )
    cross_evidence_attribution = _claim_ablation_mean(
        claim_assessments,
        kind="cross_evidence",
    )

    answer_equivalence, answer_equivalence_source = _answer_equivalence(
        input_row,
        baseline_scores,
        baseline_input_aligned=baseline_input_aligned,
    )
    temporal_correctness = _answer_temporal_correctness(
        answer_equivalence,
        explicit_claim_query_validity,
        candidate_claims=semantic.candidate_claims,
        reference_claims=semantic.reference_claims,
    )
    grounded_temporal_correctness = _grounded_temporal_correctness(
        answer_equivalence,
        temporal_attribution,
        candidate_claims=semantic.candidate_claims,
        reference_claims=semantic.reference_claims,
    )
    citation = _citation_scores(
        input_row,
        semantic,
        query=query,
        pair_index=pair_index,
        time_by_id=time_by_id,
        provenance_by_id=provenance_by_id,
    )
    retrieval = _retrieval_scores(
        input_row,
        semantic,
        pair_index=pair_index,
        time_by_id=time_by_id,
        provenance_by_id=provenance_by_id,
        k=10,
    )
    path_assessments = _path_assessments(
        input_row,
        semantic,
        query=query,
        pair_index=pair_index,
        time_by_id=time_by_id,
        provenance=provenance,
    )
    graph = _graph_scores(
        path_assessments,
        candidate_answer=input_row.candidate_answer,
        candidate_claims=semantic.candidate_claims,
    )
    response_decision, valid_reference_support, reference_conflict_exposure = _response_decision(
        input_row,
        semantic,
        query=query,
        pair_index=pair_index,
        time_by_id=time_by_id,
        provenance_by_id=provenance_by_id,
    )

    scores: dict[str, float | None] = {
        "tcred_answer_equivalence": answer_equivalence,
        "tcred_semantic_attribution": semantic_attribution,
        "tcred_temporal_attribution": temporal_attribution,
        "tcred_conflict_exposure": conflict_exposure,
        "tcred_temporal_attribution_global_conflict_sensitive": (
            global_conflict_sensitive_attribution
        ),
        "tcred_temporal_correctness": temporal_correctness,
        "tcred_grounded_temporal_correctness": grounded_temporal_correctness,
        "tcred_explicit_claim_time_validity": explicit_claim_time_validity,
        "tcred_explicit_claim_query_validity": explicit_claim_query_validity,
        **citation,
        **retrieval,
        **graph,
        "tcred_response_decision": response_decision,
        "tcred_valid_reference_support": valid_reference_support,
        "tcred_reference_conflict_exposure": reference_conflict_exposure,
        "tcred_ablation_temporal_attribution_no_time": no_time_attribution,
        "tcred_ablation_temporal_attribution_no_contradiction": (no_contradiction_attribution),
        "tcred_ablation_temporal_attribution_cross_evidence": cross_evidence_attribution,
    }
    aggregate_components = [
        answer_equivalence,
        temporal_attribution,
        citation["tcred_citation_quality"],
        graph["tcred_graph_answer_coverage"],
    ]
    scores["tcred_aggregate"] = (
        _geometric_all([float(value) for value in aggregate_components])
        if all(value is not None for value in aggregate_components)
        else None
    )
    scores["tcred_aggregate_partial_diagnostic"] = _geometric_available(aggregate_components)
    coverage = {
        "query_exact": query.status == "exact",
        "answer_equivalence": answer_equivalence is not None,
        "semantic_attribution": semantic_attribution is not None,
        "temporal_attribution": temporal_attribution is not None,
        "explicit_claim_time": explicit_claim_time_validity is not None,
        "citation": citation["tcred_citation_f1"] is not None,
        "retrieval": retrieval["tcred_t_ndcg_at_10"] is not None,
        "graph": graph["tcred_graph_answer_coverage"] is not None,
        "response_decision": response_decision is not None,
        "declared_provenance": provenance is not None,
    }
    return TCredMetricResult(
        metric_id=input_row.metric_id,
        mode=mode,
        query=query,
        candidate_claims=semantic.candidate_claims,
        reference_claims=semantic.reference_claims,
        evidence_time=evidence_time,
        claim_assessments=claim_assessments,
        path_assessments=path_assessments,
        scores=scores,
        coverage=coverage,
        audit={
            "semantic_model": semantic.model,
            "semantic_class_mapping": semantic.class_mapping,
            "semantic_support_policy": (
                "pairwise NLI plus exact-phrase and question-conditioned decontextualized "
                "token guards; guards require answer-specific and relation-context overlap"
            ),
            "answer_equivalence_source": answer_equivalence_source,
            "baseline_input_aligned": baseline_input_aligned,
            "aggregate_available_components": sum(
                value is not None for value in aggregate_components
            ),
            "aggregate_total_components": len(aggregate_components),
            "provenance_policy": (
                "declared_complete_map_required"
                if provenance is not None
                else "neutral_1_unavailable"
            ),
            "near_miss_policy": (
                "reported in evidence_time but excluded from headline link support when stale "
                "or future"
            ),
            "ordinal_relevance_policy": (
                "reference claims only; candidate claims cannot select the winning interval"
            ),
            "retrieval_status_relevance_threshold": _SUPPORT_THRESHOLD,
            "contradiction_activation_threshold": _CONTRADICTION_THRESHOLD,
            "response_conflict_policy": (
                "query-valid NLI contradiction with a question-conditioned claim anchor"
            ),
            "graph_answer_target_policy": (
                "candidate claims plus resolvable top-level answer items; unresolved two-item "
                "comma expressions return missing graph coverage"
            ),
            "graph_path_support_policy": (
                "coherence uses the minimal displayed segment connecting a question anchor to "
                "each candidate answer endpoint; ignored-edge invalidity is reported separately"
            ),
            "candidate_claim_count": len(semantic.candidate_claims),
            "exact_claim_time_count": sum(
                row.claim_time_status == "exact" for row in claim_assessments
            ),
            "ambiguous_claim_time_count": sum(
                row.claim_time_status == "ambiguous" for row in claim_assessments
            ),
            "aggregate_policy": (
                "complete-case only; partial diagnostic is separately named and non-comparable"
            ),
        },
    )


def _unique_textual_evidence(input_row: TaskJudgeInput) -> list[JudgeEvidence]:
    output: list[JudgeEvidence] = []
    seen: dict[str, JudgeEvidence] = {}
    for row in [*input_row.retrieved_evidence, *input_row.cited_evidence]:
        previous = seen.get(row.evidence_id)
        if previous is not None:
            if previous != row:
                raise ValueError(
                    "Conflicting payloads reuse evidence ID "
                    f"{row.evidence_id!r} in retrieved/cited evidence"
                )
            continue
        seen[row.evidence_id] = row
        output.append(row)
    return output


def _pair_index(
    pairs: Sequence[SemanticPairScore],
) -> dict[tuple[str, int, str, str], SemanticPairScore]:
    output = {}
    for row in pairs:
        key = (row.claim_source, row.claim_index, row.evidence_kind, row.evidence_id)
        if key in output:
            raise ValueError(f"Duplicate semantic link: {key}")
        output[key] = row
    return output


def _validate_semantic_pair_matrix(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
) -> None:
    evidence = semantic_evidence(input_row)
    claims_by_source = {
        "candidate": semantic.candidate_claims,
        "reference": semantic.reference_claims,
    }
    expected = {
        (source, claim_index, row.kind, row.evidence_id)
        for source, claims in claims_by_source.items()
        for claim_index in range(len(claims))
        for row in evidence
    }
    actual = set(pair_index)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(f"Incomplete semantic pair matrix: missing={missing}, extra={extra}")
    evidence_hashes = {
        (row.kind, row.evidence_id): hashlib.sha256(row.text.encode()).hexdigest()
        for row in evidence
    }
    for key, pair in pair_index.items():
        source, claim_index, kind, evidence_id = key
        expected_claim = claims_by_source[source][claim_index]
        if pair.claim != expected_claim:
            raise ValueError(f"Semantic pair claim mismatch for {key}")
        if pair.evidence_text_sha256 != evidence_hashes[(kind, evidence_id)]:
            raise ValueError(f"Semantic pair evidence hash mismatch for {key}")


def _reference_relevance_by_evidence(
    semantic: TCredSemanticRecord,
    evidence: Sequence[JudgeEvidence],
    *,
    question: str,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
) -> dict[str, float]:
    return {
        row.evidence_id: max(
            (
                _semantic_support_with_exact_mention(
                    claim,
                    row.text,
                    _textual_pair(pair_index, "reference", claim_index, row.evidence_id),
                    context_text=question,
                )
                for claim_index, claim in enumerate(semantic.reference_claims)
            ),
            default=0.0,
        )
        for row in evidence
    }


def _claim_assessments(
    semantic: TCredSemanticRecord,
    evidence: Sequence[JudgeEvidence],
    *,
    question: str,
    query: NormalizedTemporalQuery,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
) -> list[ClaimAssessment]:
    output = []
    for claim_index, claim in enumerate(semantic.candidate_claims):
        claim_time = parse_claim_time(claim)
        claim_query_compatibility = claim_query_time_compatibility(claim_time, query)
        links = []
        for row in evidence:
            pair = _textual_pair(pair_index, "candidate", claim_index, row.evidence_id)
            if pair is None:
                continue
            entailment = _semantic_support_with_exact_mention(
                claim,
                row.text,
                pair,
                context_text=question,
            )
            temporal = time_by_id[row.evidence_id]
            temporal_value = _headline_temporal_value(temporal)
            claim_time_value = claim_evidence_time_compatibility(
                claim_time,
                row.valid_time,
            )
            if claim_time.status == "absent":
                effective_claim_time = 1.0
            elif claim_time_value is None:
                effective_claim_time = 0.0
            else:
                effective_claim_time = claim_time_value
            effective_claim_query = _claim_query_value(
                claim_time,
                claim_query_compatibility,
            )
            provenance = provenance_by_id[row.evidence_id]
            support = (
                entailment
                * temporal_value
                * effective_claim_time
                * effective_claim_query
                * provenance
            )
            conflict = (
                pair.contradiction * temporal_value * effective_claim_time * effective_claim_query
                if pair.contradiction >= _CONTRADICTION_THRESHOLD
                else 0.0
            )
            links.append(
                ClaimEvidenceAssessment(
                    claim_index=claim_index,
                    claim=claim,
                    evidence_id=row.evidence_id,
                    entailment=entailment,
                    contradiction=pair.contradiction,
                    temporal_compatibility=temporal.compatibility,
                    exact_temporal_validity=temporal.exact_validity,
                    claim_temporal_compatibility=claim_time_value,
                    provenance=provenance,
                    link_support=support,
                    link_conflict=conflict,
                )
            )
        semantic_score = max((link.entailment for link in links), default=0.0)
        if not links or all(link.temporal_compatibility is None for link in links):
            temporal_score = None
            conflict_exposure = None
            global_conflict_sensitive = None
        else:
            temporal_score = max(
                (_clamp(link.link_support - link.link_conflict) for link in links),
                default=0.0,
            )
            conflict_exposure = max(
                (link.link_conflict for link in links),
                default=0.0,
            )
            global_conflict_sensitive = _clamp(
                max((link.link_support for link in links), default=0.0) - conflict_exposure
            )
        best_support = max(links, key=lambda link: link.link_support, default=None)
        best_conflict = max(links, key=lambda link: link.link_conflict, default=None)
        output.append(
            ClaimAssessment(
                claim_index=claim_index,
                claim=claim,
                claim_time_status=claim_time.status,
                claim_time_mode=claim_time.mode,
                claim_time_start=claim_time.start,
                claim_time_end=claim_time.end,
                claim_query_compatibility=claim_query_compatibility,
                semantic_attribution=semantic_score,
                temporal_attribution=temporal_score,
                conflict_exposure=conflict_exposure,
                global_conflict_sensitive_attribution=global_conflict_sensitive,
                best_support_evidence_id=(best_support.evidence_id if best_support else None),
                best_conflict_evidence_id=(best_conflict.evidence_id if best_conflict else None),
                links=links,
            )
        )
    return output


def _explicit_claim_time_validity(
    claims: Sequence[ClaimAssessment],
) -> float | None:
    explicit = [claim for claim in claims if claim.claim_time_status == "exact"]
    if not explicit:
        return None
    values = []
    for claim in explicit:
        query_value = (
            0.0 if claim.claim_query_compatibility is None else (claim.claim_query_compatibility)
        )
        compatible = [
            link.claim_temporal_compatibility * query_value
            for link in claim.links
            if (
                link.claim_temporal_compatibility is not None
                and link.entailment >= _SUPPORT_THRESHOLD
            )
        ]
        values.append(max(compatible, default=0.0))
    return mean(values) if values else None


def _explicit_claim_query_validity(
    claims: Sequence[ClaimAssessment],
) -> float | None:
    values = [
        claim.claim_query_compatibility
        for claim in claims
        if (claim.claim_time_status == "exact" and claim.claim_query_compatibility is not None)
    ]
    return mean(values) if values else None


def _claim_ablation_mean(
    claims: Sequence[ClaimAssessment],
    *,
    kind: str,
) -> float | None:
    values = []
    for claim in claims:
        if not claim.links:
            continue
        if kind == "no_time":
            values.append(
                max(
                    _clamp(
                        link.entailment * link.provenance
                        - (
                            link.contradiction
                            if link.contradiction >= _CONTRADICTION_THRESHOLD
                            else 0.0
                        )
                    )
                    for link in claim.links
                )
            )
        elif kind == "no_contradiction":
            if all(link.temporal_compatibility is None for link in claim.links):
                continue
            values.append(max(link.link_support for link in claim.links))
        elif kind == "cross_evidence":
            known = [
                _headline_temporal_from_link(link)
                * (
                    1.0
                    if claim.claim_time_status == "absent"
                    else (link.claim_temporal_compatibility or 0.0)
                )
                * (
                    1.0
                    if claim.claim_time_status == "absent"
                    else (claim.claim_query_compatibility or 0.0)
                )
                for link in claim.links
                if link.temporal_compatibility is not None
            ]
            if not known:
                continue
            semantic_support = max(link.entailment * link.provenance for link in claim.links)
            semantic_conflict = max(
                (
                    link.contradiction if link.contradiction >= _CONTRADICTION_THRESHOLD else 0.0
                    for link in claim.links
                ),
                default=0.0,
            )
            temporal = max(known)
            values.append(_clamp(semantic_support * temporal - semantic_conflict * temporal))
        else:
            raise ValueError(f"Unknown claim ablation: {kind}")
    return _mean_or_none(values)


def _answer_equivalence(
    input_row: TaskJudgeInput,
    baseline_scores: Mapping[str, float | None] | None,
    *,
    baseline_input_aligned: bool,
) -> tuple[float, str]:
    if (
        baseline_input_aligned
        and baseline_scores is not None
        and baseline_scores.get("pedants_probability") is not None
    ):
        value = float(baseline_scores["pedants_probability"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"PEDANTS probability outside [0,1]: {value}")
        return value, "pedants_probability"
    fallback = reference_answer_scores(input_row.candidate_answer, input_row.reference_answer)
    source = (
        "normalized_token_f1_fallback_no_pedants"
        if baseline_scores is None or baseline_scores.get("pedants_probability") is None
        else "normalized_token_f1_fallback_unverified_baseline"
    )
    return fallback["token_f1"], source


def _answer_temporal_correctness(
    answer_equivalence: float,
    explicit_claim_query_validity: float | None,
    *,
    candidate_claims: Sequence[str],
    reference_claims: Sequence[str],
) -> float | None:
    if not candidate_claims and not reference_claims:
        return None
    if explicit_claim_query_validity is None:
        return answer_equivalence
    return min(answer_equivalence, explicit_claim_query_validity)


def _grounded_temporal_correctness(
    answer_equivalence: float,
    temporal_attribution: float | None,
    *,
    candidate_claims: Sequence[str],
    reference_claims: Sequence[str],
) -> float | None:
    if not candidate_claims and not reference_claims:
        return None
    if not candidate_claims or not reference_claims:
        return min(answer_equivalence, temporal_attribution or 0.0)
    if temporal_attribution is None:
        return None
    return min(answer_equivalence, temporal_attribution)


def _citation_scores(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    *,
    query: NormalizedTemporalQuery,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
) -> dict[str, float | None]:
    cited = input_row.cited_evidence
    claims = semantic.candidate_claims
    claim_times = [parse_claim_time(claim) for claim in claims]
    claim_query_values = [
        claim_query_time_compatibility(claim_time, query) for claim_time in claim_times
    ]
    cited_support_by_claim: list[float] = []
    cited_support_by_evidence: list[float] = []
    cited_semantic_by_evidence: list[float] = []
    cited_time_by_evidence: list[float | None] = []
    for claim_index, _claim in enumerate(claims):
        claim_time = claim_times[claim_index]
        values = [
            _link_value(
                pair_index,
                claim_source="candidate",
                claim_index=claim_index,
                evidence_kind="cited",
                evidence_id=row.evidence_id,
                time_by_id=time_by_id,
                provenance_by_id=provenance_by_id,
                use_time=True,
                claim_time=claim_time,
                claim_query_compatibility=claim_query_values[claim_index],
                evidence_interval=row.valid_time,
                evidence_text=row.text,
                claim=claims[claim_index],
                context_text=input_row.question,
            )
            for row in cited
        ]
        cited_support_by_claim.append(max(values, default=0.0))
    for row in cited:
        links = [
            pair_index.get(("candidate", claim_index, "cited", row.evidence_id))
            for claim_index in range(len(claims))
        ]
        pairs = [pair for pair in links if pair is not None]
        semantic_values = [
            _semantic_support_with_exact_mention(
                claims[pair.claim_index],
                row.text,
                pair,
                context_text=input_row.question,
            )
            for pair in pairs
        ]
        temporal = time_by_id.get(row.evidence_id)
        temporal_value = _headline_temporal_value(temporal) if temporal else 0.0
        cited_support_by_evidence.append(
            max(
                (
                    semantic_values[pair_index_in_evidence]
                    * temporal_value
                    * _claim_time_value(
                        claim_times[pair.claim_index],
                        row.valid_time,
                    )
                    * _claim_query_value(
                        claim_times[pair.claim_index],
                        claim_query_values[pair.claim_index],
                    )
                    * provenance_by_id[row.evidence_id]
                    for pair_index_in_evidence, pair in enumerate(pairs)
                ),
                default=0.0,
            )
        )
        cited_semantic_by_evidence.append(max(semantic_values, default=0.0))
        cited_time_by_evidence.append(
            max(
                (
                    (temporal.exact_validity or 0.0)
                    * _claim_time_value(
                        claim_times[pair.claim_index],
                        row.valid_time,
                    )
                    * _claim_query_value(
                        claim_times[pair.claim_index],
                        claim_query_values[pair.claim_index],
                    )
                    for pair_index_in_evidence, pair in enumerate(pairs)
                    if semantic_values[pair_index_in_evidence] >= _SUPPORT_THRESHOLD
                ),
                default=0.0,
            )
        )

    precision = _mean_or_none(cited_support_by_evidence)
    retrieved_support_by_claim = [
        _max_claim_support(
            pair_index,
            claim_index=claim_index,
            rows=input_row.retrieved_evidence,
            kind="retrieved",
            time_by_id=time_by_id,
            provenance_by_id=provenance_by_id,
            use_time=True,
            claim_time=claim_times[claim_index],
            claim_query_compatibility=claim_query_values[claim_index],
            claim=claims[claim_index],
            context_text=input_row.question,
        )
        for claim_index in range(len(claims))
    ]
    supported_claims = [value >= _SUPPORT_THRESHOLD for value in retrieved_support_by_claim]
    if claims and (cited or input_row.retrieved_evidence):
        completeness = mean(cited_support_by_claim) if cited else 0.0
    else:
        completeness = None
    f1 = _harmonic_optional(precision, completeness)
    relevant_citations = [
        index
        for index, value in enumerate(cited_semantic_by_evidence)
        if value >= _SUPPORT_THRESHOLD
    ]
    invalid_rate = None
    if relevant_citations:
        invalid_rate = mean(
            float(cited_time_by_evidence[index] != 1.0) for index in relevant_citations
        )
    required_count = sum(supported_claims)
    missing_rate = None
    if required_count:
        missing_rate = (
            sum(
                supported and cited_support_by_claim[index] < _SUPPORT_THRESHOLD
                for index, supported in enumerate(supported_claims)
            )
            / required_count
        )

    no_time_precision = _citation_no_time_precision(
        input_row,
        semantic,
        pair_index,
        provenance_by_id,
        question=input_row.question,
    )
    no_time_completeness = _citation_no_time_completeness(
        input_row,
        semantic,
        pair_index,
        provenance_by_id,
        question=input_row.question,
    )
    citation_quality = completeness if precision is None else f1
    return {
        "tcred_citation_precision": precision,
        "tcred_citation_completeness": completeness,
        "tcred_citation_f1": f1,
        "tcred_citation_quality": citation_quality,
        "tcred_citation_invalid_rate": invalid_rate,
        "tcred_missing_required_citation_rate": missing_rate,
        "tcred_ablation_citation_f1_no_time": _harmonic_optional(
            no_time_precision,
            no_time_completeness,
        ),
    }


def _citation_no_time_precision(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    provenance_by_id: Mapping[str, float],
    *,
    question: str,
) -> float | None:
    values = []
    for row in input_row.cited_evidence:
        values.append(
            max(
                (
                    _semantic_support_with_exact_mention(
                        semantic.candidate_claims[index],
                        row.text,
                        pair_index[("candidate", index, "cited", row.evidence_id)],
                        context_text=question,
                    )
                    * provenance_by_id[row.evidence_id]
                    for index in range(len(semantic.candidate_claims))
                    if ("candidate", index, "cited", row.evidence_id) in pair_index
                ),
                default=0.0,
            )
        )
    return _mean_or_none(values)


def _citation_no_time_completeness(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    provenance_by_id: Mapping[str, float],
    *,
    question: str,
) -> float | None:
    if not semantic.candidate_claims:
        return None
    values = []
    for claim_index in range(len(semantic.candidate_claims)):
        values.append(
            max(
                (
                    _semantic_support_with_exact_mention(
                        semantic.candidate_claims[claim_index],
                        row.text,
                        pair_index[("candidate", claim_index, "cited", row.evidence_id)],
                        context_text=question,
                    )
                    * provenance_by_id[row.evidence_id]
                    for row in input_row.cited_evidence
                    if ("candidate", claim_index, "cited", row.evidence_id) in pair_index
                ),
                default=0.0,
            )
        )
    return mean(values)


def _retrieval_scores(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    *,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
    k: int,
) -> dict[str, float | None]:
    if not semantic.reference_claims or not input_row.retrieved_evidence:
        empty = {
            f"tcred_t_ndcg_at_{k}": None,
            f"tcred_t_precision_at_{k}": None,
            f"tcred_temporal_cleanliness_at_{k}": None,
            f"tcred_ablation_retrieval_ndcg_no_time_at_{k}": None,
        }
        empty.update(temporal_status_rates([], k=k))
        return empty
    temporal_gains = []
    semantic_gains = []
    ordered_assessments = []
    semantic_relevances = []
    for row in input_row.retrieved_evidence:
        entailment = max(
            (
                _semantic_support_with_exact_mention(
                    claim,
                    row.text,
                    pair_index.get(("reference", index, "retrieved", row.evidence_id)),
                    context_text=input_row.question,
                )
                for index, claim in enumerate(semantic.reference_claims)
            ),
            default=0.0,
        )
        provenance = provenance_by_id[row.evidence_id]
        semantic_gain = entailment * provenance
        temporal = time_by_id[row.evidence_id]
        temporal_gain = semantic_gain * _headline_temporal_value(temporal)
        semantic_gains.append(semantic_gain)
        semantic_relevances.append(entailment)
        temporal_gains.append(temporal_gain)
        ordered_assessments.append(temporal)
    temporal_ndcg = _within_list_ndcg(temporal_gains, k=k)
    if temporal_ndcg is None and any(gain > 0.0 for gain in semantic_gains):
        temporal_ndcg = 0.0
    scores = {
        f"tcred_t_ndcg_at_{k}": temporal_ndcg,
        f"tcred_t_precision_at_{k}": sum(temporal_gains[:k]) / k,
        f"tcred_ablation_retrieval_ndcg_no_time_at_{k}": _within_list_ndcg(
            semantic_gains,
            k=k,
        ),
    }
    status_rates = temporal_status_rates(
        ordered_assessments,
        semantic_relevance=semantic_relevances,
        k=k,
    )
    scores.update(status_rates)
    invalid_relevant_rates = [
        status_rates[f"tcred_stale_rate_at_{k}"],
        status_rates[f"tcred_future_invalid_rate_at_{k}"],
        status_rates[f"tcred_unknown_time_rate_at_{k}"],
    ]
    scores[f"tcred_temporal_cleanliness_at_{k}"] = 1.0 - sum(
        float(value) for value in invalid_relevant_rates if value is not None
    )
    return scores


def _path_assessments(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    *,
    query,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance: Mapping[str, float] | None,
) -> list[PathAssessment]:
    path_ids = [path.path_id for path in input_row.graph_paths]
    if len(path_ids) != len(set(path_ids)):
        raise ValueError(f"Duplicate graph path IDs for {input_row.metric_id}")
    path_edge_times: dict[str, EvidenceTemporalAssessment] = {}
    temporal_target_by_path: dict[str, str] = {}
    tied_answer_count_by_edge: dict[str, int] = {}
    temporal_competitor_count = 0
    if query.basis == "world_valid_time":
        (
            path_edge_times,
            temporal_target_by_path,
            tied_answer_count_by_edge,
            temporal_competitor_count,
        ) = _world_path_temporal_assessments(input_row, query)
    return [
        _assess_path(
            input_row,
            path,
            semantic,
            query=query,
            pair_index=pair_index,
            time_by_id=time_by_id,
            path_edge_times=path_edge_times,
            temporal_target_edge_id=temporal_target_by_path.get(path.path_id),
            temporal_competitor_count=temporal_competitor_count,
            tied_answer_count_by_edge=tied_answer_count_by_edge,
            provenance=provenance,
        )
        for path in input_row.graph_paths
    ]


def _assess_path(
    input_row: TaskJudgeInput,
    path: JudgeGraphPath,
    semantic: TCredSemanticRecord,
    *,
    query,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    path_edge_times: Mapping[str, EvidenceTemporalAssessment],
    temporal_target_edge_id: str | None,
    temporal_competitor_count: int,
    tied_answer_count_by_edge: Mapping[str, int],
    provenance: Mapping[str, float] | None,
) -> PathAssessment:
    structural = _path_structurally_valid(path)
    endpoints = [path.nodes[0], path.nodes[-1]] if path.nodes else []
    answer_indices = _answer_endpoint_indices(endpoints, input_row)
    answer_index = answer_indices[0] if answer_indices else 0
    answer_endpoint_ids = {
        endpoints[index].id for index in answer_indices if index < len(endpoints)
    }
    answer_labels = [endpoints[index].label for index in answer_indices if index < len(endpoints)]
    answer_scores = [_mention_score(label, input_row.candidate_answer) for label in answer_labels]
    answer_anchor = max(answer_scores, default=0.0)
    question_candidates = [
        _mention_score(node.label, input_row.question)
        for node in path.nodes
        if node.id not in answer_endpoint_ids
    ]
    if not question_candidates:
        question_candidates = [
            _mention_score(node.label, input_row.question) for node in path.nodes
        ]
    question_anchor = max(question_candidates, default=0.0)
    relevant_edge_indices = _relevant_path_edge_indices(
        path,
        input_row,
        endpoint_answer_indices=answer_indices,
    )
    relation_relevance = _relation_relevance(
        path,
        input_row.question,
        edge_indices=relevant_edge_indices,
    )
    edge_semantics = []
    edge_times: list[float | None] = []
    edge_provenance = []
    ignored_edge_times: list[float | None] = []
    supporting_edge_ids: list[str] = []
    ignored_edge_ids: list[str] = []
    for index, edge in enumerate(path.edges):
        edge_id = path_edge_evidence_id(path.path_id, index, edge.fact_id)
        edge_semantic = max(
            (
                _semantic_support_with_exact_mention(
                    claim,
                    edge.evidence_text,
                    pair_index.get(("reference", claim_index, "path", edge_id)),
                    context_text=input_row.question,
                )
                for claim_index, claim in enumerate(semantic.reference_claims)
            ),
            default=0.0,
        )
        temporal = None
        if query.basis in {"snapshot_observation", "document_revision"}:
            temporal = time_by_id.get(edge.fact_id)
        elif query.basis == "world_valid_time":
            temporal = path_edge_times.get(edge_id)
        if temporal is None:
            temporal = assess_evidence_times(
                query,
                [
                    JudgeEvidence(
                        evidence_id=edge.fact_id or edge_id,
                        text=edge.evidence_text,
                        valid_time=edge.valid_time,
                    )
                ],
            )[0]
        edge_time = None if temporal.compatibility is None else _headline_temporal_value(temporal)
        if index in relevant_edge_indices:
            supporting_edge_ids.append(edge_id)
            edge_semantics.append(edge_semantic)
            edge_times.append(edge_time)
            edge_provenance.append(_provenance_value(edge.fact_id, provenance))
        else:
            ignored_edge_ids.append(edge_id)
            ignored_edge_times.append(edge_time)
    semantic_relevance = _mean_or_zero(edge_semantics)
    relation_components = [question_anchor, answer_anchor, semantic_relevance]
    if relation_relevance > 0:
        relation_components.append(relation_relevance)
    path_relevance = _geometric_all(relation_components)
    path_provenance = _geometric_all(edge_provenance) if edge_provenance else 0.0
    extraneous_invalid_edge_rate = (
        mean(float(value is None or value <= 0.0) for value in ignored_edge_times)
        if ignored_edge_times
        else None
    )
    if not structural:
        path_time = 0.0
        coherence = 0.0
        status = "structurally_invalid"
        explanation = "The displayed traversal does not form one continuous path."
    elif not edge_times:
        path_time = 0.0
        coherence = 0.0
        status = "empty_path"
        explanation = "The path contains no edge."
    elif any(value is None for value in edge_times):
        path_time = None
        coherence = None
        status = "unknown_path_time"
        explanation = "At least one edge lacks time licensed by the question's temporal basis."
    else:
        path_time = min(float(value) for value in edge_times if value is not None)
        coherence = path_relevance * path_time * path_provenance
        status = "coherent" if path_time > 0 else "temporally_invalid"
        explanation = (
            "Path continuity, endpoint anchoring, semantic relevance, and time were combined."
        )
    return PathAssessment(
        path_id=path.path_id,
        structural_valid=structural,
        answer_endpoint_label=(
            endpoints[answer_index].label if endpoints and answer_scores else ""
        ),
        answer_endpoint_labels=answer_labels,
        question_anchor=question_anchor,
        answer_anchor=answer_anchor,
        semantic_relevance=semantic_relevance,
        path_relevance=path_relevance,
        path_time=path_time,
        path_provenance=path_provenance,
        coherence=coherence,
        temporal_target_edge_id=temporal_target_edge_id,
        temporal_competitor_count=temporal_competitor_count,
        temporal_tied_answer_count=tied_answer_count_by_edge.get(
            temporal_target_edge_id or "",
            0,
        ),
        supporting_edge_ids=supporting_edge_ids,
        ignored_edge_ids=ignored_edge_ids,
        extraneous_invalid_edge_rate=extraneous_invalid_edge_rate,
        status=status,
        explanation=explanation,
    )


def _world_path_temporal_assessments(
    input_row: TaskJudgeInput,
    query,
) -> tuple[
    dict[str, EvidenceTemporalAssessment],
    dict[str, str],
    dict[str, int],
    int,
]:
    """Evaluate ordinal path edges jointly instead of declaring each singleton a winner."""

    edge_rows: dict[str, JudgeEvidence] = {}
    target_by_path: dict[str, str] = {}
    answer_label_by_target: dict[str, str] = {}
    non_target_by_path: dict[str, list[str]] = {}
    for path in input_row.graph_paths:
        if not path.edges or not _path_structurally_valid(path):
            continue
        target_index, answer_label = _answer_side_edge(path, input_row)
        non_target_by_path[path.path_id] = []
        for index, edge in enumerate(path.edges):
            edge_id = path_edge_evidence_id(path.path_id, index, edge.fact_id)
            edge_rows[edge_id] = JudgeEvidence(
                evidence_id=edge_id,
                text=edge.evidence_text,
                valid_time=edge.valid_time,
            )
            if index == target_index:
                target_by_path[path.path_id] = edge_id
                answer_label_by_target[edge_id] = answer_label
            else:
                non_target_by_path[path.path_id].append(edge_id)

    if query.operator not in _ORDINAL_OPERATORS:
        assessments = assess_evidence_times(query, list(edge_rows.values()))
        return (
            {row.evidence_id: row for row in assessments},
            target_by_path,
            {},
            0,
        )

    target_rows = [edge_rows[edge_id] for edge_id in target_by_path.values()]
    target_assessments = assess_evidence_times(query, target_rows)
    result = {row.evidence_id: row for row in target_assessments}
    selected_ids = {row.evidence_id for row in target_assessments if row.exact_validity == 1.0}
    selected_labels = {
        _normalize_text(answer_label_by_target[edge_id])
        for edge_id in selected_ids
        if _normalize_text(answer_label_by_target[edge_id])
    }
    tied_count = len(selected_labels)
    covered_labels = {
        label
        for edge_id, label in (
            (edge_id, _normalize_text(answer_label_by_target[edge_id])) for edge_id in selected_ids
        )
        if label and _mention_score(label, input_row.candidate_answer) == 1.0
    }
    tie_coverage = len(covered_labels) / tied_count if tied_count else 0.0
    tied_answer_count_by_edge: dict[str, int] = {}
    if tied_count > 1 and tie_coverage < 1.0:
        for edge_id in selected_ids:
            tied_answer_count_by_edge[edge_id] = tied_count
            result[edge_id] = result[edge_id].model_copy(
                update={
                    "compatibility": tie_coverage,
                    "exact_validity": tie_coverage,
                    "near_miss": 0.0,
                    "status": "temporally_ambiguous",
                    "explanation": (
                        "The ordinal rule produced multiple answer entities; temporal support "
                        "was discounted by the candidate's coverage of that tied answer set."
                    ),
                }
            )

    for path in input_row.graph_paths:
        target_id = target_by_path.get(path.path_id)
        if target_id is None:
            continue
        target = result[target_id]
        event_anchor = (
            target.effective_end
            if query.operator in {"before", "expired", "previous"}
            else target.effective_start
        )
        for edge_id in non_target_by_path.get(path.path_id, []):
            if event_anchor is None:
                result[edge_id] = EvidenceTemporalAssessment(
                    evidence_id=edge_id,
                    status="temporally_ambiguous",
                    temporal_source="valid_time",
                    explanation=(
                        "The ordinal target has no usable event date for checking this bridge edge."
                    ),
                )
                continue
            bridge_query = query.model_copy(
                update={
                    "operator": "as_of",
                    "query_start": event_anchor,
                    "query_end": event_anchor,
                    "evaluation_time": event_anchor,
                    "granularity": "day",
                    "interval_requires_coverage": False,
                    "explanation": (
                        "Bridge edges are checked at the selected ordinal edge's event time."
                    ),
                }
            )
            result[edge_id] = assess_evidence_times(
                bridge_query,
                [edge_rows[edge_id]],
            )[0]
    return (
        result,
        target_by_path,
        tied_answer_count_by_edge,
        len(target_rows),
    )


def _answer_side_edge(path: JudgeGraphPath, input_row: TaskJudgeInput) -> tuple[int, str]:
    """Return the edge adjacent to the answer-side endpoint and that endpoint's label."""

    endpoints = [path.nodes[0], path.nodes[-1]]
    answer_index = _answer_endpoint_index(endpoints, input_row)
    endpoint = endpoints[answer_index]
    return (0, endpoint.label) if answer_index == 0 else (len(path.edges) - 1, endpoint.label)


def _answer_endpoint_index(
    endpoints: Sequence[JudgePathNode],
    input_row: TaskJudgeInput,
) -> int:
    return _answer_endpoint_indices(endpoints, input_row)[0]


def _answer_endpoint_indices(
    endpoints: Sequence[JudgePathNode],
    input_row: TaskJudgeInput,
) -> list[int]:
    if len(endpoints) < 2:
        return [0] if endpoints else []
    answer_scores = [_mention_score(node.label, input_row.candidate_answer) for node in endpoints]
    question_scores = [_mention_score(node.label, input_row.question) for node in endpoints]
    specificity = [
        answer_score - question_score
        for answer_score, question_score in zip(answer_scores, question_scores, strict=True)
    ]
    best = max(specificity)
    tied = [index for index, score in enumerate(specificity) if score == best]
    if len(tied) == 1:
        return tied
    least_question_overlap = min(question_scores[index] for index in tied)
    return [index for index in tied if question_scores[index] == least_question_overlap]


def _graph_scores(
    paths: Sequence[PathAssessment],
    *,
    candidate_answer: str,
    candidate_claims: Sequence[str],
) -> dict[str, float | None]:
    if not paths:
        return {
            "tcred_best_path_coherence": None,
            "tcred_mean_top3_path_coherence": None,
            "tcred_graph_answer_coverage": None,
            "tcred_invalid_path_rate": None,
            "tcred_structural_invalid_path_rate": None,
            "tcred_temporally_invalid_path_rate": None,
            "tcred_unknown_path_time_rate": None,
            "tcred_extraneous_invalid_edge_rate": None,
            "tcred_ablation_graph_mean_top3_no_time": None,
            "tcred_ablation_graph_answer_coverage_no_time": None,
        }
    known = sorted(
        (row.coherence for row in paths if row.coherence is not None),
        reverse=True,
    )
    no_time = sorted(
        (_path_no_time_support(row) for row in paths),
        reverse=True,
    )
    answer_coverage = _graph_answer_coverage(
        paths,
        candidate_answer=candidate_answer,
        candidate_claims=candidate_claims,
        use_time=True,
    )
    answer_coverage_no_time = _graph_answer_coverage(
        paths,
        candidate_answer=candidate_answer,
        candidate_claims=candidate_claims,
        use_time=False,
    )
    return {
        "tcred_best_path_coherence": max(known) if known else None,
        "tcred_mean_top3_path_coherence": mean(known[:3]) if known else None,
        "tcred_graph_answer_coverage": answer_coverage,
        "tcred_invalid_path_rate": mean(
            float(not row.structural_valid or (row.path_time is not None and row.path_time <= 0.0))
            for row in paths
        ),
        "tcred_structural_invalid_path_rate": mean(
            float(not row.structural_valid) for row in paths
        ),
        "tcred_temporally_invalid_path_rate": mean(
            float(row.structural_valid and row.path_time is not None and row.path_time <= 0.0)
            for row in paths
        ),
        "tcred_unknown_path_time_rate": mean(float(row.path_time is None) for row in paths),
        "tcred_extraneous_invalid_edge_rate": _mean_or_none(
            [
                row.extraneous_invalid_edge_rate
                for row in paths
                if row.extraneous_invalid_edge_rate is not None
            ]
        ),
        "tcred_ablation_graph_mean_top3_no_time": mean(no_time[:3]) if no_time else None,
        "tcred_ablation_graph_answer_coverage_no_time": answer_coverage_no_time,
    }


def _graph_answer_coverage(
    paths: Sequence[PathAssessment],
    *,
    candidate_answer: str,
    candidate_claims: Sequence[str],
    use_time: bool,
) -> float | None:
    """Macro-average best path support over candidate claims, including missing-path zeros."""

    targets = _graph_answer_targets(
        paths,
        candidate_answer=candidate_answer,
        candidate_claims=candidate_claims,
    )
    if targets is None:
        return None
    if not targets:
        return 0.0
    best_by_answer: list[float] = []
    for target in targets:
        values: list[float | None] = []
        for row in paths:
            labels = row.answer_endpoint_labels or [row.answer_endpoint_label]
            if not any(
                raw_label and _mention_score(raw_label, target) == 1.0 for raw_label in labels
            ):
                continue
            values.append(row.coherence if use_time else _path_no_time_support(row))
        if not values:
            best_by_answer.append(0.0)
            continue
        known = [value for value in values if value is not None]
        if not known:
            return None
        best_by_answer.append(max(known))
    return mean(best_by_answer)


def _path_no_time_support(path: PathAssessment) -> float:
    """Remove only temporal validity while retaining the path's structural contract."""

    if not path.structural_valid:
        return 0.0
    return path.path_relevance * path.path_provenance


def _graph_answer_targets(
    paths: Sequence[PathAssessment],
    *,
    candidate_answer: str,
    candidate_claims: Sequence[str],
) -> list[str] | None:
    if len(candidate_claims) > 1:
        return list(candidate_claims)
    fallback = list(candidate_claims) or ([candidate_answer] if candidate_answer.strip() else [])
    endpoint_labels = {
        _normalize_text(label)
        for path in paths
        for label in (path.answer_endpoint_labels or [path.answer_endpoint_label])
        if _normalize_text(label)
    }
    normalized_answer = _normalize_text(candidate_answer)
    if normalized_answer in endpoint_labels:
        return fallback

    items, has_conjunction = split_top_level_answer_items(candidate_answer)
    if len(items) < 2:
        return fallback
    matched = [
        any(_mention_score(label, item) == 1.0 for label in endpoint_labels) for item in items
    ]
    if all(matched) or (has_conjunction and any(matched)):
        return items
    if any(matched):
        # A two-part comma expression may be either a list or one qualified entity name. Without
        # full endpoint coverage the automatic metric cannot resolve that ambiguity soundly.
        return None
    return fallback


def _response_decision(
    input_row: TaskJudgeInput,
    semantic: TCredSemanticRecord,
    *,
    query: NormalizedTemporalQuery,
    pair_index: Mapping[tuple[str, int, str, str], SemanticPairScore],
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
) -> tuple[float, float, float]:
    if not semantic.reference_claims or not input_row.retrieved_evidence:
        valid_support = 0.0
        conflict_exposure = 0.0
    else:
        assessments = [
            _reference_claim_answerability(
                input_row,
                pair_index,
                claim_index=index,
                claim=semantic.reference_claims[index],
                sibling_claims=[
                    sibling
                    for sibling_index, sibling in enumerate(semantic.reference_claims)
                    if sibling_index != index
                ],
                query=query,
                time_by_id=time_by_id,
                provenance_by_id=provenance_by_id,
            )
            for index in range(len(semantic.reference_claims))
        ]
        # Answering requires support for the complete reference claim set. A well-founded
        # refusal needs only one material claim to remain unresolved.
        valid_support = min((row[0] for row in assessments), default=0.0)
        conflict_exposure = max((row[1] for row in assessments), default=0.0)
    refuses = is_refusal(input_row.candidate_answer)
    score = 1.0 - valid_support if refuses else valid_support
    return score, valid_support, conflict_exposure


def _reference_claim_answerability(
    input_row: TaskJudgeInput,
    pairs: Mapping[tuple[str, int, str, str], SemanticPairScore],
    *,
    claim_index: int,
    claim: str,
    sibling_claims: Sequence[str],
    query: NormalizedTemporalQuery,
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
) -> tuple[float, float]:
    claim_time = parse_claim_time(claim)
    claim_query = claim_query_time_compatibility(claim_time, query)
    support_values: list[float] = []
    conflict_values: list[float] = []
    for row in input_row.retrieved_evidence:
        pair = pairs.get(("reference", claim_index, "retrieved", row.evidence_id))
        if pair is None:
            continue
        temporal = _headline_temporal_value(time_by_id[row.evidence_id])
        claim_temporal = _claim_time_value(claim_time, row.valid_time)
        query_temporal = _claim_query_value(claim_time, claim_query)
        provenance = provenance_by_id[row.evidence_id]
        support_values.append(
            _semantic_support_with_exact_mention(
                claim,
                row.text,
                pair,
                context_text=input_row.question,
            )
            * temporal
            * claim_temporal
            * query_temporal
            * provenance
        )
        # Restrict contradiction to evidence that actually mentions this claim. This keeps
        # evidence for another valid list item from cancelling the current item.
        if pair.contradiction >= _CONTRADICTION_THRESHOLD and _claim_anchor_is_mentioned(
            claim,
            row.text,
            context_text=input_row.question,
            sibling_claims=sibling_claims,
        ):
            conflict_values.append(
                pair.contradiction * temporal * claim_temporal * query_temporal * provenance
            )
    support = max(support_values, default=0.0)
    conflict = max(conflict_values, default=0.0)
    return _clamp(support - conflict), conflict


def _path_structurally_valid(path: JudgeGraphPath) -> bool:
    if not path.edges or len(path.nodes) != len(path.edges) + 1:
        return False
    # Public path payloads are already oriented in traversal order. The direction flag records
    # whether that displayed traversal follows or opposes the canonical relation.
    traversals = [(edge.source.id, edge.target.id) for edge in path.edges]
    if any(
        traversals[index][1] != traversals[index + 1][0] for index in range(len(traversals) - 1)
    ):
        return False
    node_ids = [node.id for node in path.nodes]
    traversal_nodes = [traversals[0][0], *[target for _source, target in traversals]]
    return node_ids == traversal_nodes


def _relevant_path_edge_indices(
    path: JudgeGraphPath,
    input_row: TaskJudgeInput,
    *,
    endpoint_answer_indices: Sequence[int],
) -> set[int]:
    """Return the minimal displayed segment joining question and answer anchors."""

    if not path.edges or not path.nodes or not endpoint_answer_indices:
        return set(range(len(path.edges)))
    answer_node_indices = {
        0 if endpoint_index == 0 else len(path.nodes) - 1
        for endpoint_index in endpoint_answer_indices
    }
    question_scores = [
        _mention_score(node.label, input_row.question)
        if index not in answer_node_indices
        else 0.0
        for index, node in enumerate(path.nodes)
    ]
    best_question = max(question_scores, default=0.0)
    if best_question <= 0.0:
        return set(range(len(path.edges)))
    question_indices = [
        index for index, score in enumerate(question_scores) if score == best_question
    ]
    relevant: set[int] = set()
    for answer_index in answer_node_indices:
        question_index = min(question_indices, key=lambda index: abs(index - answer_index))
        relevant.update(range(min(question_index, answer_index), max(question_index, answer_index)))
    return relevant or set(range(len(path.edges)))


def _relation_relevance(
    path: JudgeGraphPath,
    question: str,
    *,
    edge_indices: set[int] | None = None,
) -> float:
    question_tokens = _tokens(question) - _QUESTION_STOPWORDS
    values = []
    selected = set(range(len(path.edges))) if edge_indices is None else edge_indices
    for index, edge in enumerate(path.edges):
        if index not in selected:
            continue
        relation_tokens = _tokens(edge.relation_label or edge.relation) - _QUESTION_STOPWORDS
        if relation_tokens:
            values.append(len(relation_tokens & question_tokens) / len(relation_tokens))
    return max(values, default=0.0)


def _mention_score(label: str, text: str) -> float:
    normalized_label = _normalize_text(label)
    normalized_text = _normalize_text(text)
    if not normalized_label:
        return 0.0
    if _contains_normalized_phrase(normalized_label, normalized_text):
        return 1.0
    label_tokens = set(normalized_label.split()) - _QUESTION_STOPWORDS
    text_tokens = set(normalized_text.split()) - _QUESTION_STOPWORDS
    return len(label_tokens & text_tokens) / len(label_tokens) if label_tokens else 0.0


def _normalize_text(value: str) -> str:
    ascii_value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    return " ".join(_TOKEN.findall(ascii_value))


def _tokens(value: str) -> set[str]:
    return set(_normalize_text(value).split())


def _textual_pair(
    pairs: Mapping[tuple[str, int, str, str], SemanticPairScore],
    source: str,
    claim_index: int,
    evidence_id: str,
) -> SemanticPairScore | None:
    return pairs.get((source, claim_index, "retrieved", evidence_id)) or pairs.get(
        (source, claim_index, "cited", evidence_id)
    )


def _link_value(
    pairs: Mapping[tuple[str, int, str, str], SemanticPairScore],
    *,
    claim_source: str,
    claim_index: int,
    evidence_kind: str,
    evidence_id: str,
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
    use_time: bool,
    claim_time: ClaimTimeConstraint | None = None,
    claim_query_compatibility: float | None = None,
    evidence_interval: VisibleInterval | None = None,
    evidence_text: str | None = None,
    claim: str | None = None,
    context_text: str | None = None,
) -> float:
    pair = pairs.get((claim_source, claim_index, evidence_kind, evidence_id))
    if pair is None:
        return 0.0
    temporal = _headline_temporal_value(time_by_id[evidence_id]) if use_time else 1.0
    claim_temporal = (
        _claim_time_value(claim_time, evidence_interval)
        if use_time and claim_time is not None and evidence_interval is not None
        else 1.0
    )
    claim_query = (
        _claim_query_value(claim_time, claim_query_compatibility)
        if use_time and claim_time is not None
        else 1.0
    )
    semantic = (
        _semantic_support_with_exact_mention(
            claim,
            evidence_text,
            pair,
            context_text=context_text,
        )
        if claim is not None and evidence_text is not None
        else pair.entailment
    )
    return semantic * temporal * claim_temporal * claim_query * provenance_by_id[evidence_id]


def _max_claim_support(
    pairs: Mapping[tuple[str, int, str, str], SemanticPairScore],
    *,
    claim_index: int,
    rows: Sequence[JudgeEvidence],
    kind: str,
    time_by_id: Mapping[str, EvidenceTemporalAssessment],
    provenance_by_id: Mapping[str, float],
    use_time: bool,
    claim_time: ClaimTimeConstraint | None = None,
    claim_query_compatibility: float | None = None,
    claim: str,
    context_text: str,
) -> float:
    return max(
        (
            _link_value(
                pairs,
                claim_source="candidate",
                claim_index=claim_index,
                evidence_kind=kind,
                evidence_id=row.evidence_id,
                time_by_id=time_by_id,
                provenance_by_id=provenance_by_id,
                use_time=use_time,
                claim_time=claim_time,
                claim_query_compatibility=claim_query_compatibility,
                evidence_interval=row.valid_time,
                evidence_text=row.text,
                claim=claim,
                context_text=context_text,
            )
            for row in rows
        ),
        default=0.0,
    )


def _semantic_support_with_exact_mention(
    claim: str,
    evidence_text: str,
    pair: SemanticPairScore | None,
    *,
    context_text: str | None = None,
) -> float:
    entailment = pair.entailment if pair is not None else 0.0
    normalized_claim = _normalize_text(claim)
    normalized_evidence = _normalize_text(evidence_text)
    exact_phrase = bool(
        normalized_claim
        and _contains_normalized_phrase(normalized_claim, normalized_evidence)
    )
    contextual_coverage = bool(
        context_text is not None
        and _contextual_claim_coverage(claim, evidence_text, context_text=context_text)
    )
    if not exact_phrase and not contextual_coverage:
        return entailment
    if context_text is not None:
        context_tokens = _tokens(context_text) - _QUESTION_STOPWORDS
        evidence_tokens = _tokens(evidence_text) - _QUESTION_STOPWORDS
        if context_tokens and not context_tokens & evidence_tokens:
            return entailment
    contradiction = pair.contradiction if pair is not None else 0.0
    return max(entailment, 1.0 - contradiction)


def _contextual_claim_coverage(
    claim: str,
    evidence_text: str,
    *,
    context_text: str,
) -> bool:
    """License a lexical rescue only with answer-specific and relation-context coverage."""

    claim_tokens = _tokens(claim) - _QUESTION_STOPWORDS - _DISCOURSE_SUPPORT_TOKENS
    context_tokens = _tokens(context_text) - _QUESTION_STOPWORDS - _DISCOURSE_SUPPORT_TOKENS
    evidence_tokens = _tokens(evidence_text) - _QUESTION_STOPWORDS - _DISCOURSE_SUPPORT_TOKENS
    specific = claim_tokens - context_tokens
    if not specific or not context_tokens:
        return False
    specific_overlap = len(specific & evidence_tokens)
    context_overlap = len(context_tokens & evidence_tokens)
    required_specific = 1 if len(specific) == 1 else 2
    required_context = 1 if len(context_tokens) <= 3 else 2
    return bool(
        specific_overlap >= required_specific
        and specific_overlap / len(specific) >= 0.6
        and context_overlap >= required_context
        and context_overlap / len(context_tokens) >= 0.4
    )


def _claim_anchor_is_mentioned(
    claim: str,
    evidence_text: str,
    *,
    context_text: str,
    sibling_claims: Sequence[str] = (),
) -> bool:
    """Require a claim-specific anchor before treating NLI contradiction as direct conflict."""

    normalized_claim = _normalize_text(claim)
    normalized_evidence = _normalize_text(evidence_text)
    if _contains_normalized_phrase(normalized_claim, normalized_evidence):
        return True

    claim_tokens = _tokens(claim) - _QUESTION_STOPWORDS
    evidence_tokens = _tokens(evidence_text) - _QUESTION_STOPWORDS
    context_tokens = _tokens(context_text) - _QUESTION_STOPWORDS
    sibling_tokens = {
        token
        for sibling in sibling_claims
        for token in (_tokens(sibling) - _QUESTION_STOPWORDS)
    }
    claim_specific_tokens = claim_tokens - context_tokens - sibling_tokens
    if not claim_specific_tokens:
        claim_specific_tokens = claim_tokens - sibling_tokens
    if not claim_tokens or not claim_specific_tokens:
        return False
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    return bool(claim_specific_tokens & evidence_tokens) and overlap >= 0.5


def _contains_normalized_phrase(phrase: str, text: str) -> bool:
    phrase_tokens = phrase.split()
    text_tokens = text.split()
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def _headline_temporal_value(row: EvidenceTemporalAssessment) -> float:
    if row.status in {"stale", "future_invalid", "unknown_valid_time", "publication_only"}:
        return 0.0
    if row.compatibility is None:
        return 0.0
    if row.exact_validity != 1.0 and (row.near_miss or 0.0) > 0.0:
        return 0.0
    return row.compatibility


def _headline_temporal_from_link(row: ClaimEvidenceAssessment) -> float:
    if row.exact_temporal_validity == 1.0:
        return float(row.temporal_compatibility) if row.temporal_compatibility is not None else 0.0
    return 0.0


def _claim_time_value(
    claim_time: ClaimTimeConstraint,
    interval: VisibleInterval,
) -> float:
    value = claim_evidence_time_compatibility(claim_time, interval)
    if claim_time.status == "absent":
        return 1.0
    return 0.0 if value is None else value


def _claim_query_value(
    claim_time: ClaimTimeConstraint,
    value: float | None,
) -> float:
    if claim_time.status == "absent":
        return 1.0
    return 0.0 if value is None else value


def _provenance_value(evidence_id: str, provenance: Mapping[str, float] | None) -> float:
    if provenance is None:
        return 1.0
    if not evidence_id or evidence_id not in provenance:
        raise ValueError(
            "A supplied provenance map must contain every scored evidence and graph-edge ID; "
            f"missing {evidence_id!r}"
        )
    value = float(provenance[evidence_id])
    if not 0 <= value <= 1:
        raise ValueError(f"Provenance outside [0,1] for {evidence_id}: {value}")
    return value


def _within_list_ndcg(gains: Sequence[float], *, k: int) -> float | None:
    top = list(gains[:k])
    ideal = sorted(gains, reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    if idcg == 0:
        return None
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(top, start=1))
    return dcg / idcg


def _harmonic_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return 2 * left * right / (left + right) if left + right else 0.0


def _geometric_available(values: Sequence[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return _geometric_all(available) if available else None


def _geometric_all(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def _mean_or_zero(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
