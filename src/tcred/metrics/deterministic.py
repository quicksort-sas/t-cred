from __future__ import annotations

import math
import re
import string
from collections import Counter

from tcred.metrics.models import ClaimJudgeResult

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Apply the conventional SQuAD answer normalization."""
    lowered = text.casefold().translate(_PUNCTUATION)
    without_articles = _ARTICLES.sub(" ", lowered)
    return _WHITESPACE.sub(" ", without_articles).strip()


def reference_answer_scores(candidate: str, reference: str) -> dict[str, float]:
    candidate_tokens = normalize_answer(candidate).split()
    reference_tokens = normalize_answer(reference).split()
    exact_match = float(candidate_tokens == reference_tokens)

    overlap = sum((Counter(candidate_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(candidate_tokens) if candidate_tokens else float(not reference_tokens)
    recall = overlap / len(reference_tokens) if reference_tokens else float(not candidate_tokens)
    token_f1 = _harmonic_mean(precision, recall)

    lcs = _lcs_length(candidate_tokens, reference_tokens)
    rouge_precision = (
        lcs / len(candidate_tokens) if candidate_tokens else float(not reference_tokens)
    )
    rouge_recall = lcs / len(reference_tokens) if reference_tokens else float(not candidate_tokens)
    return {
        "exact_match": exact_match,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": token_f1,
        "rouge_1": _rouge_n_f1(candidate_tokens, reference_tokens, n=1),
        "rouge_2": _rouge_n_f1(candidate_tokens, reference_tokens, n=2),
        "rouge_l": _harmonic_mean(rouge_precision, rouge_recall),
    }


def ranked_retrieval_scores(
    *,
    relevance: list[bool],
    required_count: int,
    k: int = 10,
) -> dict[str, float | None]:
    """Compute binary P/R, Hit, MRR, and nDCG without rewarding duplicate evidence."""
    if required_count <= 0:
        return {
            f"retrieval_precision_at_{k}": None,
            f"retrieval_recall_at_{k}": None,
            f"retrieval_hit_at_{k}": None,
            f"retrieval_average_precision_at_{k}": None,
            "retrieval_mrr": None,
            "retrieval_r_precision": None,
            f"retrieval_ndcg_at_{k}": None,
        }
    top = relevance[:k]
    relevant_retrieved = sum(top)
    precision = relevant_retrieved / k
    recall = relevant_retrieved / required_count
    first_rank = next((index for index, value in enumerate(relevance, start=1) if value), None)
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, value in enumerate(top, start=1) if value)
    ideal_hits = min(required_count, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    running_hits = 0
    precision_sum = 0.0
    for rank, value in enumerate(top, start=1):
        if value:
            running_hits += 1
            precision_sum += running_hits / rank
    r_hits = sum(relevance[:required_count])
    return {
        f"retrieval_precision_at_{k}": precision,
        f"retrieval_recall_at_{k}": recall,
        f"retrieval_hit_at_{k}": float(relevant_retrieved > 0),
        f"retrieval_average_precision_at_{k}": precision_sum / ideal_hits,
        "retrieval_mrr": 1.0 / first_rank if first_rank else 0.0,
        "retrieval_r_precision": r_hits / required_count,
        f"retrieval_ndcg_at_{k}": dcg / idcg if idcg else None,
    }


def set_precision_recall(
    *,
    predicted: set[tuple[str, ...]],
    required: set[tuple[str, ...]],
    prefix: str,
) -> dict[str, float | None]:
    intersection = len(predicted & required)
    return {
        f"{prefix}_precision": intersection / len(predicted) if predicted else None,
        f"{prefix}_recall": intersection / len(required) if required else None,
    }


def claim_judge_scores(
    result: ClaimJudgeResult,
    *,
    retrieved_count: int,
    cited_count: int,
) -> dict[str, float | None]:
    """Apply the published RAGChecker formulas to structured claim judgments."""
    candidate = result.candidate_claims
    reference = result.reference_claims
    candidate_count = len(candidate)
    reference_count = len(reference)

    precision = sum(claim.reference_supported for claim in candidate) / candidate_count
    recall = sum(claim.candidate_supported for claim in reference) / reference_count
    overall_f1 = _harmonic_mean(precision, recall)

    relevant_chunks = {
        index
        for claim in reference
        for index in claim.retrieved_support_indices
        if 1 <= index <= retrieved_count
    }
    retrieved_reference_claims = [claim for claim in reference if claim.retrieved_support_indices]
    faithful = sum(bool(claim.retrieved_support_indices) for claim in candidate) / candidate_count
    hallucination = (
        sum(
            not claim.reference_supported and not claim.retrieved_support_indices
            for claim in candidate
        )
        / candidate_count
    )
    self_knowledge = (
        sum(
            claim.reference_supported and not claim.retrieved_support_indices for claim in candidate
        )
        / candidate_count
    )

    relevant_noise = (
        sum(
            not claim.reference_supported
            and bool(set(claim.retrieved_support_indices) & relevant_chunks)
            for claim in candidate
        )
        / candidate_count
    )
    irrelevant_noise = (
        sum(
            not claim.reference_supported
            and any(index not in relevant_chunks for index in claim.retrieved_support_indices)
            for claim in candidate
        )
        / candidate_count
    )

    context_utilization = None
    if retrieved_reference_claims:
        context_utilization = sum(
            claim.candidate_supported for claim in retrieved_reference_claims
        ) / len(retrieved_reference_claims)

    cited_chunks = {
        index
        for claim in candidate
        for index in claim.cited_support_indices
        if 1 <= index <= cited_count
    }
    citation_completeness = (
        sum(bool(claim.cited_support_indices) for claim in candidate) / candidate_count
        if cited_count
        else None
    )
    citation_precision = len(cited_chunks) / cited_count if cited_count else None

    return {
        "g_eval_answer_correctness": result.answer_correctness / 4.0,
        "g_eval_answer_relevance": result.answer_relevance / 4.0,
        "ragchecker_precision": precision,
        "ragchecker_recall": recall,
        "ragchecker_f1": overall_f1,
        "ragchecker_claim_recall": sum(bool(claim.retrieved_support_indices) for claim in reference)
        / reference_count,
        "ragchecker_context_precision": (
            len(relevant_chunks) / retrieved_count if retrieved_count else None
        ),
        "ragchecker_faithfulness": faithful,
        "ragchecker_hallucination": hallucination,
        "ragchecker_non_hallucination": 1.0 - hallucination,
        "ragchecker_self_knowledge": self_knowledge,
        "ragchecker_relevant_noise_sensitivity": relevant_noise,
        "ragchecker_irrelevant_noise_sensitivity": irrelevant_noise,
        "ragchecker_context_utilization": context_utilization,
        "alce_citation_completeness": citation_completeness,
        "alce_citation_precision": citation_precision,
    }


def _harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0


def _rouge_n_f1(candidate: list[str], reference: list[str], *, n: int) -> float:
    candidate_ngrams = _ngram_counts(candidate, n=n)
    reference_ngrams = _ngram_counts(reference, n=n)
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    overlap = sum((candidate_ngrams & reference_ngrams).values())
    precision = overlap / sum(candidate_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    return _harmonic_mean(precision, recall)


def _ngram_counts(tokens: list[str], *, n: int) -> Counter[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
