from __future__ import annotations

from tcred.external_evaluations.sabet_tkgqa.display_labels import (
    ResolvedAnswerText,
    resolve_answer_text,
)
from tcred.external_evaluations.sabet_tkgqa.label_bundle import TimeQuestionsLabelResolver
from tcred.external_evaluations.sabet_tkgqa.schema import (
    AnswerMetricRecord,
    SabetPredictionRecord,
)
from tcred.metrics.deterministic import reference_answer_scores


def score_prediction(
    record: SabetPredictionRecord,
    *,
    resolved_text: ResolvedAnswerText | None = None,
    resolver: TimeQuestionsLabelResolver | None = None,
) -> AnswerMetricRecord:
    """Score top-1 text against each reference and retain the best score per metric."""

    gold_ids = set(record.gold_answer_ids)
    native_hit_at_1 = float(record.predicted_answer_ids[0] in gold_ids)
    native_hit_at_10 = float(bool(set(record.predicted_answer_ids[:10]) & gold_ids))
    text = resolved_text or resolve_answer_text(record, resolver=resolver)
    readable_references = text.references
    candidate_available = text.candidate is not None
    if not readable_references or not candidate_available:
        return AnswerMetricRecord(
            run_id=record.run_id,
            dataset=record.dataset,
            model=record.model,
            variant=record.variant,
            seed=record.seed,
            qid=record.qid,
            source_index=record.source_index,
            question_type=record.question_type,
            answer_type=record.answer_type,
            native_hit_at_1=native_hit_at_1,
            native_hit_at_10=native_hit_at_10,
            scores={
                name: None
                for name in (
                    "exact_match",
                    "token_precision",
                    "token_recall",
                    "token_f1",
                    "rouge_1",
                    "rouge_2",
                    "rouge_l",
                )
            },
            winning_reference_by_metric={},
            applicability=_applicability(
                reference_available=bool(readable_references),
                candidate_available=candidate_available,
            ),
        )
    assert text.candidate is not None
    reference_rows = [
        reference_answer_scores(text.candidate.text, reference.text)
        for reference in readable_references
    ]
    scores: dict[str, float | None] = {}
    winners: dict[str, int] = {}
    for metric_name in reference_rows[0]:
        values = [row[metric_name] for row in reference_rows]
        best_index, best_value = max(enumerate(values), key=lambda item: item[1])
        scores[metric_name] = best_value
        original_index = readable_references[best_index].original_reference_index
        assert original_index is not None
        winners[metric_name] = original_index
    return AnswerMetricRecord(
        run_id=record.run_id,
        dataset=record.dataset,
        model=record.model,
        variant=record.variant,
        seed=record.seed,
        qid=record.qid,
        source_index=record.source_index,
        question_type=record.question_type,
        answer_type=record.answer_type,
        native_hit_at_1=native_hit_at_1,
        native_hit_at_10=native_hit_at_10,
        scores=scores,
        winning_reference_by_metric=winners,
        applicability=_applicability(
            reference_available=True,
            candidate_available=True,
        ),
    )


def _applicability(
    *, reference_available: bool, candidate_available: bool
) -> dict[str, bool]:
    answer_text_available = reference_available and candidate_available
    return {
        "reference_text_available": reference_available,
        "candidate_text_available": candidate_available,
        "answer_equivalence": answer_text_available,
        "temporal_correctness_answer_only": answer_text_available,
        "semantic_attribution": False,
        "temporal_attribution": False,
        "citation_quality": False,
        "temporal_retrieval": False,
        "graph_path_quality": False,
        "response_decision": False,
        "complete_tcred_aggregate": False,
    }
