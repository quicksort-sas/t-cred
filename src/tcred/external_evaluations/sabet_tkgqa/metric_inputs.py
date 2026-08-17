from __future__ import annotations

import hashlib
from collections import defaultdict

from tcred.external_evaluations.sabet_tkgqa.display_labels import (
    ResolvedAnswerText,
    resolve_answer_text,
)
from tcred.external_evaluations.sabet_tkgqa.label_bundle import TimeQuestionsLabelResolver
from tcred.external_evaluations.sabet_tkgqa.schema import SabetPredictionRecord
from tcred.metrics.models import MetricInput
from tcred.metrics.task_judge_models import TaskJudgeInput
from tcred.metrics.tcred_models import TCredMetricResult, TCredSemanticRecord
from tcred.metrics.tcred_semantic import semantic_claims, semantic_input_hash
from tcred.metrics.tcred_suite import score_tcred_suite


def build_pairwise_metric_inputs(
    record: SabetPredictionRecord,
    *,
    resolved_text: ResolvedAnswerText | None = None,
    resolver: TimeQuestionsLabelResolver | None = None,
) -> list[MetricInput]:
    """Build one metric input per distinct gold label without exposing answer IDs as text."""

    text = resolved_text or resolve_answer_text(record, resolver=resolver)
    if text.candidate is None:
        return []
    gold_ids_by_label: dict[str, list[str]] = defaultdict(list)
    gold_indices_by_label: dict[str, list[int]] = defaultdict(list)
    gold_sources_by_label: dict[str, set[str]] = defaultdict(set)
    gold_revisions_by_label: dict[str, set[int]] = defaultdict(set)
    gold_canonical_qids_by_label: dict[str, set[str]] = defaultdict(set)
    for reference in text.references:
        assert reference.original_reference_index is not None
        gold_ids_by_label[reference.text].append(reference.answer_id)
        gold_indices_by_label[reference.text].append(reference.original_reference_index)
        gold_sources_by_label[reference.text].add(reference.source)
        if reference.wikidata_lastrevid is not None:
            gold_revisions_by_label[reference.text].add(reference.wikidata_lastrevid)
        if reference.wikidata_canonical_qid is not None:
            gold_canonical_qids_by_label[reference.text].add(
                reference.wikidata_canonical_qid
            )
    native_hit_at_1 = record.predicted_answer_ids[0] in set(record.gold_answer_ids)
    inputs = []
    for reference_index, (label, answer_ids) in enumerate(gold_ids_by_label.items()):
        metric_id = (
            f"sabet:{record.run_id}:{record.source_index}:reference:{reference_index}"
        )
        inputs.append(
            MetricInput(
                metric_id=metric_id,
                population="system_full",
                dataset_family=f"sabet_tkgqa:{record.dataset}",
                source_kind="external_sabet_tkgqa",
                system_name=f"{record.model}:{record.variant}:seed{record.seed}",
                unit_id=f"{record.run_id}:{record.source_index}",
                qid=record.qid,
                scenario_id=record.dataset,
                question=record.question,
                reference_answer=label,
                candidate_answer=text.candidate.text,
                gold_labels={"native_hit_at_1": str(int(native_hit_at_1))},
                gold_provenance={
                    "external_identity_oracle": {
                        "gold_answer_ids": answer_ids,
                        "gold_answer_indices": gold_indices_by_label[label],
                        "predicted_answer_id": record.predicted_answer_ids[0],
                        "answer_type": record.answer_type,
                        "question_type": record.question_type,
                    },
                    "display_label_resolution": {
                        "candidate_source": text.candidate.source,
                        "candidate_wikidata_lastrevid": (
                            text.candidate.wikidata_lastrevid
                        ),
                        "candidate_wikidata_canonical_qid": (
                            text.candidate.wikidata_canonical_qid
                        ),
                        "reference_sources": sorted(gold_sources_by_label[label]),
                        "reference_wikidata_lastrevids": sorted(
                            gold_revisions_by_label[label]
                        ),
                        "reference_wikidata_canonical_qids": sorted(
                            gold_canonical_qids_by_label[label]
                        ),
                    },
                },
            )
        )
    return inputs


def build_answer_only_tcred_input(row: MetricInput) -> TaskJudgeInput:
    """Convert an external answer pair without fabricating unavailable evidence."""

    return TaskJudgeInput(
        metric_id=row.metric_id,
        population=row.population,
        dataset_family=row.dataset_family,
        source_kind=row.source_kind,
        system_name=row.system_name,
        unit_id=row.unit_id,
        qid=row.qid,
        scenario_id=row.scenario_id,
        question=row.question,
        reference_answer=row.reference_answer,
        candidate_answer=row.candidate_answer,
        applicable_fields=["answer_correct", "temporal_correct"],
        context_note=(
            "External closed-world answer ranking; no retrieved evidence, citations, or graph "
            "trace were exported by the evaluated system."
        ),
        gold_labels=row.gold_labels,
        gold_provenance=row.gold_provenance,
        source_question_sha256=_text_sha256(row.question),
        source_reference_answer_sha256=_text_sha256(row.reference_answer),
        source_candidate_answer_sha256=_text_sha256(row.candidate_answer),
        presentation_contract="sabet-answer-only-v1",
    )


def score_answer_only_tcred(
    row: MetricInput,
    *,
    baseline_scores: dict[str, float | None],
) -> TCredMetricResult:
    """Score the applicable T-CRED vector while retaining unsupported fields as missing."""

    task_input = build_answer_only_tcred_input(row)
    candidate_claims, reference_claims = semantic_claims(task_input)
    semantic = TCredSemanticRecord(
        metric_id=task_input.metric_id,
        input_sha256=semantic_input_hash(task_input),
        model="not_applicable:no_exported_evidence",
        class_mapping={"entailment": 0, "neutral": 1, "contradiction": 2},
        candidate_claims=candidate_claims,
        reference_claims=reference_claims,
        pairs=[],
    )
    result = score_tcred_suite(
        task_input,
        semantic,
        baseline_scores=baseline_scores,
        baseline_input_aligned=True,
        provenance=None,
    )
    applicable_scores = {
        "tcred_answer_equivalence",
        "tcred_temporal_correctness",
        "tcred_explicit_claim_time_validity",
        "tcred_explicit_claim_query_validity",
    }
    scores = {
        name: value if name in applicable_scores else None
        for name, value in result.scores.items()
    }
    coverage = {
        **result.coverage,
        "semantic_attribution": False,
        "temporal_attribution": False,
        "citation": False,
        "retrieval": False,
        "graph": False,
        "response_decision": False,
        "declared_provenance": False,
    }
    audit = {
        **result.audit,
        "external_applicability_mask": (
            "No exported evidence, citations, graph paths, retrieval trace, provenance map, or "
            "response decision; unavailable components are masked to null rather than zero."
        ),
        "unmasked_scores_retained": sorted(applicable_scores),
    }
    return result.model_copy(update={"scores": scores, "coverage": coverage, "audit": audit})


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
