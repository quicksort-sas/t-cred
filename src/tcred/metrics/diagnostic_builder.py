from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import orjson

from tcred.dataset.graph import fact_answer_id
from tcred.dataset.io import load_bundle
from tcred.dataset.models import AnswerVariant, DatasetBundle, Fact, Question
from tcred.dataset.solver import fact_visible
from tcred.human_eval.export import (
    blind_public_payload,
    controlled_internal_unit,
    public_evidence,
)
from tcred.metrics.deterministic import ranked_retrieval_scores, set_precision_recall
from tcred.metrics.diagnostic_models import (
    DiagnosticCase,
    DiagnosticPair,
    DiagnosticSuite,
    diagnostic_inference_cluster_ids,
)
from tcred.metrics.models import EvidenceText, MetricInput
from tcred.metrics.task_judge_models import JUDGED_FIELDS, TaskJudgeInput
from tcred.qa.corpus import dataset_content_hash

DEFAULT_DIAGNOSTIC_SEED = 20260815
DEFAULT_PAIR_CAP = 40

_ORACLE_FIELD_MAP = {
    "answer_correct": "answer_correct",
    "temporal_correct": "temporal_correct",
    "evidence_supports_answer": "evidence_supports_answer",
    "citation_temporally_valid": "citation_temporally_valid",
    "graph_evidence_sufficient": "graph_path_sufficient",
    "response_decision_appropriate": "refusal_appropriate",
}
_BASELINE_TYPES = {"correct_supported", "correct_refusal"}
_REFUSAL_TYPES = {"correct_refusal", "inappropriate_refusal"}
_SEVERITY_INCORRECT_TYPES = {
    "future_invalid_answer",
    "overconfident_should_refuse",
    "stale_answer",
    "unsupported_hallucinated_answer",
    "wrong_operator_answer",
}
_VARIANT_TESTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "correct_answer_invalid_evidence": (
        ("citation_correctness", "citation_temporally_valid", "left_higher"),
        ("answer_correctness", "answer_correct", "equal"),
        ("temporal_correctness", "temporal_correct", "equal"),
        ("evidence_support", "evidence_supports_answer", "equal"),
    ),
    "stale_answer": (
        ("answer_correctness", "answer_correct", "left_higher"),
        ("temporal_correctness", "temporal_correct", "left_higher"),
        ("evidence_support", "evidence_supports_answer", "equal"),
    ),
    "outdated_source_answer": (
        ("answer_correctness", "answer_correct", "left_higher"),
        ("temporal_correctness", "temporal_correct", "left_higher"),
        ("evidence_support", "evidence_supports_answer", "equal"),
    ),
    "future_invalid_answer": (
        ("answer_correctness", "answer_correct", "left_higher"),
        ("temporal_correctness", "temporal_correct", "left_higher"),
        ("evidence_support", "evidence_supports_answer", "equal"),
    ),
    "wrong_operator_answer": (
        ("answer_correctness", "answer_correct", "left_higher"),
        ("temporal_correctness", "temporal_correct", "left_higher"),
        ("evidence_support", "evidence_supports_answer", "equal"),
    ),
    "partial_answer": (("answer_correctness", "answer_correct", "left_higher"),),
    "unsupported_hallucinated_answer": (("answer_correctness", "answer_correct", "left_higher"),),
    "overconfident_should_refuse": (
        ("response_decision", "response_decision_appropriate", "left_higher"),
        ("answer_correctness", "answer_correct", "left_higher"),
    ),
    "inappropriate_refusal": (
        ("answer_correctness", "answer_correct", "left_higher"),
    ),
}
_PHENOMENON_NAMES = {
    "correct_answer_invalid_evidence": "temporally_invalid_citation",
    "stale_answer": "stale_answer",
    "outdated_source_answer": "outdated_source_answer",
    "future_invalid_answer": "future_answer",
    "wrong_operator_answer": "wrong_temporal_operator",
    "partial_answer": "partial_answer",
    "unsupported_hallucinated_answer": "unsupported_answer",
    "overconfident_should_refuse": "answer_when_evidence_insufficient",
    "inappropriate_refusal": "refuse_when_answer_supported",
}
_LABEL_SCORE = {"no": 0, "partial": 1, "yes": 2}
_INTERNAL_RENDER_CACHE: dict[tuple[int, str], dict[str, object]] = {}


@dataclass(frozen=True)
class _VariantContext:
    family: str
    bundle: DatasetBundle
    question: Question
    answer: AnswerVariant
    baseline: AnswerVariant


def build_diagnostic_suite(
    dataset_root: Path,
    *,
    seed: int = DEFAULT_DIAGNOSTIC_SEED,
    pair_cap_per_phenomenon: int = DEFAULT_PAIR_CAP,
    source_split: str = "test_auto",
) -> DiagnosticSuite:
    if pair_cap_per_phenomenon < 10:
        raise ValueError("pair_cap_per_phenomenon must be at least 10")
    _INTERNAL_RENDER_CACHE.clear()

    families = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    bundles = {family: load_bundle(dataset_root / family) for family in families}
    missing_split = [
        family for family, bundle in bundles.items() if source_split not in bundle.splits
    ]
    if missing_split:
        raise ValueError(
            f"Diagnostic source split {source_split!r} is missing from: {missing_split}"
        )
    hashes = {family: dataset_content_hash(dataset_root / family) for family in sorted(bundles)}
    contexts = _eligible_variant_contexts(bundles, source_split=source_split)
    cases: dict[str, DiagnosticCase] = {}
    pairs: list[DiagnosticPair] = []
    internal_by_case: dict[str, dict[str, object]] = {}

    by_variant: defaultdict[str, list[_VariantContext]] = defaultdict(list)
    for context in contexts:
        by_variant[str(context.answer.variant_type)].append(context)

    selected_contexts: list[_VariantContext] = []
    for variant_type, tests in _VARIANT_TESTS.items():
        selected = _balanced_sample(
            by_variant[variant_type],
            cap=pair_cap_per_phenomenon,
            seed=seed,
            salt=variant_type,
        )
        selected_contexts.extend(selected)
        for context in selected:
            baseline_case = _canonical_case(context, answer=context.baseline)
            candidate_case = _canonical_case(context, answer=context.answer)
            cases.setdefault(baseline_case.case_id, baseline_case)
            cases.setdefault(candidate_case.case_id, candidate_case)
            internal_by_case.setdefault(
                baseline_case.case_id,
                _render_internal(context.bundle, context.baseline),
            )
            internal_by_case.setdefault(
                candidate_case.case_id,
                _render_internal(context.bundle, context.answer),
            )

            left_id, right_id = _matched_style_pair(
                context,
                baseline_case=baseline_case,
                candidate_case=candidate_case,
                cases=cases,
                internal_by_case=internal_by_case,
                seed=seed,
            )
            phenomenon = _PHENOMENON_NAMES[variant_type]
            for construct, field, relation in tests:
                left_label = baseline_case.oracle_labels.get(field)
                right_label = candidate_case.oracle_labels.get(field)
                if not _labels_support_relation(left_label, right_label, relation=relation):
                    continue
                pairs.append(
                    _pair(
                        construct=construct,
                        phenomenon=phenomenon,
                        context=context,
                        left_case_id=left_id,
                        right_case_id=right_id,
                        expected_relation="left_higher" if relation == "left_higher" else "equal",
                        target_fields=[field],
                        changed_components=_changed_components(
                            internal_by_case[left_id], internal_by_case[right_id]
                        ),
                        oracle_basis=(
                            "Validated AnswerVariant semantic-oracle labels; the pair shares a "
                            "question and differs by one generator-controlled failure mode."
                        ),
                    )
                )

    _add_answer_invariance_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    severity_pair_count = _add_severity_pairs(
        contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    if severity_pair_count == 0:
        raise ValueError(
            "The diagnostic source split cannot construct a partial-versus-incorrect "
            "severity test"
        )
    _add_inappropriate_refusal_decision_pairs(
        by_variant["inappropriate_refusal"],
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_evidence_order_invariance_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_irrelevant_evidence_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_wrong_evidence_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_temporally_invalid_retrieval_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_graph_only_pairs(
        by_variant["invalid_graph_path_answer"],
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_graph_time_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_citation_omission_pairs(
        selected_contexts,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
    )
    _add_update_invariance_pairs(
        bundles,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
        source_split=source_split,
    )
    _add_update_sensitivity_pairs(
        bundles,
        cases=cases,
        pairs=pairs,
        internal_by_case=internal_by_case,
        cap=pair_cap_per_phenomenon,
        seed=seed,
        source_split=source_split,
    )

    pairs = sorted({pair.pair_id: pair for pair in pairs}.values(), key=lambda row: row.pair_id)
    ordered_cases = sorted(cases.values(), key=lambda row: row.case_id)
    _validate_pair_isolation(ordered_cases, pairs)
    inference_cluster_ids = diagnostic_inference_cluster_ids(ordered_cases, pairs)
    return DiagnosticSuite(
        seed=seed,
        source_split=source_split,
        pair_cap_per_phenomenon=pair_cap_per_phenomenon,
        dataset_content_hashes=hashes,
        cases=ordered_cases,
        pairs=pairs,
        audit={
            "case_count": len(ordered_cases),
            "pair_count": len(pairs),
            "question_clusters": len(
                {
                    (case.metric_input.dataset_family, case.metric_input.qid)
                    for case in ordered_cases
                }
            ),
            "source_scenarios": len(
                {
                    (case.metric_input.dataset_family, case.metric_input.scenario_id)
                    for case in ordered_cases
                }
            ),
            "inference_clusters": len(set(inference_cluster_ids.values())),
            "pair_counts_by_test_type": dict(
                sorted(Counter(pair.test_type for pair in pairs).items())
            ),
            "pair_counts_by_construct": dict(
                sorted(Counter(pair.target_construct for pair in pairs).items())
            ),
            "pair_counts_by_phenomenon": dict(
                sorted(Counter(pair.phenomenon for pair in pairs).items())
            ),
            "pair_counts_by_dataset": dict(
                sorted(Counter(pair.dataset_family for pair in pairs).items())
            ),
            "selection_policy": (
                f"Fixed-hash, family-interleaved sampling from {source_split} only; no metric "
                "score is read during construction."
            ),
        },
    )


def _eligible_variant_contexts(
    bundles: dict[str, DatasetBundle],
    *,
    source_split: str,
) -> list[_VariantContext]:
    contexts: list[_VariantContext] = []
    for family, bundle in bundles.items():
        selected_scenarios = set(bundle.splits[source_split])
        question_by_id = {question.qid: question for question in bundle.questions}
        answers_by_qid: defaultdict[str, list[AnswerVariant]] = defaultdict(list)
        for answer in bundle.answer_variants:
            if answer.scenario_id in selected_scenarios:
                answers_by_qid[answer.qid].append(answer)
        for qid, answers in answers_by_qid.items():
            baseline = next(
                (answer for answer in answers if str(answer.variant_type) in _BASELINE_TYPES),
                None,
            )
            if baseline is None:
                continue
            for answer in answers:
                if str(answer.variant_type) in _BASELINE_TYPES:
                    continue
                contexts.append(
                    _VariantContext(
                        family=family,
                        bundle=bundle,
                        question=question_by_id[qid],
                        answer=answer,
                        baseline=baseline,
                    )
                )
    return contexts


def _canonical_case(context: _VariantContext, *, answer: AnswerVariant) -> DiagnosticCase:
    internal = _render_internal(context.bundle, answer)
    return _make_case(
        family=context.family,
        question=context.question,
        internal=internal,
        source_answer_id=answer.answer_id,
        transformation="canonical_answer_variant",
        oracle_labels=_answer_oracle(answer),
        oracle_basis="Dataset generator semantic oracle, validated by the release validator.",
        changed_components=[],
        case_suffix=f"answer:{answer.answer_id}",
    )


def _make_case(
    *,
    family: str,
    question: Question,
    internal: dict[str, object],
    source_answer_id: str | None,
    transformation: str,
    oracle_labels: dict[str, str],
    oracle_basis: str,
    changed_components: list[str],
    case_suffix: str,
) -> DiagnosticCase:
    public, key = blind_public_payload(_copy(internal))
    case_id = f"diagnostic:{family}:{case_suffix}"
    evidence_map = {
        str(handle): str(fact_id)
        for handle, fact_id in _mapping(key.get("evidence_handles")).items()
    }
    required = set(question.required_valid_evidence_ids)
    credited: set[str] = set()
    relevance: list[bool] = []
    for row in _list_of_mappings(public.get("retrieved_evidence")):
        raw_id = evidence_map.get(str(row.get("evidence_id", "")), "")
        relevant = raw_id in required and raw_id not in credited
        relevance.append(relevant)
        if relevant:
            credited.add(raw_id)
    cited_raw = {
        evidence_map.get(str(row.get("evidence_id", "")), "")
        for row in _list_of_mappings(public.get("cited_evidence"))
    }
    cited_raw.discard("")
    retrieval = ranked_retrieval_scores(
        relevance=relevance,
        required_count=len(required),
        k=10,
    )
    citations = set_precision_recall(
        predicted={(value,) for value in cited_raw},
        required={(value,) for value in required},
        prefix="required_citation",
    )
    citations["citation_resolution_rate"] = 1.0 if public.get("cited_evidence_ids") else None
    metric_input = MetricInput(
        metric_id=case_id,
        population="diagnostic_challenge",
        dataset_family=family,
        source_kind="formal_diagnostic",
        qid=question.qid,
        scenario_id=question.scenario_id,
        question=str(public["question"]),
        reference_answer=str(public["reference_answer"]),
        candidate_answer=str(public["answer_text"]),
        retrieved_evidence=[
            EvidenceText(evidence_id=str(row["evidence_id"]), text=str(row["text"]))
            for row in _list_of_mappings(public.get("retrieved_evidence"))
        ],
        cited_evidence=[
            EvidenceText(evidence_id=str(row["evidence_id"]), text=str(row["text"]))
            for row in _list_of_mappings(public.get("cited_evidence"))
        ],
        retrieval_metrics=retrieval,
        citation_metrics=citations,
    )
    applicable = [
        str(field) for field in public.get("applicable_fields", []) if str(field) in JUDGED_FIELDS
    ]
    if "answer_correct" not in applicable:
        applicable.insert(0, "answer_correct")
    task_input = TaskJudgeInput(
        metric_id=case_id,
        population="diagnostic_challenge",
        dataset_family=family,
        source_kind="formal_diagnostic",
        qid=question.qid,
        scenario_id=question.scenario_id,
        question=str(public["question"]),
        reference_answer=str(public["reference_answer"]),
        candidate_answer=str(public["answer_text"]),
        cited_evidence_ids=[str(value) for value in public.get("cited_evidence_ids", [])],
        cited_evidence=public.get("cited_evidence", []),
        retrieved_evidence=public.get("retrieved_evidence", []),
        graph_paths=public.get("graph_paths", []),
        context_note=str(public.get("context_note", "")),
        applicable_fields=applicable,
        source_question_sha256=_text_hash(str(public["question"])),
        source_reference_answer_sha256=_text_hash(str(public["reference_answer"])),
        source_candidate_answer_sha256=_text_hash(str(public["answer_text"])),
        presentation_changed_fields=[],
        presentation_contract="formal_diagnostic_blind_public_payload_v1",
    )
    return DiagnosticCase(
        case_id=case_id,
        source_answer_id=source_answer_id,
        transformation=transformation,
        oracle_labels=oracle_labels,
        oracle_basis=oracle_basis,
        changed_components=changed_components,
        metric_input=metric_input,
        task_judge_input=task_input,
    )


def _matched_style_pair(
    context: _VariantContext,
    *,
    baseline_case: DiagnosticCase,
    candidate_case: DiagnosticCase,
    cases: dict[str, DiagnosticCase],
    internal_by_case: dict[str, dict[str, object]],
    seed: int,
) -> tuple[str, str]:
    variant_type = str(context.answer.variant_type)
    if (
        variant_type in _REFUSAL_TYPES
        or _hash_fraction(seed, "pair-style", context.answer.answer_id) >= 0.5
    ):
        return baseline_case.case_id, candidate_case.case_id
    styled_ids: list[str] = []
    for answer, canonical in (
        (context.baseline, baseline_case),
        (context.answer, candidate_case),
    ):
        internal = _copy(internal_by_case[canonical.case_id])
        internal["answer_text"] = _answer_wrapper(str(internal["answer_text"]))
        styled = _make_case(
            family=context.family,
            question=context.question,
            internal=internal,
            source_answer_id=answer.answer_id,
            transformation="matched_answer_wrapper",
            oracle_labels=canonical.oracle_labels,
            oracle_basis="Meaning-preserving answer wrapper applied symmetrically to both sides.",
            changed_components=["candidate_answer"],
            case_suffix=f"answer:{answer.answer_id}:wrapped",
        )
        cases.setdefault(styled.case_id, styled)
        internal_by_case.setdefault(styled.case_id, internal)
        styled_ids.append(styled.case_id)
    return styled_ids[0], styled_ids[1]


def _add_answer_invariance_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    baselines = _unique_baselines(contexts)
    eligible = [item for item in baselines if str(item.baseline.variant_type) != "correct_refusal"]
    for context in _balanced_sample(eligible, cap=cap, seed=seed, salt="answer_invariance"):
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        internal = _render_internal(context.bundle, context.baseline)
        internal_by_case.setdefault(canonical.case_id, internal)
        wrapped_internal = _copy(internal)
        wrapped_internal["answer_text"] = _answer_wrapper(str(wrapped_internal["answer_text"]))
        wrapped = _make_case(
            family=context.family,
            question=context.question,
            internal=wrapped_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="answer_wrapper_invariance",
            oracle_labels=canonical.oracle_labels,
            oracle_basis="The wrapper adds no proposition and preserves the complete answer text.",
            changed_components=["candidate_answer"],
            case_suffix=f"answer:{context.baseline.answer_id}:invariance-wrapper",
        )
        cases.setdefault(wrapped.case_id, wrapped)
        internal_by_case.setdefault(wrapped.case_id, wrapped_internal)
        pairs.append(
            _pair(
                construct="answer_correctness",
                phenomenon="meaning_preserving_answer_wrapper",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=wrapped.case_id,
                expected_relation="equal",
                target_fields=["answer_correct"],
                changed_components=["candidate_answer"],
                oracle_basis=(
                    "Deterministic semantics-preserving wrapper (CheckList invariance test)."
                ),
            )
        )


def _add_severity_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> int:
    by_question: defaultdict[tuple[str, str], list[_VariantContext]] = defaultdict(list)
    for context in contexts:
        by_question[(context.family, context.question.qid)].append(context)
    candidates = _severity_context_pairs(by_question, seed=seed)
    selected = _balanced_context_pair_sample(
        candidates,
        cap=cap,
        seed=seed,
        salt="severity_partial_incorrect",
    )
    for partial_context, incorrect_context in selected:
        baseline_internal = _render_internal(partial_context.bundle, partial_context.baseline)
        partial_internal = _copy(baseline_internal)
        partial_internal["answer_text"] = partial_context.answer.answer_text
        partial_internal["applicable_fields"] = ["answer_correct"]
        incorrect_internal = _copy(partial_internal)
        incorrect_internal["answer_text"] = incorrect_context.answer.answer_text

        partial_oracle = {field: "not_applicable" for field in _ORACLE_FIELD_MAP}
        partial_oracle["answer_correct"] = "partial"
        incorrect_oracle = dict(partial_oracle)
        incorrect_oracle["answer_correct"] = "no"
        partial = _make_case(
            family=partial_context.family,
            question=partial_context.question,
            internal=partial_internal,
            source_answer_id=partial_context.answer.answer_id,
            transformation="severity_partial_answer_only",
            oracle_labels=partial_oracle,
            oracle_basis=(
                "The validated source answer has answer_correct=partial; all displayed "
                "components except the candidate answer are held fixed."
            ),
            changed_components=["candidate_answer"],
            case_suffix=f"severity:{partial_context.answer.answer_id}:partial",
        )
        incorrect = _make_case(
            family=incorrect_context.family,
            question=incorrect_context.question,
            internal=incorrect_internal,
            source_answer_id=incorrect_context.answer.answer_id,
            transformation="severity_incorrect_answer_only",
            oracle_labels=incorrect_oracle,
            oracle_basis=(
                "The validated source answer has answer_correct=no; all displayed components "
                "except the candidate answer are held fixed."
            ),
            changed_components=["candidate_answer"],
            case_suffix=f"severity:{incorrect_context.answer.answer_id}:incorrect",
        )
        cases.setdefault(partial.case_id, partial)
        cases.setdefault(incorrect.case_id, incorrect)
        internal_by_case.setdefault(partial.case_id, partial_internal)
        internal_by_case.setdefault(incorrect.case_id, incorrect_internal)
        pairs.append(
            _pair(
                construct="answer_correctness",
                phenomenon="partial_vs_incorrect_severity",
                context=partial_context,
                left_case_id=partial.case_id,
                right_case_id=incorrect.case_id,
                expected_relation="left_higher",
                target_fields=["answer_correct"],
                changed_components=["candidate_answer"],
                severity_left=1,
                severity_right=2,
                oracle_basis=(
                    "Ordinal semantic oracle: for the same question and presentation context, "
                    "answer_correct=partial is better than answer_correct=no."
                ),
            )
        )
    return len(selected)


def _severity_context_pairs(
    by_question: dict[tuple[str, str], list[_VariantContext]],
    *,
    seed: int,
) -> list[tuple[_VariantContext, _VariantContext]]:
    """Choose one partial/no pair per question without consulting metric scores."""

    output: list[tuple[_VariantContext, _VariantContext]] = []
    for key, contexts in sorted(by_question.items()):
        partials = [
            context for context in contexts if str(context.answer.variant_type) == "partial_answer"
        ]
        incorrect = [
            context
            for context in contexts
            if str(context.answer.variant_type) in _SEVERITY_INCORRECT_TYPES
            and str(context.answer.answer_correct) == "no"
        ]
        if not partials or not incorrect:
            continue
        partial = min(
            partials,
            key=lambda context: _hash_key(
                seed, "severity-partial", *key, context.answer.answer_id
            ),
        )
        wrong = min(
            incorrect,
            key=lambda context: _hash_key(
                seed, "severity-incorrect", *key, context.answer.answer_id
            ),
        )
        output.append((partial, wrong))
    return output


def _add_inappropriate_refusal_decision_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    """Test refusing an answerable question against the appropriate answer decision.

    The production annotation protocol only asks the response-decision field for refusals. For
    this paired diagnostic, the supported answer is explicitly marked as an appropriate decision
    so the same construct can be scored on both sides without changing displayed content.
    """

    selected = _balanced_sample(
        contexts,
        cap=cap,
        seed=seed,
        salt="inappropriate_refusal_decision",
    )
    for context in selected:
        baseline_internal = _render_internal(context.bundle, context.baseline)
        decision_internal = _copy(baseline_internal)
        applicable = set(decision_internal.get("applicable_fields", []))
        applicable.add("response_decision_appropriate")
        decision_internal["applicable_fields"] = [
            field for field in JUDGED_FIELDS if field in applicable
        ]
        baseline_oracle = _answer_oracle(context.baseline)
        baseline_oracle["response_decision_appropriate"] = "yes"
        appropriate_answer = _make_case(
            family=context.family,
            question=context.question,
            internal=decision_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="answer_decision_applicability",
            oracle_labels=baseline_oracle,
            oracle_basis=(
                "The formal solver establishes that the displayed evidence supports the reference "
                "answer, so answering rather than refusing is the appropriate response decision."
            ),
            changed_components=["applicability_contract"],
            case_suffix=f"answer:{context.baseline.answer_id}:decision-applicable",
        )
        refusal = _canonical_case(context, answer=context.answer)
        cases.setdefault(appropriate_answer.case_id, appropriate_answer)
        cases.setdefault(refusal.case_id, refusal)
        internal_by_case.setdefault(appropriate_answer.case_id, decision_internal)
        internal_by_case.setdefault(
            refusal.case_id,
            _render_internal(context.bundle, context.answer),
        )
        pairs.append(
            _pair(
                construct="response_decision",
                phenomenon="refuse_when_answer_supported",
                context=context,
                left_case_id=appropriate_answer.case_id,
                right_case_id=refusal.case_id,
                expected_relation="left_higher",
                target_fields=["response_decision_appropriate"],
                changed_components=["candidate_answer", "response_decision"],
                oracle_basis=(
                    "A supported direct answer is an appropriate response decision; refusing the "
                    "same answerable question is not."
                ),
            )
        )


def _add_evidence_order_invariance_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    eligible: list[_VariantContext] = []
    for context in _unique_baselines(contexts):
        internal = _render_internal(context.bundle, context.baseline)
        if len(internal.get("retrieved_evidence", [])) > 1:
            eligible.append(context)
    for context in _balanced_sample(eligible, cap=cap, seed=seed, salt="evidence_order"):
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        internal = _render_internal(context.bundle, context.baseline)
        internal_by_case.setdefault(canonical.case_id, internal)
        reordered_internal = _copy(internal)
        reordered_internal["retrieved_evidence"] = list(
            reversed(reordered_internal.get("retrieved_evidence", []))
        )
        reordered_internal["cited_evidence"] = list(
            reversed(reordered_internal.get("cited_evidence", []))
        )
        reordered_internal["cited_evidence_ids"] = list(
            reversed(reordered_internal.get("cited_evidence_ids", []))
        )
        reordered_internal["graph_paths"] = list(
            reversed(reordered_internal.get("graph_paths", []))
        )
        reordered = _make_case(
            family=context.family,
            question=context.question,
            internal=reordered_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="evidence_order_reversal",
            oracle_labels=canonical.oracle_labels,
            oracle_basis="Only the presentation order of the same evidence and paths changes.",
            changed_components=["evidence_order", "path_order"],
            case_suffix=f"answer:{context.baseline.answer_id}:evidence-order",
        )
        cases.setdefault(reordered.case_id, reordered)
        internal_by_case.setdefault(reordered.case_id, reordered_internal)
        for construct, field in (
            ("answer_correctness", "answer_correct"),
            ("evidence_support", "evidence_supports_answer"),
            ("citation_correctness", "citation_temporally_valid"),
            ("graph_sufficiency", "graph_evidence_sufficient"),
        ):
            if field not in canonical.task_judge_input.applicable_fields:
                continue
            pairs.append(
                _pair(
                    construct=construct,
                    phenomenon="evidence_order_invariance",
                    context=context,
                    left_case_id=canonical.case_id,
                    right_case_id=reordered.case_id,
                    expected_relation="equal",
                    target_fields=[field],
                    changed_components=["evidence_order", "path_order"],
                    oracle_basis="Permutation invariance over an unchanged evidence set.",
                )
            )


def _add_irrelevant_evidence_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    candidates: list[tuple[_VariantContext, Fact]] = []
    for context in _unique_baselines(contexts):
        if str(context.baseline.variant_type) != "correct_supported":
            continue
        fact = _irrelevant_visible_fact(context)
        if fact is not None:
            candidates.append((context, fact))
    ordered = _hash_order(candidates, seed=seed, salt="irrelevant_evidence")[:cap]
    for context, fact in ordered:
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        internal = _render_internal(context.bundle, context.baseline)
        internal_by_case.setdefault(canonical.case_id, internal)
        noisy_internal = _copy(internal)
        noisy_internal["retrieved_evidence"] = [
            public_evidence(fact),
            *noisy_internal.get("retrieved_evidence", []),
        ]
        noisy = _make_case(
            family=context.family,
            question=context.question,
            internal=noisy_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="irrelevant_evidence_insertion",
            oracle_labels=canonical.oracle_labels,
            oracle_basis=(
                "The inserted visible fact is outside the question's required evidence set; "
                "the original complete support remains present."
            ),
            changed_components=["retrieved_evidence"],
            case_suffix=f"answer:{context.baseline.answer_id}:irrelevant-evidence",
        )
        cases.setdefault(noisy.case_id, noisy)
        internal_by_case.setdefault(noisy.case_id, noisy_internal)
        pairs.append(
            _pair(
                construct="evidence_support",
                phenomenon="irrelevant_evidence_robustness",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=noisy.case_id,
                expected_relation="equal",
                target_fields=["evidence_supports_answer"],
                changed_components=["retrieved_evidence"],
                oracle_basis="Support-preserving distractor insertion (CheckList invariance test).",
            )
        )
        pairs.append(
            _pair(
                construct="retrieval_quality",
                phenomenon="irrelevant_evidence_rank_penalty",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=noisy.case_id,
                expected_relation="left_higher",
                target_fields=["retrieval_ranking"],
                changed_components=["retrieved_evidence"],
                oracle_basis=(
                    "A non-required item is inserted at rank one while relevant ranks shift down."
                ),
            )
        )


def _add_wrong_evidence_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    eligible = [
        context
        for context in contexts
        if str(context.answer.variant_type)
        in {"stale_answer", "future_invalid_answer", "wrong_operator_answer"}
        and _cited_facts_exclude_gold(context)
    ]
    for context in _balanced_sample(eligible, cap=cap, seed=seed, salt="wrong_evidence"):
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        baseline_internal = _render_internal(context.bundle, context.baseline)
        bad_internal = _render_internal(context.bundle, context.answer)
        internal_by_case.setdefault(canonical.case_id, baseline_internal)
        wrong_internal = _copy(baseline_internal)
        wrong_internal["cited_evidence"] = _copy(bad_internal.get("cited_evidence", []))
        wrong_internal["cited_evidence_ids"] = list(bad_internal.get("cited_evidence_ids", []))
        wrong_internal["retrieved_evidence"] = _copy(bad_internal.get("cited_evidence", []))
        wrong_internal["graph_paths"] = []
        wrong_internal["context_note"] = ""
        applicable = {"answer_correct", "evidence_supports_answer"}
        wrong_internal["applicable_fields"] = [
            field for field in JUDGED_FIELDS if field in applicable
        ]
        oracle = dict(canonical.oracle_labels)
        oracle.update(
            {
                "evidence_supports_answer": "no",
                "temporal_correct": "not_applicable",
                "citation_temporally_valid": "not_applicable",
                "graph_evidence_sufficient": "not_applicable",
            }
        )
        wrong = _make_case(
            family=context.family,
            question=context.question,
            internal=wrong_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="wrong_evidence_swap",
            oracle_labels=oracle,
            oracle_basis=(
                "The candidate remains the gold answer, while every displayed cited fact has an "
                "answer entity disjoint from the gold answer and excludes required evidence."
            ),
            changed_components=["cited_evidence", "retrieved_evidence", "graph_paths"],
            case_suffix=f"answer:{context.baseline.answer_id}:wrong-evidence:{context.answer.answer_id}",
        )
        cases.setdefault(wrong.case_id, wrong)
        internal_by_case.setdefault(wrong.case_id, wrong_internal)
        pairs.append(
            _pair(
                construct="evidence_support",
                phenomenon="answer_preserved_wrong_evidence",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=wrong.case_id,
                expected_relation="left_higher",
                target_fields=["evidence_supports_answer"],
                changed_components=["cited_evidence", "retrieved_evidence", "graph_paths"],
                oracle_basis="Formal answer-entity disjointness and required-evidence exclusion.",
            )
        )


def _add_temporally_invalid_retrieval_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    """Hold semantic content fixed and invalidate only exposed temporal metadata."""

    eligible = []
    for context in _unique_baselines(contexts):
        if str(context.baseline.variant_type) != "correct_supported":
            continue
        internal = _render_internal(context.bundle, context.baseline)
        if not internal.get("retrieved_evidence"):
            continue
        invalid_interval = _invalid_interval_for_question(context.question)
        if invalid_interval is None:
            continue
        basis = str(context.question.program.temporal_basis)
        visible_rows = _list_of_mappings(internal.get("retrieved_evidence"))
        if basis in {"snapshot_observation", "document_revision"}:
            has_visible_time = any(row.get("publication_time") for row in visible_rows)
        else:
            has_visible_time = any(
                _mapping(row.get("valid_time")).get("start") for row in visible_rows
            )
        if has_visible_time:
            eligible.append((context, invalid_interval))

    selected = _hash_order(
        eligible,
        seed=seed,
        salt="temporally_invalid_retrieval",
    )[:cap]
    for context, invalid_interval in selected:
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        internal = _render_internal(context.bundle, context.baseline)
        internal_by_case.setdefault(canonical.case_id, internal)
        invalid_internal = _copy(internal)
        basis = str(context.question.program.temporal_basis)
        for field in ("retrieved_evidence", "cited_evidence"):
            for row in _list_of_mappings(invalid_internal.get(field)):
                _invalidate_visible_evidence_time(
                    row,
                    basis=basis,
                    invalid_interval=invalid_interval,
                )

        oracle = dict(canonical.oracle_labels)
        if invalid_internal.get("cited_evidence"):
            oracle["citation_temporally_valid"] = "no"
        if oracle.get("response_decision_appropriate") in _LABEL_SCORE:
            oracle["response_decision_appropriate"] = "no"
        invalid = _make_case(
            family=context.family,
            question=context.question,
            internal=invalid_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="temporally_invalid_retrieval_metadata",
            oracle_labels=oracle,
            oracle_basis=(
                "Candidate, evidence text, evidence IDs, and retrieval order are fixed; only "
                "visible valid-time or licensed revision-time metadata is moved outside the "
                "query's temporal constraint."
            ),
            changed_components=["evidence_time_metadata"],
            case_suffix=f"answer:{context.baseline.answer_id}:invalid-retrieval-time",
        )
        cases.setdefault(invalid.case_id, invalid)
        internal_by_case.setdefault(invalid.case_id, invalid_internal)
        pair_arguments = {
            "context": context,
            "left_case_id": canonical.case_id,
            "right_case_id": invalid.case_id,
            "changed_components": ["evidence_time_metadata"],
            "oracle_basis": (
                "A content-identical evidence list loses only query-compatible temporal "
                "metadata."
            ),
        }
        pairs.extend(
            [
                _pair(
                    construct="temporal_attribution",
                    phenomenon="semantically_supported_temporally_invalid_evidence",
                    expected_relation="left_higher",
                    target_fields=["temporal_attribution"],
                    **pair_arguments,
                ),
                _pair(
                    construct="retrieval_quality",
                    phenomenon="temporally_invalid_retrieval_metadata",
                    expected_relation="left_higher",
                    target_fields=["retrieval_temporal_validity"],
                    **pair_arguments,
                ),
                _pair(
                    construct="temporal_correctness",
                    phenomenon="evidence_time_answer_invariance",
                    expected_relation="equal",
                    target_fields=["temporal_correct"],
                    **pair_arguments,
                ),
                _pair(
                    construct="evidence_support",
                    phenomenon="evidence_time_semantic_invariance",
                    expected_relation="equal",
                    target_fields=["evidence_supports_answer"],
                    **pair_arguments,
                ),
            ]
        )


def _invalid_interval_for_question(question: Question) -> tuple[str, str] | None:
    query = question.program.query_time
    operator = str(question.program.operator)
    if operator == "first" or (operator in {"latest", "last"} and query.end is None):
        return None
    if operator in {"before", "previous", "expired", "latest", "last"}:
        anchor = query.end or query.start
        direction = 1
    else:
        anchor = query.start or query.end
        direction = -1
    if anchor is None:
        return None
    year = min(9998, anchor.year + 100) if direction > 0 else max(1, anchor.year - 100)
    start = date(year, 1, 1).isoformat()
    end = date(year, 12, 31).isoformat()
    return start, end


def _invalidate_visible_evidence_time(
    row: dict[str, object],
    *,
    basis: str,
    invalid_interval: tuple[str, str],
) -> None:
    start, end = invalid_interval
    if basis in {"snapshot_observation", "document_revision"}:
        row["publication_time"] = start
        return
    row["valid_time"] = {
        "type": "interval",
        "start": start,
        "end": end,
        "granularity": "day",
    }


def _add_graph_only_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    eligible = [
        context
        for context in contexts
        if _render_internal(context.bundle, context.answer).get("graph_paths")
    ]
    for context in _balanced_sample(eligible, cap=cap, seed=seed, salt="graph_only"):
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        baseline_internal = _render_internal(context.bundle, context.baseline)
        invalid_internal = _render_internal(context.bundle, context.answer)
        internal_by_case.setdefault(canonical.case_id, baseline_internal)
        graph_internal = _copy(baseline_internal)
        graph_internal["graph_paths"] = _copy(invalid_internal["graph_paths"])
        applicable = set(graph_internal.get("applicable_fields", []))
        applicable.add("graph_evidence_sufficient")
        graph_internal["applicable_fields"] = [
            field for field in JUDGED_FIELDS if field in applicable
        ]
        oracle = dict(canonical.oracle_labels)
        oracle["graph_evidence_sufficient"] = "no"
        graph_case = _make_case(
            family=context.family,
            question=context.question,
            internal=graph_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="invalid_graph_path_only",
            oracle_labels=oracle,
            oracle_basis=(
                "The answer and textual evidence are held fixed; only a validator-certified "
                "non-supporting graph path replaces the supporting path."
            ),
            changed_components=["graph_paths"],
            case_suffix=f"answer:{context.baseline.answer_id}:invalid-path:{context.answer.answer_id}",
        )
        cases.setdefault(graph_case.case_id, graph_case)
        internal_by_case.setdefault(graph_case.case_id, graph_internal)
        pairs.append(
            _pair(
                construct="graph_sufficiency",
                phenomenon="invalid_graph_path_only",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=graph_case.case_id,
                expected_relation="left_higher",
                target_fields=["graph_evidence_sufficient"],
                changed_components=["graph_paths"],
                oracle_basis="Graph-path support flag certified by dataset validation.",
            )
        )


def _add_graph_time_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    eligible = []
    for context in _unique_baselines(contexts):
        internal = _render_internal(context.bundle, context.baseline)
        if (
            context.question.program.temporal_basis == "world_valid_time"
            and internal.get("graph_paths")
            and context.baseline.graph_path_sufficient == "yes"
        ):
            eligible.append(context)
    selected = _balanced_sample(eligible, cap=cap, seed=seed, salt="graph_time_only")
    for context in selected:
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        internal = _render_internal(context.bundle, context.baseline)
        internal_by_case.setdefault(canonical.case_id, internal)
        invalid_internal = _copy(internal)
        for path in _list_of_mappings(invalid_internal.get("graph_paths")):
            for edge in _list_of_mappings(path.get("edges")):
                edge["valid_time"] = {
                    "schema_version": "3.0",
                    "type": "interval",
                    "start": "1000-01-01",
                    "end": "1000-12-31",
                    "granularity": "day",
                }
        applicable = set(invalid_internal.get("applicable_fields", []))
        applicable.add("graph_evidence_sufficient")
        invalid_internal["applicable_fields"] = [
            field for field in JUDGED_FIELDS if field in applicable
        ]
        oracle = dict(canonical.oracle_labels)
        oracle["graph_evidence_sufficient"] = "no"
        graph_case = _make_case(
            family=context.family,
            question=context.question,
            internal=invalid_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="temporally_invalid_graph_path_only",
            oracle_labels=oracle,
            oracle_basis=(
                "Answer, text, path topology, and endpoints are fixed; only every displayed "
                "edge interval is shifted outside the query time."
            ),
            changed_components=["graph_paths"],
            case_suffix=f"answer:{context.baseline.answer_id}:invalid-path-time",
        )
        cases.setdefault(graph_case.case_id, graph_case)
        internal_by_case.setdefault(graph_case.case_id, invalid_internal)
        pairs.append(
            _pair(
                construct="graph_sufficiency",
                phenomenon="temporally_invalid_graph_path_only",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=graph_case.case_id,
                expected_relation="left_higher",
                target_fields=["graph_evidence_sufficient"],
                changed_components=["graph_paths"],
                oracle_basis="Explicit edge intervals are incompatible with the query time.",
            )
        )


def _add_citation_omission_pairs(
    contexts: list[_VariantContext],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
) -> None:
    eligible = []
    for context in _unique_baselines(contexts):
        internal = _render_internal(context.bundle, context.baseline)
        if internal.get("cited_evidence") and context.question.required_valid_evidence_ids:
            eligible.append(context)
    for context in _balanced_sample(eligible, cap=cap, seed=seed, salt="citation_omission"):
        canonical = _canonical_case(context, answer=context.baseline)
        cases.setdefault(canonical.case_id, canonical)
        internal = _render_internal(context.bundle, context.baseline)
        internal_by_case.setdefault(canonical.case_id, internal)
        omitted_internal = _copy(internal)
        omitted_internal["cited_evidence"] = []
        omitted_internal["cited_evidence_ids"] = []
        applicable = set(omitted_internal.get("applicable_fields", []))
        applicable.discard("citation_temporally_valid")
        omitted_internal["applicable_fields"] = [
            field for field in JUDGED_FIELDS if field in applicable
        ]
        oracle = dict(canonical.oracle_labels)
        oracle["citation_temporally_valid"] = "not_applicable"
        omitted = _make_case(
            family=context.family,
            question=context.question,
            internal=omitted_internal,
            source_answer_id=context.baseline.answer_id,
            transformation="citation_omission",
            oracle_labels=oracle,
            oracle_basis=(
                "All required citation links are removed while answer and retrieved support stay "
                "fixed."
            ),
            changed_components=["cited_evidence"],
            case_suffix=f"answer:{context.baseline.answer_id}:citation-omission",
        )
        cases.setdefault(omitted.case_id, omitted)
        internal_by_case.setdefault(omitted.case_id, omitted_internal)
        pairs.append(
            _pair(
                construct="citation_correctness",
                phenomenon="required_citation_omission",
                context=context,
                left_case_id=canonical.case_id,
                right_case_id=omitted.case_id,
                expected_relation="left_higher",
                target_fields=["citation_completeness"],
                changed_components=["cited_evidence"],
                oracle_basis=(
                    "Required-citation set difference computed from the formal question program."
                ),
            )
        )


def _add_update_invariance_pairs(
    bundles: dict[str, DatasetBundle],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
    source_split: str,
) -> None:
    candidates: list[tuple[_VariantContext, _VariantContext]] = []
    for family, bundle in bundles.items():
        test_scenarios = set(bundle.splits[source_split])
        correct_by_qid = {
            answer.qid: answer
            for answer in bundle.answer_variants
            if answer.scenario_id in test_scenarios and str(answer.variant_type) in _BASELINE_TYPES
        }
        by_series: defaultdict[str, list[Question]] = defaultdict(list)
        for question in bundle.questions:
            if (
                question.scenario_id in test_scenarios
                and question.semantic_series_id
                and question.qid in correct_by_qid
            ):
                by_series[question.semantic_series_id].append(question)
        for questions in by_series.values():
            ordered = sorted(questions, key=lambda row: row.program.snapshot_id)
            for left, right in zip(ordered, ordered[1:], strict=False):
                if (
                    left.gold_answer_text != right.gold_answer_text
                    or left.should_abstain != right.should_abstain
                ):
                    continue
                left_context = _VariantContext(
                    family=family,
                    bundle=bundle,
                    question=left,
                    answer=correct_by_qid[left.qid],
                    baseline=correct_by_qid[left.qid],
                )
                right_context = _VariantContext(
                    family=family,
                    bundle=bundle,
                    question=right,
                    answer=correct_by_qid[right.qid],
                    baseline=correct_by_qid[right.qid],
                )
                candidates.append((left_context, right_context))
    for left_context, right_context in _hash_order(candidates, seed=seed, salt="update_invariance")[
        :cap
    ]:
        left = _canonical_case(left_context, answer=left_context.baseline)
        right = _canonical_case(right_context, answer=right_context.baseline)
        cases.setdefault(left.case_id, left)
        cases.setdefault(right.case_id, right)
        internal_by_case.setdefault(
            left.case_id,
            _render_internal(left_context.bundle, left_context.baseline),
        )
        internal_by_case.setdefault(
            right.case_id,
            _render_internal(right_context.bundle, right_context.baseline),
        )
        shared = _VariantContext(
            family=left_context.family,
            bundle=left_context.bundle,
            question=left_context.question,
            answer=right_context.answer,
            baseline=left_context.baseline,
        )
        pairs.append(
            _pair(
                construct="answer_correctness",
                phenomenon="answer_preserving_snapshot_update",
                context=shared,
                left_case_id=left.case_id,
                right_case_id=right.case_id,
                expected_relation="equal",
                target_fields=["answer_correct"],
                changed_components=["question_snapshot", "available_evidence"],
                oracle_basis=(
                    "Consecutive questions share a semantic series and identical formal gold "
                    "answer; "
                    "only the evidence snapshot changes."
                ),
            )
        )


def _add_update_sensitivity_pairs(
    bundles: dict[str, DatasetBundle],
    *,
    cases: dict[str, DiagnosticCase],
    pairs: list[DiagnosticPair],
    internal_by_case: dict[str, dict[str, object]],
    cap: int,
    seed: int,
    source_split: str,
) -> None:
    """Hold the candidate answer fixed across an update that changes the gold answer."""

    candidates: list[tuple[_VariantContext, _VariantContext]] = []
    for family, bundle in bundles.items():
        test_scenarios = set(bundle.splits[source_split])
        baseline_by_qid: dict[str, AnswerVariant] = {
            answer.qid: answer
            for answer in bundle.answer_variants
            if answer.scenario_id in test_scenarios
            and str(answer.variant_type) in _BASELINE_TYPES
        }
        by_series: defaultdict[str, list[Question]] = defaultdict(list)
        for question in bundle.questions:
            if (
                question.scenario_id in test_scenarios
                and question.semantic_series_id
                and baseline_by_qid.get(question.qid) is not None
            ):
                by_series[question.semantic_series_id].append(question)
        for questions in by_series.values():
            ordered = sorted(questions, key=lambda row: row.program.snapshot_id)
            for left, right in zip(ordered, ordered[1:], strict=False):
                if left.gold_answer_text == right.gold_answer_text:
                    continue
                left_answer = baseline_by_qid[left.qid]
                right_baseline = baseline_by_qid[right.qid]
                left_context = _VariantContext(
                    family=family,
                    bundle=bundle,
                    question=left,
                    answer=left_answer,
                    baseline=left_answer,
                )
                right_context = _VariantContext(
                    family=family,
                    bundle=bundle,
                    question=right,
                    answer=right_baseline,
                    baseline=right_baseline,
                )
                candidates.append((left_context, right_context))

    for left_context, right_context in _balanced_context_pair_sample(
        candidates,
        cap=cap,
        seed=seed,
        salt="update_sensitivity",
    ):
        left = _canonical_case(left_context, answer=left_context.answer)
        right_internal = _copy(
            _render_internal(right_context.bundle, right_context.baseline)
        )
        right_internal["answer_text"] = left.metric_input.candidate_answer
        applicable = {"answer_correct"}
        oracle = {field: "not_applicable" for field in _ORACLE_FIELD_MAP}
        oracle["answer_correct"] = "no"
        left_temporal = left.oracle_labels.get("temporal_correct")
        right_temporal = _answer_oracle(right_context.baseline).get("temporal_correct")
        if left_temporal in _LABEL_SCORE and right_temporal in _LABEL_SCORE:
            applicable.add("temporal_correct")
            oracle["temporal_correct"] = "no"
        if left_context.question.should_abstain != right_context.question.should_abstain:
            applicable.add("response_decision_appropriate")
            oracle["response_decision_appropriate"] = "no"
        right_internal["applicable_fields"] = [
            field for field in JUDGED_FIELDS if field in applicable
        ]
        right = _make_case(
            family=right_context.family,
            question=right_context.question,
            internal=right_internal,
            source_answer_id=left_context.answer.answer_id,
            transformation="answer_changing_snapshot_update",
            oracle_labels=oracle,
            oracle_basis=(
                "The formal solver changes the gold answer after the snapshot update, while the "
                "candidate is deterministically held at the earlier correct answer."
            ),
            changed_components=[
                "question_snapshot",
                "reference_answer",
                "available_evidence",
            ],
            case_suffix=(
                f"update:{left_context.question.qid}:candidate-on:{right_context.question.qid}"
            ),
        )
        if _normalized_text(left.metric_input.candidate_answer) != _normalized_text(
            right.metric_input.candidate_answer
        ):
            raise ValueError("Update-sensitivity pair failed to hold the candidate answer fixed")
        cases.setdefault(left.case_id, left)
        cases.setdefault(right.case_id, right)
        internal_by_case.setdefault(
            left.case_id,
            _render_internal(left_context.bundle, left_context.answer),
        )
        internal_by_case.setdefault(
            right.case_id,
            right_internal,
        )
        for construct, field in (
            ("answer_correctness", "answer_correct"),
            ("temporal_correctness", "temporal_correct"),
            ("response_decision", "response_decision_appropriate"),
        ):
            if not _labels_support_relation(
                left.oracle_labels.get(field),
                right.oracle_labels.get(field),
                relation="left_higher",
            ):
                continue
            pairs.append(
                _pair(
                    construct=construct,
                    phenomenon="answer_changing_snapshot_update",
                    context=right_context,
                    left_case_id=left.case_id,
                    right_case_id=right.case_id,
                    expected_relation="left_higher",
                    target_fields=[field],
                    changed_components=[
                        "question_snapshot",
                        "reference_answer",
                        "available_evidence",
                    ],
                    oracle_basis=(
                        "Consecutive questions share a semantic series. The candidate is held "
                        "fixed, while the formal gold answer changes after the snapshot update."
                    ),
                )
            )


def _pair(
    *,
    construct: str,
    phenomenon: str,
    context: _VariantContext,
    left_case_id: str,
    right_case_id: str,
    expected_relation: str,
    target_fields: list[str],
    changed_components: list[str],
    oracle_basis: str,
    severity_left: int | None = None,
    severity_right: int | None = None,
) -> DiagnosticPair:
    identity = "|".join([construct, phenomenon, left_case_id, right_case_id, expected_relation])
    pair_id = f"dp_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    return DiagnosticPair(
        pair_id=pair_id,
        test_type="directional" if expected_relation == "left_higher" else "invariance",
        target_construct=construct,
        phenomenon=phenomenon,
        dataset_family=context.family,
        qid=context.question.qid,
        scenario_id=context.question.scenario_id,
        left_case_id=left_case_id,
        right_case_id=right_case_id,
        expected_relation=expected_relation,
        target_fields=target_fields,
        changed_components=changed_components,
        severity_left=severity_left,
        severity_right=severity_right,
        oracle_basis=oracle_basis,
    )


def _answer_oracle(answer: AnswerVariant) -> dict[str, str]:
    return {public: str(getattr(answer, source)) for public, source in _ORACLE_FIELD_MAP.items()}


def _labels_support_relation(
    left: str | None,
    right: str | None,
    *,
    relation: str,
) -> bool:
    if left not in _LABEL_SCORE or right not in _LABEL_SCORE:
        return False
    if relation == "left_higher":
        return _LABEL_SCORE[left] > _LABEL_SCORE[right]
    return _LABEL_SCORE[left] == _LABEL_SCORE[right]


def _changed_components(
    left: dict[str, object],
    right: dict[str, object],
) -> list[str]:
    components = {
        "question": "question",
        "reference_answer": "reference_answer",
        "answer_text": "candidate_answer",
        "cited_evidence": "cited_evidence",
        "retrieved_evidence": "retrieved_evidence",
        "graph_paths": "graph_paths",
        "context_note": "context_note",
    }
    return [
        public for source, public in components.items() if left.get(source) != right.get(source)
    ]


def _balanced_sample(
    rows: list[_VariantContext],
    *,
    cap: int,
    seed: int,
    salt: str,
) -> list[_VariantContext]:
    by_family: defaultdict[str, list[_VariantContext]] = defaultdict(list)
    for row in rows:
        by_family[row.family].append(row)
    for family, values in by_family.items():
        by_family[family] = sorted(
            values,
            key=lambda row: _hash_key(seed, salt, family, row.answer.answer_id),
        )
    selected: list[_VariantContext] = []
    families = sorted(by_family)
    while len(selected) < cap and any(by_family.values()):
        for family in families:
            if by_family[family] and len(selected) < cap:
                selected.append(by_family[family].pop(0))
    return selected


def _balanced_context_pair_sample(
    rows: list[tuple[_VariantContext, _VariantContext]],
    *,
    cap: int,
    seed: int,
    salt: str,
) -> list[tuple[_VariantContext, _VariantContext]]:
    by_family: defaultdict[str, list[tuple[_VariantContext, _VariantContext]]] = defaultdict(list)
    for row in rows:
        by_family[row[0].family].append(row)
    for family, values in by_family.items():
        by_family[family] = sorted(
            values,
            key=lambda row: _hash_key(
                seed,
                salt,
                family,
                row[0].question.qid,
                row[1].question.qid,
            ),
        )
    selected: list[tuple[_VariantContext, _VariantContext]] = []
    families = sorted(by_family)
    while len(selected) < cap and any(by_family.values()):
        for family in families:
            if by_family[family] and len(selected) < cap:
                selected.append(by_family[family].pop(0))
    return selected


def _unique_baselines(contexts: list[_VariantContext]) -> list[_VariantContext]:
    output: dict[tuple[str, str], _VariantContext] = {}
    for context in contexts:
        output.setdefault((context.family, context.baseline.answer_id), context)
    return list(output.values())


def _irrelevant_visible_fact(context: _VariantContext) -> Fact | None:
    internal = _render_internal(context.bundle, context.baseline)
    used = {
        str(row.get("evidence_id", ""))
        for key in ("cited_evidence", "retrieved_evidence")
        for row in _list_of_mappings(internal.get(key))
    }
    required = set(context.question.required_valid_evidence_ids)
    candidates = [
        fact
        for fact in context.bundle.facts
        if fact.scenario_id == context.question.scenario_id
        and fact.fact_id not in used | required
        and fact_visible(fact, context.question.program.snapshot_id)
    ]
    return sorted(candidates, key=lambda fact: fact.fact_id)[0] if candidates else None


def _cited_facts_exclude_gold(context: _VariantContext) -> bool:
    if not context.answer.cited_evidence_ids:
        return False
    fact_by_id = {fact.fact_id: fact for fact in context.bundle.facts}
    cited = [fact_by_id.get(fact_id) for fact_id in context.answer.cited_evidence_ids]
    if any(fact is None for fact in cited):
        return False
    gold = set(context.question.gold_answer_entity_ids)
    return all(
        fact.fact_id not in context.question.required_valid_evidence_ids
        and fact_answer_id(fact) not in gold
        for fact in cited
        if fact is not None
    )


def _hash_order(values: list[object], *, seed: int, salt: str) -> list[object]:
    return sorted(values, key=lambda value: _hash_key(seed, salt, _stable_identity(value)))


def _stable_identity(value: object) -> str:
    if isinstance(value, _VariantContext):
        return f"{value.family}:{value.question.qid}:{value.answer.answer_id}"
    if isinstance(value, Fact):
        return value.fact_id
    if isinstance(value, tuple):
        return "|".join(_stable_identity(item) for item in value)
    return str(value)


def _hash_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()


def _hash_fraction(seed: int, *parts: str) -> float:
    return int(_hash_key(seed, *parts)[:12], 16) / float(16**12)


def _answer_wrapper(answer: str) -> str:
    return f"Based on the available evidence, the answer is {answer.rstrip('.')}."


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split()).rstrip(".")


def _validate_pair_isolation(cases: list[DiagnosticCase], pairs: list[DiagnosticPair]) -> None:
    by_id = {case.case_id: case for case in cases}
    for pair in pairs:
        left = by_id[pair.left_case_id]
        right = by_id[pair.right_case_id]
        if pair.expected_relation == "equal":
            for field in pair.target_fields:
                if (
                    field in left.oracle_labels
                    and field in right.oracle_labels
                    and left.oracle_labels[field] != right.oracle_labels[field]
                ):
                    raise ValueError(f"Invariance oracle changed for {pair.pair_id}: {field}")
        if pair.expected_relation == "left_higher":
            for field in pair.target_fields:
                if field not in left.oracle_labels or field not in right.oracle_labels:
                    continue
                left_label = left.oracle_labels[field]
                right_label = right.oracle_labels[field]
                if (
                    left_label in _LABEL_SCORE
                    and right_label in _LABEL_SCORE
                    and _LABEL_SCORE[left_label] <= _LABEL_SCORE[right_label]
                ):
                    raise ValueError(f"Directional oracle is not ordered: {pair.pair_id}")


def _render_internal(bundle: DatasetBundle, answer: AnswerVariant) -> dict[str, object]:
    key = (id(bundle), answer.answer_id)
    if key not in _INTERNAL_RENDER_CACHE:
        _INTERNAL_RENDER_CACHE[key] = controlled_internal_unit(bundle, answer)
    return _INTERNAL_RENDER_CACHE[key]


def _copy(value: object) -> object:
    return orjson.loads(orjson.dumps(value))


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return value


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
