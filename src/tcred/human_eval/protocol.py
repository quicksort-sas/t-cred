from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "4.2"

LabelValue = Literal["yes", "partial", "no", "not_applicable", "unjudgeable"]
WorkflowStage = Literal["evidence_review", "answer_review"]

CATEGORICAL_FIELDS = (
    "answer_correct",
    "temporal_correct",
    "evidence_supports_answer",
    "citation_temporally_valid",
    "graph_evidence_sufficient",
    "response_decision_appropriate",
)
EVIDENCE_STAGE_FIELDS = (
    "temporal_correct",
    "evidence_supports_answer",
    "citation_temporally_valid",
    "graph_evidence_sufficient",
)
ANSWER_STAGE_FIELDS = ("answer_correct", "response_decision_appropriate")
VISIBLE_LABEL_OPTIONS = ("yes", "partial", "no", "unjudgeable")
REASON_REQUIRED_LABELS = {"partial", "unjudgeable"}
ORDINAL_LABEL_VALUES = {"yes": 0, "partial": 1, "no": 2}


class ReasonOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    label: str


class FieldSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    prompt: str
    scope_note: str
    stage: WorkflowStage
    definitions: dict[str, str]
    reasons: dict[str, tuple[ReasonOption, ...]] = Field(default_factory=dict)
    decision_notes: tuple[str, ...] = ()


class IssueOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    label: str
    requires_detail: bool = False


class AnnotationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: dict[str, str] = Field(default_factory=dict)
    reasons: dict[str, str] = Field(default_factory=dict)
    issue_code: str = ""
    issue_detail: str = ""
    comment: str = ""
    workflow_stage: WorkflowStage = "evidence_review"


FIELD_SPECS: dict[str, FieldSpec] = {
    "answer_correct": FieldSpec(
        label="Answer correct",
        prompt="Does the candidate fully answer the exact question?",
        scope_note=(
            "Compare the candidate with the revealed reference. Judge every material part of the "
            "answer, including required list items, dates, qualifications, and material extra "
            "claims. Yes permits no material false claim. Partial requires a genuinely correct "
            "material answer component; merely mentioning the reference as a rejected alternative "
            "does not qualify."
        ),
        stage="answer_review",
        definitions={
            "yes": "All material answer content is correct and complete for the question.",
            "partial": (
                "The displayed information is sufficient to decide the answer, and a meaningful "
                "answer component is correct, but a material part is missing, mixed, or slightly "
                "wrong."
            ),
            "no": (
                "The central answer is wrong, unsupported, contradictory, or answers another "
                "question."
            ),
            "unjudgeable": (
                "The reference and displayed material are insufficient, ambiguous, or "
                "conflicting, so correctness cannot be decided."
            ),
        },
        reasons={
            "partial": (
                ReasonOption(code="incomplete_answer", label="A required answer part is missing"),
                ReasonOption(
                    code="mixed_correct_incorrect",
                    label="The answer mixes correct and incorrect material",
                ),
                ReasonOption(
                    code="minor_nonfatal_error", label="A minor error prevents full correctness"
                ),
            ),
            "unjudgeable": (
                ReasonOption(
                    code="ambiguous_question",
                    label="The question has more than one plausible reading",
                ),
                ReasonOption(
                    code="reference_evidence_conflict",
                    label="The reference conflicts with the displayed evidence",
                ),
                ReasonOption(
                    code="insufficient_displayed_information",
                    label="The displayed material is insufficient",
                ),
            ),
        },
        decision_notes=(
            (
                "Score the requested result, not merely correct setup or intermediate facts. "
                "Correctly identifying the subject while withholding the requested value does "
                "not earn Partial."
            ),
            (
                "A refusal that supplies none of the requested answer is No here, even when "
                "refusing was appropriate for the shown context. A genuinely correct subset of "
                "a requested list can still be Partial."
            ),
        ),
    ),
    "temporal_correct": FieldSpec(
        label="Temporal correct",
        prompt="Does the candidate respect the requested date, interval, or temporal operator?",
        scope_note=(
            "Judge the timing of claims the candidate actually makes. Timing attached only to a "
            "different person, entity, relation, or answer cannot establish the candidate's "
            "temporal validity. A missing answer item is handled by Answer correct after the "
            "reference is revealed; it does not by itself make this field Partial."
        ),
        stage="evidence_review",
        definitions={
            "yes": (
                "Every material answer claim is valid for the requested time or temporal operator."
            ),
            "partial": (
                "The displayed times are sufficient to decide the field, and only part of the "
                "answer's time scope or set of claims is temporally valid."
            ),
            "no": "A material claim is stale, future-invalid, out of range, or in the wrong order.",
            "unjudgeable": (
                "Temporal validity applies, but required dates are missing, ambiguous, or "
                "conflicting, so it cannot be decided."
            ),
        },
        reasons={
            "partial": (
                ReasonOption(
                    code="partial_time_coverage",
                    label="Only part of the requested interval is covered",
                ),
                ReasonOption(
                    code="vague_time_scope",
                    label="The candidate's time scope is incomplete or vague",
                ),
                ReasonOption(
                    code="mixed_temporal_validity",
                    label="Some answer claims are valid and others are not",
                ),
            ),
            "unjudgeable": (
                ReasonOption(
                    code="missing_valid_time", label="Required valid-time information is missing"
                ),
                ReasonOption(
                    code="ambiguous_time_expression", label="The time expression is ambiguous"
                ),
                ReasonOption(
                    code="conflicting_time_information",
                    label="Displayed time information conflicts",
                ),
                ReasonOption(
                    code="candidate_time_not_established",
                    label="Displayed timing does not apply to the candidate claim",
                ),
            ),
        },
        decision_notes=(
            "Point and closed-range boundaries are inclusive: start <= date <= end.",
            (
                "\"At any time\" requires any overlap with the requested interval; "
                "\"throughout\" requires coverage of the whole interval."
            ),
            (
                "\"First after\" and \"most recent before\" use strict cutoffs. "
                "\"By\" includes the cutoff."
            ),
            (
                "When several relationships tie for first, last, previous, or next, all tied "
                "items count."
            ),
            (
                "Tied items may appear in arbitrary textual order without implying a sequence. "
                "But ordering words such as \"then\" or \"followed by\" assert a sequence; if the "
                "items have the same timestamp, score that temporal assertion as wrong."
            ),
        ),
    ),
    "evidence_supports_answer": FieldSpec(
        label="Evidence supports claim content",
        prompt=(
            "If all dates and validity intervals were hidden, would the evidence text support the "
            "candidate's factual claims?"
        ),
        scope_note=(
            "This is a content-only check. Deliberately ignore whether the evidence is valid at "
            "the question's date. Stale evidence may support claim content while Temporal correct "
            "and Citation time coverage are No. If hiding dates makes mutually exclusive central "
            "claims impossible to resolve, choose Cannot judge. Judge claims the candidate states; "
            "an omitted reference item is assessed later under Answer correct."
        ),
        stage="evidence_review",
        definitions={
            "yes": "All central candidate claims are directly supported by displayed evidence.",
            "partial": (
                "The evidence is sufficient to decide support and directly supports some, but not "
                "all, material candidate claims."
            ),
            "no": (
                "The central claim is unsupported, irrelevant to the evidence, or contradicted "
                "by it."
            ),
            "unjudgeable": (
                "The evidence is missing, incomplete, unreadable, ambiguous, or conflicting, so "
                "the claim-evidence relationship cannot be decided."
            ),
        },
        reasons={
            "partial": (
                ReasonOption(
                    code="some_claims_supported", label="Only some material claims are supported"
                ),
                ReasonOption(
                    code="unstated_inference_required",
                    label="Support requires an unstated or uncertain inference",
                ),
                ReasonOption(
                    code="secondary_claim_unsupported",
                    label="A secondary material claim lacks support",
                ),
            ),
            "unjudgeable": (
                ReasonOption(
                    code="evidence_text_incomplete",
                    label="The evidence text is incomplete or unclear",
                ),
                ReasonOption(code="evidence_conflict", label="Displayed evidence items conflict"),
                ReasonOption(
                    code="unclear_claim_link",
                    label="The link between claim and evidence is unclear",
                ),
            ),
        },
        decision_notes=(
            (
                "Do not lower this field merely because the candidate omits another answer item. "
                "Use Partial only when a material claim it actually makes has only partial support."
            ),
            "A supported claim can still be stale; temporal fields handle dates separately.",
        ),
    ),
    "citation_temporally_valid": FieldSpec(
        label="Citation time coverage",
        prompt="Do the cited evidence intervals cover the time required by the question?",
        scope_note=(
            "Inspect only evidence marked as cited and judge only its displayed interval. This "
            "field does not ask whether the citation's words, entities, or relation support the "
            "candidate; Evidence supports claim content handles that separately."
        ),
        stage="evidence_review",
        definitions={
            "yes": (
                "Every displayed cited interval covers the requested or claimed time."
            ),
            "partial": (
                "The displayed citation times are sufficient to decide the field, and their "
                "validity or interval coverage is mixed or incomplete."
            ),
            "no": (
                "A displayed cited interval is stale, future-invalid, or outside the required "
                "time."
            ),
            "unjudgeable": (
                "A citation exists, but required timing is missing, ambiguous, or conflicting, so "
                "its time coverage cannot be decided."
            ),
        },
        reasons={
            "partial": (
                ReasonOption(
                    code="mixed_citation_validity",
                    label="Some citations are valid and others are not",
                ),
                ReasonOption(
                    code="partial_interval_coverage",
                    label="The citations cover only part of the interval",
                ),
                ReasonOption(
                    code="citation_time_imprecise",
                    label="Citation time is too imprecise for a full judgment",
                ),
            ),
            "unjudgeable": (
                ReasonOption(
                    code="citation_valid_time_missing", label="The citation's valid time is missing"
                ),
                ReasonOption(
                    code="citation_identity_unresolved",
                    label="A cited evidence identifier is unresolved",
                ),
                ReasonOption(
                    code="conflicting_citation_times",
                    label="Cited evidence has conflicting time information",
                ),
            ),
        },
        decision_notes=(
            (
                "A citation can have Yes time coverage while Evidence supports claim content is "
                "No because the cited interval is in range but its content is irrelevant."
            ),
            (
                "Do not penalize an omitted answer item here. Judge the intervals of citations "
                "the candidate supplied."
            ),
        ),
    ),
    "graph_evidence_sufficient": FieldSpec(
        label="Graph evidence sufficient",
        prompt=(
            "Does the displayed path set collectively support the answer with correct nodes, "
            "directions, relations, and times?"
        ),
        scope_note=(
            "Judge all displayed candidate paths together, independently of the evidence prose. "
            "Yes requires at least one complete connection, or a combination that supplies every "
            "connection required for the candidate's stated claims. Irrelevant extra paths do not "
            "cancel sufficient graph evidence. An answer item omitted from the candidate is "
            "assessed later under Answer correct."
        ),
        stage="evidence_review",
        definitions={
            "yes": (
                "The displayed path set collectively contains every material connection needed "
                "for the candidate answer."
            ),
            "partial": (
                "The path set is readable enough to decide and supports a meaningful subset of "
                "the candidate's claims, but a required connection or answer branch is missing."
            ),
            "no": (
                "The readable path set contains no sufficient connection, or contains only "
                "materially wrong nodes, relations, directions, or times."
            ),
            "unjudgeable": (
                "Graph evidence is shown, but unreadable, incomplete, ambiguous, or conflicting "
                "rendering prevents a reliable set-level decision."
            ),
        },
        reasons={
            "partial": (
                ReasonOption(
                    code="incomplete_path_set",
                    label="A required connection or answer branch is missing",
                ),
                ReasonOption(
                    code="some_subclaims_connected",
                    label="The path supports only some answer claims",
                ),
                ReasonOption(
                    code="secondary_connection_missing",
                    label="A secondary required connection is missing",
                ),
            ),
            "unjudgeable": (
                ReasonOption(code="edge_time_missing", label="A required edge time is missing"),
                ReasonOption(
                    code="direction_or_relation_unclear",
                    label="Edge direction or relation is unclear",
                ),
                ReasonOption(
                    code="graph_rendering_problem", label="The graph cannot be read reliably"
                ),
            ),
        },
        decision_notes=(
            (
                "Follow arrow direction, relation label, endpoints, and edge time. Correct prose "
                "does not repair an incorrect graph path."
            ),
            (
                "Use Partial only when the path set supports a meaningful subset of claims the "
                "candidate actually states."
            ),
        ),
    ),
    "response_decision_appropriate": FieldSpec(
        label="Answer/refusal decision appropriate",
        prompt=(
            "Was the candidate right to answer rather than refuse, or to refuse rather than "
            "answer?"
        ),
        scope_note=(
            "Judge the response decision against the displayed material. This field appears for "
            "refusals and hybrid answer/refusal responses, and for every response to a question "
            "whose correct behavior may be to abstain."
        ),
        stage="answer_review",
        definitions={
            "yes": (
                "The candidate made the correct decision to answer or refuse given the displayed "
                "information."
            ),
            "partial": (
                "The displayed information is sufficient to decide the field, and the caveat or "
                "decision is only partly justified or is mixed with a materially overconfident "
                "claim."
            ),
            "no": (
                "The candidate refuses despite sufficient information, or gives an answer when "
                "the displayed information requires refusal."
            ),
            "unjudgeable": (
                "The displayed material is insufficient, ambiguous, or conflicting, so whether "
                "the answer/refusal decision was appropriate cannot be decided."
            ),
        },
        reasons={
            "partial": (
                ReasonOption(
                    code="mixed_answer_and_refusal",
                    label="The candidate mixes an answer with a refusal",
                ),
                ReasonOption(
                    code="response_decision_partly_justified",
                    label="Only part of the answer/refusal decision is justified",
                ),
                ReasonOption(
                    code="caveat_partly_justified", label="The caveat is reasonable but overstated"
                ),
            ),
            "unjudgeable": (
                ReasonOption(
                    code="evidence_sufficiency_unclear", label="Evidence sufficiency is unclear"
                ),
                ReasonOption(
                    code="contradictory_information", label="Available information is contradictory"
                ),
                ReasonOption(
                    code="response_decision_basis_unclear",
                    label="The basis for answering or refusing is unclear",
                ),
            ),
        },
        decision_notes=(
            (
                "Answer correctness and decision appropriateness can differ. A refusal may be "
                "appropriate for the shown candidate context even when the revealed reference "
                "contains an answer."
            ),
            (
                "In that case, the refusal can receive Answer correct = No and decision = Yes: "
                "it does not provide the requested result, but it correctly avoids inventing one "
                "from insufficient shown material."
            ),
            (
                "Conversely, refusing is inappropriate when the displayed candidate context "
                "already establishes an answer."
            ),
        ),
    ),
}

ISSUE_OPTIONS = (
    IssueOption(code="ambiguous_question", label="Question wording is ambiguous"),
    IssueOption(code="reference_evidence_conflict", label="Reference conflicts with evidence"),
    IssueOption(
        code="missing_required_content", label="Required text, evidence, or path content is missing"
    ),
    IssueOption(code="unreadable_graph", label="Graph display is unreadable"),
    IssueOption(
        code="other_data_problem", label="Other data or display problem", requires_detail=True
    ),
)


def annotation_fields_manifest() -> dict[str, str]:
    return {
        **{field: "yes|partial|no|not_applicable|unjudgeable" for field in CATEGORICAL_FIELDS},
        "reason_codes": "required for partial and unjudgeable labels",
        "issue_code": "optional structured data/display problem",
        "comment": "optional free text",
    }


def protocol_payload() -> dict[str, object]:
    return {
        "version": PROTOCOL_VERSION,
        "label_decision_rule": {
            "question": "Can the displayed material decide this field?",
            "unjudgeable": "No: choose Cannot judge.",
            "yes": "Yes, and the requirement is fully satisfied: choose Yes.",
            "partial": (
                "Yes, and a meaningful part is satisfied but a material part is missing or mixed: "
                "choose Partial."
            ),
            "no": "Yes, and the central requirement fails: choose No.",
        },
        "visible_label_options": list(VISIBLE_LABEL_OPTIONS),
        "evidence_stage_fields": list(EVIDENCE_STAGE_FIELDS),
        "answer_stage_fields": list(ANSWER_STAGE_FIELDS),
        "fields": {name: spec.model_dump(mode="json") for name, spec in FIELD_SPECS.items()},
        "issue_options": [option.model_dump(mode="json") for option in ISSUE_OPTIONS],
    }


def applicable_fields(row: dict[str, object]) -> tuple[str, ...]:
    declared = row.get("applicable_fields")
    if isinstance(declared, list):
        fields = tuple(str(field) for field in declared if str(field) in CATEGORICAL_FIELDS)
        if fields:
            return fields
    return CATEGORICAL_FIELDS


def fields_for_stage(row: dict[str, object], stage: WorkflowStage) -> tuple[str, ...]:
    allowed = EVIDENCE_STAGE_FIELDS if stage == "evidence_review" else ANSWER_STAGE_FIELDS
    applicable = set(applicable_fields(row))
    return tuple(field for field in allowed if field in applicable)


def blank_state(row: dict[str, object]) -> AnnotationState:
    applicable = set(applicable_fields(row))
    labels = {
        field: "" if field in applicable else "not_applicable" for field in CATEGORICAL_FIELDS
    }
    return AnnotationState(labels=labels)


def reason_required(label: str) -> bool:
    return label in REASON_REQUIRED_LABELS


def reason_allowed(field: str, label: str, reason: str) -> bool:
    return any(option.code == reason for option in FIELD_SPECS[field].reasons.get(label, ()))


def issue_allowed(code: str) -> bool:
    return not code or any(option.code == code for option in ISSUE_OPTIONS)


def issue_requires_detail(code: str) -> bool:
    return any(option.code == code and option.requires_detail for option in ISSUE_OPTIONS)


def label_distance(left: str, right: str) -> float:
    """Return the preregistered ordinal disagreement distance for human labels."""
    if left == right:
        return 0.0
    if left in ORDINAL_LABEL_VALUES and right in ORDINAL_LABEL_VALUES:
        return ((ORDINAL_LABEL_VALUES[left] - ORDINAL_LABEL_VALUES[right]) / 2) ** 2
    return 1.0


def stage_complete(row: dict[str, object], state: AnnotationState, stage: WorkflowStage) -> bool:
    return all(_field_complete(field, state) for field in fields_for_stage(row, stage))


def annotation_complete(row: dict[str, object], state: AnnotationState) -> bool:
    return (
        state.workflow_stage == "answer_review"
        and stage_complete(row, state, "evidence_review")
        and stage_complete(row, state, "answer_review")
        and (not issue_requires_detail(state.issue_code) or bool(state.issue_detail.strip()))
        and not logical_conflicts(row, state)
    )


def logical_conflicts(
    row: dict[str, object],
    state: AnnotationState,
) -> tuple[dict[str, object], ...]:
    applicable = set(applicable_fields(row))
    conflicts: list[dict[str, object]] = []
    if (
        {"answer_correct", "temporal_correct"} <= applicable
        and state.labels.get("answer_correct") == "yes"
        and state.labels.get("temporal_correct") == "no"
    ):
        conflicts.append(
            {
                "code": "fully_correct_but_temporally_incorrect",
                "fields": ["answer_correct", "temporal_correct"],
                "message": (
                    "Answer correct is Yes while Temporal correct is No. A fully correct answer "
                    "to this time-specific question cannot also be temporally incorrect."
                ),
            }
        )
    return tuple(conflicts)


def _field_complete(field: str, state: AnnotationState) -> bool:
    label = state.labels.get(field, "")
    if label not in VISIBLE_LABEL_OPTIONS:
        return False
    if reason_required(label):
        return reason_allowed(field, label, state.reasons.get(field, ""))
    return True
