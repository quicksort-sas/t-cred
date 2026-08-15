from __future__ import annotations

import math
import re
from calendar import monthrange
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from tcred.metrics.task_judge_models import JudgeEvidence, VisibleInterval
from tcred.metrics.tcred_models import (
    EvidenceTemporalAssessment,
    NormalizedTemporalQuery,
)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_PATTERN = "|".join(_MONTHS)
_LONG_DATE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_FIRST_DATE = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTH_PATTERN})\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_MONTH_YEAR = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
_YEAR = re.compile(r"(?<![\d-])(?P<year>(?:1[5-9]|20)\d{2})(?![\d-])")
_SNAPSHOT_DATE = re.compile(
    rf"\b(?:evidence|wikidata)\s+snapshot\s+dated\s+"
    rf"(?P<date>(?:{_MONTH_PATTERN})\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})",
    re.IGNORECASE,
)
_DOCUMENT_DATE = re.compile(
    rf"\bdocument\s+revision\b.*?\bdated\s+"
    rf"(?P<date>(?:{_MONTH_PATTERN})\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _DateMention:
    start: int
    end: int
    lower: date
    upper: date
    granularity: str


@dataclass(frozen=True)
class ClaimTimeConstraint:
    status: str
    mode: str
    start: date | None = None
    end: date | None = None
    granularity: str = "unknown"
    explanation: str = ""


_PARENTHETICAL = re.compile(r"\(([^()]*)\)")
_RANGE_SEPARATOR = re.compile(r"\b(?:to|through|until)\b|[\u2013\u2014]", re.I)
_START_BOUNDARY = re.compile(r"\b(?:starting(?:\s+on)?|began(?:\s+on)?|since)\b", re.I)
_POINT_CUE = re.compile(r"\b(?:on|as\s+of)\b", re.I)
_RELATION_TIME_CUE = re.compile(
    r"\b(?:served|held|employed|affiliated|member|effective|active|appointed|began|ended)\b",
    re.I,
)
_BARE_TEMPORAL_VALUE = re.compile(
    rf"^\s*(?:"
    rf"\d{{4}}-\d{{2}}-\d{{2}}|"
    rf"(?:{_MONTH_PATTERN})\s+\d{{1,2}},?\s+\d{{4}}|"
    rf"\d{{1,2}}\s+(?:{_MONTH_PATTERN})\s+\d{{4}}|"
    rf"(?:{_MONTH_PATTERN})\s+\d{{4}}|"
    rf"(?:1[5-9]|20)\d{{2}}"
    rf")\s*[.]?\s*$",
    re.IGNORECASE,
)


def normalize_temporal_query(question: str) -> NormalizedTemporalQuery:
    """Parse the temporal contract visible in an English T-CRED question.

    This parser deliberately does not read the private QuestionProgram. Snapshot and document
    anchors are handled before world-valid-time dates so a retrieval snapshot cannot be mistaken
    for the date about which the question asks.
    """

    normalized = " ".join(question.split())
    lowered = normalized.casefold()

    document = _DOCUMENT_DATE.search(normalized)
    if document:
        mention = _single_date_mention(document.group("date"))
        return _anchored_query(
            mention,
            operator="as_of",
            basis="document_revision",
            explanation="The question explicitly targets a dated document revision.",
        )

    snapshot = _SNAPSHOT_DATE.search(normalized)
    is_world_snapshot = snapshot is not None and "evidence snapshot" in snapshot.group(0).casefold()
    if snapshot and not is_world_snapshot:
        mention = _single_date_mention(snapshot.group("date"))
        return _anchored_query(
            mention,
            operator="as_of",
            basis="snapshot_observation",
            explanation="The question explicitly targets a dated Wikidata snapshot.",
        )

    world_text = normalized
    if snapshot and is_world_snapshot:
        world_text = f"{normalized[: snapshot.start()]} {normalized[snapshot.end() :]}"
    mentions = _date_mentions(world_text)
    operator = _world_operator(lowered)

    if operator == "not_temporal":
        return NormalizedTemporalQuery(
            status="unparseable",
            operator=operator,
            basis="not_temporal",
            explanation="No supported temporal operator or anchor was found.",
        )
    if operator == "ambiguous":
        return NormalizedTemporalQuery(
            status="ambiguous_temporal_operator",
            operator=operator,
            basis="world_valid_time",
            explanation="The question contains multiple incompatible temporal operators.",
        )

    interval_required = operator == "between"
    anchor_optional = operator in {"first", "latest", "last"}
    if interval_required and len(mentions) < 2:
        return NormalizedTemporalQuery(
            status="missing_temporal_anchor",
            operator=operator,
            basis="world_valid_time",
            interval_requires_coverage=operator == "during",
            explanation="The interval operator does not expose two parseable dates.",
        )
    if not mentions and not anchor_optional:
        return NormalizedTemporalQuery(
            status="missing_temporal_anchor",
            operator=operator,
            basis="world_valid_time",
            explanation="The temporal operator does not expose a parseable date.",
        )

    if interval_required or (operator == "during" and len(mentions) >= 2):
        first, second = mentions[-2:]
        query_start = min(first.lower, second.lower)
        query_end = max(first.upper, second.upper)
        granularity = (
            first.granularity if first.granularity == second.granularity else "day"
        )
    elif mentions:
        selected = mentions[-1]
        query_start, query_end = selected.lower, selected.upper
        granularity = selected.granularity
    else:
        query_start = query_end = None
        granularity = "unknown"

    return NormalizedTemporalQuery(
        status="exact",
        operator=operator,
        basis="world_valid_time",
        query_start=query_start,
        query_end=query_end,
        granularity=granularity,
        interval_requires_coverage=operator == "during",
        explanation=(
            "The world-valid-time operator and anchor were parsed after removing any evidence "
            "snapshot date."
        ),
    )


def assess_evidence_times(
    query: NormalizedTemporalQuery,
    evidence: Sequence[JudgeEvidence],
    *,
    semantic_relevance: Mapping[str, float] | None = None,
) -> list[EvidenceTemporalAssessment]:
    """Assess every evidence item under one query without treating unknown time as valid."""

    relevance = semantic_relevance or {}
    if query.basis == "not_temporal":
        return [
            EvidenceTemporalAssessment(
                evidence_id=row.evidence_id,
                compatibility=1.0,
                exact_validity=1.0,
                near_miss=0.0,
                status="not_temporal",
                temporal_source="none",
                explanation="The question has no temporal validity requirement.",
            )
            for row in evidence
        ]

    if query.basis in {"snapshot_observation", "document_revision"}:
        return [_assess_publication_anchor(query, row) for row in evidence]

    known = {
        row.evidence_id: _visible_bounds(row.valid_time)
        for row in evidence
        if _visible_bounds(row.valid_time) is not None
    }
    selected = _ordinal_selection(query, known, relevance)
    return [
        _assess_world_interval(query, row, bounds=known.get(row.evidence_id), selected=selected)
        for row in evidence
    ]


def parse_claim_time(claim: str) -> ClaimTimeConstraint:
    """Extract only explicit, structurally unambiguous valid-time assertions from a claim."""

    if _BARE_TEMPORAL_VALUE.fullmatch(claim):
        return ClaimTimeConstraint(
            status="absent",
            mode="none",
            explanation="The complete answer is a temporal value, not a scoped proposition.",
        )

    parenthetical_candidates = []
    for match in _PARENTHETICAL.finditer(claim):
        content = match.group(1)
        mentions = _date_mentions(content)
        if mentions:
            parenthetical_candidates.append((content, mentions))
    if len(parenthetical_candidates) > 1:
        return ClaimTimeConstraint(
            status="ambiguous",
            mode="unknown",
            explanation="Multiple parenthetical time expressions apply to one automatic claim.",
        )
    if parenthetical_candidates:
        content, mentions = parenthetical_candidates[0]
        return _constraint_from_mentions(content, mentions, parenthetical=True)

    cleaned = _DOCUMENT_DATE.sub("", claim)
    cleaned = _SNAPSHOT_DATE.sub("", cleaned)
    mentions = _date_mentions(cleaned)
    if not mentions:
        return ClaimTimeConstraint(
            status="absent",
            mode="none",
            explanation="The claim contains no explicit valid-time assertion.",
        )
    boundary = list(_START_BOUNDARY.finditer(cleaned))
    if boundary:
        scoped = [mention for mention in mentions if mention.start >= boundary[-1].end()]
        if len(scoped) == 1:
            return _constraint_from_mentions(
                cleaned[boundary[-1].start() :],
                scoped,
                parenthetical=False,
            )
    if len(mentions) == 1 and (_START_BOUNDARY.search(cleaned) or _POINT_CUE.search(cleaned)):
        return _constraint_from_mentions(cleaned, mentions, parenthetical=False)
    if (
        len(mentions) == 2
        and _RANGE_SEPARATOR.search(cleaned[mentions[0].end : mentions[1].start])
        and (
            _RELATION_TIME_CUE.search(cleaned)
            or re.search(r"\bfrom\s*$", cleaned[: mentions[0].start], re.IGNORECASE)
        )
    ):
        return _constraint_from_mentions(cleaned, mentions, parenthetical=False)
    return ClaimTimeConstraint(
        status="ambiguous",
        mode="unknown",
        explanation=(
            "The claim contains dates, but their role cannot be separated conservatively from "
            "question, snapshot, or discourse context."
        ),
    )


def claim_evidence_time_compatibility(
    constraint: ClaimTimeConstraint,
    interval: VisibleInterval,
) -> float | None:
    """Return whether evidence valid time entails an explicit claim-time constraint."""

    if constraint.status == "absent":
        return 1.0
    if constraint.status != "exact" or constraint.start is None or constraint.end is None:
        return None
    bounds = _visible_bounds(interval)
    if bounds is None:
        return None
    evidence_start, evidence_end = bounds
    if constraint.mode == "start_boundary":
        return float(evidence_start == constraint.start)
    if constraint.mode == "point":
        return float(evidence_start <= constraint.start and evidence_end >= constraint.end)
    if constraint.mode == "interval":
        return float(evidence_start <= constraint.start and evidence_end >= constraint.end)
    return None


def claim_query_time_compatibility(
    constraint: ClaimTimeConstraint,
    query: NormalizedTemporalQuery,
) -> float | None:
    """Check an explicit claim time against the temporal operation in the question.

    ``None`` means that no sound direct comparison is licensed. Evidence-level temporal
    selection still applies in those cases; the caller must not reinterpret ``None`` as a
    successful parse.
    """

    if constraint.status == "absent":
        return 1.0
    if constraint.status != "exact" or constraint.start is None or constraint.end is None:
        return None
    if query.basis != "world_valid_time" or query.query_start is None or query.query_end is None:
        return None

    claim_start = constraint.start
    claim_end = date.max if constraint.mode == "start_boundary" else constraint.end
    query_start = query.query_start
    query_end = query.query_end

    if query.operator in {"as_of", "current", "effective", "during"}:
        return float(claim_start <= query_start and claim_end >= query_end)
    if query.operator == "between":
        return float(claim_start <= query_end and query_start <= claim_end)
    if query.operator in {"before", "previous", "expired"}:
        return float(claim_end < query_start)
    if query.operator in {"after", "next"}:
        return float(claim_start > query_end)
    if query.operator in {"latest", "last"}:
        # Ordinal winner selection is performed over the displayed evidence set. The claim time
        # must agree with its evidence. A supplied cutoff still licenses an eligibility check.
        return float(query.query_end is None or claim_start <= query.query_end)
    if query.operator == "first":
        # Earliest selection has no independent anchor constraint in the supported grammar.
        return 1.0
    return None


def _constraint_from_mentions(
    text: str,
    mentions: Sequence[_DateMention],
    *,
    parenthetical: bool,
) -> ClaimTimeConstraint:
    if len(mentions) == 1:
        mention = mentions[0]
        mode = "start_boundary" if _START_BOUNDARY.search(text) else "point"
        return ClaimTimeConstraint(
            status="exact",
            mode=mode,
            start=mention.lower,
            end=mention.upper,
            granularity=mention.granularity,
            explanation=(
                "A parenthetical claim time was parsed."
                if parenthetical
                else "An explicit point or start-boundary claim time was parsed."
            ),
        )
    if len(mentions) == 2 and _RANGE_SEPARATOR.search(
        text[mentions[0].end : mentions[1].start]
    ):
        return ClaimTimeConstraint(
            status="exact",
            mode="interval",
            start=mentions[0].lower,
            end=mentions[1].upper,
            granularity=(
                "year" if all(row.granularity == "year" for row in mentions) else "day"
            ),
            explanation="An explicit claim-time interval was parsed.",
        )
    return ClaimTimeConstraint(
        status="ambiguous",
        mode="unknown",
        explanation="The explicit time expression is not a single supported point or interval.",
    )


def temporal_status_rates(
    assessments: Sequence[EvidenceTemporalAssessment],
    *,
    semantic_relevance: Sequence[float] | None = None,
    relevance_threshold: float = 0.5,
    k: int = 10,
) -> dict[str, float | None]:
    top = list(assessments[:k])
    if not top:
        return {
            f"tcred_stale_rate_at_{k}": None,
            f"tcred_future_invalid_rate_at_{k}": None,
            f"tcred_unknown_time_rate_at_{k}": None,
            f"tcred_raw_stale_rate_at_{k}": None,
            f"tcred_raw_future_invalid_rate_at_{k}": None,
            f"tcred_raw_unknown_time_rate_at_{k}": None,
        }
    if semantic_relevance is None:
        relevance = [1.0] * len(top)
    else:
        relevance = list(semantic_relevance[: len(top)])
        if len(relevance) != len(top):
            raise ValueError("Temporal status rows and relevance scores must align")
    denominator = len(top)
    relevant = [value >= relevance_threshold for value in relevance]
    unknown_statuses = {"unknown_valid_time", "publication_only"}
    return {
        f"tcred_stale_rate_at_{k}": sum(
            is_relevant and row.status == "stale"
            for row, is_relevant in zip(top, relevant, strict=True)
        )
        / denominator,
        f"tcred_future_invalid_rate_at_{k}": (
            sum(
                is_relevant and row.status == "future_invalid"
                for row, is_relevant in zip(top, relevant, strict=True)
            )
            / denominator
        ),
        f"tcred_unknown_time_rate_at_{k}": (
            sum(
                is_relevant and row.status in unknown_statuses
                for row, is_relevant in zip(top, relevant, strict=True)
            )
            / denominator
        ),
        f"tcred_raw_stale_rate_at_{k}": (
            sum(row.status == "stale" for row in top) / denominator
        ),
        f"tcred_raw_future_invalid_rate_at_{k}": (
            sum(row.status == "future_invalid" for row in top) / denominator
        ),
        f"tcred_raw_unknown_time_rate_at_{k}": (
            sum(row.status in unknown_statuses for row in top) / denominator
        ),
    }


def _anchored_query(
    mention: _DateMention | None,
    *,
    operator: str,
    basis: str,
    explanation: str,
) -> NormalizedTemporalQuery:
    if mention is None:
        return NormalizedTemporalQuery(
            status="missing_temporal_anchor",
            operator=operator,
            basis=basis,
            explanation=f"{explanation} Its date could not be parsed.",
        )
    return NormalizedTemporalQuery(
        status="exact",
        operator=operator,
        basis=basis,
        query_start=mention.lower,
        query_end=mention.upper,
        evaluation_time=mention.upper,
        granularity=mention.granularity,
        explanation=explanation,
    )


def _world_operator(lowered: str) -> str:
    patterns = (
        ("previous", ("ended most recently before the relationship", "previous relationship")),
        ("next", ("began first after the relationship", "next relationship")),
        ("expired", ("expired",)),
        ("during", ("held throughout", "throughout the period", "during")),
        ("between", ("at any time from", "between")),
        ("after", ("began first after", " after ")),
        ("before", ("ended most recently before", " before ")),
        ("first", ("first recorded", "earliest recorded")),
        ("latest", ("latest recorded", "most recent recorded")),
        ("last", ("last recorded", "relationship to begin")),
        ("effective", ("in effect", "effective")),
        ("current", ("current on", "currently", "was current")),
        ("as_of", (" as of ", " on ", "using the evidence snapshot")),
    )
    matches = {
        operator
        for operator, needles in patterns
        if any(needle in lowered for needle in needles)
    }
    # Specific phrases contain generic operator substrings. Remove only those licensed aliases;
    # any remaining multi-operator combination is genuinely underdetermined.
    if "previous" in matches:
        matches.discard("before")
        matches.discard("effective")
        matches.discard("current")
    if "next" in matches:
        matches.discard("after")
        matches.discard("effective")
        matches.discard("current")
    if matches - {"as_of"}:
        matches.discard("as_of")
    if not matches:
        return "not_temporal"
    if len(matches) > 1:
        return "ambiguous"
    return next(iter(matches))


def _date_mentions(text: str) -> list[_DateMention]:
    mentions: list[_DateMention] = []
    occupied: list[tuple[int, int]] = []
    for pattern in (_ISO_DATE, _LONG_DATE, _DAY_FIRST_DATE):
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            try:
                parsed = _match_date(match)
            except ValueError:
                # Malformed external input is an unparseable anchor, not a metric crash.
                occupied.append((match.start(), match.end()))
                continue
            mentions.append(
                _DateMention(match.start(), match.end(), parsed, parsed, "day")
            )
            occupied.append((match.start(), match.end()))
    for match in _MONTH_YEAR.finditer(text):
        if any(match.start() < end and start < match.end() for start, end in occupied):
            continue
        year = int(match.group("year"))
        month = _MONTHS[match.group("month").casefold()]
        mentions.append(
            _DateMention(
                match.start(),
                match.end(),
                date(year, month, 1),
                date(year, month, monthrange(year, month)[1]),
                "month",
            )
        )
        occupied.append((match.start(), match.end()))
    for match in _YEAR.finditer(text):
        if any(match.start() < end and start < match.end() for start, end in occupied):
            continue
        year = int(match.group("year"))
        mentions.append(
            _DateMention(match.start(), match.end(), date(year, 1, 1), date(year, 12, 31), "year")
        )
    return sorted(mentions, key=lambda row: row.start)


def _single_date_mention(text: str) -> _DateMention | None:
    mentions = _date_mentions(text)
    return mentions[0] if mentions else None


def _match_date(match: re.Match[str]) -> date:
    month_text = match.group("month")
    month = int(month_text) if month_text.isdigit() else _MONTHS[month_text.casefold()]
    return date(int(match.group("year")), month, int(match.group("day")))


def _visible_bounds(interval: VisibleInterval) -> tuple[date, date] | None:
    if interval.type == "unknown" or interval.start is None:
        return None
    start = _iso_date(interval.start)
    end = _iso_date(interval.end) if interval.end else date.max
    if start is None or end is None or start > end:
        return None
    return start, end


def _iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _assess_publication_anchor(
    query: NormalizedTemporalQuery,
    evidence: JudgeEvidence,
) -> EvidenceTemporalAssessment:
    publication = _iso_date(evidence.publication_time)
    if publication is None:
        return EvidenceTemporalAssessment(
            evidence_id=evidence.evidence_id,
            status="unknown_valid_time",
            temporal_source="none",
            explanation="This snapshot/revision item has no publication or revision date.",
        )
    if query.query_start is None or query.query_end is None:
        return EvidenceTemporalAssessment(
            evidence_id=evidence.evidence_id,
            status="temporally_ambiguous",
            temporal_source="publication_time",
            effective_start=publication,
            effective_end=publication,
            explanation="The question's snapshot/revision date is missing.",
        )
    exact = float(query.query_start <= publication <= query.query_end)
    if exact:
        status = "valid"
    elif publication < query.query_start:
        status = "stale"
    else:
        status = "future_invalid"
    gap = _point_interval_gap(publication, query.query_start, query.query_end)
    near = _near_miss(gap) if not exact else 0.0
    return EvidenceTemporalAssessment(
        evidence_id=evidence.evidence_id,
        compatibility=max(exact, near),
        exact_validity=exact,
        near_miss=near,
        status=status,
        effective_start=publication,
        effective_end=publication,
        temporal_source="publication_time",
        explanation="Publication/revision time was compared with the licensed snapshot anchor.",
    )


def _ordinal_selection(
    query: NormalizedTemporalQuery,
    known: Mapping[str, tuple[date, date]],
    relevance: Mapping[str, float],
) -> set[str] | None:
    if query.operator not in {
        "after",
        "before",
        "expired",
        "first",
        "latest",
        "last",
        "previous",
        "next",
    }:
        return None
    candidates = (
        [
            (evidence_id, bounds)
            for evidence_id, bounds in known.items()
            if relevance.get(evidence_id, 0.0) >= 0.25
        ]
        if relevance
        else list(known.items())
    )
    if not candidates:
        return set()
    start = query.query_start
    end = query.query_end
    eligible: list[tuple[str, tuple[date, date]]]
    key_index: int
    if query.operator == "first":
        eligible = candidates
        key_index = 0
        selected_value = min(bounds[key_index] for _evidence_id, bounds in eligible)
    elif query.operator in {"latest", "last"}:
        eligible = [row for row in candidates if end is None or row[1][0] <= end]
        key_index = 0
        if not eligible:
            return set()
        selected_value = max(bounds[key_index] for _evidence_id, bounds in eligible)
    elif query.operator in {"before", "expired", "previous"}:
        if start is None:
            return set()
        eligible = [row for row in candidates if row[1][1] != date.max and row[1][1] < start]
        key_index = 1
        if not eligible:
            return set()
        selected_value = max(bounds[key_index] for _evidence_id, bounds in eligible)
    else:
        if end is None:
            return set()
        eligible = [row for row in candidates if row[1][0] > end]
        key_index = 0
        if not eligible:
            return set()
        selected_value = min(bounds[key_index] for _evidence_id, bounds in eligible)
    return {
        evidence_id
        for evidence_id, bounds in eligible
        if bounds[key_index] == selected_value
    }


def _assess_world_interval(
    query: NormalizedTemporalQuery,
    evidence: JudgeEvidence,
    *,
    bounds: tuple[date, date] | None,
    selected: set[str] | None,
) -> EvidenceTemporalAssessment:
    if bounds is None:
        status = "publication_only" if evidence.publication_time else "unknown_valid_time"
        return EvidenceTemporalAssessment(
            evidence_id=evidence.evidence_id,
            status=status,
            temporal_source="none",
            explanation="World-valid time is unavailable; publication time is not a substitute.",
        )
    start, end = bounds
    display_end = None if end == date.max else end
    if selected is not None:
        exact = float(evidence.evidence_id in selected)
        gap = _ordinal_gap_days(query, start, end)
        near = _near_miss(gap) if not exact else 0.0
        status = "valid" if exact else _relative_status(query, start, end)
        return EvidenceTemporalAssessment(
            evidence_id=evidence.evidence_id,
            compatibility=max(exact, near),
            exact_validity=exact,
            near_miss=near,
            status=status,
            effective_start=start,
            effective_end=display_end,
            temporal_source="valid_time",
            explanation="The interval was evaluated by the query's deterministic ordinal rule.",
        )
    if query.query_start is None or query.query_end is None:
        return EvidenceTemporalAssessment(
            evidence_id=evidence.evidence_id,
            status="temporally_ambiguous",
            effective_start=start,
            effective_end=display_end,
            temporal_source="valid_time",
            explanation="Evidence time is known but the query interval is not.",
        )
    overlap_start = max(start, query.query_start)
    overlap_end = min(end, query.query_end)
    overlaps = overlap_start <= overlap_end
    if query.operator == "between":
        exact = float(overlaps)
        coverage = exact
    elif query.operator == "during":
        overlap_days = max(0, (overlap_end - overlap_start).days + 1) if overlaps else 0
        query_days = (query.query_end - query.query_start).days + 1
        coverage = overlap_days / query_days
        exact = float(coverage == 1.0)
    else:
        exact = float(start <= query.query_start and end >= query.query_end)
        coverage = exact
    gap = _interval_gap_days(start, end, query.query_start, query.query_end)
    near = _near_miss(gap) if not overlaps else 0.0
    status = "valid" if exact else _relative_status(query, start, end)
    if query.operator == "during" and overlaps and not exact:
        status = "temporally_ambiguous"
    return EvidenceTemporalAssessment(
        evidence_id=evidence.evidence_id,
        compatibility=max(coverage, near),
        exact_validity=exact,
        near_miss=near,
        status=status,
        effective_start=start,
        effective_end=display_end,
        temporal_source="valid_time",
        explanation="Valid-time overlap was evaluated against the parsed query interval.",
    )


def _relative_status(query: NormalizedTemporalQuery, start: date, end: date) -> str:
    if query.query_start is not None and end != date.max and end < query.query_start:
        return "stale"
    if query.query_end is not None and start > query.query_end:
        return "future_invalid"
    return "temporally_ambiguous"


def _near_miss(gap_days: int | None) -> float:
    if gap_days is None:
        return 0.0
    return math.exp(-gap_days / 365.0) * 0.25


def _point_interval_gap(point: date, start: date, end: date) -> int:
    if point < start:
        return (start - point).days
    if point > end:
        return (point - end).days
    return 0


def _interval_gap_days(start: date, end: date, query_start: date, query_end: date) -> int:
    if end != date.max and end < query_start:
        return (query_start - end).days
    if start > query_end:
        return (start - query_end).days
    return 0


def _ordinal_gap_days(query: NormalizedTemporalQuery, start: date, end: date) -> int | None:
    if query.operator in {"before", "expired", "previous"} and query.query_start is not None:
        if end == date.max:
            return None
        return abs((query.query_start - end).days)
    if query.operator in {"after", "next", "latest", "last"} and query.query_end is not None:
        return abs((start - query.query_end).days)
    return 0
