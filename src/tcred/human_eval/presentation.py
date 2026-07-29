from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tcred.dataset.text import annotation_plain_text

ANNOTATION_TEXT_REPAIR_VERSION = "1.0"

_DATE_PATTERN = r"[A-Za-z]+ \d{1,2}, \d{4}"
_PAT_PREFIX = re.compile(
    rf"^According to the Wikidata snapshot dated (?P<date>{_DATE_PATTERN}), "
    r"(?P<body>.+)$",
    flags=re.IGNORECASE,
)
_HOH_PREFIX = re.compile(
    rf"^According to the document revision for (?P<document>.+) dated "
    rf"(?P<date>{_DATE_PATTERN}), (?P<body>.+)$",
    flags=re.IGNORECASE,
)
_ANNOTATION_TEXT_KEYS = frozenset(
    {
        "answer_text",
        "context_note",
        "evidence_text",
        "label",
        "object_label",
        "question",
        "reference_answer",
        "relation_label",
        "text",
    }
)
_HOH_TEXT_REPAIRS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[Ss]akaSaga Dawa\b"), "Saga Dawa"),
    (re.compile(r"\(\s*;\s*(?:;\s*)?born\b"), "(born"),
    (re.compile(r"\bBud Gaugh He now\b"), "Bud Gaugh. He now"),
    (
        re.compile(
            r"At 9:47p\.m\., after traveling nearly , while cruising at near the city "
            r"of Saginaw, Michigan,"
        ),
        "At 9:47 p.m., while cruising near the city of Saginaw, Michigan,",
    ),
)
_TEXT_QUALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("concatenated Saga Dawa name", re.compile(r"\b[Ss]akaSaga Dawa\b")),
    ("empty source parenthetical", re.compile(r"\(\s*;")),
    ("missing sentence boundary after Bud Gaugh", re.compile(r"\bBud Gaugh He now\b")),
    (
        "removed quantities left an ungrammatical sentence",
        re.compile(r"traveling nearly ,|cruising at near"),
    ),
    ("Unicode replacement character", re.compile("\ufffd")),
)


def annotation_question_text(question: str, *, dataset_family: str) -> str:
    """Return source-scoped wording intended for human annotation.

    QA outputs remain tied to the frozen benchmark question. This display-only normalization
    removes the misleading combination of a historical source date with unqualified words such as
    ``current`` and ``currently`` while preserving the original question semantics.
    """

    text = annotation_plain_text(question)
    if dataset_family == "tcred_pat":
        return _pat_snapshot_question(text)
    if dataset_family == "tcred_hoh":
        return _hoh_revision_question(text)
    return text


def normalize_annotation_payload(
    row: Mapping[str, object],
    *,
    dataset_family: str,
) -> dict[str, object]:
    """Return a deep-copied annotation view with audited display-only text repairs.

    The benchmark source, QA outputs, and graph semantics remain unchanged. Repairs are restricted
    to evaluator-visible strings and an explicit source-family allowlist so presentation cleanup
    cannot silently rewrite arbitrary facts.
    """

    copied: dict[str, object] = deepcopy(dict(row))
    if dataset_family != "tcred_hoh":
        return copied
    return _normalize_visible_text(copied, dataset_family=dataset_family)


def annotation_text_quality_issues(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return known extraction/corruption defects still visible in an annotation payload."""

    issues: set[str] = set()
    for text in _iter_strings(row):
        for description, pattern in _TEXT_QUALITY_PATTERNS:
            if pattern.search(text):
                issues.add(description)
    return tuple(sorted(issues))


def displayed_evidence(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the cited-first union shown to annotators.

    Cited and retrieved evidence are separate provenance views of the same public objects. A cited
    item must never disappear merely because the retrieval list is non-empty.
    """

    return merge_evidence(
        cited=row.get("cited_evidence"),
        retrieved=row.get("retrieved_evidence"),
    )


def merge_evidence(*, cited: object, retrieved: object) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_rows in (cited, retrieved):
        if not isinstance(raw_rows, list):
            continue
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            evidence_id = str(row.get("evidence_id", "")).strip()
            if evidence_id and evidence_id in seen_ids:
                continue
            if evidence_id:
                seen_ids.add(evidence_id)
            merged.append(row)
    return merged


def missing_visible_citation_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    declared = row.get("cited_evidence_ids")
    cited_ids = (
        {str(evidence_id) for evidence_id in declared if str(evidence_id)}
        if isinstance(declared, list)
        else set()
    )
    visible_ids = {
        str(evidence.get("evidence_id"))
        for evidence in displayed_evidence(row)
        if evidence.get("evidence_id")
    }
    return tuple(sorted(cited_ids - visible_ids))


def require_visible_citations(row: Mapping[str, object]) -> None:
    missing = missing_visible_citation_ids(row)
    if missing:
        unit_id = str(row.get("unit_id", "<unknown>"))
        raise ValueError(
            f"Human-evaluation unit {unit_id} cites evidence absent from the displayed union: "
            f"{', '.join(missing)}"
        )


def _pat_snapshot_question(question: str) -> str:
    match = _PAT_PREFIX.fullmatch(question)
    if match is None:
        return question
    body = _rewrite_pat_body(match.group("body"))
    return f"In the Wikidata snapshot dated {match.group('date')}, {_lower_initial(body)}"


def _rewrite_pat_body(body: str) -> str:
    rewrites: tuple[tuple[str, str], ...] = (
        (
            r"Who was the last person to hold the position that (.+) holds currently\?",
            r"Who held, immediately before \1, the position that \1 is recorded as holding?",
        ),
        (
            r"Where was the current (.+) born\?",
            r"Where was the person recorded as \1 born?",
        ),
        (
            r"Who did the current (.+) succeed in office\?",
            r"Whom did the person recorded as \1 succeed in office?",
        ),
        (
            r"Who did the previous (.+) succeed in office\?",
            r"Whom did the person recorded as \1 succeed in office?",
        ),
        (
            r"Which team does (.+) play for currently\?",
            r"Which team is \1 recorded as playing for?",
        ),
        (
            r"Which employer does (.+) work for currently\?",
            r"Which employer is \1 recorded as working for?",
        ),
        (
            r"What is the current political affiliation of (.+)\?",
            r"What political affiliation is recorded for \1?",
        ),
        (
            r"What is the home venue of the team that (.+) play for currently\?",
            r"What is the home venue of the team that \1 is recorded as playing for?",
        ),
        (
            r"Who is the head coach of the team that (.+) play for currently\?",
            r"Who is recorded as head coach of the team that \1 is recorded as playing for?",
        ),
        (
            r"Who is the chair of the political party which the (.+) belongs to currently\?",
            r"Who is recorded as chair of the political party to which \1 belongs?",
        ),
        (
            r"Who is the spouse of the current (.+)\?",
            r"Who is recorded as the spouse of the person recorded as \1?",
        ),
        (
            r"Which position does (.+) hold currently\?",
            r"Which position is \1 recorded as holding?",
        ),
        (
            r"Where is the headquarters of the political party (.+) belongs to currently\?",
            r"Where is the recorded headquarters of the political party to which \1 belongs?",
        ),
        (
            r"Where is the headquarters of the current employer of (.+)\?",
            r"Where is the recorded headquarters of the employer listed for \1?",
        ),
        (
            r"Where is the headquarters of the current owner of (.+) located at\?",
            r"Where is the recorded headquarters of the owner listed for \1?",
        ),
        (
            r"Who is the owner of the current employer of (.+)\?",
            r"Who is recorded as the owner of the employer listed for \1?",
        ),
        (
            r"Which school did the current (.+) attend\?",
            r"Which school did the person recorded as \1 attend?",
        ),
        (
            r"Who is (.+) currently\?",
            r"Who is recorded as \1?",
        ),
    )
    for pattern, replacement in rewrites:
        if re.fullmatch(pattern, body, flags=re.IGNORECASE):
            return re.sub(pattern, replacement, body, flags=re.IGNORECASE)

    clarified = re.sub(r"\bcurrently\b", "in that snapshot", body, flags=re.IGNORECASE)
    clarified = re.sub(
        r"\bthe current\b",
        "the person or value recorded in that snapshot as",
        clarified,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\bcurrent\b",
        "recorded in that snapshot",
        clarified,
        flags=re.IGNORECASE,
    )


def _hoh_revision_question(question: str) -> str:
    match = _HOH_PREFIX.fullmatch(question)
    if match is None:
        return question
    body = match.group("body")
    body = re.sub(
        r"^Who is the current (.+)\?$",
        r"Who does that revision identify as \1?",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(
        r"^What is the current (.+)\?$",
        r"What does that revision identify as \1?",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"\bcurrently\b", "in that revision", body, flags=re.IGNORECASE)
    return (
        f"In the document revision for {match.group('document')} dated {match.group('date')}, "
        f"{_lower_initial(body)}"
    )


def _lower_initial(value: str) -> str:
    if not value:
        return value
    return value[0].lower() + value[1:]


def _normalize_visible_text(value: Any, *, dataset_family: str, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            child_key: _normalize_visible_text(
                child_value,
                dataset_family=dataset_family,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_visible_text(item, dataset_family=dataset_family, key=key)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _normalize_visible_text(item, dataset_family=dataset_family, key=key)
            for item in value
        )
    if not isinstance(value, str) or key not in _ANNOTATION_TEXT_KEYS:
        return value
    if dataset_family == "tcred_hoh":
        for pattern, replacement in _HOH_TEXT_REPAIRS:
            value = pattern.sub(replacement, value)
    return value


def _iter_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, Mapping):
        for child in value.values():
            strings.extend(_iter_strings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            strings.extend(_iter_strings(child))
    elif isinstance(value, str):
        strings.append(value)
    return strings
