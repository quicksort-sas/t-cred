from __future__ import annotations

import re
from collections.abc import Sequence

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_INITIALISM = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_LIST_BOUNDARY = re.compile(r"\s*(?:;|\n\s*[-*\u2022]\s+)\s*")
_PARENTHETICAL_CONTENT = re.compile(r"\([^()]+\)")
_DATE_WITH_COMMA = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
_DATE_TRAILING_COMMA = re.compile(
    r"(?:\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}|\b\d{4}-\d{2}-\d{2}|\b\d{4})"
    r"(?P<comma>,)(?=\s+(?:to|through|until|and\s+again|as|with|who|which|but)\b)",
    re.IGNORECASE,
)
_CLAUSAL_COMMA = re.compile(
    r",(?=\s+(?:according|as|during|for|from|in|is|on|starting|that|the|was|were|when|"
    r"where|which|who|with)\b)",
    re.IGNORECASE,
)
_TEMPORAL_CONTEXT_COMMA = re.compile(
    r"\b(?:date|period|revision|snapshot|timeframe)(?P<comma>,)(?=\s+\S)",
    re.IGNORECASE,
)
_TEMPORAL_ITEM_START = re.compile(
    r"\b(?:"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|(?:1[5-9]|20)\d{2}"
    r")\b",
    re.IGNORECASE,
)
_EVIDENCE_ID_LIST = re.compile(
    r"\bE\d+(?:\s*,\s*E\d+){1,}(?:\s*,?\s+and\s+E\d+)?\b",
    re.IGNORECASE,
)
_ANSWER_WRAPPER = re.compile(
    r"^\s*(?:based on|according to)\b.{0,120}?\b(?:the answer is|answer:)\s*",
    re.IGNORECASE,
)
_PREDICATE_LIST = re.compile(
    r"^(?P<prefix>.+?\b(?:are|were|include(?:s|d)?|consist(?:s|ed)?\s+of|"
    r"lists?\b.{0,80}?\bas)\s+)(?P<tail>.+)$",
    re.IGNORECASE,
)
_FINITE_VERB = re.compile(
    r"\b(?:am|are|is|was|were|be|been|being|has|have|had|does|do|did|"
    r"will|would|shall|should|can|could|may|might|must)\b",
    re.IGNORECASE,
)
_PLURAL_LIST_CUE = re.compile(
    r"\b(?:answers|claims|officers|organizations|parties|positions|professions|"
    r"relationships|roles|values|were|are|include(?:s|d)?|consist(?:s|ed)?)\b",
    re.IGNORECASE,
)
_COORDINATED_TITLE = re.compile(
    r"^(?P<head>minister|ministry|department|secretary|committee)\s+(?:for|of|on)\b",
    re.IGNORECASE,
)
_CLAUSE_CONNECTOR = re.compile(
    r"^(?:although|but|however|since|therefore|though|while)\b",
    re.IGNORECASE,
)
_REFUSAL_PATTERNS = (
    re.compile(r"\b(?:cannot|can't|unable to)\s+(?:determine|answer|establish|infer)\b", re.I),
    re.compile(r"\binsufficient\s+(?:evidence|information)\b", re.I),
    re.compile(r"\bnot enough\s+(?:evidence|information)\b", re.I),
    re.compile(r"\b(?:evidence|information)\b.{0,40}\bnot\b.{0,30}\bsufficient\b", re.I),
    re.compile(r"\bno answer is supported\b", re.I),
    re.compile(r"\bthe evidence (?:does not|doesn't) (?:establish|support|determine)\b", re.I),
    re.compile(r"^\s*(?:(?:the\s+)?answer\s+is\s+)?unknown[.!]?\s*$", re.I),
    re.compile(r"\b(?:answer|identity|result|value)\s+(?:is|remains)\s+unknown\b", re.I),
    re.compile(r"\bit\s+(?:is|remains)\s+unknown\b", re.I),
)


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.split())
    return any(pattern.search(normalized) for pattern in _REFUSAL_PATTERNS)


def decompose_claims(
    text: str,
    *,
    evidence: Sequence[tuple[str, str]] = (),
) -> list[str]:
    """Return deterministic automatic claims shared by scoring and model inference."""

    stripped = _ANSWER_WRAPPER.sub("", text.strip()).strip()
    if not stripped or is_refusal(stripped):
        return []
    claims: list[str] = []
    for sentence in _split_sentences(stripped):
        for part in _LIST_BOUNDARY.split(sentence):
            for claim in _context_preserving_list_items(part):
                normalized = claim.strip(" \t\r\n-*")
                if normalized:
                    claims.append(normalized)
    if evidence:
        if len(claims) > 1:
            resolved_claims = _evidence_validated_temporal_items(claims, evidence)
            if resolved_claims is not None:
                return resolved_claims
        else:
            resolved = _evidence_resolved_comma_items(stripped, evidence)
            if resolved is not None:
                return resolved
    return claims


def _split_sentences(text: str) -> list[str]:
    protected = _INITIALISM.sub(
        lambda match: match.group(0).replace(".", "<tcred-period>"),
        text,
    )
    return [part.replace("<tcred-period>", ".") for part in _SENTENCE_BOUNDARY.split(protected)]


def _evidence_resolved_comma_items(
    text: str,
    evidence: Sequence[tuple[str, str]],
) -> list[str] | None:
    """Resolve otherwise ambiguous comma lists from distinct visible evidence items.

    Claim decomposition is itself a source of metric error. We therefore split an ambiguous
    answer only when every proposed contiguous item has an exact token-phrase match and the items
    can be assigned to distinct displayed evidence IDs. This preserves names such as
    ``Tipperary, Ireland`` while recovering set-valued answers whose members have separate support.
    """

    parts = _top_level_comma_parts(text.strip().strip("."))
    if not 2 <= len(parts) <= 8:
        return None
    evidence_by_id = {
        str(evidence_id): _match_tokens(evidence_text)
        for evidence_id, evidence_text in evidence
        if str(evidence_id).strip() and str(evidence_text).strip()
    }
    if len(evidence_by_id) < 2:
        return None

    candidates: list[tuple[tuple[int, int], list[str]]] = []
    boundary_count = len(parts) - 1
    for mask in range(1, 1 << boundary_count):
        groups: list[str] = []
        start = 0
        for index in range(boundary_count):
            if mask & (1 << index):
                groups.append(", ".join(parts[start : index + 1]))
                start = index + 1
        groups.append(", ".join(parts[start:]))
        match_phrases = [_evidence_match_phrase(group) for group in groups]
        matches = [
            {
                evidence_id
                for evidence_id, tokens in evidence_by_id.items()
                if _contains_token_phrase(_match_tokens(phrase), tokens)
            }
            for phrase in match_phrases
        ]
        if all(matches) and _has_distinct_evidence_assignment(matches):
            resolved = _repeat_temporal_item_context(groups)
            candidates.append(
                (
                    (len(resolved), min(len(_match_tokens(group)) for group in resolved)),
                    resolved,
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]


def _evidence_match_phrase(group: str) -> str:
    normalized = re.sub(r"^(?:and|or)\s+", "", group.strip(), flags=re.IGNORECASE)
    temporal = _TEMPORAL_ITEM_START.search(normalized)
    return normalized[temporal.start() :] if temporal else normalized


def _repeat_temporal_item_context(groups: Sequence[str]) -> list[str]:
    """Restore the relational prefix on comma-separated interval items.

    For example, ``Alice served from 2001 to 2002, 2005 to 2006`` becomes two
    independently checkable claims. The rewrite is licensed only when all later groups begin
    with a date expression; the caller has already required distinct exact evidence matches.
    """

    if len(groups) < 2:
        return list(groups)
    first = groups[0].strip()
    first_time = _TEMPORAL_ITEM_START.search(first)
    later = [re.sub(r"^(?:and|or)\s+", "", group.strip(), flags=re.I) for group in groups[1:]]
    if first_time is None or not all(_TEMPORAL_ITEM_START.match(group) for group in later):
        return [re.sub(r"^(?:and|or)\s+", "", group.strip(), flags=re.I) for group in groups]
    prefix = first[: first_time.start()].rstrip()
    return [first, *[f"{prefix} {group}".strip() for group in later]]


def _has_distinct_evidence_assignment(matches: Sequence[set[str]]) -> bool:
    ordered = sorted(matches, key=len)

    def assign(index: int, used: set[str]) -> bool:
        if index == len(ordered):
            return True
        return any(
            assign(index + 1, used | {evidence_id})
            for evidence_id in ordered[index] - used
        )

    return assign(0, set())


def _evidence_validated_temporal_items(
    claims: Sequence[str],
    evidence: Sequence[tuple[str, str]],
) -> list[str] | None:
    normalized = [claim.strip().strip(".") for claim in claims]
    resolved = _repeat_temporal_item_context(normalized)
    if resolved == normalized:
        return None
    evidence_by_id = {
        str(evidence_id): _match_tokens(evidence_text)
        for evidence_id, evidence_text in evidence
        if str(evidence_id).strip() and str(evidence_text).strip()
    }
    matches = [
        {
            evidence_id
            for evidence_id, tokens in evidence_by_id.items()
            if _contains_token_phrase(_match_tokens(_evidence_match_phrase(claim)), tokens)
        }
        for claim in resolved
    ]
    return resolved if all(matches) and _has_distinct_evidence_assignment(matches) else None


def _match_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _contains_token_phrase(phrase: Sequence[str], text: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(text):
        return False
    width = len(phrase)
    return any(
        tuple(text[index : index + width]) == tuple(phrase)
        for index in range(len(text) - width + 1)
    )


def split_top_level_answer_items(text: str) -> tuple[list[str], bool]:
    """Expose conservative top-level items for graph answer-target resolution.

    The boolean records whether an explicit ``and``/``or`` conjunction was present. Callers
    still need contextual entity evidence before interpreting a two-item comma expression as an
    enumeration rather than a single name such as ``Tipperary, Ireland``.
    """

    stripped = _ANSWER_WRAPPER.sub("", text.strip()).strip()
    items, _delimiter_count, conjunction_count = _split_top_level_items(stripped)
    return items, conjunction_count > 0


def _context_preserving_list_items(text: str) -> list[str]:
    match = _PREDICATE_LIST.match(text.strip())
    if match is None:
        return _comma_list_items(text)
    prefix = match.group("prefix").strip()
    tail = match.group("tail").strip()
    items, delimiter_count, conjunction_count = _split_top_level_items(tail)
    comma_count = delimiter_count - conjunction_count
    if len(items) < 2:
        return _comma_list_items(text)
    if len(items) == 2 and conjunction_count == 0:
        return _comma_list_items(text)
    if (
        len(items) == 2
        and comma_count == 0
        and not _PARENTHETICAL_CONTENT.search(tail)
        and not _PLURAL_LIST_CUE.search(prefix)
    ):
        return _comma_list_items(text)
    title = _COORDINATED_TITLE.match(tail)
    if title and len(re.findall(rf"\b{re.escape(title.group('head'))}\b", tail, re.I)) == 1:
        return [text]
    if any(_FINITE_VERB.search(item) for item in items[1:]):
        return _comma_list_items(text)
    if any(_CLAUSE_CONNECTOR.search(item) for item in items[1:]):
        return _comma_list_items(text)
    return [f"{prefix} {item}" for item in items]


def _split_top_level_items(text: str) -> tuple[list[str], int, int]:
    """Split coordinated list items without touching punctuation inside brackets or quotes."""

    items: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    delimiter_count = 0
    conjunction_count = 0
    protected_commas = _protected_comma_indices(text)
    index = 0
    while index < len(text):
        character = text[index]
        if character == '"':
            quote = None if quote == character else character if quote is None else quote
        elif quote is None and character in "([{":
            depth += 1
        elif quote is None and character in ")]}" and depth:
            depth -= 1
        if (
            quote is None
            and depth == 0
            and character == ","
            and index not in protected_commas
        ):
            _append_list_item(items, current)
            current = []
            delimiter_count += 1
            index += 1
            continue
        conjunction = _top_level_conjunction(text, index, depth=depth, quote=quote)
        if conjunction:
            _append_list_item(items, current)
            current = []
            delimiter_count += 1
            conjunction_count += 1
            index += len(conjunction)
            continue
        current.append(character)
        index += 1
    _append_list_item(items, current)
    return items, delimiter_count, conjunction_count


def _top_level_conjunction(
    text: str,
    index: int,
    *,
    depth: int,
    quote: str | None,
) -> str:
    if depth or quote is not None:
        return ""
    for conjunction in (" and ", " or "):
        if text[index : index + len(conjunction)].casefold() == conjunction:
            return conjunction
    return ""


def _append_list_item(items: list[str], characters: list[str]) -> None:
    item = "".join(characters).strip().strip(".")
    item = re.sub(r"^(?:and|or)\s+", "", item, flags=re.IGNORECASE)
    if item:
        items.append(item)


def _comma_list_items(text: str) -> list[str]:
    parts = _top_level_comma_parts(text)
    if len(parts) < 3:
        return [text]

    first_words = parts[0].split()
    for prefix_length in range(min(3, len(first_words)), 0, -1):
        prefix = " ".join(first_words[:prefix_length]).casefold()
        if not any(part.casefold().startswith(f"{prefix} ") for part in parts[1:]):
            continue
        items: list[str] = []
        current = parts[0]
        for part in parts[1:]:
            if part.casefold().startswith(f"{prefix} "):
                items.append(current)
                current = part
            else:
                current = f"{current}, {part}"
        items.append(current)
        return items

    if parts[-1].casefold().startswith(("and ", "or ")):
        items = [re.sub(r"^(?:and|or)\s+", "", part, flags=re.I) for part in parts]
        if _compact_nominal_items(items):
            return items
    if (
        all(" and " not in part.casefold() and " or " not in part.casefold() for part in parts)
        and _compact_nominal_items(parts)
    ):
        return parts
    return [text]


def _top_level_comma_parts(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    protected_commas = _protected_comma_indices(text)
    for index, character in enumerate(text):
        if character == '"':
            quote = None if quote == character else character if quote is None else quote
        elif quote is None and character in "([{":
            depth += 1
        elif quote is None and character in ")]}" and depth:
            depth -= 1
        if (
            character == ","
            and quote is None
            and depth == 0
            and index not in protected_commas
        ):
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(character)
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def _protected_comma_indices(text: str) -> set[int]:
    protected = {
        index
        for match in _DATE_WITH_COMMA.finditer(text)
        for index in range(match.start(), match.end())
        if text[index] == ","
    }
    protected.update(match.start("comma") for match in _DATE_TRAILING_COMMA.finditer(text))
    protected.update(match.start() for match in _CLAUSAL_COMMA.finditer(text))
    protected.update(
        match.start("comma") for match in _TEMPORAL_CONTEXT_COMMA.finditer(text)
    )
    for temporal in _TEMPORAL_ITEM_START.finditer(text):
        if re.fullmatch(r"(?:1[5-9]|20)\d{2}", temporal.group(0)):
            prefix = text[max(0, temporal.start() - 32) : temporal.start()]
            if not re.search(
                r"\b(?:as\s+of|before|after|by|date|during|from|on|revision|since|"
                r"snapshot|starting|through|to)\s*$",
                prefix,
                re.IGNORECASE,
            ):
                continue
        comma = temporal.end()
        if comma >= len(text) or text[comma] != ",":
            continue
        following = text[comma + 1 :].lstrip()
        next_item = re.sub(r"^(?:and|or)\s+", "", following, flags=re.I)
        if not _TEMPORAL_ITEM_START.match(next_item):
            protected.add(comma)
    protected.update(
        index
        for match in _EVIDENCE_ID_LIST.finditer(text)
        for index in range(match.start(), match.end())
        if text[index] == ","
    )
    protected.update(
        index
        for index, character in enumerate(text)
        if (
            character == ","
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        )
    )
    return protected


def _compact_nominal_items(items: list[str]) -> bool:
    if not 2 <= len(items) <= 12:
        return False
    if sum(len(item.split()) for item in items) > 40:
        return False
    return all(not _FINITE_VERB.search(item) for item in items)
