from __future__ import annotations

import re
from typing import Literal

ResponseDecisionKind = Literal["answer", "refusal", "hybrid"]

_REFUSAL_CUE = re.compile(
    r"""
    (?:
        can(?:not|'t)\s+(?:be\s+)?determin(?:e|ed)
        |could\s+not\s+(?:be\s+)?determin(?:e|ed)
        |unable\s+to\s+(?:answer|determine)
        |insufficient(?:\s+\w+){0,3}\s+(?:to\s+)?(?:answer|determine|establish)
        |not\s+enough\s+(?:information|evidence)
        |no(?:\s+\w+){0,2}\s+(?:information|evidence)
            (?:\s+(?:is|was))?\s+(?:available|provided|shown|found)
        |not(?:\s+\w+){0,3}\s+sufficient\s+to\s+(?:answer|determine|establish)
        |no\s+(?:supported\s+)?answer
        |not\s+possible\s+to\s+determine
        |does\s+not(?:\s+\w+){0,2}\s+(?:provide|state|specify|establish)
        |do\s+not(?:\s+\w+){0,2}\s+(?:provide|state|specify|establish)
        |not\s+(?:provided|stated|specified|established|available)
        |(?:evidence|information)\s+(?:is|was)\s+(?:missing|unavailable)
        |unclear(?:\s+\w+){0,6}\s+(?:evidence|information)
        |conflicting\s+(?:evidence|claims|information|ages|values)
        |conflicts?\s+(?:on|with|about)
        |no\s+definitive\s+answer
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9]+")
_CONTRAST = re.compile(r"\b(?:but|however|although|yet)\b", flags=re.IGNORECASE)


def response_decision_kind(answer_text: str) -> ResponseDecisionKind:
    """Classify whether a response answers, refuses, or combines both.

    This detector controls field applicability only. It never supplies a correctness label.
    """

    text = " ".join(answer_text.split())
    refusal_matches = list(_REFUSAL_CUE.finditer(text))
    if not refusal_matches:
        return "answer"

    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT.split(text) if sentence.strip()]
    if any(
        not _REFUSAL_CUE.search(sentence) and len(_WORD.findall(sentence)) >= 4
        for sentence in sentences
    ):
        return "hybrid"

    first_refusal = refusal_matches[0]
    prefix = text[: first_refusal.start()].strip(" ,;:-")
    if (
        ("(" in prefix or _CONTRAST.search(text[: first_refusal.start()]))
        and len(_WORD.findall(prefix)) >= 4
        and not re.search(r"\b(?:is|was|are|were|does|do)\s*$", prefix, flags=re.IGNORECASE)
    ):
        return "hybrid"

    contrast = _CONTRAST.search(text)
    if contrast and contrast.start() < first_refusal.start():
        asserted = text[: contrast.start()].strip(" ,;:-")
        if len(_WORD.findall(asserted)) >= 4:
            return "hybrid"

    return "refusal"
