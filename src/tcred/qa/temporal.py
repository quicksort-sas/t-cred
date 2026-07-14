from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime

from tcred.dataset.models import TemporalInterval
from tcred.qa.corpus import RuntimeCorpus, RuntimeFact
from tcred.qa.models import RetrievalHit, TemporalIntent

_DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
_ON_DATE_PATTERN = re.compile(
    rf"\bon\s+{_DATE_PATTERN.pattern}",
    re.IGNORECASE,
)
_SNAPSHOT_DATE_CLAUSE = re.compile(
    rf"\busing\s+the\s+evidence\s+snapshot\s+dated\s+{_DATE_PATTERN.pattern}\s*,?\s*",
    re.IGNORECASE,
)


def parse_temporal_intent(question: str) -> TemporalIntent:
    query_text = _SNAPSHOT_DATE_CLAUSE.sub("", question)
    lowered = query_text.casefold()
    dates = [_parse_date(value) for value in _DATE_PATTERN.findall(query_text)]
    operator = _operator(lowered)
    if operator == "unknown" and dates and _ON_DATE_PATTERN.search(query_text):
        operator = "as_of"
    start = dates[0] if dates else None
    end = dates[1] if len(dates) > 1 else start
    confidence = "high" if operator != "unknown" else ("medium" if dates else "low")
    if operator in {"during", "between"} and len(dates) < 2:
        confidence = "medium"
    return TemporalIntent(
        operator=operator,
        query_start=start,
        query_end=end,
        confidence=confidence,
        explanation=(
            f"Parsed operator={operator}; dates="
            f"{', '.join(day.isoformat() for day in dates) if dates else 'none'}"
            + ("; ignored evidence snapshot date" if query_text != question else "")
        ),
    )


class TemporalRanker:
    def __init__(self, corpus: RuntimeCorpus) -> None:
        self.corpus = corpus

    def rank(
        self,
        hits: list[RetrievalHit],
        *,
        intent: TemporalIntent,
        top_k: int,
        snapshot_id: str,
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        by_group: defaultdict[tuple[str, str, str], list[RuntimeFact]] = defaultdict(list)
        for index in self.corpus.visible_indices(snapshot_id):
            fact = self.corpus.documents[index].fact
            by_group[self.corpus.fact_group_key(fact)].append(fact)

        max_hybrid = max(hit.score for hit in hits) or 1.0
        scored: list[tuple[bool, float, RetrievalHit]] = []
        for hit in hits:
            fact = self.corpus.fact_by_id[hit.fact_id]
            peers = by_group[self.corpus.fact_group_key(fact)]
            temporal_score = score_temporal_compatibility(fact, intent=intent, peers=peers)
            combined = 0.55 * (hit.score / max_hybrid) + 0.45 * temporal_score
            accepted = temporal_score > 0.0 or intent.confidence == "low"
            scored.append(
                (
                    accepted,
                    combined,
                    hit.model_copy(
                        update={
                            "score": combined,
                            "temporal_score": temporal_score,
                        }
                    ),
                )
            )

        pool = [item for item in scored if item[0]]
        ordered = sorted(pool, key=lambda item: (-item[1], item[2].fact_id))[:top_k]
        return [item[2].model_copy(update={"rank": rank}) for rank, item in enumerate(ordered, 1)]


def score_temporal_compatibility(
    fact: RuntimeFact,
    *,
    intent: TemporalIntent,
    peers: list[RuntimeFact],
) -> float:
    interval = fact.valid_time
    if interval.type == "unknown" or interval.start is None:
        return 0.15

    operator = intent.operator
    reference = intent.query_start
    if operator in {"during", "between"} and reference and intent.query_end:
        query_interval = TemporalInterval(start=reference, end=intent.query_end)
        return 1.0 if interval.overlaps(query_interval) else 0.0
    if operator in {"current", "as_of", "effective"} and reference:
        return 1.0 if interval.contains(reference) else 0.0
    if operator in {"previous", "before", "expired"} and reference:
        return _before_score(interval, reference)
    if operator in {"next", "after"} and reference:
        return _after_score(interval, reference)
    if operator in {"latest", "last"} and reference:
        if interval.start > reference:
            return 0.0
        return _relative_recency(fact, peers, latest=True, cutoff=reference)
    if operator == "first":
        return _relative_recency(fact, peers, latest=False)
    if operator in {"current", "latest", "last"}:
        return _relative_recency(fact, peers, latest=True)
    if operator == "previous":
        return _relative_ordinal(fact, peers, position_from_end=2)
    if operator == "next":
        return _relative_ordinal(fact, peers, position_from_end=1)
    return 0.5


def _operator(question: str) -> str:
    checks = (
        ("immediately before", "previous"),
        ("previous", "previous"),
        ("immediately after", "next"),
        ("most recently expired", "expired"),
        ("expired", "expired"),
        ("between", "between"),
        ("during", "during"),
        (" before ", "before"),
        (" after ", "after"),
        ("first recorded", "first"),
        ("latest recorded", "latest"),
        ("effective on", "effective"),
        ("as of", "current" if "current" in question else "as_of"),
        ("currently", "current"),
        ("current", "current"),
        ("latest", "latest"),
        ("last", "last"),
    )
    for marker, operator in checks:
        if marker in question:
            return operator
    return "unknown"


def _parse_date(value: str) -> date:
    return datetime.strptime(value.title(), "%B %d, %Y").date()


def _before_score(interval: TemporalInterval, reference: date) -> float:
    if interval.end is None or interval.end >= reference:
        return 0.0
    years = max((reference - interval.end).days / 365.25, 0.0)
    return 1.0 / (1.0 + years)


def _after_score(interval: TemporalInterval, reference: date) -> float:
    if interval.start is None or interval.start <= reference:
        return 0.0
    years = max((interval.start - reference).days / 365.25, 0.0)
    return 1.0 / (1.0 + years)


def _relative_recency(
    fact: RuntimeFact,
    peers: list[RuntimeFact],
    *,
    latest: bool,
    cutoff: date | None = None,
) -> float:
    eligible = [
        peer
        for peer in peers
        if peer.valid_time.start is not None and (cutoff is None or peer.valid_time.start <= cutoff)
    ]
    if not eligible or fact not in eligible:
        return 0.0
    ordered = sorted(eligible, key=lambda peer: (peer.valid_time.start, peer.fact_id))
    target = ordered[-1] if latest else ordered[0]
    if fact.fact_id == target.fact_id:
        return 1.0
    distance = abs(ordered.index(fact) - ordered.index(target))
    return max(0.2, 1.0 - 0.25 * distance)


def _relative_ordinal(
    fact: RuntimeFact,
    peers: list[RuntimeFact],
    *,
    position_from_end: int,
) -> float:
    eligible = [peer for peer in peers if peer.valid_time.start is not None]
    ordered = sorted(eligible, key=lambda peer: (peer.valid_time.start, peer.fact_id))
    if len(ordered) < position_from_end:
        return 0.4
    target = ordered[-position_from_end]
    return 1.0 if fact.fact_id == target.fact_id else 0.25
