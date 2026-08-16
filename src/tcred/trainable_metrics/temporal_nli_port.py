from __future__ import annotations

import csv
import hashlib
import html
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from tcred.trainable_metrics.source_io import file_sha256

PORT_VERSION = "tcred-temporal-nli-uds-relation-port-v1"
_HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", re.IGNORECASE)

_RELATION_TEMPLATES = (
    (0, 1, "started", "started"),
    (0, 1, "started", "ended"),
    (0, 1, "ended", "started"),
    (0, 1, "ended", "ended"),
    (1, 0, "started", "started"),
    (1, 0, "started", "ended"),
    (1, 0, "ended", "started"),
    (1, 0, "ended", "ended"),
)


@dataclass(frozen=True)
class UdToken:
    form: str
    lemma: str
    upos: str


def build_public_uds_temporal_nli(
    *,
    uds_t_path: Path,
    ud_ewt_dir: Path,
    output_path: Path,
    split: str = "train",
    minimum_relation_confidence: float = 3.0,
    require_verb_events: bool = True,
) -> dict[str, Any]:
    sentences = _load_ud_sentences(ud_ewt_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    input_rows = 0
    skipped = Counter()
    seen_event_pairs: set[str] = set()

    fields = [
        "context",
        "hypothesis",
        "pair-id",
        "source_id",
        "type-of-inference",
        "relation-template",
        "split",
        "label",
    ]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with (
        uds_t_path.open("r", encoding="utf-8", newline="") as source_handle,
        temporary.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(source_handle, delimiter="\t")
        writer = csv.DictWriter(output_handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get("Split") != split:
                continue
            input_rows += 1
            try:
                confidence = float(row.get("Relation.Confidence", "4"))
            except ValueError:
                skipped["invalid_relation_confidence"] += 1
                continue
            if confidence < minimum_relation_confidence:
                skipped["relation_confidence_below_threshold"] += 1
                continue
            event_pair_id = f"{row['Event1.ID']}|{row['Event2.ID']}"
            if event_pair_id in seen_event_pairs:
                skipped["duplicate_event_pair_annotation"] += 1
                continue
            seen_event_pairs.add(event_pair_id)
            converted = _convert_row(
                row,
                sentences=sentences,
                require_verb_events=require_verb_events,
            )
            if not converted:
                skipped["invalid_or_ambiguous_event_pair"] += 1
                continue
            for item in converted:
                writer.writerow(item)
                counts[item["label"]] += 1
    temporary.replace(output_path)

    manifest = {
        "schema_version": "tcred-temporal-nli-port-manifest-v1",
        "port_version": PORT_VERSION,
        "created_at": datetime.now(UTC),
        "paper_method": (
            "Vashishtha et al. (Findings of EMNLP 2020), UDS-T temporal-order "
            "recasting with eight start/end templates"
        ),
        "split": split,
        "quality_filters": {
            "minimum_relation_confidence": minimum_relation_confidence,
            "require_verb_events": require_verb_events,
            "reject_unresolved_html_entities": True,
        },
        "input_rows": input_rows,
        "source_groups": len(seen_event_pairs),
        "output_rows": sum(counts.values()),
        "labels": dict(counts),
        "skipped": dict(skipped),
        "source_files": {
            "uds_t": {"path": str(uds_t_path), "sha256": file_sha256(uds_t_path)},
            "ud_ewt": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for path in sorted(ud_ewt_dir.glob("en-ud-*.conllu"))
            ],
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
        },
        "deviations_from_legacy_author_code": [
            (
                "The semantic relation vector, train threshold, and eight templates are preserved; "
                "the modern port reads CoNLL-U directly instead of loading the obsolete "
                "decomp stack."
            ),
            (
                "Event surfaces use the released UDS-T predicate span plus deterministic gerund "
                "inflection instead of PredPatt dependency expansion. This avoids unsupported old "
                "dependencies and is recorded as a protocol amendment, not claimed byte-identical."
            ),
            (
                "Quoted 'event' wrappers replace the legacy 'the <gerund>' surface so generated "
                "hypotheses remain readable without changing start/end relation semantics."
            ),
            (
                "The restricted RED/LDC corpus and non-UDS source corpora are not used "
                "in this file."
            ),
        ],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    return manifest


def _convert_row(
    row: dict[str, str],
    *,
    sentences: dict[str, list[UdToken]],
    require_verb_events: bool,
) -> list[dict[str, str]]:
    sentence_ids = (row["Sentence1.ID"], row["Sentence2.ID"])
    if any(sentence_id not in sentences for sentence_id in sentence_ids):
        return []
    first_tokens = sentences[sentence_ids[0]]
    second_tokens = sentences[sentence_ids[1]]
    context_tokens = (
        first_tokens
        if sentence_ids[0] == sentence_ids[1]
        else first_tokens + second_tokens
    )
    context = html.unescape(_detokenize([token.form for token in context_tokens]))

    event_tokens: list[UdToken] = []
    for sentence_id, token_value in (
        (row["Sentence1.ID"], row["Pred1.Token"]),
        (row["Sentence2.ID"], row["Pred2.Token"]),
    ):
        try:
            event_tokens.append(sentences[sentence_id][int(token_value)])
        except (IndexError, ValueError):
            return []
    event_phrases = (
        html.unescape(_event_phrase(row["Pred1.Text"], row["Pred1.Lemma"], event_tokens[0].upos)),
        html.unescape(_event_phrase(row["Pred2.Text"], row["Pred2.Lemma"], event_tokens[1].upos)),
    )
    if require_verb_events and any(token.upos != "VERB" for token in event_tokens):
        return []
    if (
        not all(event_phrases)
        or event_phrases[0].casefold() == event_phrases[1].casefold()
        or _HTML_ENTITY.search(context)
        or any(_HTML_ENTITY.search(phrase) for phrase in event_phrases)
    ):
        return []
    relation_vector = _relation_vector(row)
    if relation_vector is None:
        return []
    event_pair_id = f"{row['Event1.ID']}|{row['Event2.ID']}"
    result: list[dict[str, str]] = []
    for index, (first, second, first_boundary, second_boundary) in enumerate(
        _RELATION_TEMPLATES
    ):
        hypothesis = (
            f"{_event_subject(event_phrases[first], event_tokens[first].upos)} "
            f"{first_boundary} before "
            f"{_event_subject(event_phrases[second], event_tokens[second].upos)} "
            f"{second_boundary}."
        )
        result.append(
            {
                "context": context,
                "hypothesis": hypothesis[0].upper() + hypothesis[1:],
                "pair-id": f"{event_pair_id}:template:{index}",
                "source_id": event_pair_id,
                "type-of-inference": "temporal-relation",
                "relation-template": str(index),
                "split": row["Split"],
                "label": "entailed" if relation_vector[index] else "not-entailed",
            }
        )
    return result


def _relation_vector(row: dict[str, str]) -> tuple[bool, ...] | None:
    try:
        b1, e1, b2, e2 = (
            float(row["Pred1.Beg"]),
            float(row["Pred1.End"]),
            float(row["Pred2.Beg"]),
            float(row["Pred2.End"]),
        )
    except (KeyError, ValueError):
        return None
    minimum = min(b1, e1, b2, e2)
    adjusted = [value - minimum for value in (b1, e1, b2, e2)]
    maximum = max(adjusted)
    if maximum:
        b1, e1, b2, e2 = (round(value / maximum, 4) for value in adjusted)
    else:
        b1 = e1 = b2 = e2 = 0.0
    return (
        b1 < b2,
        b1 < e2,
        e1 < b2,
        e1 < e2,
        b2 < b1,
        b2 < e1,
        e2 < b1,
        e2 < e1,
    )


def _load_ud_sentences(ud_ewt_dir: Path) -> dict[str, list[UdToken]]:
    result: dict[str, list[UdToken]] = {}
    for path in sorted(ud_ewt_dir.glob("en-ud-*.conllu")):
        sentence_number = 0
        tokens: list[UdToken] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if not stripped:
                    if tokens:
                        sentence_number += 1
                        result[f"{path.name} {sentence_number}"] = tokens
                        tokens = []
                    continue
                if stripped.startswith("#"):
                    continue
                fields = stripped.split("\t")
                if len(fields) < 4 or not fields[0].isdigit():
                    continue
                tokens.append(UdToken(form=fields[1], lemma=fields[2], upos=fields[3]))
        if tokens:
            sentence_number += 1
            result[f"{path.name} {sentence_number}"] = tokens
    if not result:
        raise ValueError(f"No CoNLL-U sentences found under {ud_ewt_dir}")
    return result


def _event_phrase(surface: str, lemma: str, upos: str) -> str:
    normalized = " ".join(surface.split())
    if not normalized:
        return ""
    words = normalized.split()
    if len(words) > 1 and words[0].casefold() in {"am", "are", "be", "been", "is", "was", "were"}:
        return " ".join(words[1:])
    if upos == "VERB" and len(words) == 1:
        return _gerund(lemma or normalized)
    return normalized


def _gerund(lemma: str) -> str:
    try:
        from lemminflect import getInflection

        values = getInflection(lemma, tag="VBG")
        if values:
            return str(values[0])
    except ImportError:
        pass
    word = lemma.casefold()
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith(("ee", "ye")):
        return word[:-1] + "ing"
    return word + "ing"


def _event_subject(event_phrase: str, upos: str) -> str:
    del upos
    return f'the event "{event_phrase}"'


def _detokenize(tokens: list[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    for punctuation in (".", ",", ";", ":", "!", "?", ")", "]", "}"):
        text = text.replace(f" {punctuation}", punctuation)
    for punctuation in ("(", "[", "{"):
        text = text.replace(f"{punctuation} ", punctuation)
    return " ".join(text.split())


def converter_fingerprint() -> str:
    return hashlib.sha256(PORT_VERSION.encode()).hexdigest()
