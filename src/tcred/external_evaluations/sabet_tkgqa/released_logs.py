from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tcred.external_evaluations.sabet_tkgqa.schema import ReleasedEvaluation

_CONFIG_HEADER = "===== ARGUMENTS ====="
_CONFIG_FOOTER = "====================="
_SPLIT = re.compile(r"^Split\s+(.+?)\s*$")
_HITS = re.compile(r"^Hits at (1|10):\s*([0-9]*\.?[0-9]+)\s*$")
_GROUP = re.compile(
    r"^(.+?)\s+([0-9]*\.?[0-9]+)\s+total questions:\s*([0-9]+)\s*$"
)


def parse_released_log(
    path: Path,
    *,
    artifact_root: Path | None = None,
) -> list[ReleasedEvaluation]:
    """Parse every complete evaluation block without guessing which run is authoritative."""

    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    config = _parse_first_config(text)
    source_path = str(path.relative_to(artifact_root)) if artifact_root else str(path)
    dataset = config.get("dataset_name") or _dataset_from_path(path)
    artifact_label = path.stem
    source_sha256 = hashlib.sha256(payload).hexdigest()

    evaluations: list[ReleasedEvaluation] = []
    current_split: str | None = None
    current: dict[int, float] = {}
    groups: dict[int, dict[str, float]] = {1: {}, 10: {}}
    counts: dict[str, int] = {}

    def flush() -> None:
        nonlocal current_split, current, groups, counts
        if current_split is not None and 1 in current and 10 in current:
            evaluations.append(
                ReleasedEvaluation(
                    source_path=source_path,
                    source_sha256=source_sha256,
                    dataset=dataset,
                    artifact_label=artifact_label,
                    config=config,
                    split=current_split,
                    ordinal=len(evaluations),
                    hits_at_1=current[1],
                    hits_at_10=current[10],
                    group_scores_at_1=groups[1],
                    group_scores_at_10=groups[10],
                    group_counts=counts,
                    inferred_example_count=_infer_example_count(counts),
                )
            )
        current_split = None
        current = {}
        groups = {1: {}, 10: {}}
        counts = {}

    active_k: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        split_match = _SPLIT.fullmatch(line)
        if split_match:
            flush()
            current_split = split_match.group(1)
            active_k = None
            continue
        if current_split is None:
            continue
        hits_match = _HITS.fullmatch(line)
        if hits_match:
            active_k = int(hits_match.group(1))
            current[active_k] = float(hits_match.group(2))
            continue
        group_match = _GROUP.fullmatch(line)
        if group_match and active_k in {1, 10}:
            name = group_match.group(1).strip()
            groups[active_k][name] = float(group_match.group(2))
            count = int(group_match.group(3))
            previous = counts.setdefault(name, count)
            if previous != count:
                raise ValueError(f"Inconsistent group count for {name!r} in {path}")
    flush()
    return evaluations


def final_test_evaluation(evaluations: list[ReleasedEvaluation]) -> ReleasedEvaluation:
    tests = [row for row in evaluations if row.split.casefold() == "test"]
    if not tests:
        raise ValueError("Released log contains no complete test evaluation")
    return tests[-1]


def _parse_first_config(text: str) -> dict[str, str]:
    start = text.find(_CONFIG_HEADER)
    if start < 0:
        return {}
    end = text.find(_CONFIG_FOOTER, start + len(_CONFIG_HEADER))
    if end < 0:
        return {}
    output: dict[str, str] = {}
    for line in text[start + len(_CONFIG_HEADER) : end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        output[key.strip()] = value.strip()
    return output


def _dataset_from_path(path: Path) -> str:
    parts = path.parts
    try:
        return parts[parts.index("logs") + 1]
    except (ValueError, IndexError):
        return "unknown"


def _infer_example_count(counts: dict[str, int]) -> int | None:
    for key in ("simple", "complex"):
        if key in counts and {"simple", "complex"}.issubset(counts):
            return counts["simple"] + counts["complex"]
    if "complex" in counts:
        return counts["complex"]
    if {"entity", "time"}.issubset(counts):
        return counts["entity"] + counts["time"]
    return None
