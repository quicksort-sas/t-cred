from __future__ import annotations

import re


def normalize_visible_text(value: str) -> str:
    """Normalize text for human-visible identity and duplicate checks."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def annotation_plain_text(value: str) -> str:
    """Render model text exactly as it is exposed in blind annotation cards."""

    value = re.sub(r"[`*_#>]", "", value)
    value = re.sub(r"(?m)^\s*[-+]\s+", "", value)
    return " ".join(value.split())
