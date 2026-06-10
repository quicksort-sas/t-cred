from __future__ import annotations

from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts" / "dataset"


def load_prompt(name: str) -> str:
    path = PROMPT_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
