from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


class NearDuplicateIndex:
    """MinHash candidate index with exact word-trigram Jaccard verification."""

    def __init__(
        self,
        *,
        threshold: float,
        candidate_threshold: float,
        num_perm: int,
        seed: int = 20260816,
    ) -> None:
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError as exc:  # pragma: no cover - optional environment
            raise RuntimeError("Install the metrics-trainable optional dependencies") from exc
        if not 0.0 < candidate_threshold <= threshold <= 1.0:
            raise ValueError(
                "Expected 0 < candidate_threshold <= exact threshold <= 1"
            )
        self.threshold = threshold
        self.candidate_threshold = candidate_threshold
        self.num_perm = num_perm
        self.seed = seed
        self._MinHash = MinHash
        self._lsh = MinHashLSH(threshold=candidate_threshold, num_perm=num_perm)
        self._tokens_by_key: dict[str, frozenset[str]] = {}

    def add(self, key: str, text: str) -> None:
        if key in self._tokens_by_key:
            raise ValueError(f"Near-duplicate index key already exists: {key}")
        tokens = word_trigram_shingles(text)
        self._lsh.insert(key, self._signature(tokens))
        self._tokens_by_key[key] = tokens

    def matches(self, text: str) -> list[tuple[str, float]]:
        tokens = word_trigram_shingles(text)
        signature = self._signature(tokens)
        matches = []
        for key in sorted(self._lsh.query(signature)):
            similarity = jaccard(tokens, self._tokens_by_key[key])
            if similarity >= self.threshold:
                matches.append((key, similarity))
        return matches

    def has_match(self, text: str) -> bool:
        return bool(self.matches(text))

    def __len__(self) -> int:
        return len(self._tokens_by_key)

    def _signature(self, tokens: frozenset[str]) -> Any:
        signature = self._MinHash(num_perm=self.num_perm, seed=self.seed)
        for token in sorted(tokens):
            signature.update(token.encode("utf-8"))
        return signature


def word_trigram_shingles(text: str) -> frozenset[str]:
    tokens = _TOKEN.findall(text.casefold())
    if len(tokens) < 3:
        return frozenset(tokens or ["<empty>"])
    return frozenset("\x1f".join(tokens[index : index + 3]) for index in range(len(tokens) - 2))


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    return len(first & second) / len(first | second)
