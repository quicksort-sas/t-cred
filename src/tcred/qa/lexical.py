from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from numpy.typing import NDArray

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """Small exact BM25 index used as the sparse half of hybrid retrieval."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.k1 = k1
        self.b = b
        self.document_count = len(documents)
        tokenized = [tokenize(document) for document in documents]
        self.lengths = np.asarray([len(tokens) for tokens in tokenized], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) or 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for index, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            document_frequency.update(counts)
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))
        self.idf = {
            token: math.log(1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> NDArray[np.float32]:
        scores = np.zeros(self.document_count, dtype=np.float32)
        for token in dict.fromkeys(tokenize(query)):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, frequency in self.postings[token]:
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / self.average_length
                )
                scores[index] += idf * numerator / denominator
        return scores
