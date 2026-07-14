from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
from pathlib import Path

import httpx
import numpy as np
import orjson
from dotenv import load_dotenv
from numpy.typing import NDArray

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


class EmbeddingClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 90.0,
        max_attempts: int = 8,
    ) -> None:
        load_dotenv()
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.max_attempts = max_attempts
        if provider == "mistral":
            endpoint = "https://api.mistral.ai/v1"
            self.api_key = os.getenv("LLM_MISTRAL_API_KEY")
            if dimensions != 1024:
                raise ValueError("mistral-embed produces fixed 1024-dimensional vectors")
        elif provider == "openai":
            endpoint = "https://api.openai.com/v1"
            self.api_key = os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        if not self.api_key:
            raise RuntimeError(f"An API key is required for embedding provider {provider}")
        self._client = httpx.AsyncClient(
            base_url=endpoint,
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def __aenter__(self) -> EmbeddingClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str], *, batch_size: int = 128) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        batches: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), batch_size):
            batches.append(await self._embed_batch(texts[start : start + batch_size]))
        matrix = np.vstack(batches).astype(np.float32, copy=False)
        if matrix.shape != (len(texts), self.dimensions):
            raise RuntimeError(
                f"Embedding response shape {matrix.shape} does not match "
                f"({len(texts)}, {self.dimensions})"
            )
        return normalize_rows(matrix)

    async def _embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        payload: dict[str, object] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.provider == "openai":
            payload["dimensions"] = self.dimensions
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.post("/embeddings", json=payload)
                if response.is_error:
                    if response.status_code != 429 and response.status_code < 500:
                        raise PermanentEmbeddingError(
                            f"Embedding API returned {response.status_code}: {response.text[:500]}"
                        )
                    error_code = (response.json().get("error") or {}).get("code")
                    if error_code == "insufficient_quota":
                        raise PermanentEmbeddingError(
                            f"Embedding API quota is unavailable: {response.text[:500]}"
                        )
                    response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                return np.asarray([item["embedding"] for item in data], dtype=np.float32)
            except PermanentEmbeddingError:
                raise
            except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        f"Embedding request failed after {self.max_attempts} attempts"
                    ) from exc
                retry_after = (
                    response.headers.get("retry-after") if "response" in locals() else None
                )
                try:
                    delay = float(retry_after) if retry_after else float(2**attempt)
                except ValueError:
                    delay = float(2**attempt)
                await asyncio.sleep(min(delay + random.random(), 30.0))
        raise AssertionError("unreachable")


async def load_or_create_embeddings(
    *,
    client: EmbeddingClient,
    ids: list[str],
    texts: list[str],
    cache_dir: Path,
    namespace: str,
) -> NDArray[np.float32]:
    if len(ids) != len(texts):
        raise ValueError("Embedding ids and texts must have the same length")
    cache_dir.mkdir(parents=True, exist_ok=True)
    content_hash = embedding_content_hash(
        ids=ids,
        texts=texts,
        provider=client.provider,
        model=client.model,
        dimensions=client.dimensions,
    )
    safe_model = _SAFE_NAME.sub("_", client.model)
    path = cache_dir / (
        f"{namespace}.{client.provider}.{safe_model}.{client.dimensions}.{content_hash[:16]}.npz"
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            cached_ids = cached["ids"].tolist()
            matrix = cached["embeddings"].astype(np.float32, copy=False)
        if (
            cached_ids == ids
            and matrix.shape == (len(ids), client.dimensions)
            and np.isfinite(matrix).all()
        ):
            return matrix

    matrix = await client.embed(texts)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        ids=np.asarray(ids, dtype=np.str_),
        embeddings=matrix,
    )
    temporary.replace(path)
    metadata = {
        "format_version": 1,
        "namespace": namespace,
        "provider": client.provider,
        "model": client.model,
        "dimensions": client.dimensions,
        "count": len(ids),
        "content_sha256": content_hash,
        "cache_file": str(path),
    }
    path.with_suffix(".json").write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return matrix


def embedding_content_hash(
    *,
    ids: list[str],
    texts: list[str],
    provider: str,
    model: str,
    dimensions: int,
) -> str:
    digest = hashlib.sha256(f"{provider}:{model}:{dimensions}".encode())
    for item_id, text in zip(ids, texts, strict=True):
        digest.update(b"\0")
        digest.update(item_id.encode())
        digest.update(b"\0")
        digest.update(text.encode())
    return digest.hexdigest()


def normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


class PermanentEmbeddingError(RuntimeError):
    pass
