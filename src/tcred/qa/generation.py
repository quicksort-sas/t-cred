from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from tcred.dataset.graph import fact_endpoint_ids, relation_label
from tcred.qa.corpus import RuntimeCorpus
from tcred.qa.models import (
    LLMAnswerPayload,
    QASystemName,
    RetrievalResult,
    TokenUsage,
)

PROMPT_VERSION = "qa_grounded_v4_no_internal_handles"
STRUCTURED_OUTPUT_RETRY_TOKEN_LIMITS = (384, 512, 640, 640, 640, 640)
INTERNAL_PATH_REFERENCE = re.compile(
    r"(?i)(?<!\w)(?:(?:retrieved|retrieval|candidate|graph)\s*[_-]?\s*)?"
    r"path\s*[_-]?\s*\d+\b"
)
SYSTEM_INSTRUCTION = """Answer the question using only the supplied evidence.
Do not use outside knowledge. Resolve conflicts using the evidence itself. If the evidence is
insufficient or does not establish an answer for the requested time, say that the answer cannot be
determined. Keep the answer concise. Cite only supplied evidence handles such as E1 that directly
support the answer. Put handles only in cited_evidence_ids; do not write handles in answer_text.
Never mention internal path labels (for example, retrieved_path_01 or Path 1) in answer_text.
Return the required JSON object."""


@dataclass(frozen=True)
class GenerationResult:
    payload: LLMAnswerPayload
    usage: TokenUsage
    latency_ms: int
    prompt_sha256: str


class ChatAnswerClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str,
        timeout_seconds: float,
        seed: int,
        max_attempts: int = 8,
    ) -> None:
        load_dotenv()
        endpoint, key = _provider(provider)
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            base_url=endpoint,
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {key}"},
        )

    async def __aenter__(self) -> ChatAnswerClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def answer(self, *, question: str, context: str) -> GenerationResult:
        user_prompt, prompt_hash = answer_prompt(question=question, context=context)
        started = time.perf_counter()
        parse_error: Exception | None = None
        for parse_attempt, max_tokens in enumerate(STRUCTURED_OUTPUT_RETRY_TOKEN_LIMITS):
            request = answer_request_payload(
                provider=self.provider,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed + parse_attempt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            response = await self._post_with_retry(request)
            try:
                payload, usage = parse_generation_body(response.json())
            except (ValidationError, ValueError, KeyError) as exc:
                parse_error = exc
                continue
            return GenerationResult(
                payload=payload,
                usage=usage,
                latency_ms=round((time.perf_counter() - started) * 1000),
                prompt_sha256=prompt_hash,
            )
        raise RuntimeError(
            "Answer remained invalid after structured-output retries"
        ) from parse_error

    async def _post_with_retry(self, payload: dict[str, object]) -> httpx.Response:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError("Answer request failed after transport retries") from exc
                await asyncio.sleep(min(2**attempt + random.random(), 30.0))
                continue
            if not response.is_error:
                return response
            if response.status_code != 429 and response.status_code < 500:
                raise RuntimeError(
                    f"Answer API returned {response.status_code}: {response.text[:1000]}"
                )
            if attempt == self.max_attempts:
                raise RuntimeError(
                    f"Answer API remained unavailable: {response.status_code} {response.text[:500]}"
                )
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else float(2**attempt)
            except ValueError:
                delay = float(2**attempt)
            await asyncio.sleep(min(delay + random.random(), 30.0))
        raise AssertionError("unreachable")


def render_retrieval_context(
    *,
    retrieval: RetrievalResult,
    corpus: RuntimeCorpus,
) -> str:
    include_time = retrieval.system_name in {
        QASystemName.TEMPORAL_FILTER_RAG,
        QASystemName.TEMPORAL_GRAPH_RAG,
    }
    graph_mode = retrieval.system_name in {
        QASystemName.GRAPH_RAG_NO_TIME,
        QASystemName.TEMPORAL_GRAPH_RAG,
    }
    handle_by_fact = {fact_id: handle for handle, fact_id in retrieval.evidence_handle_map.items()}
    sections: list[str] = []
    if graph_mode and retrieval.graph_paths:
        for path in retrieval.graph_paths:
            lines = [f"Path {path.path_id}:"]
            directions = path.traversal_directions or ["forward"] * len(path.fact_ids)
            for fact_id, direction in zip(path.fact_ids, directions, strict=True):
                lines.append(
                    _fact_line(
                        corpus,
                        fact_id,
                        handle=handle_by_fact[fact_id],
                        include_time=include_time,
                        traversal_direction=direction,
                    )
                )
            sections.append("\n".join(lines))
    path_fact_ids = {fact_id for path in retrieval.graph_paths for fact_id in path.fact_ids}
    remaining = [hit for hit in retrieval.hits if hit.fact_id not in path_fact_ids]
    if remaining:
        lines = ["Additional evidence:"] if graph_mode else []
        lines.extend(
            _fact_line(
                corpus,
                hit.fact_id,
                handle=handle_by_fact[hit.fact_id],
                include_time=include_time,
            )
            for hit in remaining
        )
        sections.append("\n".join(lines))
    return (
        "\n\n".join(section for section in sections if section).strip() or "No evidence retrieved."
    )


def context_sha256(context: str) -> str:
    return hashlib.sha256(context.encode()).hexdigest()


def answer_json_schema() -> dict[str, object]:
    """Return the minimal Pydantic schema without descriptive title fields."""
    schema = LLMAnswerPayload.model_json_schema()
    _remove_titles(schema)
    return schema


def answer_prompt(*, question: str, context: str) -> tuple[str, str]:
    user_prompt = f"Question:\n{question}\n\nEvidence:\n{context}"
    prompt_hash = hashlib.sha256(
        f"{PROMPT_VERSION}\0{SYSTEM_INSTRUCTION}\0{user_prompt}".encode()
    ).hexdigest()
    return user_prompt, prompt_hash


def answer_request_payload(
    *,
    provider: str,
    model: str,
    reasoning_effort: str,
    seed: int,
    user_prompt: str,
    max_tokens: int = 384,
) -> dict[str, object]:
    schema = answer_json_schema()
    if provider == "groq" and model.startswith("openai/gpt-oss-"):
        response_format: dict[str, object] = {
            "type": "json_schema",
            "json_schema": {
                "name": "qa_answer",
                "strict": True,
                "schema": schema,
            },
        }
    elif provider == "mistral":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "qa_answer",
                "schema": schema,
            },
        }
    else:
        response_format = {"type": "json_object"}
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": response_format,
        "temperature": 0,
    }
    if provider == "mistral":
        payload.update(max_tokens=max_tokens, random_seed=seed)
    else:
        payload.update(max_completion_tokens=max_tokens, seed=seed)
    if provider == "groq" and model.startswith("openai/gpt-oss-"):
        payload["reasoning_effort"] = reasoning_effort
    return payload


def parse_generation_body(body: dict[str, object]) -> tuple[LLMAnswerPayload, TokenUsage]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Generation response contains no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("Generation response contains no text content")
    payload = LLMAnswerPayload.model_validate_json(content)
    if INTERNAL_PATH_REFERENCE.search(payload.answer_text):
        raise ValueError("Generated answer exposes an internal retrieval-path label")
    raw_usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    details = (
        raw_usage.get("completion_tokens_details")
        if isinstance(raw_usage.get("completion_tokens_details"), dict)
        else {}
    )
    usage = TokenUsage(
        input_tokens=int(raw_usage.get("prompt_tokens") or 0),
        output_tokens=int(raw_usage.get("completion_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        total_tokens=int(raw_usage.get("total_tokens") or 0),
    )
    return payload, usage


def evidence_handle_map(retrieval: RetrievalResult) -> dict[str, str]:
    """Assign per-question opaque handles in display order."""
    ordered: list[str] = []
    for path in retrieval.graph_paths:
        ordered.extend(path.fact_ids)
    ordered.extend(hit.fact_id for hit in retrieval.hits)
    unique = list(dict.fromkeys(ordered))
    return {f"E{index}": fact_id for index, fact_id in enumerate(unique, start=1)}


def merge_inline_citation_handles(answer_text: str, cited_handles: list[str]) -> list[str]:
    """Preserve explicit inline citations even when the structured list omits them."""
    inline = re.findall(r"\bE\d+\b", answer_text)
    return list(dict.fromkeys([*cited_handles, *inline]))


def canonicalize_inline_citations(
    answer_text: str,
    *,
    supplied_handles: dict[str, str],
    cited_handles: list[str],
) -> str:
    """Resolve inline handles before reindexing and mark unresolved citations."""
    inline = set(re.findall(r"\bE\d+\b", answer_text))
    undeclared = inline - set(cited_handles)
    if undeclared:
        raise ValueError(
            "Inline evidence handles must be merged into cited_evidence_ids before "
            f"canonicalization: undeclared={sorted(undeclared)}"
        )
    canonical = answer_text
    for handle in sorted(inline, key=len, reverse=True):
        canonical = re.sub(
            rf"\b{re.escape(handle)}\b",
            supplied_handles.get(handle, f"unresolved citation {handle}"),
            canonical,
        )
    return canonical


def _fact_line(
    corpus: RuntimeCorpus,
    fact_id: str,
    *,
    handle: str,
    include_time: bool,
    traversal_direction: str = "forward",
) -> str:
    fact = corpus.fact_by_id[fact_id]
    source_id, target_id = fact_endpoint_ids(
        fact,
        traversal_direction=traversal_direction,
    )
    source = corpus.entities.get(source_id)
    target = corpus.entities.get(target_id)
    qualifier = corpus.entities.get(fact.object_id or "")
    relation = fact.source_relation_label or relation_label(
        fact.relation,
        object_label=qualifier.name if qualifier else "",
    )
    if traversal_direction == "reverse":
        relation = f"reverse traversal of {relation}"
    direction = (
        f"{source.name if source else source_id} --"
        f"{relation}--> "
        f"{target.name if target else target_id}"
    )
    evidence = fact.paraphrased_evidence or fact.canonical_evidence
    timing = ""
    if include_time:
        start = fact.valid_time.start.isoformat() if fact.valid_time.start else "unknown"
        end = fact.valid_time.end.isoformat() if fact.valid_time.end else "open"
        publication = fact.publication_time.isoformat() if fact.publication_time else "unknown"
        timing = f" [valid={start}..{end}; published={publication}]"
    return f"[{handle}] {direction}.{timing} {evidence}"


def _provider(provider: str) -> tuple[str, str]:
    settings = {
        "groq": ("https://api.groq.com/openai/v1", "LLM_GROQ_API_KEY"),
        "openai": ("https://api.openai.com/v1", "LLM_OPENAI_API_KEY"),
        "mistral": ("https://api.mistral.ai/v1", "LLM_MISTRAL_API_KEY"),
    }
    if provider not in settings:
        raise ValueError(f"Unsupported answer provider: {provider}")
    endpoint, variable = settings[provider]
    key = os.getenv(variable)
    if provider == "openai":
        key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"{variable} is required for provider {provider}")
    return endpoint, key


def _remove_titles(value: object) -> None:
    if isinstance(value, dict):
        value.pop("title", None)
        for child in value.values():
            _remove_titles(child)
    elif isinstance(value, list):
        for child in value:
            _remove_titles(child)
