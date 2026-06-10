from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

import httpx
import orjson
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from tcred.llm.prompts import load_prompt

ProviderName = Literal["openai", "anthropic", "mistral", "groq"]


@dataclass(frozen=True)
class ParaphraseResult:
    provider: str
    model: str
    prompt_name: str
    input_text: str
    output_text: str
    raw_response_excerpt: str = ""


class ParaphraseEquivalenceDecision(BaseModel):
    """Structured LLM-judge output for semantic equivalence checks."""

    equivalent: bool
    operator_preserved: bool
    time_preserved: bool
    entities_preserved: bool
    reason: str = Field(min_length=1, max_length=300)


class LLMParaphraseClient:
    def __init__(self, provider: ProviderName, model: str, timeout_seconds: int = 60) -> None:
        load_dotenv()
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def paraphrase_question(self, text: str) -> ParaphraseResult:
        return await self._run(prompt_name="question_paraphrase.md", input_text=text)

    async def paraphrase_evidence(self, text: str) -> ParaphraseResult:
        return await self._run(prompt_name="evidence_paraphrase.md", input_text=text)

    async def check_equivalence(
        self,
        *,
        canonical_text: str,
        paraphrased_text: str,
    ) -> ParaphraseEquivalenceDecision:
        system_prompt = load_prompt("paraphrase_equivalence_check.md")
        user_prompt = f"Canonical text:\n{canonical_text}\n\nParaphrased text:\n{paraphrased_text}"
        output_text = await self._complete(system_prompt=system_prompt, user_prompt=user_prompt)
        return _parse_equivalence_decision(output_text)

    async def _run(self, *, prompt_name: str, input_text: str) -> ParaphraseResult:
        system_prompt = load_prompt(prompt_name)
        output_text = await self._complete(system_prompt=system_prompt, user_prompt=input_text)
        cleaned = _clean_plain_text(output_text)
        return ParaphraseResult(
            provider=self.provider,
            model=self.model,
            prompt_name=prompt_name,
            input_text=input_text,
            output_text=cleaned,
            raw_response_excerpt=output_text[:500],
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
    )
    async def _complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            return await self._openai_response(system_prompt, user_prompt)
        if self.provider == "anthropic":
            return await self._anthropic_message(system_prompt, user_prompt)
        if self.provider == "mistral":
            return await self._openai_compatible_chat(
                base_url="https://api.mistral.ai/v1",
                key_env="LLM_MISTRAL_API_KEY",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        if self.provider == "groq":
            return await self._openai_compatible_chat(
                base_url="https://api.groq.com/openai/v1",
                key_env="LLM_GROQ_API_KEY",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    async def _openai_response(self, system_prompt: str, user_prompt: str) -> str:
        key = os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key is not configured")
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": 1000,
            "reasoning": {"effort": "minimal"},
            "store": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            _raise_for_status(response)
            payload = response.json()
        if "output_text" in payload:
            return str(payload["output_text"])
        chunks: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    chunks.append(str(content.get("text", "")))
        text = "".join(chunks).strip()
        if not text:
            excerpt = orjson.dumps(payload).decode("utf-8")[:1000]
            raise ValueError(f"OpenAI response did not contain output text: {excerpt}")
        return text

    async def _anthropic_message(self, system_prompt: str, user_prompt: str) -> str:
        key = os.getenv("LLM_ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Anthropic API key is not configured")
        payload = {
            "model": self.model,
            "max_tokens": 400,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )
            _raise_for_status(response)
            payload = response.json()
        chunks = [
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ]
        text = "".join(chunks).strip()
        if not text:
            raise ValueError("Anthropic response did not contain text")
        return text

    async def _openai_compatible_chat(
        self,
        *,
        base_url: str,
        key_env: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        key = os.getenv(key_env)
        if not key:
            raise ValueError(f"{key_env} is not configured")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            _raise_for_status(response)
            payload = response.json()
        try:
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError) as exc:
            raise ValueError("Chat-completions response did not contain content") from exc


def _parse_equivalence_decision(text: str) -> ParaphraseEquivalenceDecision:
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if not match:
        raise ValueError(f"Equivalence response did not contain a JSON object: {text[:500]}")
    try:
        return ParaphraseEquivalenceDecision.model_validate_json(match.group(0))
    except ValueError as exc:
        raise ValueError(f"Equivalence response did not match schema: {text[:500]}") from exc


def _clean_plain_text(text: str) -> str:
    text = text.strip()
    fence_match = re.fullmatch(r"```(?:text|markdown)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text[:1000]
        raise httpx.HTTPStatusError(
            f"{exc}; provider response body: {body}",
            request=response.request,
            response=response,
        ) from exc
