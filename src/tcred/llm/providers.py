from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelListResult:
    provider: str
    status: str
    models: list[str]
    message: str = ""


async def list_available_models() -> list[ModelListResult]:
    """List models for configured providers without exposing API keys."""
    load_dotenv()
    tasks = [
        _openai_models(),
        _mistral_models(),
        _groq_models(),
        _anthropic_probe(),
    ]
    return await asyncio.gather(*tasks)


async def _openai_models() -> ModelListResult:
    key = os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return ModelListResult("openai", "missing_key", [], "No OpenAI key configured")
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            models = sorted(item["id"] for item in response.json().get("data", []))
            return ModelListResult("openai", "ok", models)
        except Exception as exc:  # noqa: BLE001 - model probe should report provider failures
            return ModelListResult("openai", "error", [], str(exc))


async def _mistral_models() -> ModelListResult:
    key = os.getenv("LLM_MISTRAL_API_KEY")
    if not key:
        return ModelListResult("mistral", "missing_key", [], "No Mistral key configured")
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            models = sorted(item["id"] for item in response.json().get("data", []))
            return ModelListResult("mistral", "ok", models)
        except Exception as exc:  # noqa: BLE001
            return ModelListResult("mistral", "error", [], str(exc))


async def _groq_models() -> ModelListResult:
    key = os.getenv("LLM_GROQ_API_KEY")
    if not key:
        return ModelListResult("groq", "missing_key", [], "No Groq key configured")
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            models = sorted(item["id"] for item in response.json().get("data", []))
            return ModelListResult("groq", "ok", models)
        except Exception as exc:  # noqa: BLE001
            return ModelListResult("groq", "error", [], str(exc))


async def _anthropic_probe() -> ModelListResult:
    key = os.getenv("LLM_ANTHROPIC_API_KEY")
    if not key:
        return ModelListResult("anthropic", "missing_key", [], "No Anthropic key configured")
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
            )
            response.raise_for_status()
            payload = response.json()
            models = sorted(item["id"] for item in payload.get("data", []))
            return ModelListResult("anthropic", "ok", models)
        except Exception as exc:  # noqa: BLE001
            return ModelListResult("anthropic", "error", [], str(exc))
