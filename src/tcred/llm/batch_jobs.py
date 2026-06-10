from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import httpx
import orjson
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from tcred.llm.batch import BatchProvider

BatchEndpoint = Literal[
    "/v1/responses",
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/completions",
    "/v1/moderations",
]


class BatchJobSubmission(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    provider: BatchProvider
    batch_id: str
    uploaded_file_id: str | None = None
    status: str | None = None
    raw_response: dict[str, object]


class BatchJobStatus(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    provider: BatchProvider
    batch_id: str
    status: str | None
    output_file_id: str | None = None
    error_file_id: str | None = None
    results_url: str | None = None
    raw_response: dict[str, object]


class BatchJobClient:
    def __init__(
        self,
        provider: BatchProvider,
        *,
        api_key: str | None = None,
        timeout_seconds: int = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        load_dotenv()
        self.provider = provider
        self.api_key = api_key or _api_key_for_provider(provider)
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def submit(
        self,
        *,
        request_file: Path,
        model: str | None = None,
        endpoint: BatchEndpoint | str | None = None,
        completion_window: str = "24h",
        timeout_hours: int = 24,
        metadata: dict[str, str] | None = None,
    ) -> BatchJobSubmission:
        if self.provider == "anthropic":
            return await self._submit_anthropic(request_file=request_file)

        endpoint = endpoint or _default_endpoint(self.provider)
        if self.provider == "mistral" and not model:
            raise ValueError("Mistral batch submission requires --model")
        uploaded = await self._upload_file(request_file)
        response = await self._create_file_batch(
            file_id=str(uploaded["id"]),
            endpoint=str(endpoint),
            model=model,
            completion_window=completion_window,
            timeout_hours=timeout_hours,
            metadata=metadata,
        )
        return BatchJobSubmission(
            provider=self.provider,
            batch_id=str(response["id"]),
            uploaded_file_id=str(uploaded["id"]),
            status=_status_from_response(response),
            raw_response=response,
        )

    async def retrieve(self, batch_id: str) -> BatchJobStatus:
        async with self._client() as client:
            if self.provider == "anthropic":
                response = await client.get(f"/v1/messages/batches/{batch_id}")
            elif self.provider == "mistral":
                response = await client.get(f"/v1/batch/jobs/{batch_id}")
            else:
                response = await client.get(f"/v1/batches/{batch_id}")
            _raise_for_status(response)
            payload = response.json()
        return _status_from_payload(provider=self.provider, payload=payload)

    async def download_results(
        self,
        *,
        output_path: Path,
        file_id: str | None = None,
        results_url: str | None = None,
    ) -> Path:
        if self.provider == "anthropic":
            if not results_url:
                raise ValueError("Anthropic result download requires --results-url")
            return await self._download_absolute_url(
                results_url=results_url,
                output_path=output_path,
            )
        if not file_id:
            raise ValueError(f"{self.provider} result download requires --file-id")
        async with self._client() as client:
            response = await client.get(f"/v1/files/{file_id}/content")
            _raise_for_status(response)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
        return output_path

    async def _submit_anthropic(self, *, request_file: Path) -> BatchJobSubmission:
        body = orjson.loads(request_file.read_bytes())
        async with self._client() as client:
            response = await client.post("/v1/messages/batches", json=body)
            _raise_for_status(response)
            payload = response.json()
        return BatchJobSubmission(
            provider=self.provider,
            batch_id=str(payload["id"]),
            uploaded_file_id=None,
            status=str(payload.get("processing_status", "")),
            raw_response=payload,
        )

    async def _upload_file(self, request_file: Path) -> dict[str, object]:
        async with self._client() as client:
            response = await client.post(
                "/v1/files",
                data={"purpose": "batch"},
                files={
                    "file": (
                        request_file.name,
                        request_file.read_bytes(),
                        "application/jsonl",
                    )
                },
            )
            _raise_for_status(response)
            return response.json()

    async def _create_file_batch(
        self,
        *,
        file_id: str,
        endpoint: str,
        model: str | None,
        completion_window: str,
        timeout_hours: int,
        metadata: dict[str, str] | None,
    ) -> dict[str, object]:
        async with self._client() as client:
            if self.provider == "mistral":
                body: dict[str, object] = {
                    "input_files": [file_id],
                    "endpoint": endpoint,
                    "model": model,
                    "timeout_hours": timeout_hours,
                }
            else:
                body = {
                    "input_file_id": file_id,
                    "endpoint": endpoint,
                    "completion_window": completion_window,
                }
            if metadata:
                body["metadata"] = metadata
            path = "/v1/batch/jobs" if self.provider == "mistral" else "/v1/batches"
            response = await client.post(path, json=body)
            _raise_for_status(response)
            return response.json()

    async def _download_absolute_url(self, *, results_url: str, output_path: Path) -> Path:
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.get(results_url, headers=self._headers())
            _raise_for_status(response)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
        return output_path

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=_base_url(self.provider),
            timeout=self.timeout_seconds,
            headers=self._headers(),
            transport=self.transport,
        )

    def _headers(self) -> dict[str, str]:
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
        return {"Authorization": f"Bearer {self.api_key}"}


def _api_key_for_provider(provider: BatchProvider) -> str:
    key = {
        "openai": os.getenv("LLM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("LLM_ANTHROPIC_API_KEY"),
        "mistral": os.getenv("LLM_MISTRAL_API_KEY"),
        "groq": os.getenv("LLM_GROQ_API_KEY"),
    }[provider]
    if not key:
        raise ValueError(f"API key is not configured for provider {provider}")
    return key


def _base_url(provider: BatchProvider) -> str:
    return {
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com",
        "mistral": "https://api.mistral.ai",
        "groq": "https://api.groq.com/openai",
    }[provider]


def _default_endpoint(provider: BatchProvider) -> str:
    return "/v1/responses" if provider == "openai" else "/v1/chat/completions"


def _status_from_payload(*, provider: BatchProvider, payload: dict[str, object]) -> BatchJobStatus:
    return BatchJobStatus(
        provider=provider,
        batch_id=str(payload["id"]),
        status=_status_from_response(payload),
        output_file_id=_string_or_none(payload.get("output_file_id") or payload.get("output_file")),
        error_file_id=_string_or_none(payload.get("error_file_id") or payload.get("error_file")),
        results_url=_string_or_none(payload.get("results_url")),
        raw_response=payload,
    )


def _status_from_response(payload: dict[str, object]) -> str | None:
    value = payload.get("status") or payload.get("processing_status")
    return str(value) if value is not None else None


def _string_or_none(value: object) -> str | None:
    return str(value) if value else None


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
