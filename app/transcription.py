from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, runtime_checkable

import httpx

_config = import_module("app.config")

Settings = _config.Settings


class TranscriptionError(RuntimeError):
    pass


class TranscriptionConfigError(TranscriptionError):
    pass


@runtime_checkable
class TranscriptionClient(Protocol):
    async def transcribe(self, audio_bytes: bytes, *, filename: str | None = None) -> str: ...


@dataclass(frozen=True)
class _GradiumRequestConfig:
    base_url: str
    api_key: str
    path: str
    model: str | None
    timeout_seconds: float


class GradiumTranscriber:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.gradium_transcription_transport.lower() != "rest":
            raise TranscriptionConfigError("Gradium transcription transport must be rest")

        if not settings.gradium_api_base_url:
            raise TranscriptionConfigError("GRADIUM_API_BASE_URL is required")

        if not settings.gradium_api_key:
            raise TranscriptionConfigError("GRADIUM_API_KEY is required")

        self._config = _GradiumRequestConfig(
            base_url=settings.gradium_api_base_url,
            api_key=settings.gradium_api_key,
            path=settings.gradium_transcription_path,
            model=settings.gradium_transcription_model,
            timeout_seconds=float(settings.gradium_timeout_seconds),
        )
        self._client = client or httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.timeout_seconds),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def transcribe(self, audio_bytes: bytes, *, filename: str | None = None) -> str:
        file_name = filename or "audio"
        files: dict[str, tuple[str, bytes, str]] = {
            "file": (file_name, audio_bytes, "application/octet-stream"),
        }
        data: dict[str, str] = {}
        if self._config.model:
            data["model"] = self._config.model

        try:
            response = await self._client.post(
                self._config.path,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                data=data,
                files=files,
            )
        except httpx.TimeoutException as exc:
            raise TranscriptionError("Gradium transcription request timed out") from exc
        except httpx.HTTPError as exc:
            raise TranscriptionError("Gradium transcription request failed") from exc

        if response.is_error:
            raise TranscriptionError(
                f"Gradium transcription failed with HTTP {response.status_code}: "
                f"{_response_excerpt(response)}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise TranscriptionError(
                f"Gradium transcription returned invalid JSON: {_response_excerpt(response)}"
            ) from exc

        if not isinstance(payload, dict):
            raise TranscriptionError(
                f"Gradium transcription returned unexpected payload: {_response_excerpt(response)}"
            )

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError(
                f"Gradium transcription response missing text: {_response_excerpt(response)}"
            )

        return text.strip()


def _response_excerpt(response: httpx.Response, limit: int = 240) -> str:
    try:
        content = response.text.strip()
    except Exception:
        content = ""

    if not content:
        return "<empty response>"

    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."
