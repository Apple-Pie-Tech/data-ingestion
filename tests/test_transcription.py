from __future__ import annotations

from importlib import import_module

import httpx
import pytest

_config = import_module("app.config")
_transcription = import_module("app.transcription")

Settings = _config.Settings
GradiumTranscriber = _transcription.GradiumTranscriber
TranscriptionError = _transcription.TranscriptionError


def make_settings() -> Settings:
    return Settings(
        gradium_api_base_url="https://gradium.example",
        gradium_api_key="super-secret-key",
        gradium_transcription_model="whisper-1",
        gradium_transcription_path="/v1/audio/transcriptions",
        gradium_transcription_transport="rest",
        gradium_timeout_seconds=3,
    )


@pytest.mark.asyncio
async def test_gradium_transcriber_returns_transcript_and_sends_expected_request() -> None:
    seen_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request["method"] = request.method
        seen_request["url"] = str(request.url)
        seen_request["auth"] = request.headers.get("authorization", "")
        seen_request["content_type"] = request.headers.get("content-type", "")
        seen_request["body"] = request.content
        return httpx.Response(200, json={"text": "hello apple pie"})

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        transcript = await transcriber.transcribe(b"fake-audio-bytes", filename="sample.wav")
    finally:
        await transcriber.aclose()

    assert transcript == "hello apple pie"
    body = seen_request["body"]
    assert seen_request["method"] == "POST"
    assert seen_request["url"] == "https://gradium.example/v1/audio/transcriptions"
    assert seen_request["auth"] == "Bearer super-secret-key"
    assert isinstance(body, bytes)
    assert b'name="file"; filename="sample.wav"' in body
    assert b'name="model"' in body
    assert b"whisper-1" in body


@pytest.mark.asyncio
async def test_gradium_transcriber_raises_sanitized_error_for_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "temporary outage"})

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        with pytest.raises(TranscriptionError) as exc_info:
            await transcriber.transcribe(b"fake-audio-bytes")
    finally:
        await transcriber.aclose()

    message = str(exc_info.value)
    assert "503" in message
    assert "temporary outage" in message
    assert "super-secret-key" not in message


@pytest.mark.asyncio
async def test_gradium_transcriber_raises_for_missing_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"duration": 1.2})

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        with pytest.raises(TranscriptionError, match="missing text"):
            await transcriber.transcribe(b"fake-audio-bytes")
    finally:
        await transcriber.aclose()


@pytest.mark.asyncio
async def test_gradium_transcriber_raises_for_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        with pytest.raises(TranscriptionError, match="timed out"):
            await transcriber.transcribe(b"fake-audio-bytes")
    finally:
        await transcriber.aclose()
