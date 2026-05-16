from __future__ import annotations

from io import BytesIO
from importlib import import_module
import wave

import httpx
import numpy as np
import pytest

_config = import_module("app.config")
_transcription = import_module("app.transcription")

Settings = _config.Settings
GradiumTranscriber = _transcription.GradiumTranscriber
TranscriptionError = _transcription.TranscriptionError


def make_settings(*, gradium_transcription_path: str = "/post/speech/asr") -> Settings:
    return Settings(
        gradium_api_base_url="https://gradium.example",
        gradium_api_key="super-secret-key",
        gradium_transcription_model="default",
        gradium_transcription_path=gradium_transcription_path,
        gradium_transcription_transport="rest",
        gradium_timeout_seconds=3,
    )


@pytest.mark.asyncio
async def test_gradium_transcriber_returns_transcript_and_sends_expected_request() -> None:
    seen_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request["method"] = request.method
        seen_request["url"] = str(request.url)
        seen_request["api_key"] = request.headers.get("x-api-key", "")
        seen_request["content_type"] = request.headers.get("content-type", "")
        seen_request["body"] = request.content
        return httpx.Response(
            200,
            text='{"type":"text","text":"hello"}\n\n'
            '{"type":"text","text":"apple pie"}\n'
            '{"type":"end_text","stop_s":1.2,"stream_id":0}\n',
            headers={"Content-Type": "application/x-ndjson"},
        )

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        transcript = await transcriber.transcribe(
            b"fake-audio-bytes",
            filename="sample.pcm",
            content_type="audio/pcm",
        )
    finally:
        await transcriber.aclose()

    assert transcript == "hello apple pie"
    body = seen_request["body"]
    assert seen_request["method"] == "POST"
    assert seen_request["url"] == "https://gradium.example/post/speech/asr?model_name=default"
    assert seen_request["api_key"] == "super-secret-key"
    assert seen_request["content_type"] == "audio/pcm"
    assert isinstance(body, bytes)
    assert body == b"fake-audio-bytes"


@pytest.mark.asyncio
async def test_gradium_transcriber_normalizes_wav_and_remaps_legacy_rest_path() -> None:
    seen_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_request["url"] = str(request.url)
        seen_request["path"] = request.url.path
        seen_request["params"] = dict(request.url.params)
        seen_request["content_type"] = request.headers.get("content-type", "")
        seen_request["body"] = request.content
        return httpx.Response(
            200,
            text='{"type":"text","text":"hello"}\n',
            headers={"Content-Type": "application/x-ndjson"},
        )

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(
        make_settings(gradium_transcription_path="/v1/audio/transcriptions"),
        client=client,
    )
    wav_bytes = build_wav_bytes(sample_rate=48_000, sample_count=4_800)

    try:
        transcript = await transcriber.transcribe(
            wav_bytes,
            filename="sample.wav",
            content_type="audio/wav",
        )
    finally:
        await transcriber.aclose()

    body = seen_request["body"]
    assert transcript == "hello"
    assert seen_request["path"] == "/post/speech/asr"
    assert seen_request["params"] == {
        "model_name": "default",
        "input_format": "pcm_16000",
    }
    assert seen_request["content_type"] == "audio/pcm"
    assert isinstance(body, bytes)
    assert len(body) < len(wav_bytes)


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
            await transcriber.transcribe(
                b"fake-audio-bytes",
                filename="sample.wav",
                content_type="audio/wav",
            )
    finally:
        await transcriber.aclose()

    message = str(exc_info.value)
    assert "503" in message
    assert "temporary outage" in message
    assert "super-secret-key" not in message


@pytest.mark.asyncio
async def test_gradium_transcriber_raises_for_missing_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='{"type":"end_text","stop_s":1.2,"stream_id":0}\n',
            headers={"Content-Type": "application/x-ndjson"},
        )

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        with pytest.raises(TranscriptionError, match="missing text"):
            await transcriber.transcribe(
                b"fake-audio-bytes",
                filename="sample.wav",
                content_type="audio/wav",
            )
    finally:
        await transcriber.aclose()


@pytest.mark.asyncio
async def test_gradium_transcriber_raises_for_stream_error_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='{"type":"error","message":"bad audio"}\n',
            headers={"Content-Type": "application/x-ndjson"},
        )

    client = httpx.AsyncClient(
        base_url="https://gradium.example",
        transport=httpx.MockTransport(handler),
    )
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        with pytest.raises(TranscriptionError, match="bad audio"):
            await transcriber.transcribe(
                b"fake-audio-bytes",
                filename="sample.wav",
                content_type="audio/wav",
            )
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
            await transcriber.transcribe(
                b"fake-audio-bytes",
                filename="sample.wav",
                content_type="audio/wav",
            )
    finally:
        await transcriber.aclose()


@pytest.mark.asyncio
async def test_gradium_transcriber_requires_audio_content_type_or_filename() -> None:
    client = httpx.AsyncClient(base_url="https://gradium.example")
    transcriber = GradiumTranscriber(make_settings(), client=client)

    try:
        with pytest.raises(TranscriptionError, match="content type"):
            await transcriber.transcribe(b"fake-audio-bytes")
    finally:
        await transcriber.aclose()


def build_wav_bytes(*, sample_rate: int, sample_count: int) -> bytes:
    time = np.linspace(0, sample_count / sample_rate, num=sample_count, endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * time) * 32767).astype("<i2")

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()
