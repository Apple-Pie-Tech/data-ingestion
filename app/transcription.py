from __future__ import annotations

from io import BytesIO
import json
import mimetypes
import wave
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

_config = import_module("app.config")

Settings = _config.Settings


class TranscriptionError(RuntimeError):
    pass


class TranscriptionConfigError(TranscriptionError):
    pass


@runtime_checkable
class TranscriptionClient(Protocol):
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class _GradiumRequestConfig:
    base_url: str
    api_key: str
    path: str
    model: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class _PreparedAudioRequest:
    body: bytes
    content_type: str
    input_format: str | None


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
            path=_normalize_transcription_path(settings.gradium_transcription_path),
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

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        prepared_request = _prepare_audio_request(
            audio_bytes,
            filename=filename,
            content_type=content_type,
        )
        params: dict[str, str] = {}
        if self._config.model:
            params["model_name"] = self._config.model
        if prepared_request.input_format:
            params["input_format"] = prepared_request.input_format

        try:
            response = await self._client.post(
                self._config.path,
                params=params,
                headers={
                    "x-api-key": self._config.api_key,
                    "Content-Type": prepared_request.content_type,
                },
                content=prepared_request.body,
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

        transcript_parts: list[str] = []
        for raw_line in response.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TranscriptionError(
                    f"Gradium transcription returned invalid NDJSON: {_response_excerpt(response)}"
                ) from exc

            if not isinstance(payload, dict):
                raise TranscriptionError(
                    f"Gradium transcription returned unexpected payload: {_response_excerpt(response)}"
                )

            message_type = payload.get("type")
            if message_type == "text":
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise TranscriptionError(
                        "Gradium transcription text event missing text: "
                        f"{_response_excerpt(response)}"
                    )
                transcript_parts.append(text.strip())
                continue

            if message_type == "error":
                message = payload.get("message")
                if isinstance(message, str) and message.strip():
                    raise TranscriptionError(
                        f"Gradium transcription stream failed: {message.strip()}"
                    )
                raise TranscriptionError(
                    "Gradium transcription stream failed without an error message"
                )

        transcript = " ".join(part for part in transcript_parts if part)
        if not transcript.strip():
            raise TranscriptionError(
                f"Gradium transcription response missing text: {_response_excerpt(response)}"
            )

        return transcript.strip()


def _resolve_audio_content_type(*, filename: str | None, content_type: str | None) -> str:
    if content_type and content_type.strip():
        return content_type.strip()

    if filename:
        guessed_content_type, _ = mimetypes.guess_type(filename)
        if guessed_content_type:
            return guessed_content_type

    raise TranscriptionError(
        "Gradium transcription requires an audio content type or a recognizable filename"
    )


def _normalize_transcription_path(path: str) -> str:
    normalized = path.strip() or "/post/speech/asr"
    if normalized == "/v1/audio/transcriptions":
        return "/post/speech/asr"
    return normalized


def _prepare_audio_request(
    audio_bytes: bytes,
    *,
    filename: str | None,
    content_type: str | None,
) -> _PreparedAudioRequest:
    resolved_content_type = _resolve_audio_content_type(
        filename=filename,
        content_type=content_type,
    )
    if not _is_wav_content_type(resolved_content_type):
        return _PreparedAudioRequest(
            body=audio_bytes,
            content_type=resolved_content_type,
            input_format=None,
        )

    converted_bytes = _convert_wav_to_pcm_16000(audio_bytes)
    if converted_bytes is None:
        return _PreparedAudioRequest(
            body=audio_bytes,
            content_type=resolved_content_type,
            input_format=None,
        )

    return _PreparedAudioRequest(
        body=converted_bytes,
        content_type="audio/pcm",
        input_format="pcm_16000",
    )


def _is_wav_content_type(content_type: str) -> bool:
    normalized = content_type.strip().lower()
    return normalized in {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
    }


def _convert_wav_to_pcm_16000(audio_bytes: bytes) -> bytes | None:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except wave.Error:
        return None

    samples = _decode_pcm_frames(
        raw_frames,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    mono_samples = _mix_to_mono(samples)
    resampled_samples = _resample_audio(mono_samples, source_rate=frame_rate, target_rate=16_000)
    clipped_samples = np.clip(np.rint(resampled_samples), -32768, 32767).astype("<i2")
    return clipped_samples.tobytes()


def _decode_pcm_frames(raw_frames: bytes, *, sample_width: int, channel_count: int) -> np.ndarray:
    if sample_width == 2:
        decoded = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
    elif sample_width == 3:
        frame_bytes = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
        decoded = (
            frame_bytes[:, 0].astype(np.int32)
            | (frame_bytes[:, 1].astype(np.int32) << 8)
            | (frame_bytes[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        decoded = ((decoded ^ sign_mask) - sign_mask).astype(np.float32)
    elif sample_width == 4:
        decoded = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32)
    else:
        raise TranscriptionError(f"unsupported WAV sample width: {sample_width}")

    if decoded.size % channel_count != 0:
        raise TranscriptionError("invalid WAV frame data")
    return decoded.reshape(-1, channel_count)


def _mix_to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] == 1:
        return samples[:, 0]
    return samples.mean(axis=1)


def _resample_audio(samples: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise TranscriptionError("WAV sample rate must be positive")
    if source_rate == target_rate or samples.size == 0:
        return samples

    target_length = max(1, round(samples.size * target_rate / source_rate))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.linspace(0, samples.size - 1, num=target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples)


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
