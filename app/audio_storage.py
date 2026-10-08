from __future__ import annotations

import asyncio
import mimetypes
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

_config = import_module("app.config")

Settings = _config.Settings


class AudioStorageError(RuntimeError):
    pass


class AudioStorageConfigurationError(AudioStorageError):
    pass


@dataclass(frozen=True)
class StoredAudio:
    object_key: str
    url: str


@runtime_checkable
class AudioStorageClient(Protocol):
    async def store_audio(
        self,
        audio_bytes: bytes,
        *,
        input_id: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> StoredAudio: ...

    async def aclose(self) -> None: ...

    async def delete_audio(self, stored_audio: StoredAudio) -> None: ...


class S3AudioStorage:
    """Stores ingested audio in S3 under ``audio/{input_id}/source{suffix}``.

    boto3 is synchronous, so every call hops to a worker thread. The service's
    other AWS client (``app/embeddings.py``) does the same: one idiom for both is
    worth more than adding aioboto3 for the two calls this makes per request.

    The bucket is created by Terraform, so there is no create-if-missing step
    here. That is deliberate -- an application that can create its own bucket
    needs ``s3:CreateBucket``, and the whole point of the per-function IAM roles
    is that this one can only write to the one bucket it owns.
    """

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        if not settings.audio_storage_enabled:
            raise AudioStorageConfigurationError("audio storage is disabled")

        if not settings.s3_audio_bucket:
            raise AudioStorageConfigurationError("S3_AUDIO_BUCKET is required")

        self._bucket = settings.s3_audio_bucket
        self._region = settings.aws_region

        if client is not None:
            self._client = client
            return

        try:
            boto3 = import_module("boto3")
        except ModuleNotFoundError as exc:
            raise AudioStorageConfigurationError(
                "boto3 dependency is required for audio storage"
            ) from exc

        self._client = boto3.client("s3", region_name=settings.aws_region)

    async def store_audio(
        self,
        audio_bytes: bytes,
        *,
        input_id: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> StoredAudio:
        object_key = _build_object_key(
            input_id=input_id,
            filename=filename,
            content_type=content_type,
        )
        resolved_content_type = _resolve_content_type(content_type, object_key)

        try:
            # put_object overwrites by default, which is the behaviour the Azure
            # path asked for explicitly with overwrite=True. No ACL is passed:
            # object keys derive from a client-supplied input_id, so a public
            # object would expose guessable URLs to private voice recordings.
            # Readers get a presigned URL instead.
            await asyncio.to_thread(
                self._client.put_object,
                Bucket=self._bucket,
                Key=object_key,
                Body=audio_bytes,
                ContentType=resolved_content_type,
            )
        except Exception as exc:
            raise AudioStorageError(
                f"failed to store {object_key} in s3://{self._bucket}: {exc}"
            ) from exc

        return StoredAudio(object_key=object_key, url=self._object_url(object_key))

    async def aclose(self) -> None:
        """Release the botocore client's connection pool.

        ``IngestionService.aclose`` discovers this by name. The async Azure
        client leaked an aiohttp session on every request without it
        (ISSUES.md E58); botocore's pool is smaller but the seam still has to
        exist, or ``_close_resource`` silently skips this dependency again.
        """
        client = getattr(self, "_client", None)
        close = getattr(client, "close", None)
        if close is None:
            return
        await asyncio.to_thread(close)

    async def delete_audio(self, stored_audio: StoredAudio) -> None:
        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=self._bucket,
            Key=stored_audio.object_key,
        )

    def _object_url(self, object_key: str) -> str:
        """The canonical **unsigned** URL for an object.

        The bucket is private, so this is not fetchable on its own --
        data-provision-api mints a presigned token at read time. Persisting the
        unsigned form is what stops stored records rotting into expired links
        (ISSUES.md E52), so this must stay unsigned.
        """
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{quote(object_key)}"


# `mimetypes` answers from the host MIME database (/etc/mime.types), so the same
# upload yields `audio/wav` on one machine and `audio/x-wav` on another -- proven
# live against S3. The value is persisted on the object and served to browsers,
# and the suffix becomes part of the object key, so neither may depend on which
# base image the service happens to be running. Pin the formats this service
# accepts and fall back to `mimetypes` only for anything outside the list.
_AUDIO_CONTENT_TYPE_BY_SUFFIX: dict[str, str] = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
_SUFFIX_BY_AUDIO_CONTENT_TYPE: dict[str, str] = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
}


def _build_object_key(*, input_id: str, filename: str | None, content_type: str | None) -> str:
    return f"audio/{_sanitize_path_segment(input_id)}/source{_resolve_suffix(filename, content_type)}"


def _resolve_suffix(filename: str | None, content_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix.strip()
        if suffix:
            return suffix.lower()

    if content_type:
        normalized = content_type.strip().split(";")[0].strip().lower()
        pinned = _SUFFIX_BY_AUDIO_CONTENT_TYPE.get(normalized)
        if pinned:
            return pinned

        guessed_extension = mimetypes.guess_extension(normalized, strict=False)
        if guessed_extension:
            return guessed_extension.lower()

    return ".bin"


def _resolve_content_type(content_type: str | None, object_key: str) -> str:
    if content_type and content_type.strip():
        return content_type.strip()

    suffix = Path(object_key).suffix.lower()
    pinned = _AUDIO_CONTENT_TYPE_BY_SUFFIX.get(suffix)
    if pinned:
        return pinned

    guessed_content_type, _ = mimetypes.guess_type(object_key)
    if guessed_content_type:
        return guessed_content_type

    return "application/octet-stream"


def _sanitize_path_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized or "audio"
