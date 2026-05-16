from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

_config = import_module("app.config")

Settings = _config.Settings


class AudioStorageError(RuntimeError):
    pass


class AudioStorageConfigurationError(AudioStorageError):
    pass


@dataclass(frozen=True)
class StoredAudio:
    blob_name: str
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

    async def delete_audio(self, stored_audio: StoredAudio) -> None: ...


class AzureBlobAudioStorage:
    def __init__(self, settings: Settings) -> None:
        if not settings.audio_storage_enabled:
            raise AudioStorageConfigurationError("audio storage is disabled")

        if not settings.azure_blob_connection_string:
            raise AudioStorageConfigurationError("AZURE_BLOB_CONNECTION_STRING is required")

        if not settings.azure_blob_container:
            raise AudioStorageConfigurationError("AZURE_BLOB_CONTAINER is required")

        try:
            azure_blob = import_module("azure.storage.blob")
            azure_blob_async = import_module("azure.storage.blob.aio")
            azure_core_exceptions = import_module("azure.core.exceptions")
        except ModuleNotFoundError as exc:
            raise AudioStorageConfigurationError(
                "azure-storage-blob dependency is required for audio storage"
            ) from exc

        self._blob_service_client = azure_blob_async.BlobServiceClient.from_connection_string(
            settings.azure_blob_connection_string
        )
        self._content_settings_cls = azure_blob.ContentSettings
        self._public_access_blob = azure_blob.PublicAccess.Blob
        self._resource_exists_error = azure_core_exceptions.ResourceExistsError
        self._container_name = settings.azure_blob_container
        self._container_ready = False

    async def store_audio(
        self,
        audio_bytes: bytes,
        *,
        input_id: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> StoredAudio:
        await self._ensure_container()
        blob_name = _build_blob_name(
            input_id=input_id,
            filename=filename,
            content_type=content_type,
        )
        blob_client = self._blob_service_client.get_blob_client(
            container=self._container_name,
            blob=blob_name,
        )
        resolved_content_type = _resolve_content_type(content_type, blob_name)
        content_settings = self._content_settings_cls(content_type=resolved_content_type)
        await blob_client.upload_blob(audio_bytes, overwrite=True, content_settings=content_settings)
        return StoredAudio(blob_name=blob_name, url=blob_client.url)

    async def delete_audio(self, stored_audio: StoredAudio) -> None:
        blob_client = self._blob_service_client.get_blob_client(
            container=self._container_name,
            blob=stored_audio.blob_name,
        )
        await blob_client.delete_blob(delete_snapshots="include")

    async def _ensure_container(self) -> None:
        if self._container_ready:
            return

        container_client = self._blob_service_client.get_container_client(self._container_name)
        try:
            await container_client.create_container(public_access=self._public_access_blob)
        except self._resource_exists_error:
            pass

        await container_client.set_container_access_policy(public_access=self._public_access_blob)

        self._container_ready = True


def _build_blob_name(*, input_id: str, filename: str | None, content_type: str | None) -> str:
    return f"audio/{_sanitize_path_segment(input_id)}/source{_resolve_suffix(filename, content_type)}"


def _resolve_suffix(filename: str | None, content_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix.strip()
        if suffix:
            return suffix.lower()

    if content_type:
        guessed_extension = mimetypes.guess_extension(content_type.strip(), strict=False)
        if guessed_extension:
            return guessed_extension.lower()

    return ".bin"


def _resolve_content_type(content_type: str | None, blob_name: str) -> str:
    if content_type and content_type.strip():
        return content_type.strip()

    guessed_content_type, _ = mimetypes.guess_type(blob_name)
    if guessed_content_type:
        return guessed_content_type

    return "application/octet-stream"


def _sanitize_path_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized or "audio"
