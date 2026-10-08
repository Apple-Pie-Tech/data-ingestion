from __future__ import annotations

import pytest

from app.audio_storage import AzureBlobAudioStorage


class _ContainerClient:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def create_container(self, **kwargs: object) -> None:
        self._calls.append("create_container")
        public_access = kwargs.get("public_access")
        if public_access is not None:
            raise AssertionError(
                f"container must not be created with public access, got {public_access!r}"
            )

    async def set_container_access_policy(self, *args: object, **kwargs: object) -> None:
        self._calls.append("set_container_access_policy")


class _ServiceClient:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def get_container_client(self, _name: str) -> _ContainerClient:
        return _ContainerClient(self._calls)


def _storage_with(calls: list[str]) -> AzureBlobAudioStorage:
    """Build the storage object without running __init__'s SDK imports."""
    storage = AzureBlobAudioStorage.__new__(AzureBlobAudioStorage)
    storage._blob_service_client = _ServiceClient(calls)  # type: ignore[attr-defined]
    storage._container_name = "ingest-audio"  # type: ignore[attr-defined]
    storage._container_ready = False  # type: ignore[attr-defined]
    storage._resource_exists_error = FileExistsError  # type: ignore[attr-defined]
    return storage


@pytest.mark.asyncio
async def test_ensure_container_does_not_make_the_container_public() -> None:
    """User voice recordings must not be world-readable.

    ``_ensure_container`` used to create the container with ``PublicAccess.Blob``
    and then call ``set_container_access_policy(public_access=...)`` on every
    startup. Blob names derive from a client-supplied ``input_id``, so public
    access would have produced guessable URLs to private recordings. Containers
    stay private; readers get a read-time SAS instead.
    """
    calls: list[str] = []
    await _storage_with(calls)._ensure_container()

    assert "set_container_access_policy" not in calls
    assert calls == ["create_container"]


@pytest.mark.asyncio
async def test_ensure_container_survives_an_existing_container() -> None:
    """Creating an existing container must be tolerated, not fatal."""
    calls: list[str] = []
    storage = _storage_with(calls)

    class _Existing(_ContainerClient):
        async def create_container(self, **kwargs: object) -> None:
            self._calls.append("create_container")
            raise FileExistsError

    storage._blob_service_client.get_container_client = (  # type: ignore[attr-defined]
        lambda _name: _Existing(calls)
    )

    await storage._ensure_container()
    assert storage._container_ready is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ensure_container_is_idempotent() -> None:
    """The container is prepared once, not on every upload."""
    calls: list[str] = []
    storage = _storage_with(calls)

    await storage._ensure_container()
    await storage._ensure_container()

    assert calls == ["create_container"]


@pytest.mark.asyncio
async def test_audio_storage_closes_its_blob_client() -> None:
    """The async BlobServiceClient owns an aiohttp session that must be released.

    ``IngestionService.aclose`` closes each dependency by looking for an
    ``aclose``/``close`` attribute. ``AzureBlobAudioStorage`` had neither, so
    ``_close_resource`` silently skipped it and every request leaked a session --
    visible as ``Unclosed client session`` / ``Unclosed connector`` in the logs
    (ISSUES.md E58).
    """
    closed: list[str] = []

    class _Client:
        async def close(self) -> None:
            closed.append("closed")

    storage = AzureBlobAudioStorage.__new__(AzureBlobAudioStorage)
    storage._blob_service_client = _Client()  # type: ignore[attr-defined]

    assert hasattr(storage, "aclose"), "AzureBlobAudioStorage must expose aclose()"
    await storage.aclose()

    assert closed == ["closed"]
