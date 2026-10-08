from __future__ import annotations

from typing import Any

import pytest

from app.audio_storage import (
    AudioStorageConfigurationError,
    AudioStorageError,
    S3AudioStorage,
    StoredAudio,
)
from app.config import Settings


class FakeS3:
    """Stands in for boto3's S3 client, recording the exact kwargs it is sent."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.closed = 0
        self._fail_with = fail_with

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if self._fail_with is not None:
            raise self._fail_with
        self.put_calls.append(kwargs)
        return {}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_calls.append(kwargs)
        return {}

    def close(self) -> None:
        self.closed += 1


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "audio_storage_enabled": True,
        "s3_audio_bucket": "applepie-audio",
        "aws_region": "us-east-1",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _storage(client: FakeS3 | None = None) -> tuple[S3AudioStorage, FakeS3]:
    fake = client or FakeS3()
    return S3AudioStorage(_settings(), client=fake), fake


def test_disabled_audio_storage_is_a_configuration_error() -> None:
    with pytest.raises(AudioStorageConfigurationError, match="disabled"):
        S3AudioStorage(_settings(audio_storage_enabled=False), client=FakeS3())


def test_missing_bucket_is_a_configuration_error() -> None:
    """Failing here beats a NoSuchBucket on the first upload of a deployment."""
    with pytest.raises(AudioStorageConfigurationError, match="S3_AUDIO_BUCKET"):
        S3AudioStorage(_settings(s3_audio_bucket=None), client=FakeS3())


@pytest.mark.asyncio
async def test_store_audio_keeps_the_existing_key_layout() -> None:
    """The key layout is a compatibility surface, not an implementation detail.

    ``audio/{input_id}/source{suffix}`` is what the Azure path wrote, what the
    ingest rollback deletes by, and what the E2E matrix asserts on.
    """
    storage, fake = _storage()

    stored = await storage.store_audio(
        b"riff-bytes",
        input_id="sample-input-001",
        filename="recording.WAV",
        content_type="audio/wav",
    )

    assert stored.object_key == "audio/sample-input-001/source.wav"
    assert fake.put_calls == [
        {
            "Bucket": "applepie-audio",
            "Key": "audio/sample-input-001/source.wav",
            "Body": b"riff-bytes",
            "ContentType": "audio/wav",
        }
    ]


@pytest.mark.asyncio
async def test_store_audio_sets_a_real_content_type() -> None:
    """boto3's kwarg is ``ContentType``, and getting it wrong is silent.

    The sync Azure SDK ignored a bare ``content_type=`` kwarg, so every blob was
    stored as application/octet-stream and nothing failed (ISSUES.md E47-adjacent).
    Assert the value, not merely that a kwarg was passed.
    """
    storage, fake = _storage()

    await storage.store_audio(b"bytes", input_id="x", filename="clip.mp3")

    assert fake.put_calls[0]["ContentType"] == "audio/mpeg"


@pytest.mark.asyncio
async def test_store_audio_falls_back_to_octet_stream_for_unknown_types() -> None:
    storage, fake = _storage()

    stored = await storage.store_audio(b"bytes", input_id="x", filename="clip.unknownext")

    assert stored.object_key == "audio/x/source.unknownext"
    assert fake.put_calls[0]["ContentType"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_store_audio_sanitises_a_client_supplied_input_id() -> None:
    """input_id comes from the request body, so it must not shape the key freely."""
    storage, fake = _storage()

    stored = await storage.store_audio(
        b"bytes",
        input_id="../../etc/passwd",
        filename="recording.wav",
    )

    assert stored.object_key == "audio/etc-passwd/source.wav"
    assert ".." not in fake.put_calls[0]["Key"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "expected_key"),
    [
        ("audio/wav", "audio/x/source.wav"),
        ("audio/x-wav", "audio/x/source.wav"),
        ("audio/mpeg", "audio/x/source.mp3"),
        ("audio/webm;codecs=opus", "audio/x/source.webm"),
        ("audio/vnd.nonexistent", "audio/x/source.bin"),
    ],
)
async def test_a_content_type_only_upload_resolves_a_stable_suffix(
    content_type: str, expected_key: str
) -> None:
    """A browser recording arrives with a content type and no filename.

    ``mimetypes.guess_extension("audio/wav")`` returns None on this platform and
    ``.wav`` on others, so relying on it made the object key depend on the host
    MIME database. These suffixes are pinned instead; the parameter list is the
    contract.
    """
    storage, _ = _storage()

    stored = await storage.store_audio(b"bytes", input_id="x", content_type=content_type)

    assert stored.object_key == expected_key


@pytest.mark.asyncio
async def test_content_type_does_not_depend_on_the_host_mime_database() -> None:
    """Verified live: a .wav filename was stored as `audio/x-wav`, not `audio/wav`.

    ``mimetypes.guess_type`` answers from /etc/mime.types, so the same upload
    produced different stored metadata on the dev machine and would again in a
    different base image. The value is persisted on the object and served to
    browsers, so it has to be deterministic.
    """
    storage, fake = _storage()

    await storage.store_audio(b"bytes", input_id="x", filename="source.wav")

    assert fake.put_calls[0]["ContentType"] == "audio/wav"


@pytest.mark.asyncio
async def test_store_audio_never_sets_an_acl() -> None:
    """Object keys are guessable, so a public object leaks private recordings.

    The Azure path created its container with ``PublicAccess.Blob`` until that
    was fixed; the S3 equivalent would be passing ``ACL="public-read"`` here.
    """
    storage, fake = _storage()

    await storage.store_audio(b"bytes", input_id="x", filename="clip.wav")

    assert "ACL" not in fake.put_calls[0]


@pytest.mark.asyncio
async def test_stored_url_is_unsigned_and_addresses_the_object() -> None:
    """Storage holds the canonical unsigned URL; readers presign it (E52).

    A signed URL here would be persisted into the vector payload and rot into a
    dead link as soon as the token expired.
    """
    storage, _ = _storage()

    stored = await storage.store_audio(b"bytes", input_id="x", filename="clip.wav")

    assert stored.url == (
        "https://applepie-audio.s3.us-east-1.amazonaws.com/audio/x/source.wav"
    )
    assert "X-Amz-Signature" not in stored.url


@pytest.mark.asyncio
async def test_store_audio_wraps_transport_failures() -> None:
    """IngestionService maps AudioStorageError to a 503; a raw ClientError is a 500."""
    storage, _ = _storage(FakeS3(fail_with=RuntimeError("AccessDenied")))

    with pytest.raises(AudioStorageError, match="s3://applepie-audio"):
        await storage.store_audio(b"bytes", input_id="x", filename="clip.wav")


@pytest.mark.asyncio
async def test_delete_audio_removes_the_stored_key() -> None:
    """The ingest rollback path depends on this (app/ingestion.py)."""
    storage, fake = _storage()

    await storage.delete_audio(
        StoredAudio(object_key="audio/x/source.wav", url="https://example.invalid/x")
    )

    assert fake.delete_calls == [
        {"Bucket": "applepie-audio", "Key": "audio/x/source.wav"}
    ]


@pytest.mark.asyncio
async def test_audio_storage_closes_its_client() -> None:
    """``IngestionService.aclose`` closes dependencies by looking for ``aclose``.

    The Azure client had neither ``aclose`` nor ``close``, so ``_close_resource``
    silently skipped it and every request leaked an aiohttp session
    (ISSUES.md E58). The seam has to survive the port.
    """
    storage, fake = _storage()

    assert hasattr(storage, "aclose"), "S3AudioStorage must expose aclose()"
    await storage.aclose()

    assert fake.closed == 1
