import json
from importlib import import_module

from fastapi.testclient import TestClient

_ingestion = import_module("app.ingestion")
_main = import_module("app.main")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")
_story_labeling = import_module("app.story_labeling")
_vector_store = import_module("app.vector_store")
_audio_storage = import_module("app.audio_storage")

AudioFile = _ingestion.AudioFile
IngestionService = _ingestion.IngestionService
app = _main.app
create_app = _main.create_app
get_ingestion_service = _main.get_ingestion_service
get_settings = _main.get_settings
get_story_labeling_trigger = _main.get_story_labeling_trigger
Settings = _main.Settings
IngestMetadata = _schemas.IngestMetadata
IngestResult = _schemas.IngestResult
Chunk = _semantic_chunking.Chunk
StoryLabelingTriggerResult = _story_labeling.StoryLabelingTriggerResult
VectorStoreError = _vector_store.VectorStoreError
AudioStorageError = _audio_storage.AudioStorageError

client = TestClient(app)


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def ingest(
        self,
        *,
        metadata: IngestMetadata,
        text: str | None,
        audio_file: AudioFile | None,
    ) -> IngestResult:
        self.calls.append(
            {
                "metadata": metadata,
                "text": text,
                "audio_file": audio_file,
            }
        )
        return IngestResult(
            input_id=metadata.input_id,
            status="indexed",
            chunks=1,
            audio_url=None,
        )


class FakeStoryLabelingTrigger:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def trigger_cluster_labels(
        self,
        *,
        metadata: IngestMetadata,
        source: str,
    ) -> StoryLabelingTriggerResult:
        self.calls.append({"metadata": metadata, "source": source})
        return StoryLabelingTriggerResult(
            status="completed",
            points_read=5,
            points_clustered=4,
            clusters_found=2,
            noise_points=1,
            points_updated=6,
        )


class FakeChunker:
    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(
                text=text,
                chunk_index=0,
                metadata={
                    "similarity_threshold": 0.72,
                    "overlap_sentences": 1,
                },
            )
        ]


class FakeEmbeddingsClient:
    model_name = "text-embedding-3-large"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeTranscriber:
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        return "transcribed speech"


class FakeVectorStore:
    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: str,
        embedding_model: str,
        audio_url: str | None,
    ) -> int:
        return len(chunks)


class FailingVectorStore:
    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: str,
        embedding_model: str,
        audio_url: str | None,
    ) -> int:
        raise VectorStoreError("Qdrant upsert failed")


class FailingAudioStorage:
    async def store_audio(
        self,
        audio_bytes: bytes,
        *,
        input_id: str,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> object:
        raise AudioStorageError("blob upload failed")

    async def delete_audio(self, stored_audio: object) -> None:
        raise AssertionError("delete_audio should not be called when upload fails")


def _multipart_fields(
    metadata: dict[str, object],
    text: str | None = None,
    audio: bytes | None = None,
) -> dict[str, tuple[None | str, str | bytes, str | None]]:
    fields: dict[str, tuple[None | str, str | bytes, str | None]] = {
        "metadata": (None, json.dumps(metadata), None),
    }
    if text is not None:
        fields["text"] = (None, text, None)
    if audio is not None:
        fields["audio"] = ("sample.wav", audio, "audio/wav")
    return fields


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_preflight_returns_cors_headers_for_allowed_origin() -> None:
    cors_app = create_app(Settings(cors_allow_origins="http://127.0.0.1:8081"))
    cors_client = TestClient(cors_app)

    response = cors_client.options(
        "/ingest",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8081"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_text_only_ingest_uses_fake_service() -> None:
    fake_service = FakeIngestionService()
    fake_trigger = FakeStoryLabelingTrigger()
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service
    app.dependency_overrides[get_story_labeling_trigger] = lambda: fake_trigger

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                text="hello apple pie",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "input_id": "sample-input-001",
        "status": "indexed",
        "chunks": 1,
    }
    assert len(fake_service.calls) == 1
    assert fake_service.calls[0]["text"] == "hello apple pie"
    assert fake_service.calls[0]["audio_file"] is None
    assert fake_trigger.calls == [
        {
            "metadata": IngestMetadata.model_validate(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                }
            ),
            "source": "text",
        }
    ]


def test_text_wins_over_audio() -> None:
    fake_service = FakeIngestionService()
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                text="hello apple pie",
                audio=b"fake-audio-bytes",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_service.calls[0]["text"] == "hello apple pie"
    assert fake_service.calls[0]["audio_file"] is None


def test_text_wins_over_oversized_audio() -> None:
    fake_service = FakeIngestionService()
    app.dependency_overrides[get_settings] = lambda: Settings(max_audio_bytes=5)
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                text="hello apple pie",
                audio=b"x" * 10,
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_service.calls[0]["text"] == "hello apple pie"
    assert fake_service.calls[0]["audio_file"] is None


def test_audio_upload_preserves_filename_and_content_type() -> None:
    fake_service = FakeIngestionService()
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                audio=b"fake-audio-bytes",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_service.calls[0]["text"] is None
    assert fake_service.calls[0]["audio_file"] == AudioFile(
        content=b"fake-audio-bytes",
        filename="sample.wav",
        content_type="audio/wav",
    )


def test_successful_ingest_response_has_required_json_keys() -> None:
    service = IngestionService(
        Settings(),
        chunker=FakeChunker(),
        embeddings=FakeEmbeddingsClient(),
        vector_store=FakeVectorStore(),
    )
    app.dependency_overrides[get_ingestion_service] = lambda: service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                text="hello apple pie",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert set(response.json()) == {"input_id", "status", "chunks"}
    assert response.json() == {
        "input_id": "sample-input-001",
        "status": "indexed",
        "chunks": 1,
    }


def test_audio_ingest_returns_audio_url_when_storage_is_enabled() -> None:
    class FakeAudioStorage:
        async def store_audio(
            self,
            audio_bytes: bytes,
            *,
            input_id: str,
            filename: str | None = None,
                content_type: str | None = None,
        ) -> object:
            return _audio_storage.StoredAudio(
                blob_name="audio/sample-input-001/source.wav",
                url=(
                    "https://example.blob.core.windows.net/ingest-audio/"
                    "audio/sample-input-001/source.wav"
                )
            )

        async def delete_audio(self, stored_audio: object) -> None:
            raise AssertionError("delete_audio should not be called on successful ingest")

    service = IngestionService(
        Settings(audio_storage_enabled=True),
        transcriber=FakeTranscriber(),
        chunker=FakeChunker(),
        embeddings=FakeEmbeddingsClient(),
        vector_store=FakeVectorStore(),
        audio_storage=FakeAudioStorage(),
    )
    app.dependency_overrides[get_ingestion_service] = lambda: service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                audio=b"fake-audio-bytes",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "input_id": "sample-input-001",
        "status": "indexed",
        "chunks": 1,
        "audio_url": (
            "https://example.blob.core.windows.net/ingest-audio/"
            "audio/sample-input-001/source.wav"
        ),
    }


def test_audio_storage_failure_returns_503() -> None:
    service = IngestionService(
        Settings(audio_storage_enabled=True),
        chunker=FakeChunker(),
        embeddings=FakeEmbeddingsClient(),
        vector_store=FakeVectorStore(),
        audio_storage=FailingAudioStorage(),
    )
    app.dependency_overrides[get_ingestion_service] = lambda: service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                audio=b"fake-audio-bytes",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "audio_storage_unavailable"}


def test_vector_store_failure_returns_503() -> None:
    service = IngestionService(
        Settings(),
        chunker=FakeChunker(),
        embeddings=FakeEmbeddingsClient(),
        vector_store=FailingVectorStore(),
    )
    app.dependency_overrides[get_ingestion_service] = lambda: service

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                text="hello apple pie",
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "vector_store_unavailable" in response.json()["detail"]


def test_invalid_metadata_json_returns_400() -> None:
    response = client.post(
        "/ingest",
        files={
            "metadata": (None, "not-json", None),
            "text": (None, "hello", None),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "metadata must be valid JSON"


def test_missing_content_returns_400() -> None:
    response = client.post(
        "/ingest",
        files={
            "metadata": (
                None,
                json.dumps(
                    {
                        "input_id": "sample-input-001",
                        "user_id": "user-123",
                        "timestamp": "2026-05-16T12:00:00Z",
                    }
                ),
                None,
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "text or audio required"


def test_audio_size_is_checked() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_audio_bytes=5)
    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()

    try:
        response = client.post(
            "/ingest",
            files=_multipart_fields(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                },
                audio=b"x" * 10,
            ),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "audio exceeds maximum size"
