import json
from importlib import import_module

from fastapi.testclient import TestClient

_ingestion = import_module("app.ingestion")
_main = import_module("app.main")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")
_vector_store = import_module("app.vector_store")

AudioFile = _ingestion.AudioFile
IngestionService = _ingestion.IngestionService
app = _main.app
get_ingestion_service = _main.get_ingestion_service
get_settings = _main.get_settings
Settings = _main.Settings
IngestMetadata = _schemas.IngestMetadata
IngestResult = _schemas.IngestResult
Chunk = _semantic_chunking.Chunk
VectorStoreError = _vector_store.VectorStoreError

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


class FakeVectorStore:
    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: str,
        embedding_model: str,
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
    ) -> int:
        raise VectorStoreError("Qdrant upsert failed")


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


def test_text_only_ingest_uses_fake_service() -> None:
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
