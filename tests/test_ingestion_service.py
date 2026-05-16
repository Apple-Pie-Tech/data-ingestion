from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module

import pytest

_config = import_module("app.config")
_ingestion = import_module("app.ingestion")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")

Settings = _config.Settings
AudioFile = _ingestion.AudioFile
IngestionService = _ingestion.IngestionService
IngestionValidationError = _ingestion.IngestionValidationError
IngestMetadata = _schemas.IngestMetadata
Chunk = _semantic_chunking.Chunk


def make_metadata() -> IngestMetadata:
    return IngestMetadata(
        input_id="sample-input-001",
        user_id="user-123",
        timestamp=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
    )


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="First semantic chunk.",
            chunk_index=0,
            metadata={
                "similarity_threshold": 0.72,
                "overlap_sentences": 1,
            },
        ),
        Chunk(
            text="Second semantic chunk.",
            chunk_index=1,
            metadata={
                "similarity_threshold": 0.72,
                "overlap_sentences": 1,
            },
        ),
    ]


class FakeTranscriber:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "filename": filename,
                "content_type": content_type,
            }
        )
        return self.transcript

    async def aclose(self) -> None:
        self.closed = True


class FakeChunker:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []
        self.closed = False

    def chunk(self, text: str) -> list[Chunk]:
        self.calls.append(text)
        return self.chunks

    async def aclose(self) -> None:
        self.closed = True


class FakeEmbeddingsClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[list[str]] = []
        self.model_name = "text-embedding-3-large"
        self.closed = False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self.embeddings

    async def aclose(self) -> None:
        self.closed = True


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: str,
        embedding_model: str,
    ) -> int:
        self.calls.append(
            {
                "metadata": metadata,
                "chunks": chunks,
                "embeddings": embeddings,
                "source": source,
                "embedding_model": embedding_model,
            }
        )
        return len(chunks)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_text_wins_over_audio_and_transcriber_is_not_called() -> None:
    transcriber = FakeTranscriber("ignored transcript")
    chunker = FakeChunker(make_chunks())
    embeddings = FakeEmbeddingsClient(
        embeddings=[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )
    vector_store = FakeVectorStore()
    service = IngestionService(
        Settings(),
        transcriber=transcriber,
        chunker=chunker,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = await service.ingest(
        metadata=make_metadata(),
        text="  Hello\n\napple pie  ",
        audio_file=AudioFile(
            content=b"fake-audio-bytes",
            filename="sample.wav",
            content_type="audio/wav",
        ),
    )

    assert result.input_id == "sample-input-001"
    assert result.status == "indexed"
    assert result.chunks == 2
    assert transcriber.calls == []
    assert chunker.calls == ["Hello apple pie"]
    assert embeddings.calls == [["First semantic chunk.", "Second semantic chunk."]]
    assert vector_store.calls[0]["source"] == "text"


@pytest.mark.asyncio
async def test_audio_only_transcribes_once_and_indexes_audio_source() -> None:
    transcriber = FakeTranscriber("  transcribed\n speech  ")
    chunker = FakeChunker(make_chunks()[:1])
    embeddings = FakeEmbeddingsClient(embeddings=[[0.1, 0.2, 0.3, 0.4]])
    vector_store = FakeVectorStore()
    service = IngestionService(
        Settings(),
        transcriber=transcriber,
        chunker=chunker,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    result = await service.ingest(
        metadata=make_metadata(),
        text=None,
        audio_file=AudioFile(
            content=b"fake-audio-bytes",
            filename="sample.wav",
            content_type="audio/wav",
        ),
    )

    assert result.input_id == "sample-input-001"
    assert result.status == "indexed"
    assert result.chunks == 1
    assert transcriber.calls == [
        {
            "audio_bytes": b"fake-audio-bytes",
            "filename": "sample.wav",
            "content_type": "audio/wav",
        }
    ]
    assert chunker.calls == ["transcribed speech"]
    assert vector_store.calls[0]["source"] == "audio"


@pytest.mark.asyncio
async def test_empty_normalized_transcript_is_rejected() -> None:
    service = IngestionService(
        Settings(),
        transcriber=FakeTranscriber("   \n   "),
        chunker=FakeChunker(make_chunks()),
        embeddings=FakeEmbeddingsClient(embeddings=[[0.1, 0.2, 0.3, 0.4]]),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(IngestionValidationError, match="transcript must not be empty"):
        await service.ingest(
            metadata=make_metadata(),
            text=None,
            audio_file=AudioFile(
                content=b"fake-audio-bytes",
                filename="sample.wav",
                content_type="audio/wav",
            ),
        )


@pytest.mark.asyncio
async def test_aclose_closes_all_lazily_created_dependencies() -> None:
    transcriber = FakeTranscriber("transcript")
    chunker = FakeChunker(make_chunks())
    embeddings = FakeEmbeddingsClient(embeddings=[[0.1, 0.2, 0.3, 0.4]])
    vector_store = FakeVectorStore()
    service = IngestionService(
        Settings(),
        transcriber=transcriber,
        chunker=chunker,
        embeddings=embeddings,
        vector_store=vector_store,
    )

    await service.aclose()

    assert transcriber.closed is True
    assert chunker.closed is True
    assert embeddings.closed is True
    assert vector_store.closed is True
