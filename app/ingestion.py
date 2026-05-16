from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol

_config = import_module("app.config")
_embeddings = import_module("app.embeddings")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")
_transcription = import_module("app.transcription")
_vector_store = import_module("app.vector_store")

Settings = _config.Settings
EmbeddingClient = _embeddings.EmbeddingClient
EmbeddingError = _embeddings.EmbeddingError
IngestMetadata = _schemas.IngestMetadata
IngestResult = _schemas.IngestResult
Chunk = _semantic_chunking.Chunk
SemanticChunkerAdapter = _semantic_chunking.SemanticChunkerAdapter
normalize_text = _semantic_chunking.normalize_text
GradiumTranscriber = _transcription.GradiumTranscriber
TranscriptionClient = _transcription.TranscriptionClient
TranscriptionError = _transcription.TranscriptionError
QdrantVectorStore = _vector_store.QdrantVectorStore
VectorStoreClient = _vector_store.VectorStoreClient
VectorStoreError = _vector_store.VectorStoreError


class Chunker(Protocol):
    def chunk(self, text: str) -> list[Chunk]: ...


class EmbeddingsClient(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class AudioFile:
    content: bytes
    filename: str | None = None


class IngestionValidationError(ValueError):
    pass


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        *,
        transcriber: TranscriptionClient | None = None,
        chunker: Chunker | None = None,
        embeddings: EmbeddingsClient | None = None,
        vector_store: VectorStoreClient | None = None,
    ) -> None:
        self._settings = settings
        self._transcriber = transcriber
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def aclose(self) -> None:
        for resource in (
            self._transcriber,
            self._chunker,
            self._embeddings,
            self._vector_store,
        ):
            await self._close_resource(resource)

    async def ingest(
        self,
        *,
        metadata: IngestMetadata,
        text: str | None,
        audio_file: AudioFile | None,
    ) -> IngestResult:
        transcript, source = await self._resolve_transcript(text=text, audio_file=audio_file)

        chunks = self._get_chunker().chunk(transcript)
        if not chunks:
            raise RuntimeError("semantic chunking produced no chunks")

        embeddings = await self._embed_chunks(chunks)
        indexed_chunks = await self._upsert_chunks(
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings,
            source=source,
        )

        if indexed_chunks != len(chunks):
            raise VectorStoreError(
                "vector store acknowledged an unexpected number of indexed chunks"
            )

        return IngestResult(
            input_id=metadata.input_id,
            status="indexed",
            chunks=indexed_chunks,
        )

    async def _resolve_transcript(
        self,
        *,
        text: str | None,
        audio_file: AudioFile | None,
    ) -> tuple[str, Literal["text", "audio"]]:
        normalized_text = normalize_text(text or "")
        if normalized_text:
            return normalized_text, "text"

        if audio_file is None or not audio_file.content:
            raise IngestionValidationError("text or audio required")

        try:
            transcript = await self._get_transcriber().transcribe(
                audio_file.content,
                filename=audio_file.filename,
            )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("transcription request failed") from exc

        normalized_transcript = normalize_text(transcript)
        if not normalized_transcript:
            raise IngestionValidationError("transcript must not be empty")

        return normalized_transcript, "audio"

    async def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        try:
            return await self._get_embeddings().embed_texts([chunk.text for chunk in chunks])
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("embedding request failed") from exc

    async def _upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
    ) -> int:
        try:
            return await self._get_vector_store().upsert_chunks(
                metadata=metadata,
                chunks=chunks,
                embeddings=embeddings,
                source=source,
                embedding_model=self._get_embeddings().model_name,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("vector store unavailable") from exc

    def _get_transcriber(self) -> TranscriptionClient:
        if self._transcriber is None:
            self._transcriber = GradiumTranscriber(self._settings)
        return self._transcriber

    def _get_chunker(self) -> Chunker:
        if self._chunker is None:
            self._chunker = SemanticChunkerAdapter(self._settings)
        return self._chunker

    def _get_embeddings(self) -> EmbeddingsClient:
        if self._embeddings is None:
            self._embeddings = EmbeddingClient(self._settings)
        return self._embeddings

    def _get_vector_store(self) -> VectorStoreClient:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(self._settings)
        return self._vector_store

    async def _close_resource(self, resource: object | None) -> None:
        if resource is None:
            return

        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result
