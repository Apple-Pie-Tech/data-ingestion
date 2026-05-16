from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol

_config = import_module("app.config")
_audio_storage = import_module("app.audio_storage")
_embeddings = import_module("app.embeddings")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")
_transcription = import_module("app.transcription")
_vector_store = import_module("app.vector_store")

Settings = _config.Settings
AudioStorageClient = _audio_storage.AudioStorageClient
AudioStorageError = _audio_storage.AudioStorageError
AzureBlobAudioStorage = _audio_storage.AzureBlobAudioStorage
StoredAudio = _audio_storage.StoredAudio
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
    content_type: str | None = None


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
        audio_storage: AudioStorageClient | None = None,
    ) -> None:
        self._settings = settings
        self._audio_storage = audio_storage
        self._transcriber = transcriber
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def aclose(self) -> None:
        for resource in (
            self._audio_storage,
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
        stored_audio = await self._store_audio_if_needed(metadata=metadata, text=text, audio_file=audio_file)
        try:
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
                audio_url=stored_audio.url if stored_audio is not None else None,
            )

            if indexed_chunks != len(chunks):
                raise VectorStoreError(
                    "vector store acknowledged an unexpected number of indexed chunks"
                )

            return IngestResult(
                input_id=metadata.input_id,
                status="indexed",
                chunks=indexed_chunks,
                audio_url=stored_audio.url if stored_audio is not None else None,
            )
        except Exception:
            await self._cleanup_stored_audio(stored_audio)
            raise

    async def _store_audio_if_needed(
        self,
        *,
        metadata: IngestMetadata,
        text: str | None,
        audio_file: AudioFile | None,
    ) -> StoredAudio | None:
        normalized_text = normalize_text(text or "")
        if normalized_text:
            return None

        if audio_file is None or not audio_file.content:
            return None

        if not self._settings.audio_storage_enabled:
            return None

        try:
            return await self._get_audio_storage().store_audio(
                audio_file.content,
                input_id=metadata.input_id,
                filename=audio_file.filename,
                content_type=audio_file.content_type,
            )
        except AudioStorageError:
            raise
        except Exception as exc:
            raise AudioStorageError("audio storage request failed") from exc

    async def _cleanup_stored_audio(self, stored_audio: StoredAudio | None) -> None:
        if stored_audio is None or not self._settings.audio_storage_enabled:
            return

        try:
            await self._get_audio_storage().delete_audio(stored_audio)
        except Exception:
            return

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
                content_type=audio_file.content_type,
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
        audio_url: str | None,
    ) -> int:
        try:
            return await self._get_vector_store().upsert_chunks(
                metadata=metadata,
                chunks=chunks,
                embeddings=embeddings,
                source=source,
                embedding_model=self._get_embeddings().model_name,
                audio_url=audio_url,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("vector store unavailable") from exc

    def _get_transcriber(self) -> TranscriptionClient:
        if self._transcriber is None:
            self._transcriber = GradiumTranscriber(self._settings)
        return self._transcriber

    def _get_audio_storage(self) -> AudioStorageClient:
        if self._audio_storage is None:
            self._audio_storage = AzureBlobAudioStorage(self._settings)
        return self._audio_storage

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
