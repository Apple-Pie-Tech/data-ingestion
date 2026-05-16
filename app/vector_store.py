from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol, cast, runtime_checkable

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

_config = import_module("app.config")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")

Settings = _config.Settings
IngestMetadata = _schemas.IngestMetadata
Chunk = _semantic_chunking.Chunk


class VectorStoreError(RuntimeError):
    pass


class VectorStoreConfigurationError(VectorStoreError):
    pass


@runtime_checkable
class QdrantCollectionsAPI(Protocol):
    async def collection_exists(self, collection_name: str, **kwargs: object) -> bool: ...

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: qdrant_models.VectorParams,
        **kwargs: object,
    ) -> bool: ...

    async def get_collection(self, collection_name: str, **kwargs: object) -> object: ...

    async def upsert(
        self,
        collection_name: str,
        points: list[qdrant_models.PointStruct],
        **kwargs: object,
    ) -> object: ...


@runtime_checkable
class VectorStoreClient(Protocol):
    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
    ) -> int: ...


class _QdrantVectorConfig(Protocol):
    size: int


class _QdrantCollectionParams(Protocol):
    vectors: _QdrantVectorConfig | dict[str, _QdrantVectorConfig] | None


class _QdrantCollectionConfig(Protocol):
    params: _QdrantCollectionParams


class _QdrantCollectionInfo(Protocol):
    config: _QdrantCollectionConfig


@dataclass(frozen=True)
class _VectorStoreConfig:
    url: str
    collection_name: str
    dimension: int


class QdrantVectorStore:
    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantCollectionsAPI | None = None,
    ) -> None:
        self._config = _VectorStoreConfig(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            dimension=settings.embedding_dim,
        )
        self._client: QdrantCollectionsAPI | AsyncQdrantClient = client or AsyncQdrantClient(
            url=self._config.url
        )
        self._owns_client = client is None
        self._collection_verified = False

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "chunk and embedding count mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        await self._ensure_collection()
        points = self._build_points(
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings,
            source=source,
            embedding_model=embedding_model,
        )

        if not points:
            return 0

        await self._client.upsert(
            collection_name=self._config.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    async def _ensure_collection(self) -> None:
        if self._collection_verified:
            return

        collection_name = self._config.collection_name
        if not await self._client.collection_exists(collection_name):
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self._config.dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            self._collection_verified = True
            return

        collection_info = await self._client.get_collection(collection_name)
        vector_size = _get_collection_vector_size(collection_info)
        if vector_size != self._config.dimension:
            raise VectorStoreConfigurationError(
                "existing Qdrant collection vector size mismatch: "
                f"expected {self._config.dimension}, got {vector_size}"
            )

        self._collection_verified = True

    def _build_points(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
    ) -> list[qdrant_models.PointStruct]:
        points: list[qdrant_models.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            actual_dimension = len(embedding)
            if actual_dimension != self._config.dimension:
                raise VectorStoreError(
                    "embedding dimension mismatch for chunk "
                    f"{chunk.chunk_index}: expected {self._config.dimension}, "
                    f"got {actual_dimension}"
                )

            points.append(
                qdrant_models.PointStruct(
                    id=f"{metadata.input_id}:{chunk.chunk_index}",
                    vector=embedding,
                    payload={
                        "input_id": metadata.input_id,
                        "user_id": metadata.user_id,
                        "timestamp": metadata.timestamp.isoformat(),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "source": source,
                        "embedding_model": embedding_model,
                        "semantic_chunking": {
                            "break_threshold": _get_required_chunk_metadata(
                                chunk.metadata,
                                "similarity_threshold",
                            ),
                            "overlap_sentences": _get_required_chunk_metadata(
                                chunk.metadata,
                                "overlap_sentences",
                            ),
                        },
                    },
                )
            )

        return points


def _get_collection_vector_size(collection_info: object) -> int:
    typed_collection_info = cast(_QdrantCollectionInfo, collection_info)
    try:
        vectors_config = typed_collection_info.config.params.vectors
    except AttributeError as exc:
        raise VectorStoreConfigurationError(
            "existing Qdrant collection vector configuration is unavailable"
        ) from exc

    if vectors_config is None:
        raise VectorStoreConfigurationError("existing Qdrant collection has no vector config")

    if isinstance(vectors_config, dict):
        raise VectorStoreConfigurationError(
            "named Qdrant vectors are not supported for this ingestion collection"
        )

    try:
        return int(vectors_config.size)
    except AttributeError as exc:
        raise VectorStoreConfigurationError(
            "existing Qdrant collection vector size is unavailable"
        ) from exc


def _get_required_chunk_metadata(metadata: dict[str, object], key: str) -> object:
    if key not in metadata:
        raise VectorStoreError(f"semantic chunk metadata missing required key: {key}")

    return metadata[key]
