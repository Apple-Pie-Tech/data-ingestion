from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol, runtime_checkable

_config = import_module("app.config")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")

Settings = _config.Settings
IngestMetadata = _schemas.IngestMetadata
Chunk = _semantic_chunking.Chunk


# Verified against the live service, not just the docs:
#
#   * metadata values must be strings, numbers, booleans or arrays -- a nested
#     object is rejected with ValidationException. The Qdrant payload nested
#     `semantic_chunking`, so those keys are flattened with a prefix.
#   * filterable metadata is capped at 2048 bytes per vector, and `text` alone
#     can exceed that, so `text` and `audio_url` are declared non-filterable on
#     the index.
#   * PutVectors takes at most 500 vectors per call.
#   * PutVectors on an existing key replaces its metadata wholesale; there is no
#     partial update.
MAX_VECTORS_PER_PUT = 500
MAX_FILTERABLE_METADATA_BYTES = 2048

# Declared when the index is created, and NOT changeable afterwards -- AWS
# documents that a non-filterable key cannot be made filterable later. Terraform
# creates the index with exactly this list, which is why it lives here.
NON_FILTERABLE_METADATA_KEYS: tuple[str, ...] = (
    "text",
    "audio_url",
    "clustering_description",
)


class VectorStoreError(RuntimeError):
    pass


class VectorStoreConfigurationError(VectorStoreError):
    pass


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
        audio_url: str | None,
    ) -> int: ...


@dataclass(frozen=True)
class _VectorStoreConfig:
    bucket: str
    index: str
    dimension: int


class S3VectorStore:
    """Writes story chunks into an S3 Vectors index.

    boto3 is synchronous, so calls hop to a worker thread, matching
    app/embeddings.py and app/audio_storage.py.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
    ) -> None:
        if not settings.s3_vector_bucket:
            raise VectorStoreConfigurationError("S3_VECTOR_BUCKET is required")
        if not settings.s3_vector_index:
            raise VectorStoreConfigurationError("S3_VECTOR_INDEX is required")

        self._config = _VectorStoreConfig(
            bucket=settings.s3_vector_bucket,
            index=settings.s3_vector_index,
            dimension=settings.embedding_dim,
        )
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            boto3 = import_module("boto3")
            self._client = boto3.client("s3vectors", region_name=settings.aws_region)
        self._index_verified = False

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close = getattr(self._client, "close", None)
        if close is None:
            return
        await asyncio.to_thread(close)

    async def upsert_chunks(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
        audio_url: str | None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "chunk and embedding count mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        await self._ensure_index()
        vectors = self._build_vectors(
            metadata=metadata,
            chunks=chunks,
            embeddings=embeddings,
            source=source,
            embedding_model=embedding_model,
            audio_url=audio_url,
        )

        if not vectors:
            return 0

        for batch in _batched(vectors, MAX_VECTORS_PER_PUT):
            try:
                await asyncio.to_thread(
                    self._client.put_vectors,
                    vectorBucketName=self._config.bucket,
                    indexName=self._config.index,
                    vectors=batch,
                )
            except Exception as exc:
                raise VectorStoreError(f"S3 Vectors put_vectors failed: {exc}") from exc

        return len(vectors)

    async def _ensure_index(self) -> None:
        """Validate the index matches the configured dimension.

        The index is created by Terraform, not here: writing to the wrong width
        is the failure worth catching, and an ingest role that could create
        indexes would defeat the point of scoping it to one. The dimension is
        also enforced server-side on every write, but failing once up front
        names the setting instead of reporting a per-vector length error.
        """
        if self._index_verified:
            return

        try:
            response = await asyncio.to_thread(
                self._client.get_index,
                vectorBucketName=self._config.bucket,
                indexName=self._config.index,
            )
        except Exception as exc:
            raise VectorStoreConfigurationError(
                f"S3 Vectors index {self._config.bucket}/{self._config.index} "
                f"is unavailable: {exc}"
            ) from exc

        index = response.get("index") or {}
        dimension = index.get("dimension")
        if dimension is None:
            raise VectorStoreConfigurationError(
                "S3 Vectors index dimension is unavailable"
            )
        if int(dimension) != self._config.dimension:
            raise VectorStoreConfigurationError(
                "existing S3 Vectors index dimension mismatch: "
                f"expected {self._config.dimension}, got {int(dimension)}"
            )

        self._index_verified = True

    def _build_vectors(
        self,
        *,
        metadata: IngestMetadata,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
        audio_url: str | None,
    ) -> list[dict[str, Any]]:
        vectors: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            actual_dimension = len(embedding)
            if actual_dimension != self._config.dimension:
                raise VectorStoreError(
                    "embedding dimension mismatch for chunk "
                    f"{chunk.chunk_index}: expected {self._config.dimension}, "
                    f"got {actual_dimension}"
                )

            chunk_metadata: dict[str, Any] = {
                "input_id": metadata.input_id,
                "user_id": metadata.user_id,
                "timestamp": metadata.timestamp.isoformat(),
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "source": source,
                "embedding_model": embedding_model,
                # Flattened: S3 Vectors rejects a nested object outright.
                "semantic_chunking_break_threshold": _get_required_chunk_metadata(
                    chunk.metadata,
                    "similarity_threshold",
                ),
                "semantic_chunking_overlap_sentences": _get_required_chunk_metadata(
                    chunk.metadata,
                    "overlap_sentences",
                ),
            }
            if audio_url is not None:
                chunk_metadata["audio_url"] = audio_url

            validate_filterable_metadata_size(chunk_metadata, label=f"chunk {chunk.chunk_index}")

            vectors.append(
                {
                    # Keys are plain strings in S3 Vectors, so the uuid5 is no
                    # longer forced by the backend. It is kept because it is the
                    # dedupe contract: re-ingesting an input_id must overwrite
                    # its chunks rather than accumulate duplicates, and
                    # pdf-parser relies on the same scheme under a `pdf:` prefix.
                    "key": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{metadata.input_id}:{chunk.chunk_index}",
                        )
                    ),
                    "data": {"float32": [float(value) for value in embedding]},
                    "metadata": chunk_metadata,
                }
            )

        return vectors


def validate_filterable_metadata_size(metadata: dict[str, Any], *, label: str) -> None:
    """Fail with a readable message before S3 Vectors returns a 400.

    Filterable metadata is capped at 2048 bytes per vector. The cap is easy to
    breach by accident -- a long LLM-written cluster theme is enough -- and the
    service's own error names neither the limit's cause nor the offending keys.
    """
    filterable = {
        key: value
        for key, value in metadata.items()
        if key not in NON_FILTERABLE_METADATA_KEYS
    }
    size = len(json.dumps(filterable, separators=(",", ":")).encode("utf-8"))
    if size > MAX_FILTERABLE_METADATA_BYTES:
        raise VectorStoreError(
            f"filterable metadata for {label} is {size} bytes, over the "
            f"{MAX_FILTERABLE_METADATA_BYTES}-byte S3 Vectors limit; "
            f"keys: {sorted(filterable)}"
        )


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _get_required_chunk_metadata(metadata: dict[str, object], key: str) -> object:
    if key not in metadata:
        raise VectorStoreError(f"semantic chunk metadata missing required key: {key}")

    return metadata[key]
