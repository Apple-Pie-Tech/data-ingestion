from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

_config = import_module("app.config")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")
_vector_store = import_module("app.vector_store")

Settings = _config.Settings
IngestMetadata = _schemas.IngestMetadata
Chunk = _semantic_chunking.Chunk
S3VectorStore = _vector_store.S3VectorStore
VectorStoreError = _vector_store.VectorStoreError
VectorStoreConfigurationError = _vector_store.VectorStoreConfigurationError
MAX_VECTORS_PER_PUT = _vector_store.MAX_VECTORS_PER_PUT
NON_FILTERABLE_METADATA_KEYS = _vector_store.NON_FILTERABLE_METADATA_KEYS


def make_settings(*, embedding_dim: int = 4, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "s3_vector_bucket": "applepie-vectors",
        "s3_vector_index": "apple-pie-story-chunks",
        "aws_region": "us-east-1",
        "embedding_dim": embedding_dim,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_metadata() -> IngestMetadata:
    return IngestMetadata(
        input_id="sample-input-001",
        user_id="user-123",
        timestamp=datetime(2025, 5, 1, 12, 30, tzinfo=UTC),
    )


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="We peeled the apples and dusted them with cinnamon.",
            chunk_index=0,
            metadata={"similarity_threshold": 0.72, "overlap_sentences": 1},
        ),
        Chunk(
            text="Then we measured vector recall against the story archive.",
            chunk_index=1,
            metadata={"similarity_threshold": 0.72, "overlap_sentences": 1},
        ),
    ]


@dataclass
class FakeS3Vectors:
    """Stands in for boto3's s3vectors client.

    Mirrors the behaviour probed against the live service: get_index reports the
    index dimension, and put_vectors records whole batches.
    """

    dimension: int = 4
    get_index_error: Exception | None = None
    put_error: Exception | None = None
    index_payload: dict[str, Any] | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    puts: list[list[dict[str, Any]]] = field(default_factory=list)

    def get_index(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_index", kwargs))
        if self.get_index_error is not None:
            raise self.get_index_error
        if self.index_payload is not None:
            return self.index_payload
        return {
            "index": {
                "indexName": kwargs["indexName"],
                "dimension": self.dimension,
                "dataType": "float32",
                "distanceMetric": "cosine",
            }
        }

    def put_vectors(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put_vectors", kwargs))
        if self.put_error is not None:
            raise self.put_error
        self.puts.append(kwargs["vectors"])
        return {}

    @property
    def vectors(self) -> list[dict[str, Any]]:
        return [vector for batch in self.puts for vector in batch]


def test_missing_bucket_is_a_configuration_error() -> None:
    with pytest.raises(VectorStoreConfigurationError, match="S3_VECTOR_BUCKET"):
        S3VectorStore(make_settings(s3_vector_bucket=None), client=FakeS3Vectors())


@pytest.mark.asyncio
async def test_deterministic_keys_and_flat_metadata_for_upserted_chunks() -> None:
    """S3 Vectors rejects nested metadata, so the Qdrant payload is flattened.

    Proven live: `Metadata values must be strings, numbers, booleans, or arrays`.
    """
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    upserted = await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=make_chunks(),
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        source="text",
        embedding_model="cohere.embed-v4:0",
        audio_url=None,
    )

    assert upserted == 2
    assert [vector["key"] for vector in fake.vectors] == [
        str(uuid5(NAMESPACE_URL, "sample-input-001:0")),
        str(uuid5(NAMESPACE_URL, "sample-input-001:1")),
    ]
    assert fake.vectors[0]["data"] == {"float32": [0.1, 0.2, 0.3, 0.4]}
    assert fake.vectors[0]["metadata"] == {
        "input_id": "sample-input-001",
        "user_id": "user-123",
        "timestamp": "2025-05-01T12:30:00+00:00",
        "chunk_index": 0,
        "text": "We peeled the apples and dusted them with cinnamon.",
        "source": "text",
        "embedding_model": "cohere.embed-v4:0",
        "semantic_chunking_break_threshold": 0.72,
        "semantic_chunking_overlap_sentences": 1,
    }


@pytest.mark.asyncio
async def test_no_metadata_value_is_a_nested_object() -> None:
    """The constraint that forced the flattening, asserted structurally."""
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=make_chunks(),
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        source="text",
        embedding_model="cohere.embed-v4:0",
        audio_url="https://applepie-audio.s3.us-east-1.amazonaws.com/audio/x/source.wav",
    )

    for vector in fake.vectors:
        for key, value in vector["metadata"].items():
            assert not isinstance(value, dict), f"{key} is a nested object"
            assert isinstance(value, str | int | float | bool | list), f"{key} is {type(value)}"


@pytest.mark.asyncio
async def test_no_credential_shaped_keys_reach_the_payload() -> None:
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=make_chunks()[:1],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        source="text",
        embedding_model="cohere.embed-v4:0",
        audio_url=None,
    )

    metadata = fake.vectors[0]["metadata"]
    assert not {"api_key", "token", "secret", "password"} & set(metadata)


@pytest.mark.asyncio
async def test_index_dimension_mismatch_raises_before_any_write() -> None:
    """Writing 1536-wide vectors into a differently sized index must not start."""
    fake = FakeS3Vectors(dimension=3)
    store = S3VectorStore(make_settings(embedding_dim=4), client=fake)

    with pytest.raises(VectorStoreConfigurationError) as exc_info:
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks()[:1],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="audio",
            embedding_model="cohere.embed-v4:0",
            audio_url=None,
        )

    assert "expected 4, got 3" in str(exc_info.value)
    assert [name for name, _ in fake.calls] == ["get_index"]
    assert fake.puts == []


@pytest.mark.asyncio
async def test_a_missing_index_is_a_configuration_error_not_a_write_failure() -> None:
    """Terraform creates the index; the service only validates it."""
    fake = FakeS3Vectors(get_index_error=RuntimeError("NotFoundException"))
    store = S3VectorStore(make_settings(), client=fake)

    with pytest.raises(VectorStoreConfigurationError, match="is unavailable"):
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks()[:1],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="text",
            embedding_model="cohere.embed-v4:0",
            audio_url=None,
        )


@pytest.mark.asyncio
async def test_the_index_is_validated_once_not_per_request() -> None:
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    for _ in range(3):
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks()[:1],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="text",
            embedding_model="cohere.embed-v4:0",
            audio_url=None,
        )

    assert [name for name, _ in fake.calls].count("get_index") == 1


@pytest.mark.asyncio
async def test_audio_url_is_stored_on_every_chunk_of_an_audio_ingest() -> None:
    audio_url = (
        "https://applepie-audio.s3.us-east-1.amazonaws.com/audio/sample-input-001/source.wav"
    )
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=make_chunks(),
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        source="audio",
        embedding_model="cohere.embed-v4:0",
        audio_url=audio_url,
    )

    assert all(vector["metadata"]["audio_url"] == audio_url for vector in fake.vectors)


@pytest.mark.asyncio
async def test_long_inputs_are_split_across_put_vectors_calls() -> None:
    """PutVectors accepts at most 500 vectors; a long document exceeds that."""
    count = MAX_VECTORS_PER_PUT + 7
    chunks = [
        Chunk(
            text=f"chunk {index}",
            chunk_index=index,
            metadata={"similarity_threshold": 0.72, "overlap_sentences": 1},
        )
        for index in range(count)
    ]
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    upserted = await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=chunks,
        embeddings=[[0.1, 0.2, 0.3, 0.4]] * count,
        source="text",
        embedding_model="cohere.embed-v4:0",
        audio_url=None,
    )

    assert upserted == count
    assert [len(batch) for batch in fake.puts] == [MAX_VECTORS_PER_PUT, 7]
    # Every chunk reaches the index exactly once, in order.
    assert [vector["metadata"]["chunk_index"] for vector in fake.vectors] == list(range(count))


@pytest.mark.asyncio
async def test_chunk_and_embedding_count_mismatch_is_rejected() -> None:
    store = S3VectorStore(make_settings(), client=FakeS3Vectors())

    with pytest.raises(VectorStoreError, match="count mismatch"):
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks(),
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="text",
            embedding_model="cohere.embed-v4:0",
            audio_url=None,
        )


@pytest.mark.asyncio
async def test_per_chunk_dimension_mismatch_names_the_chunk() -> None:
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(embedding_dim=4), client=fake)

    with pytest.raises(VectorStoreError, match="chunk 1"):
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks(),
            embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6]],
            source="text",
            embedding_model="cohere.embed-v4:0",
            audio_url=None,
        )

    assert fake.puts == []


@pytest.mark.asyncio
async def test_a_transport_failure_is_wrapped_as_a_vector_store_error() -> None:
    """app/main.py maps VectorStoreError to a 503; a raw ClientError is a 500."""
    fake = FakeS3Vectors(put_error=RuntimeError("ThrottlingException"))
    store = S3VectorStore(make_settings(), client=fake)

    with pytest.raises(VectorStoreError, match="put_vectors failed"):
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks()[:1],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="text",
            embedding_model="cohere.embed-v4:0",
            audio_url=None,
        )


@pytest.mark.asyncio
async def test_oversized_filterable_metadata_fails_with_a_readable_error() -> None:
    """Filterable metadata is capped at 2048 bytes, proven live.

    `text` is exempt because the index declares it non-filterable, which is the
    only reason a 1400-character chunk fits at all. A long value in any other
    key must name the limit instead of surfacing a bare 400.
    """
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)
    chunk = Chunk(
        text="x" * 5000,
        chunk_index=0,
        metadata={"similarity_threshold": 0.72, "overlap_sentences": 1},
    )

    # The oversized `text` alone must NOT trip the guard.
    await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=[chunk],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        source="text",
        embedding_model="cohere.embed-v4:0",
        audio_url=None,
    )
    assert len(fake.vectors) == 1

    with pytest.raises(VectorStoreError, match="2048-byte"):
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=[chunk],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="text",
            embedding_model="m" * 3000,
            audio_url=None,
        )


def test_text_and_audio_url_are_declared_non_filterable() -> None:
    """These names are frozen at index creation and cannot be changed later.

    Terraform creates the index from this list, so changing it means rebuilding
    the index and re-embedding the corpus.
    """
    assert "text" in NON_FILTERABLE_METADATA_KEYS
    assert "audio_url" in NON_FILTERABLE_METADATA_KEYS
    assert len(NON_FILTERABLE_METADATA_KEYS) <= 10


@pytest.mark.asyncio
async def test_metadata_is_json_serialisable() -> None:
    """botocore serialises the metadata document; a stray type fails at runtime."""
    fake = FakeS3Vectors()
    store = S3VectorStore(make_settings(), client=fake)

    await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=make_chunks()[:1],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        source="text",
        embedding_model="cohere.embed-v4:0",
        audio_url=None,
    )

    json.dumps(fake.vectors[0]["metadata"])
