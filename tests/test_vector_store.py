from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from typing import cast

import pytest

_config = import_module("app.config")
_schemas = import_module("app.schemas")
_semantic_chunking = import_module("app.semantic_chunking")
_vector_store = import_module("app.vector_store")
_qdrant_models = import_module("qdrant_client.http.models")

Settings = _config.Settings
IngestMetadata = _schemas.IngestMetadata
Chunk = _semantic_chunking.Chunk
QdrantVectorStore = _vector_store.QdrantVectorStore
VectorStoreConfigurationError = _vector_store.VectorStoreConfigurationError
Distance = _qdrant_models.Distance
PointStruct = _qdrant_models.PointStruct
VectorParams = _qdrant_models.VectorParams


def make_settings(
    *,
    embedding_dim: int = 4,
    qdrant_api_key: str | None = None,
) -> Settings:
    return Settings(
        qdrant_url="http://qdrant:6333",
        qdrant_api_key=qdrant_api_key,
        qdrant_collection="apple_pie_story_chunks",
        embedding_dim=embedding_dim,
    )


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
            metadata={
                "similarity_threshold": 0.72,
                "overlap_sentences": 1,
            },
        ),
        Chunk(
            text="Then we measured vector recall against the story archive.",
            chunk_index=1,
            metadata={
                "similarity_threshold": 0.72,
                "overlap_sentences": 1,
            },
        ),
    ]


@dataclass
class FakeQdrantClient:
    collection_exists_result: bool = False
    collection_size: int = 4

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.upserted_points: list[PointStruct] = []

    async def collection_exists(self, collection_name: str, **kwargs: object) -> bool:
        self.calls.append(
            ("collection_exists", {"collection_name": collection_name, **kwargs})
        )
        return self.collection_exists_result

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams,
        **kwargs: object,
    ) -> bool:
        self.calls.append(
            (
                "create_collection",
                {
                    "collection_name": collection_name,
                    "vectors_config": vectors_config,
                    **kwargs,
                },
            )
        )
        return True

    async def get_collection(self, collection_name: str, **kwargs: object) -> object:
        self.calls.append(("get_collection", {"collection_name": collection_name, **kwargs}))
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=self.collection_size),
                )
            )
        )

    async def upsert(
        self,
        collection_name: str,
        points: list[PointStruct],
        **kwargs: object,
    ) -> object:
        self.calls.append(
            (
                "upsert",
                {
                    "collection_name": collection_name,
                    "points": points,
                    **kwargs,
                },
            )
        )
        self.upserted_points = points
        return SimpleNamespace(status="acknowledged")


class AsyncQdrantClientSpy:
    created_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        type(self).created_kwargs = kwargs

    async def collection_exists(self, collection_name: str, **kwargs: object) -> bool:
        raise AssertionError("unexpected call")

    async def create_collection(self, *args: object, **kwargs: object) -> bool:
        raise AssertionError("unexpected call")

    async def get_collection(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected call")

    async def upsert(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("unexpected call")


def test_qdrant_cloud_api_key_is_passed_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_vector_store, "AsyncQdrantClient", AsyncQdrantClientSpy)

    store = QdrantVectorStore(make_settings(qdrant_api_key="qdrant-secret"), client=None)

    assert store is not None
    assert AsyncQdrantClientSpy.created_kwargs == {
        "url": "http://qdrant:6333",
        "api_key": "qdrant-secret",
    }


@pytest.mark.asyncio
async def test_deterministic_ids_and_payload_shape_for_upserted_chunks() -> None:
    fake_client = FakeQdrantClient(collection_exists_result=False)
    store = QdrantVectorStore(make_settings(), client=fake_client)

    upserted = await store.upsert_chunks(
        metadata=make_metadata(),
        chunks=make_chunks(),
        embeddings=[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        source="text",
        embedding_model="text-embedding-3-large",
    )

    assert upserted == 2
    create_call = fake_client.calls[1]
    assert create_call[0] == "create_collection"
    vectors_config = cast(VectorParams, create_call[1]["vectors_config"])
    assert vectors_config.size == 4
    assert vectors_config.distance == Distance.COSINE

    assert [point.id for point in fake_client.upserted_points] == [
        "sample-input-001:0",
        "sample-input-001:1",
    ]

    first_payload = fake_client.upserted_points[0].payload
    assert first_payload == {
        "input_id": "sample-input-001",
        "user_id": "user-123",
        "timestamp": "2025-05-01T12:30:00+00:00",
        "chunk_index": 0,
        "text": "We peeled the apples and dusted them with cinnamon.",
        "source": "text",
        "embedding_model": "text-embedding-3-large",
        "semantic_chunking": {
            "break_threshold": 0.72,
            "overlap_sentences": 1,
        },
    }
    assert not {"api_key", "token", "secret", "password"} & set(first_payload)


@pytest.mark.asyncio
async def test_wrong_dimension_existing_collection_raises_configuration_error() -> None:
    fake_client = FakeQdrantClient(collection_exists_result=True, collection_size=3)
    store = QdrantVectorStore(make_settings(embedding_dim=4), client=fake_client)

    with pytest.raises(VectorStoreConfigurationError) as exc_info:
        await store.upsert_chunks(
            metadata=make_metadata(),
            chunks=make_chunks()[:1],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            source="audio",
            embedding_model="text-embedding-3-large",
        )

    assert "expected 4, got 3" in str(exc_info.value)
    assert [name for name, _ in fake_client.calls] == ["collection_exists", "get_collection"]
    assert fake_client.upserted_points == []
