from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import pytest

_config = import_module("app.config")
_embeddings = import_module("app.embeddings")

Settings = _config.Settings
EmbeddingClient = _embeddings.EmbeddingClient
EmbeddingDimensionError = _embeddings.EmbeddingDimensionError


def make_settings(*, embedding_dim: int = 4) -> Settings:
    return Settings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret-key",
        azure_openai_api_version="2024-10-21",
        azure_openai_embeddings_deployment="embeddings-deployment",
        embedding_model="text-embedding-3-large",
        embedding_dim=embedding_dim,
    )


@dataclass(frozen=True)
class _FakeEmbeddingRecord:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingRecord]


class FakeEmbeddingsAPI:
    def __init__(self, responses: list[list[float]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, *, input: list[str], model: str, dimensions: int) -> object:
        self.calls.append(
            {
                "input": input,
                "model": model,
                "dimensions": dimensions,
            }
        )
        return _FakeEmbeddingResponse(
            data=[
                _FakeEmbeddingRecord(index=index, embedding=embedding)
                for index, embedding in enumerate(self._responses)
            ]
        )


@pytest.mark.asyncio
async def test_embedding_client_uses_deployment_name_as_model_value() -> None:
    fake_api = FakeEmbeddingsAPI(
        responses=[
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )
    client = EmbeddingClient(make_settings(), embeddings_api=fake_api)

    embeddings = await client.embed_texts(["apple pie", "vector database"])

    assert embeddings == [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    assert fake_api.calls == [
        {
            "input": ["apple pie", "vector database"],
            "model": "embeddings-deployment",
            "dimensions": 4,
        }
    ]


@pytest.mark.asyncio
async def test_dimension_mismatch_raises_embedding_dimension_error() -> None:
    fake_api = FakeEmbeddingsAPI(responses=[[0.1, 0.2, 0.3]])
    client = EmbeddingClient(make_settings(embedding_dim=4), embeddings_api=fake_api)

    with pytest.raises(EmbeddingDimensionError) as exc_info:
        await client.embed_texts(["apple pie"])

    assert exc_info.value.expected == 4
    assert exc_info.value.actual == 3


@pytest.mark.asyncio
async def test_empty_inputs_short_circuit_without_api_call() -> None:
    fake_api = FakeEmbeddingsAPI(responses=[])
    client = EmbeddingClient(make_settings(), embeddings_api=fake_api)

    embeddings = await client.embed_texts([])

    assert embeddings == []
    assert fake_api.calls == []
