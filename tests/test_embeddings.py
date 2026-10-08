from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module

import pytest

_config = import_module("app.config")
_embeddings = import_module("app.embeddings")

Settings = _config.Settings
EmbeddingClient = _embeddings.EmbeddingClient
EmbeddingError = _embeddings.EmbeddingError
EmbeddingDimensionError = _embeddings.EmbeddingDimensionError
BedrockEmbeddingsAPI = _embeddings.BedrockEmbeddingsAPI


def make_settings(*, embedding_dim: int = 4) -> Settings:
    return Settings(
        _env_file=None,
        aws_region="us-east-1",
        bedrock_embeddings_model="cohere.embed-v4:0",
        embedding_model="cohere.embed-v4:0",
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


class _FakeBody:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class FakeBedrockRuntime:
    """Stands in for boto3's bedrock-runtime client.

    Returns one vector per input text so batching can be asserted on exactly.
    """

    def __init__(self, *, dimension: int = 4) -> None:
        self.requests: list[dict[str, object]] = []
        self._dimension = dimension

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, object]:
        request = json.loads(body)
        self.requests.append({"modelId": modelId, **request})
        vectors = [
            [float(index)] * self._dimension for index in range(len(request["texts"]))
        ]
        return {"body": _FakeBody({"embeddings": {"float": vectors}})}


@pytest.mark.asyncio
async def test_embedding_client_sends_the_bedrock_model_id() -> None:
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
            "model": "cohere.embed-v4:0",
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


@pytest.mark.asyncio
async def test_long_input_lists_are_split_across_requests() -> None:
    """Cohere on Bedrock rejects input lists past its per-request ceiling.

    The previous Azure implementation sent the whole list in one call and relied on
    OpenAI's far larger limit, so a long document worked there and would fail here.
    Indices must stay globally correct across the sub-requests, otherwise chunks get
    silently paired with the wrong vectors.
    """
    runtime = FakeBedrockRuntime(dimension=4)
    api = BedrockEmbeddingsAPI(region="us-east-1", client=runtime, max_texts_per_request=2)

    response = await api.create(
        input=["a", "b", "c", "d", "e"],
        model="cohere.embed-v4:0",
        dimensions=4,
    )

    assert [len(request["texts"]) for request in runtime.requests] == [2, 2, 1]
    assert [record.index for record in response.data] == [0, 1, 2, 3, 4]
    # Every request is sent with the configured output width and document input type.
    assert {request["output_dimension"] for request in runtime.requests} == {4}
    assert {request["input_type"] for request in runtime.requests} == {"search_document"}


@pytest.mark.asyncio
async def test_batched_requests_keep_client_dimension_checks_working() -> None:
    runtime = FakeBedrockRuntime(dimension=4)
    api = BedrockEmbeddingsAPI(region="us-east-1", client=runtime, max_texts_per_request=2)
    client = EmbeddingClient(make_settings(embedding_dim=4), embeddings_api=api)

    embeddings = await client.embed_texts(["a", "b", "c"])

    assert len(embeddings) == 3
    assert all(len(vector) == 4 for vector in embeddings)


@pytest.mark.asyncio
async def test_unexpected_bedrock_payload_raises_embedding_error() -> None:
    class BadRuntime:
        def invoke_model(self, *, modelId: str, body: str) -> dict[str, object]:
            return {"body": _FakeBody({"message": "something else entirely"})}

    api = BedrockEmbeddingsAPI(region="us-east-1", client=BadRuntime())

    with pytest.raises(EmbeddingError, match="unexpected Bedrock embedding payload"):
        await api.create(input=["a"], model="cohere.embed-v4:0", dimensions=4)


@pytest.mark.asyncio
async def test_bedrock_transport_failure_is_wrapped_as_embedding_error() -> None:
    class ExplodingRuntime:
        def invoke_model(self, *, modelId: str, body: str) -> dict[str, object]:
            raise RuntimeError("throttled")

    api = BedrockEmbeddingsAPI(region="us-east-1", client=ExplodingRuntime())

    with pytest.raises(EmbeddingError, match="Bedrock embedding request failed"):
        await api.create(input=["a"], model="cohere.embed-v4:0", dimensions=4)
