from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast, runtime_checkable

_config = import_module("app.config")

Settings = _config.Settings

# Cohere's Bedrock embedding endpoint rejects input lists longer than this. The
# previous Azure implementation sent the whole list in one request and relied on
# OpenAI's much larger ceiling, so a long document silently worked there and would
# fail here. Longer lists are split and stitched back together.
COHERE_MAX_TEXTS_PER_REQUEST = 96


class EmbeddingError(RuntimeError):
    pass


class EmbeddingConfigurationError(EmbeddingError):
    pass


class EmbeddingDimensionError(EmbeddingError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"embedding dimension mismatch: expected {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


@runtime_checkable
class EmbeddingsAPI(Protocol):
    async def create(
        self,
        *,
        input: list[str],
        model: str,
        dimensions: int,
    ) -> object: ...


class EmbeddingRecord(Protocol):
    index: int
    embedding: list[float]


class EmbeddingResponse(Protocol):
    data: list[EmbeddingRecord]


@dataclass(frozen=True)
class _Record:
    index: int
    embedding: list[float]


@dataclass(frozen=True)
class _Response:
    data: list[_Record]


def invoke_cohere_embeddings(
    client: Any,
    texts: list[str],
    *,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    """One synchronous Cohere-on-Bedrock embedding call.

    Shared with the Chonkie adapter in ``semantic_chunking.py``, which needs the same
    request on a synchronous code path. Callers are responsible for batching to
    ``COHERE_MAX_TEXTS_PER_REQUEST``.
    """
    body = json.dumps(
        {
            "texts": texts,
            # Stored vectors are documents; queries must use "search_query" so
            # Cohere's asymmetric embeddings land in the same space.
            "input_type": "search_document",
            "output_dimension": dimensions,
            "embedding_types": ["float"],
        }
    )

    try:
        response = client.invoke_model(modelId=model, body=body)
        payload = json.loads(response["body"].read())
    except Exception as exc:
        raise EmbeddingError(f"Bedrock embedding request failed: {exc}") from exc

    embeddings = payload.get("embeddings")
    # Embed v4 keys the result by embedding type; v3 returns a bare list.
    if isinstance(embeddings, dict):
        embeddings = embeddings.get("float")

    if not isinstance(embeddings, list):
        raise EmbeddingError(f"unexpected Bedrock embedding payload, keys: {sorted(payload)}")

    return [list(vector) for vector in embeddings]


class BedrockEmbeddingsAPI:
    """Cohere Embed on Bedrock, behind the interface the Azure OpenAI client used.

    Keeping the ``create(input=, model=, dimensions=)`` shape means ``EmbeddingClient``
    and every injected test double stay as they were; only the transport changed.

    boto3 is synchronous, so each request is handed to a worker thread. Calling it
    directly would block the event loop for the whole round-trip, stalling every
    other request the service is serving.
    """

    def __init__(
        self,
        *,
        region: str,
        client: Any | None = None,
        max_texts_per_request: int = COHERE_MAX_TEXTS_PER_REQUEST,
    ) -> None:
        if client is None:
            boto3 = import_module("boto3")
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self._max_texts_per_request = max_texts_per_request

    async def create(
        self,
        *,
        input: list[str],
        model: str,
        dimensions: int,
    ) -> object:
        records: list[_Record] = []
        for offset in range(0, len(input), self._max_texts_per_request):
            batch = input[offset : offset + self._max_texts_per_request]
            vectors = await asyncio.to_thread(self._invoke, batch, model, dimensions)

            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Bedrock returned {len(vectors)} embeddings for {len(batch)} texts"
                )

            records.extend(
                _Record(index=offset + position, embedding=vector)
                for position, vector in enumerate(vectors)
            )

        return _Response(data=records)

    def _invoke(self, texts: list[str], model: str, dimensions: int) -> list[list[float]]:
        return invoke_cohere_embeddings(
            self._client, texts, model=model, dimensions=dimensions
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class _EmbeddingConfig:
    region: str
    model_id: str
    model_name: str
    dimension: int


class EmbeddingClient:
    def __init__(
        self,
        settings: Settings,
        *,
        embeddings_api: EmbeddingsAPI | None = None,
    ) -> None:
        if not settings.aws_region:
            raise EmbeddingConfigurationError("AWS_REGION is required")

        if not settings.bedrock_embeddings_model:
            raise EmbeddingConfigurationError("BEDROCK_EMBEDDINGS_MODEL is required")

        self._config = _EmbeddingConfig(
            region=settings.aws_region,
            model_id=settings.bedrock_embeddings_model,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dim,
        )
        # Only a client we built ourselves gets closed; an injected one is the
        # caller's to manage.
        self._owned_api: BedrockEmbeddingsAPI | None = None
        if embeddings_api is None:
            self._owned_api = BedrockEmbeddingsAPI(region=self._config.region)
            self._embeddings_api: EmbeddingsAPI = self._owned_api
        else:
            self._embeddings_api = embeddings_api

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def dimension(self) -> int:
        return self._config.dimension

    async def aclose(self) -> None:
        if self._owned_api is None:
            return

        close = getattr(self._owned_api, "aclose", None) or getattr(
            self._owned_api, "close", None
        )
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = cast(
            EmbeddingResponse,
            await self._embeddings_api.create(
                input=texts,
                model=self._config.model_id,
                dimensions=self._config.dimension,
            ),
        )
        data = sorted(response.data, key=lambda item: item.index)
        embeddings = [list(item.embedding) for item in data]

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"embedding response size mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        for embedding in embeddings:
            actual_dimension = len(embedding)
            if actual_dimension != self._config.dimension:
                raise EmbeddingDimensionError(
                    expected=self._config.dimension,
                    actual=actual_dimension,
                )

        return embeddings
