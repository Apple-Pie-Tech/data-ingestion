from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast, runtime_checkable

from openai import AsyncAzureOpenAI

_config = import_module("app.config")

Settings = _config.Settings


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
class AzureEmbeddingsAPI(Protocol):
    async def create(
        self,
        *,
        input: list[str],
        model: str,
        dimensions: int,
    ) -> object: ...


class AzureEmbeddingRecord(Protocol):
    index: int
    embedding: list[float]


class AzureEmbeddingResponse(Protocol):
    data: list[AzureEmbeddingRecord]


@dataclass(frozen=True)
class _EmbeddingConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str
    model_name: str
    dimension: int


class EmbeddingClient:
    def __init__(
        self,
        settings: Settings,
        *,
        embeddings_api: AzureEmbeddingsAPI | None = None,
    ) -> None:
        if not settings.azure_openai_endpoint:
            raise EmbeddingConfigurationError("AZURE_OPENAI_ENDPOINT is required")

        if not settings.azure_openai_api_key:
            raise EmbeddingConfigurationError("AZURE_OPENAI_API_KEY is required")

        self._config = _EmbeddingConfig(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment=settings.azure_openai_embeddings_deployment,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dim,
        )
        self._client: AsyncAzureOpenAI | None = None
        if embeddings_api is None:
            self._client = AsyncAzureOpenAI(
                api_key=self._config.api_key,
                api_version=self._config.api_version,
                azure_endpoint=self._config.endpoint,
            )
            self._embeddings_api = self._client.embeddings
        else:
            self._embeddings_api = embeddings_api

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def dimension(self) -> int:
        return self._config.dimension

    async def aclose(self) -> None:
        if self._client is None:
            return

        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = cast(
            AzureEmbeddingResponse,
            await self._embeddings_api.create(
                input=texts,
                model=self._config.deployment,
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
