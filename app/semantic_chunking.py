from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module

_config = import_module("app.config")
_chonkie_semantic = import_module("chonkie.chunker.semantic")
_chonkie_azure_openai = import_module("chonkie.embeddings.azure_openai")
_chonkie_base = import_module("chonkie.embeddings.base")

Settings = _config.Settings
ChonkieSemanticChunker = _chonkie_semantic.SemanticChunker
AzureOpenAIEmbeddings = _chonkie_azure_openai.AzureOpenAIEmbeddings
BaseEmbeddings = _chonkie_base.BaseEmbeddings


class SemanticChunkingError(RuntimeError):
    pass


class SemanticChunkingConfigurationError(SemanticChunkingError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class _SemanticChunkingConfig:
    similarity_threshold: float
    similarity_window: int
    min_chunk_chars: int
    max_chunk_chars: int

    @classmethod
    def from_settings(cls, settings: Settings) -> _SemanticChunkingConfig:
        return cls(
            similarity_threshold=settings.semantic_similarity_threshold,
            similarity_window=settings.chunk_overlap_sentences,
            min_chunk_chars=settings.min_chunk_chars,
            max_chunk_chars=settings.max_chunk_chars,
        )


class SemanticChunkerAdapter:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: BaseEmbeddings | None = None,
        threshold: float | None = None,
        similarity_window: int | None = None,
        min_chunk_chars: int | None = None,
        max_chunk_chars: int | None = None,
    ) -> None:
        config = _SemanticChunkingConfig.from_settings(settings)
        self._config = _SemanticChunkingConfig(
            similarity_threshold=config.similarity_threshold
            if threshold is None
            else threshold,
            similarity_window=config.similarity_window
            if similarity_window is None
            else similarity_window,
            min_chunk_chars=config.min_chunk_chars
            if min_chunk_chars is None
            else min_chunk_chars,
            max_chunk_chars=config.max_chunk_chars if max_chunk_chars is None else max_chunk_chars,
        )

        self._embedding_model = embedding_model or _build_azure_embeddings(settings)

        try:
            self._chunker = ChonkieSemanticChunker(
                embedding_model=self._embedding_model,
                threshold=self._config.similarity_threshold,
                similarity_window=self._config.similarity_window,
                chunk_size=self._config.max_chunk_chars,
            )
        except Exception as exc:
            raise SemanticChunkingConfigurationError(
                "failed to initialize Chonkie semantic chunker"
            ) from exc

    async def aclose(self) -> None:
        if self._embedding_model is None:
            return

        close = getattr(self._embedding_model, "aclose", None) or getattr(
            self._embedding_model,
            "close",
            None,
        )
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    def chunk(self, text: str) -> list[Chunk]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []

        chunks = self._chunker.chunk(normalized_text)
        return [
            Chunk(
                text=chunk.text,
                chunk_index=index,
                metadata=self._build_metadata(len(chunk.text)),
            )
            for index, chunk in enumerate(chunks)
        ]

    def _build_metadata(self, chunk_length: int) -> dict[str, object]:
        return {
            "provider": "chonkie",
            "chunker": "semantic",
            "similarity_threshold": self._config.similarity_threshold,
            "overlap_sentences": self._config.similarity_window,
            "min_chunk_chars": self._config.min_chunk_chars,
            "max_chunk_chars": self._config.max_chunk_chars,
            "chunk_length": chunk_length,
        }


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def _build_azure_embeddings(settings: Settings) -> BaseEmbeddings:
    if not settings.azure_openai_endpoint:
        raise SemanticChunkingConfigurationError("AZURE_OPENAI_ENDPOINT is required")

    if not settings.azure_openai_api_key:
        raise SemanticChunkingConfigurationError("AZURE_OPENAI_API_KEY is required")

    try:
        return AzureOpenAIEmbeddings(
            model=settings.embedding_model,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment=settings.azure_openai_embeddings_deployment,
            dimension=settings.embedding_dim,
        )
    except Exception as exc:
        raise SemanticChunkingConfigurationError(
            "failed to configure Azure OpenAI embeddings for Chonkie"
        ) from exc
