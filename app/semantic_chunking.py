from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

_config = import_module("app.config")
_embeddings = import_module("app.embeddings")
_chonkie_semantic = import_module("chonkie.chunker.semantic")
_chonkie_base = import_module("chonkie.embeddings.base")

Settings = _config.Settings
ChonkieSemanticChunker = _chonkie_semantic.SemanticChunker
BaseEmbeddings = _chonkie_base.BaseEmbeddings

COHERE_MAX_TEXTS_PER_REQUEST = _embeddings.COHERE_MAX_TEXTS_PER_REQUEST
invoke_cohere_embeddings = _embeddings.invoke_cohere_embeddings


class SemanticChunkingError(RuntimeError):
    pass


class SemanticChunkingConfigurationError(SemanticChunkingError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    metadata: dict[str, object]


class BedrockCohereEmbeddings(BaseEmbeddings):
    """Chonkie embedding backend over Cohere Embed on Bedrock.

    Chonkie ships no Bedrock backend. Its own local default (`minishlab/potion-base-32M`)
    would need the `model2vec` package plus baked-in model weights, neither of which is
    installed, so it falls back to sentence-transformers and fails. Wrapping the same
    Bedrock client the stored-vector path uses keeps the image small and matches what
    this service did before the migration, where Chonkie also called a hosted
    embedding API.

    These vectors are only used to decide where to split text. They are never stored.
    """

    def __init__(
        self,
        *,
        region: str,
        model_id: str,
        dimension: int,
        tokenizer_encoding: str,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        if client is None:
            boto3 = import_module("boto3")
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self._model_id = model_id
        self._dimension = dimension
        self._tokenizer_encoding = tokenizer_encoding
        self._tokenizer: Any | None = None

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for offset in range(0, len(texts), COHERE_MAX_TEXTS_PER_REQUEST):
            batch = texts[offset : offset + COHERE_MAX_TEXTS_PER_REQUEST]
            vectors.extend(
                invoke_cohere_embeddings(
                    self._client,
                    batch,
                    model=self._model_id,
                    dimensions=self._dimension,
                )
            )

        if len(vectors) != len(texts):
            raise SemanticChunkingError(
                f"Bedrock returned {len(vectors)} embeddings for {len(texts)} texts"
            )

        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_tokenizer(self) -> Any:
        # Chonkie's SemanticChunker asks for this once, at construction, to count
        # tokens against chunk_size. Built lazily so importing this module never
        # reaches for the tiktoken BPE cache.
        if self._tokenizer is None:
            tiktoken = import_module("tiktoken")
            self._tokenizer = tiktoken.get_encoding(self._tokenizer_encoding)
        return self._tokenizer

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def __repr__(self) -> str:
        return f"BedrockCohereEmbeddings(model={self._model_id!r}, dimension={self._dimension})"


@dataclass(frozen=True)
class _SemanticChunkingConfig:
    similarity_threshold: float
    similarity_window: int
    min_chunk_chars: int
    max_chunk_chars: int
    max_chunk_tokens: int

    @classmethod
    def from_settings(cls, settings: Settings) -> _SemanticChunkingConfig:
        return cls(
            similarity_threshold=settings.semantic_similarity_threshold,
            similarity_window=settings.chunk_overlap_sentences,
            min_chunk_chars=settings.min_chunk_chars,
            max_chunk_chars=settings.max_chunk_chars,
            max_chunk_tokens=settings.max_chunk_tokens,
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
        max_chunk_tokens: int | None = None,
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
            max_chunk_tokens=config.max_chunk_tokens
            if max_chunk_tokens is None
            else max_chunk_tokens,
        )

        self._embedding_model = embedding_model or _build_bedrock_embeddings(settings)

        try:
            self._chunker = ChonkieSemanticChunker(
                embedding_model=self._embedding_model,
                threshold=self._config.similarity_threshold,
                similarity_window=self._config.similarity_window,
                # Chonkie documents chunk_size as a TOKEN budget. This previously
                # received max_chunk_chars (1400 characters), which silently asked for
                # chunks roughly 4x larger than intended.
                chunk_size=self._config.max_chunk_tokens,
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
            "max_chunk_tokens": self._config.max_chunk_tokens,
            "chunk_length": chunk_length,
        }


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def _build_bedrock_embeddings(settings: Settings) -> BaseEmbeddings:
    if not settings.aws_region:
        raise SemanticChunkingConfigurationError("AWS_REGION is required")

    if not settings.bedrock_embeddings_model:
        raise SemanticChunkingConfigurationError("BEDROCK_EMBEDDINGS_MODEL is required")

    try:
        return BedrockCohereEmbeddings(
            region=settings.aws_region,
            model_id=settings.bedrock_embeddings_model,
            dimension=settings.embedding_dim,
            tokenizer_encoding=settings.semantic_tokenizer_encoding,
        )
    except Exception as exc:
        raise SemanticChunkingConfigurationError(
            "failed to configure Bedrock embeddings for Chonkie"
        ) from exc
