from __future__ import annotations

from importlib import import_module

import numpy as np

_config = import_module("app.config")
_semantic_chunking = import_module("app.semantic_chunking")
_chonkie_base = import_module("chonkie.embeddings.base")

Settings = _config.Settings
SemanticChunkerAdapter = _semantic_chunking.SemanticChunkerAdapter
normalize_text = _semantic_chunking.normalize_text
BaseEmbeddings = _chonkie_base.BaseEmbeddings


class FakeEmbeddings(BaseEmbeddings):
    @property
    def dimension(self) -> int:
        return 2

    def get_tokenizer(self) -> str:
        return "character"

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embedding_for_text(text) for text in texts]

    def _embedding_for_text(self, text: str) -> np.ndarray:
        normalized = text.lower()
        if "apple" in normalized or "cinnamon" in normalized or "pastry" in normalized:
            return np.array([1.0, 0.0], dtype=np.float32)
        if "vector" in normalized or "database" in normalized or "embedding" in normalized:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([0.5, 0.5], dtype=np.float32)


def make_settings() -> Settings:
    return Settings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret-key",
        azure_openai_api_version="2024-10-21",
        azure_openai_embeddings_deployment="embeddings-deployment",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072,
    )


def test_topic_boundary_chunking_returns_multiple_chunks_and_preserves_normalized_text() -> None:
    adapter = SemanticChunkerAdapter(
        make_settings(),
        embedding_model=FakeEmbeddings(),
    )
    transcript = """
    We peeled the apples and mixed them with cinnamon until the filling smelled sweet.

    Then we folded the pastry over the fruit and brushed the crust with butter.
    Finally we baked the apples until the pie bubbled at the edges.

    After dessert, we compared every vector database schema for retrieval quality.
    We measured how each embedding index changed latency during search.
    The team tuned the vector database filters until the queries became predictable.
    """

    chunks = adapter.chunk(transcript)

    assert len(chunks) >= 2
    assert "baked the apples" in chunks[0].text.lower()
    assert any("vector database" in chunk.text.lower() for chunk in chunks[1:])
    assert "".join(chunk.text for chunk in chunks) == normalize_text(transcript)


def test_chunk_metadata_includes_chunk_indices_and_chunking_configuration() -> None:
    adapter = SemanticChunkerAdapter(
        make_settings(),
        embedding_model=FakeEmbeddings(),
    )
    transcript = (
        "Apples and cinnamon baked slowly in the oven until the crust browned. "
        "The vector database benchmark compared recall, latency, and embedding drift across runs."
    )

    chunks = adapter.chunk(transcript)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["provider"] == "chonkie" for chunk in chunks)
    assert all(chunk.metadata["chunker"] == "semantic" for chunk in chunks)
    assert all("similarity_threshold" in chunk.metadata for chunk in chunks)
    assert all("overlap_sentences" in chunk.metadata for chunk in chunks)


def test_chunk_metadata_uses_settings_values_for_chunking_tunables() -> None:
    settings = Settings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret-key",
        azure_openai_api_version="2024-10-21",
        azure_openai_embeddings_deployment="embeddings-deployment",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072,
        semantic_similarity_threshold=0.66,
        min_chunk_chars=123,
        max_chunk_chars=456,
        chunk_overlap_sentences=2,
    )
    adapter = SemanticChunkerAdapter(
        settings,
        embedding_model=FakeEmbeddings(),
    )

    chunks = adapter.chunk("Apples were sweet. Vector databases were fast.")

    assert chunks
    assert all(chunk.metadata["similarity_threshold"] == 0.66 for chunk in chunks)
    assert all(chunk.metadata["overlap_sentences"] == 2 for chunk in chunks)
    assert all(chunk.metadata["min_chunk_chars"] == 123 for chunk in chunks)
    assert all(chunk.metadata["max_chunk_chars"] == 456 for chunk in chunks)
