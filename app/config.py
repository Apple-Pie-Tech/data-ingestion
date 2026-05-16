from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ingest_api_key: str | None = None
    max_audio_bytes: int = 25_000_000
    request_timeout_seconds: int = 60

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "apple_pie_story_chunks"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embeddings_deployment: str = "text-embedding-3-large"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072

    semantic_similarity_threshold: float = 0.8
    min_chunk_chars: int = 350
    max_chunk_chars: int = 1400
    chunk_overlap_sentences: int = 1

    gradium_api_base_url: str | None = None
    gradium_api_key: str | None = None
    gradium_transcription_model: str | None = None
    gradium_transcription_transport: str = "rest"
    gradium_transcription_path: str = "/post/speech/asr"
    gradium_timeout_seconds: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
