from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cors_allow_origins: str = (
        "http://127.0.0.1:8081,http://localhost:8081,"
        "http://127.0.0.1:8080,http://localhost:8080"
    )
    story_labeling_enabled: bool = False
    story_labeling_api_base_url: str | None = None
    story_labeling_timeout_seconds: int = 60

    ingest_api_key: str | None = None
    max_audio_bytes: int = 25_000_000
    request_timeout_seconds: int = 60
    audio_storage_enabled: bool = False
    # Bucket created by Terraform; the service only ever writes into it. Audio
    # objects are private, so the URL recorded in the vector payload is the
    # canonical unsigned one and data-provision-api presigns it at read time.
    s3_audio_bucket: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "apple_pie_story_chunks"

    aws_region: str = "us-east-1"
    # Cohere Embed v4 defaults to 1536 dimensions, which is what the existing vector
    # index was built for. Verified against the live account before the migration:
    # changing this model almost certainly changes embedding_dim too.
    bedrock_embeddings_model: str = "cohere.embed-v4:0"
    embedding_model: str = "cohere.embed-v4:0"
    embedding_dim: int = 1536

    semantic_similarity_threshold: float = 0.8
    min_chunk_chars: int = 350
    max_chunk_chars: int = 1400
    # Chonkie's chunk_size is a TOKEN budget, not a character budget. It previously
    # received max_chunk_chars (1400), asking for chunks ~4x larger than intended;
    # 350 tokens is the token-equivalent of 1400 characters.
    max_chunk_tokens: int = 350
    # Tokenizer Chonkie counts chunk_size with. tiktoken fetches the BPE table on
    # first use, so the Docker build warms TIKTOKEN_CACHE_DIR to keep it off the
    # Lambda cold-start path.
    semantic_tokenizer_encoding: str = "cl100k_base"
    chunk_overlap_sentences: int = 1

    gradium_api_base_url: str | None = None
    gradium_api_key: str | None = None
    gradium_transcription_model: str | None = None
    gradium_transcription_transport: str = "rest"
    gradium_transcription_path: str = "/post/speech/asr"
    gradium_language: str = "en"
    gradium_timeout_seconds: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
