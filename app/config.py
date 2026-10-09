from functools import lru_cache

from app.secrets import resolve_secret_fields

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
    # The labelling function URL is IAM-authorized in deployment, so the trigger
    # has to sign its request with SigV4. Off by default so a local run against
    # a plain uvicorn still works.
    story_labeling_sign_requests: bool = False

    ingest_api_key: str | None = None
    # Set by Terraform instead of the value above: a secret value in a Lambda
    # environment variable would be recorded in Terraform state in plaintext.
    # Resolved once by get_settings(); the plain field still works for local
    # development, and wins when both are absent from Secrets Manager.
    ingest_api_key_secret_arn: str | None = None

    # Verifying the Supabase access token the UI already holds is what makes
    # the write endpoint authenticated rather than merely deterred. Unset means
    # the check is SKIPPED, matching INGEST_API_KEY above and for the same reason:
    # a fresh clone with no .env has to keep working. Terraform always sets it,
    # and create_app warns when it is missing so an open deployment is not
    # silent.
    supabase_url: str | None = None
    supabase_jwt_audience: str = "authenticated"
    # Only for a project still signing with the legacy HS256 shared secret. In
    # the asymmetric regime (ES256/RS256) the keys are published at the
    # project's JWKS endpoint and there is no secret to hold at all.
    supabase_jwt_secret: str | None = None
    supabase_jwt_secret_arn: str | None = None
    supabase_jwks_cache_seconds: int = 600
    max_audio_bytes: int = 25_000_000
    request_timeout_seconds: int = 60
    audio_storage_enabled: bool = False
    # Bucket created by Terraform; the service only ever writes into it. Audio
    # objects are private, so the URL recorded in the vector payload is the
    # canonical unsigned one and data-provision-api presigns it at read time.
    s3_audio_bucket: str | None = None

    # S3 Vectors bucket and index, both created by Terraform. The index pins the
    # dimension and the non-filterable metadata keys at creation time, so this
    # service validates rather than creates it.
    s3_vector_bucket: str | None = None
    s3_vector_index: str = "apple-pie-story-chunks"

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
    gradium_api_key_secret_arn: str | None = None
    gradium_transcription_model: str | None = None
    gradium_transcription_transport: str = "rest"
    gradium_transcription_path: str = "/post/speech/asr"
    gradium_language: str = "en"
    gradium_timeout_seconds: int = 60


# Target field -> the setting holding the ARN to resolve it from.
SECRET_FIELD_ARNS = {
    "gradium_api_key": "gradium_api_key_secret_arn",
    "ingest_api_key": "ingest_api_key_secret_arn",
    "supabase_jwt_secret": "supabase_jwt_secret_arn",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build Settings, resolving any Secrets Manager ARNs it was given.

    Cached, so the Secrets Manager calls happen once per execution environment
    rather than once per request.
    """
    settings = Settings()
    resolved = resolve_secret_fields(
        settings,
        SECRET_FIELD_ARNS,
        region=settings.aws_region,
    )
    if not resolved:
        return settings
    return settings.model_copy(update=resolved)
