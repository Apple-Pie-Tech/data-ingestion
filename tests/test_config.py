from __future__ import annotations

from pathlib import Path

from app.config import Settings

_ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _env_example_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def test_embedding_defaults_match_the_live_bedrock_model() -> None:
    """Cohere Embed v4 on Bedrock, verified to return 1536 dims against the account.

    A run without a .env must not build or validate a vector index at some other
    width; that dies in S3VectorStore._ensure_index. Changing
    bedrock_embeddings_model almost always means changing embedding_dim too.
    """
    settings = Settings(_env_file=None)

    assert settings.aws_region == "us-east-1"
    assert settings.bedrock_embeddings_model == "cohere.embed-v4:0"
    assert settings.embedding_model == "cohere.embed-v4:0"
    assert settings.embedding_dim == 1536


def test_env_example_documents_the_same_embedding_defaults() -> None:
    values = _env_example_values()

    assert values["BEDROCK_EMBEDDINGS_MODEL"] == "cohere.embed-v4:0"
    assert values["EMBEDDING_MODEL"] == "cohere.embed-v4:0"
    assert values["EMBEDDING_DIM"] == "1536"


_COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def _compose_app_environment() -> dict[str, str]:
    """Read the `app` service's `environment:` mapping without a YAML dependency."""
    values: dict[str, str] = {}
    in_environment = False
    for raw_line in _COMPOSE.read_text().splitlines():
        stripped = raw_line.strip()
        if stripped == "environment:":
            in_environment = True
            continue
        if not in_environment:
            continue
        if stripped and not raw_line.startswith(" " * 6):
            break
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        values[key.strip()] = value.strip()
    return values


def test_compose_passes_every_setting_the_service_reads() -> None:
    """Anything missing here cannot be configured on the compose path at all."""
    compose_env = _compose_app_environment()
    missing = sorted(
        name.upper() for name in Settings.model_fields if name.upper() not in compose_env
    )

    assert missing == []


def test_env_example_documents_every_setting_the_service_reads() -> None:
    values = _env_example_values()
    missing = sorted(name.upper() for name in Settings.model_fields if name.upper() not in values)

    assert missing == []


def test_compose_embedding_and_gradium_defaults_match_the_live_stack() -> None:
    compose_env = _compose_app_environment()

    assert compose_env["BEDROCK_EMBEDDINGS_MODEL"].endswith(
        "${BEDROCK_EMBEDDINGS_MODEL:-cohere.embed-v4:0}"
    )
    assert compose_env["EMBEDDING_MODEL"] == "${EMBEDDING_MODEL:-cohere.embed-v4:0}"
    assert compose_env["EMBEDDING_DIM"] == "${EMBEDDING_DIM:-1536}"
    assert (
        compose_env["GRADIUM_TRANSCRIPTION_PATH"]
        == "${GRADIUM_TRANSCRIPTION_PATH:-/post/speech/asr}"
    )
