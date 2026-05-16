from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_main = import_module("app.main")

app = _main.app
get_settings = _main.get_settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_SMOKE") != "1",
    reason="set RUN_EXTERNAL_SMOKE=1 to run external smoke tests",
)

SAMPLE_METADATA = {
    "input_id": "sample-input-001",
    "user_id": "user-001",
    "timestamp": "2026-05-16T12:00:00Z",
}
SAMPLE_TEXT = (
    "Alice followed the rabbit hole into a bright hall. "
    "She found a tiny golden key on a glass table."
)
SAMPLE_WAV_PATH = Path(__file__).parent / "fixtures" / "sample.wav"
TEXT_SMOKE_ENV_VARS = (
    "QDRANT_URL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
)
AUDIO_SMOKE_ENV_VARS = TEXT_SMOKE_ENV_VARS + (
    "GRADIUM_API_BASE_URL",
    "GRADIUM_API_KEY",
    "GRADIUM_TRANSCRIPTION_MODEL",
)


def test_text_only_external_smoke() -> None:
    _skip_if_missing_env_vars(TEXT_SMOKE_ENV_VARS)

    response = _post_ingest(
        files={
            "metadata": (None, json.dumps(SAMPLE_METADATA), None),
            "text": (None, SAMPLE_TEXT, None),
        }
    )

    assert response.status_code == 200, response.text
    assert response.json()["input_id"] == SAMPLE_METADATA["input_id"]
    assert response.json()["status"] == "indexed"
    assert response.json()["chunks"] >= 1


def test_audio_external_smoke() -> None:
    _skip_if_missing_env_vars(AUDIO_SMOKE_ENV_VARS)
    assert SAMPLE_WAV_PATH.is_file(), f"missing fixture: {SAMPLE_WAV_PATH}"

    response = _post_ingest(
        files={
            "metadata": (None, json.dumps(SAMPLE_METADATA), None),
            "audio": ("sample.wav", SAMPLE_WAV_PATH.read_bytes(), "audio/wav"),
        }
    )

    assert response.status_code == 200, response.text
    assert response.json()["input_id"] == SAMPLE_METADATA["input_id"]
    assert response.json()["status"] == "indexed"
    assert response.json()["chunks"] >= 1


def _post_ingest(
    *,
    files: dict[str, tuple[str | None, str | bytes, str | None]],
):
    get_settings.cache_clear()
    app.dependency_overrides.clear()

    try:
        with TestClient(app) as client:
            return client.post("/ingest", files=files)
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def _skip_if_missing_env_vars(names: tuple[str, ...]) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip(
            "missing required env vars for external smoke: " + ", ".join(sorted(missing))
        )
