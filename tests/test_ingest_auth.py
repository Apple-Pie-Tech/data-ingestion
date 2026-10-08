"""Auth contract for /ingest (x-api-key).

The UI sends the header as `x-api-key`
(applepie-ui/src/features/recording/ingest-client.ts), so the name is part of
the contract and must not drift.
"""

from __future__ import annotations

import json
from importlib import import_module

from fastapi.testclient import TestClient

_main = import_module("app.main")
_schemas = import_module("app.schemas")

app = _main.app
get_ingestion_service = _main.get_ingestion_service
get_settings = _main.get_settings
get_story_labeling_trigger = _main.get_story_labeling_trigger
Settings = _main.Settings
IngestMetadata = _schemas.IngestMetadata
IngestResult = _schemas.IngestResult

API_KEY = "test-ingest-key"

client = TestClient(app)


class _FakeIngestionService:
    async def ingest(
        self,
        *,
        metadata: IngestMetadata,
        text: str | None,
        audio_file: object | None,
    ) -> IngestResult:
        return IngestResult(
            input_id=metadata.input_id,
            status="indexed",
            chunks=1,
            audio_url=None,
        )


class _FakeStoryLabelingTrigger:
    async def trigger_cluster_labels(self, *, metadata: IngestMetadata, source: str) -> None:
        return None


def _text_fields() -> dict[str, tuple[None, str, None]]:
    return {
        "metadata": (
            None,
            json.dumps(
                {
                    "input_id": "sample-input-001",
                    "user_id": "user-123",
                    "timestamp": "2026-05-16T12:00:00Z",
                }
            ),
            None,
        ),
        "text": (None, "hello apple pie", None),
    }


def _override(settings: Settings) -> None:
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_service] = _FakeIngestionService
    app.dependency_overrides[get_story_labeling_trigger] = _FakeStoryLabelingTrigger


def test_ingest_without_api_key_header_is_rejected() -> None:
    _override(Settings(ingest_api_key=API_KEY))

    try:
        response = client.post("/ingest", files=_text_fields())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401


def test_ingest_with_wrong_api_key_is_rejected_without_leaking_the_expected_key() -> None:
    _override(Settings(ingest_api_key=API_KEY))

    try:
        response = client.post(
            "/ingest",
            files=_text_fields(),
            headers={"x-api-key": "not-the-key"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
    assert API_KEY not in response.text


def test_ingest_with_correct_api_key_is_accepted() -> None:
    _override(Settings(ingest_api_key=API_KEY))

    try:
        response = client.post(
            "/ingest",
            files=_text_fields(),
            headers={"x-api-key": API_KEY},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"


def test_ingest_is_open_when_no_api_key_is_configured() -> None:
    """Documented choice: an unset INGEST_API_KEY skips the check.

    Failing closed would make a fresh clone with no .env unable to ingest at
    all, which is exactly the local-dev path this service has to keep working.
    The service logs a warning at startup instead.
    """
    _override(Settings(ingest_api_key=None))

    try:
        response = client.post("/ingest", files=_text_fields())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


def test_blank_api_key_setting_also_skips_the_check() -> None:
    _override(Settings(ingest_api_key="   "))

    try:
        response = client.post("/ingest", files=_text_fields())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


def test_health_stays_unauthenticated() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(ingest_api_key=API_KEY)

    try:
        response = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_is_checked_before_the_request_body_is_parsed() -> None:
    """An unauthenticated caller must not learn anything about body validation."""
    _override(Settings(ingest_api_key=API_KEY))

    try:
        response = client.post("/ingest", files={"metadata": (None, "not-json", None)})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
