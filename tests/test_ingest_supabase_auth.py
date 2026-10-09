"""Auth contract for /ingest's Supabase bearer token.

Separate from test_ingest_auth.py on purpose: that file pins the `x-api-key`
deterrent, this one pins real authentication. They coexist during the
transition, and the two 401s are deliberately indistinguishable from outside.

Tokens are minted here with an ephemeral keypair and the JWKS fetch is
injected, so nothing in this file reaches the network or needs a live Supabase
project. The unit-level checks on the verifier itself live in
test_supabase_jwt.py; these tests are about the route.
"""

from __future__ import annotations

import json
import time
from importlib import import_module

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

_main = import_module("app.main")
_schemas = import_module("app.schemas")
_supabase_jwt = import_module("app.supabase_jwt")

app = _main.app
get_ingestion_service = _main.get_ingestion_service
get_settings = _main.get_settings
get_story_labeling_trigger = _main.get_story_labeling_trigger
get_supabase_verifier = _main.get_supabase_verifier
Settings = _main.Settings
IngestMetadata = _schemas.IngestMetadata
IngestResult = _schemas.IngestResult
SupabaseJwtVerifier = _supabase_jwt.SupabaseJwtVerifier

SUPABASE_URL = "https://project.supabase.co"
ISSUER = f"{SUPABASE_URL}/auth/v1"
# Matches the user_id in the multipart metadata below; the two must agree or the
# identity-binding check rejects the request, which is the point of that check.
USER_ID = "user-123"
KID = "ingest-test-kid"

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


class _FakeResponse:
    def __init__(self, document) -> None:
        self._document = document

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._document


class _FakeJwksClient:
    def __init__(self, document, *, error: Exception | None = None) -> None:
        self._document = document
        self._error = error

    async def get(self, url: str):
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._document)


_PRIVATE_KEY = ec.generate_private_key(ec.SECP256R1())
_JWK = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(_PRIVATE_KEY.public_key()))
_JWK |= {"kid": KID, "alg": "ES256", "use": "sig"}
_JWKS = {"keys": [_JWK]}


def _token(*, sub: str = USER_ID, key=None, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": "authenticated",
        "sub": sub,
        "role": "authenticated",
        "iat": now - 10,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return jwt.encode(claims, key or _PRIVATE_KEY, algorithm="ES256", headers={"kid": KID})


def _text_fields(user_id: str = USER_ID) -> dict[str, tuple[None, str, None]]:
    return {
        "metadata": (
            None,
            json.dumps(
                {
                    "input_id": "sample-input-001",
                    "user_id": user_id,
                    "timestamp": "2026-05-16T12:00:00Z",
                }
            ),
            None,
        ),
        "text": (None, "hello apple pie", None),
    }


def _configured(*, error: Exception | None = None) -> None:
    """SUPABASE_URL set, with the JWKS served from memory."""
    app.dependency_overrides[get_settings] = lambda: Settings(supabase_url=SUPABASE_URL)
    app.dependency_overrides[get_ingestion_service] = _FakeIngestionService
    app.dependency_overrides[get_story_labeling_trigger] = _FakeStoryLabelingTrigger
    app.dependency_overrides[get_supabase_verifier] = lambda: SupabaseJwtVerifier(
        supabase_url=SUPABASE_URL,
        audience="authenticated",
        http_client=_FakeJwksClient(_JWKS, error=error),
    )


def _unconfigured() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(supabase_url=None)
    app.dependency_overrides[get_ingestion_service] = _FakeIngestionService
    app.dependency_overrides[get_story_labeling_trigger] = _FakeStoryLabelingTrigger


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_ingest_stays_open_when_supabase_url_is_unset() -> None:
    """The documented local-dev path: unset means skip, as with the API key.

    This is what lets the service deploy with verification off and be switched
    on by a later Terraform apply, rather than needing a flag day across three
    independently deployed repositories.
    """
    _unconfigured()

    response = client.post("/ingest", files=_text_fields())

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"


def test_a_valid_bearer_token_is_accepted() -> None:
    _configured()

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": f"Bearer {_token()}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"


def test_the_bearer_scheme_is_matched_case_insensitively() -> None:
    """RFC 7235 makes the scheme case-insensitive, and clients vary."""
    _configured()

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": f"bearer {_token()}"},
    )

    assert response.status_code == 200


def test_a_missing_authorization_header_is_rejected() -> None:
    _configured()

    response = client.post("/ingest", files=_text_fields())

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_a_non_bearer_scheme_is_rejected() -> None:
    _configured()

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )

    assert response.status_code == 401


def test_a_garbage_token_is_rejected() -> None:
    _configured()

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": "Bearer not-a-jwt"},
    )

    assert response.status_code == 401


def test_a_token_signed_by_another_key_is_rejected() -> None:
    """The signature is actually checked, not just decoded."""
    _configured()
    impostor = ec.generate_private_key(ec.SECP256R1())

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": f"Bearer {_token(key=impostor)}"},
    )

    assert response.status_code == 401


def test_an_expired_token_is_rejected() -> None:
    _configured()
    now = int(time.time())

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": f"Bearer {_token(exp=now - 1, iat=now - 3600)}"},
    )

    assert response.status_code == 401


def test_ingesting_under_another_users_id_is_refused() -> None:
    """The real gain of this change, and the reason the token beats the body.

    `metadata.user_id` is client-supplied, so before verification any caller
    could file memories under someone else's id -- the only gate was a key
    published in the UI's JavaScript bundle.
    """
    _configured()

    response = client.post(
        "/ingest",
        files=_text_fields(user_id="somebody-else"),
        headers={"Authorization": f"Bearer {_token()}"},
    )

    assert response.status_code == 403


def test_an_unreachable_jwks_is_a_503_not_a_401() -> None:
    """A Supabase outage must not read as "your session is invalid".

    401 would bounce every signed-in user to the sign-in screen for a fault
    that is not theirs; 200 would make a network blip an auth bypass.
    """
    _configured(error=RuntimeError("connection refused"))

    response = client.post(
        "/ingest",
        files=_text_fields(),
        headers={"Authorization": f"Bearer {_token()}"},
    )

    assert response.status_code == 503


def test_auth_is_checked_before_the_request_body_is_parsed() -> None:
    """An unauthenticated caller's upload is never read, parsed or transcoded."""
    _configured()

    response = client.post("/ingest", files={"metadata": (None, "not-json", None)})

    assert response.status_code == 401


def test_health_stays_unauthenticated_with_supabase_configured() -> None:
    """The Lambda Web Adapter's readiness probe targets /health."""
    _configured()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
