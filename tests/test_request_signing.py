from __future__ import annotations

import httpx
import pytest

from app.request_signing import SIGNING_SERVICE, SigV4Auth


class FrozenCredentials:
    """Matches what botocore's get_frozen_credentials() returns."""

    access_key = "AKIAIOSFODNN7EXAMPLE"
    secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    token = None


def sign(request: httpx.Request) -> httpx.Request:
    auth = SigV4Auth(region="us-east-1", credentials=FrozenCredentials())
    flow = auth.auth_flow(request)
    signed = next(flow)
    flow.close()
    return signed


def test_a_request_is_signed_for_the_lambda_service() -> None:
    """Function URLs are signed against `lambda`, not `execute-api`.

    Signing for the wrong service returns a 403 whose message does not say why.
    """
    assert SIGNING_SERVICE == "lambda"

    signed = sign(
        httpx.Request("POST", "https://abc123.lambda-url.us-east-1.on.aws/cluster-labels")
    )

    authorization = signed.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 ")
    assert "/us-east-1/lambda/aws4_request" in authorization
    assert "x-amz-date" in signed.headers


def test_the_trigger_headers_are_covered_by_the_signature() -> None:
    """The ingest trace headers are part of the request, so they must be signed.

    A header outside SignedHeaders can be altered in flight without
    invalidating the signature.
    """
    signed = sign(
        httpx.Request(
            "POST",
            "https://abc123.lambda-url.us-east-1.on.aws/cluster-labels",
            headers={"x-ingest-input-id": "input-1", "x-ingest-source": "audio"},
        )
    )

    authorization = signed.headers["authorization"]
    assert "x-ingest-input-id" in authorization
    assert "x-ingest-source" in authorization


def test_connection_headers_are_not_signed() -> None:
    """httpx sets these per connection, and the server recomputes them.

    Signing them makes the signature depend on values that legitimately change
    in transit, which fails intermittently rather than outright.
    """
    signed = sign(
        httpx.Request(
            "POST",
            "https://abc123.lambda-url.us-east-1.on.aws/cluster-labels",
            headers={"connection": "keep-alive"},
        )
    )

    authorization = signed.headers["authorization"]
    assert "connection" not in authorization
    assert "content-length" not in authorization


def test_the_body_is_covered_by_the_signature() -> None:
    first = sign(
        httpx.Request("POST", "https://abc.lambda-url.us-east-1.on.aws/x", content=b"a")
    )
    second = sign(
        httpx.Request("POST", "https://abc.lambda-url.us-east-1.on.aws/x", content=b"b")
    )

    assert first.headers["authorization"] != second.headers["authorization"]


def test_missing_credentials_raise_rather_than_sending_unsigned(monkeypatch) -> None:
    """An unsigned request to an IAM-authorized URL is a 403 with no explanation."""
    import app.request_signing as request_signing

    class FakeSession:
        def get_credentials(self) -> None:
            return None

    monkeypatch.setattr(
        request_signing,
        "import_module",
        lambda name: type("M", (), {"Session": FakeSession})() if name == "boto3" else __import__(name, fromlist=["x"]),
    )
    auth = SigV4Auth(region="us-east-1")

    with pytest.raises(RuntimeError, match="no AWS credentials"):
        next(auth.auth_flow(httpx.Request("POST", "https://abc.lambda-url.us-east-1.on.aws/x")))
