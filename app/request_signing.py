"""SigV4 signing for calls to an IAM-authorized Lambda function URL.

story-labeling-api's function URL uses `AWS_IAM` authorization, because
/cluster-labels reclusters the whole corpus and makes one Bedrock call per
cluster -- an unauthenticated URL would let anyone on the internet spend the
account's money and rewrite every label. That means the post-ingest trigger has
to sign its request, which it does with the ambient execution-role credentials:
nothing is stored and nothing is rotated.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from importlib import import_module
from typing import Any

import httpx


# Lambda function URLs are signed against the `lambda` service, not `execute-api`
# (which is API Gateway) -- signing for the wrong service yields a 403 whose
# message does not say why.
SIGNING_SERVICE = "lambda"

# Hop-by-hop and length headers must not be signed: httpx sets them per
# connection, so including them makes the signature depend on values the server
# recomputes.
_UNSIGNED_HEADERS = frozenset({"connection", "content-length", "accept-encoding"})


class SigV4Auth(httpx.Auth):
    """Signs each request with SigV4 before it leaves."""

    # The signature covers the body, so httpx must hand over a fully read
    # request rather than a stream.
    requires_request_body = True

    def __init__(
        self,
        *,
        region: str,
        service: str = SIGNING_SERVICE,
        credentials: Any | None = None,
    ) -> None:
        self._region = region
        self._service = service
        self._credentials = credentials

    def auth_flow(
        self,
        request: httpx.Request,
    ) -> Generator[httpx.Request, httpx.Response, None]:
        signed_headers = self._sign(
            method=request.method,
            url=str(request.url),
            headers=request.headers,
            body=request.content,
        )
        request.headers.update(signed_headers)
        yield request

    def _sign(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> dict[str, str]:
        botocore_auth = import_module("botocore.auth")
        awsrequest = import_module("botocore.awsrequest")

        credentials = self._credentials
        if credentials is None:
            boto3 = import_module("boto3")
            resolved = boto3.Session().get_credentials()
            if resolved is None:
                raise RuntimeError(
                    "no AWS credentials available to sign the story-labeling request"
                )
            credentials = resolved.get_frozen_credentials()

        signable_headers = {
            name: value
            for name, value in headers.items()
            if name.lower() not in _UNSIGNED_HEADERS
        }

        aws_request = awsrequest.AWSRequest(
            method=method,
            url=url,
            data=body,
            headers=signable_headers,
        )
        botocore_auth.SigV4Auth(credentials, self._service, self._region).add_auth(
            aws_request
        )
        return dict(aws_request.headers)


__all__ = ["SIGNING_SERVICE", "SigV4Auth"]
