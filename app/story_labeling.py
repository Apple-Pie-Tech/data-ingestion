from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import httpx

_config = import_module("app.config")
_schemas = import_module("app.schemas")

Settings = _config.Settings
IngestMetadata = _schemas.IngestMetadata


class StoryLabelingTriggerError(RuntimeError):
    pass


class StoryLabelingTriggerConfigurationError(StoryLabelingTriggerError):
    pass


@dataclass(frozen=True)
class StoryLabelingTriggerResult:
    status: str
    points_read: int
    points_clustered: int
    clusters_found: int
    noise_points: int
    points_updated: int


class StoryLabelingTrigger(Protocol):
    async def trigger_cluster_labels(
        self,
        *,
        metadata: IngestMetadata,
        source: str,
    ) -> StoryLabelingTriggerResult: ...


class StoryLabelingTriggerClient:
    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.enabled = settings.story_labeling_enabled
        self._owns_client = client is None
        self._client = client

        if not self.enabled:
            self._base_url = None
            return

        if not settings.story_labeling_api_base_url:
            raise StoryLabelingTriggerConfigurationError(
                "STORY_LABELING_API_BASE_URL is required when story labeling is enabled"
            )

        self._base_url = settings.story_labeling_api_base_url.rstrip("/")
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(float(settings.story_labeling_timeout_seconds)),
            )

    async def trigger_cluster_labels(
        self,
        *,
        metadata: IngestMetadata,
        source: str,
    ) -> StoryLabelingTriggerResult:
        if not self.enabled:
            raise StoryLabelingTriggerConfigurationError(
                "story labeling trigger is disabled"
            )

        assert self._client is not None

        try:
            response = await self._client.post(
                "/cluster-labels",
                headers={
                    "x-ingest-input-id": metadata.input_id,
                    "x-ingest-source": source,
                },
            )
        except httpx.TimeoutException as exc:
            raise StoryLabelingTriggerError(
                "story labeling trigger timed out"
            ) from exc
        except httpx.HTTPError as exc:
            raise StoryLabelingTriggerError(
                "story labeling trigger request failed"
            ) from exc

        if response.is_error:
            raise StoryLabelingTriggerError(
                f"story labeling trigger failed with HTTP {response.status_code}: {_response_excerpt(response)}"
            )

        payload = response.json()
        try:
            return StoryLabelingTriggerResult(
                status=str(payload["status"]),
                points_read=int(payload["points_read"]),
                points_clustered=int(payload["points_clustered"]),
                clusters_found=int(payload["clusters_found"]),
                noise_points=int(payload["noise_points"]),
                points_updated=int(payload["points_updated"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StoryLabelingTriggerError(
                "story labeling trigger returned invalid JSON"
            ) from exc

    async def aclose(self) -> None:
        if not self._owns_client or self._client is None:
            return

        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result


def _response_excerpt(response: httpx.Response, limit: int = 240) -> str:
    try:
        content = response.text.strip()
    except Exception:
        content = ""

    if not content:
        return "<empty response>"

    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."
