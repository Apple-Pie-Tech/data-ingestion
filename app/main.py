import json
import logging
import secrets
from collections.abc import AsyncIterator
from functools import lru_cache
from importlib import import_module
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

_config = import_module("app.config")
_audio_storage = import_module("app.audio_storage")
_embeddings = import_module("app.embeddings")
_ingestion = import_module("app.ingestion")
_schemas = import_module("app.schemas")
_story_labeling = import_module("app.story_labeling")
_transcription = import_module("app.transcription")
_vector_store = import_module("app.vector_store")
_supabase_jwt = import_module("app.supabase_jwt")

Settings = _config.Settings
get_settings = _config.get_settings
AudioStorageError = _audio_storage.AudioStorageError
EmbeddingError = _embeddings.EmbeddingError
AudioFile = _ingestion.AudioFile
IngestionService = _ingestion.IngestionService
IngestionValidationError = _ingestion.IngestionValidationError
IngestMetadata = _schemas.IngestMetadata
IngestRequest = _schemas.IngestRequest
IngestResult = _schemas.IngestResult
StoryLabelingTrigger = _story_labeling.StoryLabelingTrigger
StoryLabelingTriggerClient = _story_labeling.StoryLabelingTriggerClient
StoryLabelingTriggerConfigurationError = _story_labeling.StoryLabelingTriggerConfigurationError
StoryLabelingTriggerError = _story_labeling.StoryLabelingTriggerError
TranscriptionError = _transcription.TranscriptionError
VectorStoreError = _vector_store.VectorStoreError
SupabaseIdentity = _supabase_jwt.SupabaseIdentity
SupabaseJwtError = _supabase_jwt.SupabaseJwtError
SupabaseJwtUnavailableError = _supabase_jwt.SupabaseJwtUnavailableError
SupabaseJwtVerifier = _supabase_jwt.SupabaseJwtVerifier

logger = logging.getLogger(__name__)


def _configured_ingest_api_key(settings: Settings) -> str:
    return (settings.ingest_api_key or "").strip()


def _configured_supabase_url(settings: Settings) -> str:
    return (settings.supabase_url or "").strip()


def _parse_cors_allow_origins(raw_value: str) -> list[str]:
    return [origin for origin in (item.strip() for item in raw_value.split(",")) if origin]


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    application = FastAPI(title="Apple Pie Data Ingestion", version="0.1.0")

    cors_allow_origins = _parse_cors_allow_origins(resolved_settings.cors_allow_origins)
    if cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_credentials=False,
            allow_headers=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_origins=cors_allow_origins,
        )

    if not _configured_ingest_api_key(resolved_settings):
        logger.warning(
            "INGEST_API_KEY is not set; /ingest accepts unauthenticated requests"
        )

    if not _configured_supabase_url(resolved_settings):
        # Worth its own line: with SUPABASE_URL unset the x-api-key above is the
        # only gate, and that key is public by construction -- the UI ships it
        # in its JavaScript bundle. So this is the difference between a
        # deterrent and authentication, not a nice-to-have.
        logger.warning(
            "SUPABASE_URL is not set; /ingest accepts requests with no verified user"
        )

    return application


app = create_app()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def require_ingest_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Verify the ``x-api-key`` header against ``INGEST_API_KEY``.

    When ``INGEST_API_KEY`` is unset or blank the check is **skipped**, not
    failed. Failing closed would leave a fresh clone with no ``.env`` unable to
    ingest anything, and local development is the path this service has to keep
    working; ``create_app`` logs a warning so an unprotected deployment is not
    silent. Set ``INGEST_API_KEY`` to turn the check on.

    The 401 carries no detail about the expected key.
    """
    expected = _configured_ingest_api_key(settings)
    if not expected:
        return

    provided = x_api_key or ""
    if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="unauthorized")


@lru_cache(maxsize=1)
def _build_supabase_verifier(
    supabase_url: str,
    audience: str,
    hs256_secret: str | None,
    cache_seconds: int,
) -> SupabaseJwtVerifier:
    """One verifier per process, so its key cache survives between requests.

    Cached on the values rather than on Settings because Settings is not
    hashable. Under the Lambda Web Adapter the uvicorn process outlives a single
    invocation, so this is also the cross-invocation cache -- see
    app/supabase_jwt.py.
    """
    return SupabaseJwtVerifier(
        supabase_url=supabase_url,
        audience=audience,
        hs256_secret=hs256_secret,
        cache_seconds=cache_seconds,
    )


def get_supabase_verifier(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SupabaseJwtVerifier | None:
    """None when SUPABASE_URL is unset, which skips verification entirely."""
    supabase_url = _configured_supabase_url(settings)
    if not supabase_url:
        return None

    return _build_supabase_verifier(
        supabase_url,
        settings.supabase_jwt_audience,
        (settings.supabase_jwt_secret or "").strip() or None,
        settings.supabase_jwks_cache_seconds,
    )


async def get_supabase_identity(
    verifier: Annotated[SupabaseJwtVerifier | None, Depends(get_supabase_verifier)],
    authorization: Annotated[str | None, Header()] = None,
) -> SupabaseIdentity | None:
    """Verify the `Authorization: Bearer <supabase access token>` header.

    Returns None when ``SUPABASE_URL`` is unset -- the same "unset means skip"
    convention as ``require_ingest_api_key``, so a fresh clone with no ``.env``
    still ingests and a deploy can ship this code before the variable is set.
    ``create_app`` warns in that case.

    Once it *is* set there is no path back to unauthenticated: a missing header,
    a non-Bearer scheme, a bad signature, an expired token, the wrong issuer or
    audience all raise 401. A JWKS endpoint that cannot be reached raises 503
    instead -- answering 401 would blame the user for a Supabase outage, and
    answering 200 would turn a network blip into an authentication bypass.

    The 401 body is identical to the one ``require_ingest_api_key`` returns, so
    the response does not reveal which of the two gates refused.
    """
    if verifier is None:
        return None

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return await verifier.verify(token.strip())
    except SupabaseJwtUnavailableError as exc:
        logger.exception("could not verify the access token: Supabase is unreachable")
        raise HTTPException(status_code=503, detail="auth_unavailable") from exc
    except SupabaseJwtError as exc:
        # Logged without the token, at warning: a rejected token is routine.
        logger.warning("rejected an access token: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_ingestion_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[IngestionService]:
    service = IngestionService(settings)
    try:
        yield service
    finally:
        await service.aclose()


async def get_story_labeling_trigger(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[StoryLabelingTrigger]:
    trigger = StoryLabelingTriggerClient(settings)
    try:
        yield trigger
    finally:
        await trigger.aclose()


async def parse_ingest_request(
    metadata: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(get_settings)],
    text: Annotated[str | None, Form()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
) -> IngestRequest:
    try:
        metadata_model = IngestMetadata.model_validate_json(metadata)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    normalized_text = text.strip() if text else ""
    if normalized_text:
        return IngestRequest(
            metadata=metadata_model,
            text=normalized_text,
            source="text",
        )

    audio_bytes: bytes | None = None
    audio_filename: str | None = None
    audio_content_type: str | None = None

    if audio is not None:
        audio_bytes = await audio.read()
        if len(audio_bytes) > settings.max_audio_bytes:
            raise HTTPException(status_code=400, detail="audio exceeds maximum size")
        audio_filename = audio.filename
        audio_content_type = audio.content_type

    if not normalized_text and not audio_bytes:
        raise HTTPException(status_code=400, detail="text or audio required")

    return IngestRequest(
        metadata=metadata_model,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        audio_content_type=audio_content_type,
        source="audio",
    )


@app.post(
    "/ingest",
    response_model=IngestResult,
    response_model_exclude_none=True,
    # Both gates listed here rather than only as handler parameters, so they run
    # before the request body is parsed -- an unauthenticated caller gets a 401
    # without this service reading, buffering or transcoding their upload.
    dependencies=[Depends(require_ingest_api_key), Depends(get_supabase_identity)],
)
async def ingest(
    background_tasks: BackgroundTasks,
    request: Annotated[IngestRequest, Depends(parse_ingest_request)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    story_labeling_trigger: Annotated[StoryLabelingTrigger, Depends(get_story_labeling_trigger)],
    # Declared again as a parameter to read the claims. FastAPI caches a
    # dependency by callable within a request, so it still runs exactly once.
    identity: Annotated[SupabaseIdentity | None, Depends(get_supabase_identity)] = None,
) -> IngestResult:
    # The body carries user_id, so without this check any authenticated caller
    # could file memories under someone else's id -- and before this change any
    # caller at all could, since the only gate was a key published in the UI
    # bundle. The token's `sub` is the one identity assertion a caller cannot
    # forge, so it wins over the body.
    if identity is not None and request.metadata.user_id != identity.user_id:
        raise HTTPException(status_code=403, detail="user_id does not match the token")

    try:
        result = await service.ingest(
            metadata=request.metadata,
            text=request.text,
            audio_file=_build_audio_file(request),
        )
        _schedule_story_labeling_trigger(
            background_tasks,
            trigger=story_labeling_trigger,
            metadata=request.metadata,
            source=request.source,
        )
        return result
    except HTTPException:
        raise
    except StoryLabelingTriggerConfigurationError:
        raise
    except IngestionValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Each handler below collapses a real failure into an opaque detail string.
    # Log the traceback first: without it a broken dependency shows up as a bare
    # 503 in the client and nothing at all in the service log, which is how the
    # audio-storage breakage (ISSUES.md E54) stayed hidden.
    except AudioStorageError as exc:
        logger.exception("ingest failed: audio storage unavailable")
        raise HTTPException(status_code=503, detail="audio_storage_unavailable") from exc
    except TranscriptionError as exc:
        logger.exception("ingest failed: transcription unavailable")
        raise HTTPException(status_code=502, detail="transcription_unavailable") from exc
    except EmbeddingError as exc:
        logger.exception("ingest failed: embeddings unavailable")
        raise HTTPException(status_code=502, detail="embedding_unavailable") from exc
    except VectorStoreError as exc:
        logger.exception("ingest failed: vector store unavailable")
        raise HTTPException(status_code=503, detail="vector_store_unavailable") from exc
    except Exception as exc:
        logger.exception("ingest failed with an unhandled error")
        raise HTTPException(status_code=500, detail="internal_server_error") from exc


def _build_audio_file(request: IngestRequest) -> AudioFile | None:
    if request.audio_bytes is None:
        return None

    return AudioFile(
        content=request.audio_bytes,
        filename=request.audio_filename,
        content_type=request.audio_content_type,
    )


def _schedule_story_labeling_trigger(
    background_tasks: BackgroundTasks,
    *,
    trigger: StoryLabelingTrigger,
    metadata: IngestMetadata,
    source: str,
) -> None:
    if isinstance(trigger, StoryLabelingTriggerClient) and not trigger.enabled:
        return

    background_tasks.add_task(_run_story_labeling_trigger, trigger, metadata, source)


async def _run_story_labeling_trigger(
    trigger: StoryLabelingTrigger,
    metadata: IngestMetadata,
    source: str,
) -> None:
    try:
        await trigger.trigger_cluster_labels(metadata=metadata, source=source)
    except StoryLabelingTriggerError as exc:
        logger.warning(
            "story labeling trigger failed for %s: %s",
            metadata.input_id,
            exc,
        )
