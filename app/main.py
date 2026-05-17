import json
import logging
from collections.abc import AsyncIterator
from importlib import import_module
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
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

logger = logging.getLogger(__name__)


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

    return application


app = create_app()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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


@app.post("/ingest", response_model=IngestResult, response_model_exclude_none=True)
async def ingest(
    background_tasks: BackgroundTasks,
    request: Annotated[IngestRequest, Depends(parse_ingest_request)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    story_labeling_trigger: Annotated[StoryLabelingTrigger, Depends(get_story_labeling_trigger)],
) -> IngestResult:
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
    except AudioStorageError as exc:
        raise HTTPException(status_code=503, detail="audio_storage_unavailable") from exc
    except TranscriptionError as exc:
        raise HTTPException(status_code=502, detail="transcription_unavailable") from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail="embedding_unavailable") from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail="vector_store_unavailable") from exc
    except Exception as exc:
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
