import json
from collections.abc import AsyncIterator
from importlib import import_module
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

_config = import_module("app.config")
_embeddings = import_module("app.embeddings")
_ingestion = import_module("app.ingestion")
_schemas = import_module("app.schemas")
_transcription = import_module("app.transcription")
_vector_store = import_module("app.vector_store")

Settings = _config.Settings
get_settings = _config.get_settings
EmbeddingError = _embeddings.EmbeddingError
AudioFile = _ingestion.AudioFile
IngestionService = _ingestion.IngestionService
IngestionValidationError = _ingestion.IngestionValidationError
IngestMetadata = _schemas.IngestMetadata
IngestRequest = _schemas.IngestRequest
IngestResult = _schemas.IngestResult
TranscriptionError = _transcription.TranscriptionError
VectorStoreError = _vector_store.VectorStoreError

app = FastAPI(title="Apple Pie Data Ingestion", version="0.1.0")


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


@app.post("/ingest", response_model=IngestResult)
async def ingest(
    request: Annotated[IngestRequest, Depends(parse_ingest_request)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestResult:
    try:
        return await service.ingest(
            metadata=request.metadata,
            text=request.text,
            audio_file=_build_audio_file(request),
        )
    except HTTPException:
        raise
    except IngestionValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
