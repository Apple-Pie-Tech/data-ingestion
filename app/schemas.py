from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestMetadata(BaseModel):
    input_id: str
    user_id: str
    timestamp: datetime


class IngestRequest(BaseModel):
    metadata: IngestMetadata
    text: str | None = None
    audio_bytes: bytes | None = Field(default=None, repr=False)
    audio_filename: str | None = None
    audio_content_type: str | None = None
    source: Literal["text", "audio"]


class IngestResult(BaseModel):
    input_id: str
    status: str
    chunks: int
