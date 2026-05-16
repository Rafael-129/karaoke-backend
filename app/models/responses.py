from __future__ import annotations

from pydantic import BaseModel

from app.models.song import SongPublic

# Standard API response payloads.


class HealthResponse(BaseModel):
    status: str


class ChunkUploadResponse(BaseModel):
    job_id: str
    status: str
    chunk: int
    total: int


class SeparateResponse(BaseModel):
    job_id: str
    download_url: str
    song: SongPublic
