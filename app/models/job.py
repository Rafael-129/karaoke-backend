from __future__ import annotations

from pydantic import BaseModel

from app.models.song import SongPublic

# Pydantic model for job status payloads.


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    song: SongPublic | None = None
