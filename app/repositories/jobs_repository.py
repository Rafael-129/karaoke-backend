from __future__ import annotations

import threading
from dataclasses import dataclass

from app.common.formatters import clamp_progress
from app.models.job import JobStatus
from app.models.song import SongPublic

# In-memory job tracking for progress reporting.


@dataclass
class _JobRecord:
    status: str
    progress: int
    message: str
    song: SongPublic | None = None


class JobsRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobRecord] = {}

    def set_status(
        self,
        job_id: str,
        status: str,
        progress: int,
        message: str,
        song: SongPublic | None = None,
    ) -> None:
        with self._lock:
            self._jobs[job_id] = _JobRecord(
                status=status,
                progress=clamp_progress(progress),
                message=message,
                song=song,
            )

    def get_status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            return JobStatus(
                job_id=job_id,
                status=record.status,
                progress=record.progress,
                message=record.message,
                song=record.song,
            )

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
