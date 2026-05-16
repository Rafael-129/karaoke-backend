from __future__ import annotations

from fastapi import APIRouter

from app.core.dependencies import get_jobs_service

# Job status endpoints.

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, object]:
    return get_jobs_service().get_status(job_id)
