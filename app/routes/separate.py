from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_jobs_service, get_storage_service

# Upload and separation endpoints.

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/separate")
async def separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    lyrics: str = Form(""),
    tags: str = Form(""),
    job_id: Optional[str] = Form(None),
    chunk_index: Optional[int] = Form(None),
    total_chunks: Optional[int] = Form(None),
) -> dict[str, Any]:
    if not file:
        raise HTTPException(status_code=400, detail="Missing file.")

    file_bytes = await file.read()
    await file.close()

    try:
        return await get_jobs_service().handle_upload(
            file_bytes=file_bytes,
            filename=file.filename,
            title=title,
            artist=artist,
            tags=tags,
            job_id=job_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            background_tasks=background_tasks,
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_text = str(exc).strip()
        logger.error("Separate error: %s", error_text)
        snippet = error_text[:300] if error_text else "Separation failed."
        raise HTTPException(status_code=500, detail=snippet) from exc


@router.get("/uploads/{job_id}/{filename}")
def get_upload_file(job_id: str, filename: str) -> StreamingResponse:
    try:
        safe_name = Path(filename).name
        payload = get_storage_service().download_upload(job_id, safe_name)
        return StreamingResponse(io.BytesIO(payload), media_type="application/octet-stream")
    except Exception:
        raise HTTPException(status_code=404, detail="File not found.")


@router.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str) -> StreamingResponse:
    try:
        safe_name = Path(filename).name
        payload = get_storage_service().download_output(job_id, safe_name)
        return StreamingResponse(io.BytesIO(payload), media_type="audio/wav")
    except Exception:
        raise HTTPException(status_code=404, detail="File not found.")
