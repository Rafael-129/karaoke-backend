from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.common.formatters import build_preview, normalize_tags
from app.common.validators import ValidationError, require_file, require_title_artist
from app.core.config import Settings
from app.models.job import JobStatus
from app.models.song import SongRecord, SongPublic, to_public
from app.repositories.jobs_repository import JobsRepository
from app.repositories.songs_repository import SongsRepository
from app.services.audio.conversion_service import AudioConversionService
from app.services.audio.separation_service import AudioSeparationService
from app.services.audio.transcription_service import TranscriptionService
from app.services.storage.storage_service import StorageService

# Orchestrates the upload -> separation -> transcription -> Supabase pipeline.

logger = logging.getLogger(__name__)


@dataclass
class _ChunkResult:
    job_id: str
    file_bytes: bytes | None
    filename: str
    chunk_index: int | None
    total_chunks: int | None


class JobsService:
    def __init__(
        self,
        settings: Settings,
        storage_service: StorageService,
        songs_repository: SongsRepository,
        jobs_repository: JobsRepository,
        conversion_service: AudioConversionService,
        separation_service: AudioSeparationService,
        transcription_service: TranscriptionService,
    ) -> None:
        self._settings = settings
        self._storage = storage_service
        self._songs = songs_repository
        self._jobs = jobs_repository
        self._conversion = conversion_service
        self._separation = separation_service
        self._transcription = transcription_service

    def _safe_filename(self, filename: str | None) -> str:
        safe_name = Path(filename).name if filename else "upload.bin"
        original_filename = re.sub(r"\.chunk_\d+$", "", safe_name)
        return original_filename or "upload.bin"

    async def _assemble_chunks(
        self,
        job_id: str,
        chunk_index: int,
        total_chunks: int,
        content: bytes,
    ) -> bytes | None:
        chunk_dir = self._settings.chunks_dir / job_id
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"chunk_{chunk_index:06d}"
        chunk_path.write_bytes(content)
        logger.info("Stored chunk %s for job %s", chunk_index, job_id)

        is_final_chunk = chunk_index == total_chunks - 1
        if not is_final_chunk:
            return None

        missing_chunks: list[int] = []
        max_retries = 5
        retry_delay = 0.5

        for attempt in range(max_retries):
            missing_chunks = []
            for index in range(total_chunks):
                chunk_file = chunk_dir / f"chunk_{index:06d}"
                if not chunk_file.exists():
                    missing_chunks.append(index)

            if not missing_chunks:
                break

            if attempt < max_retries - 1:
                logger.info("Missing chunks %s. Retry %s/%s", missing_chunks, attempt + 1, max_retries)
                await asyncio.sleep(retry_delay)

        if missing_chunks:
            raise ValidationError(
                f"Missing chunks: {missing_chunks}. Received {total_chunks - len(missing_chunks)}/{total_chunks}."
            )

        file_content = b""
        for index in range(total_chunks):
            chunk_file = chunk_dir / f"chunk_{index:06d}"
            chunk_data = chunk_file.read_bytes()
            file_content += chunk_data
            chunk_file.unlink()

        chunk_dir.rmdir()
        if not file_content:
            raise ValidationError("Reassembled file is empty.")

        return file_content

    async def _prepare_upload(
        self,
        file_bytes: bytes,
        filename: str | None,
        job_id: str | None,
        chunk_index: int | None,
        total_chunks: int | None,
    ) -> _ChunkResult:
        safe_filename = self._safe_filename(filename)
        job_id = job_id or uuid.uuid4().hex

        if chunk_index is not None and total_chunks is not None:
            assembled = await self._assemble_chunks(job_id, chunk_index, total_chunks, file_bytes)
            return _ChunkResult(
                job_id=job_id,
                file_bytes=assembled,
                filename=safe_filename,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )

        return _ChunkResult(
            job_id=job_id,
            file_bytes=file_bytes,
            filename=safe_filename,
            chunk_index=None,
            total_chunks=None,
        )

    async def handle_upload(
        self,
        *,
        file_bytes: bytes,
        filename: str | None,
        title: str | None,
        artist: str | None,
        tags: str,
        job_id: str | None,
        chunk_index: int | None,
        total_chunks: int | None,
    ) -> dict[str, object]:
        require_file(file_bytes, filename)

        chunk_result = await self._prepare_upload(
            file_bytes=file_bytes,
            filename=filename,
            job_id=job_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

        if chunk_result.file_bytes is None:
            return {
                "job_id": chunk_result.job_id,
                "status": "chunk_received",
                "chunk": chunk_result.chunk_index,
                "total": chunk_result.total_chunks,
            }

        require_title_artist(title, artist)

        self._jobs.set_status(chunk_result.job_id, "processing", 10, "Subiendo...")
        duration = self._conversion.probe_duration_label(chunk_result.file_bytes, chunk_result.filename)

        self._storage.upload_original(chunk_result.job_id, chunk_result.filename, chunk_result.file_bytes)
        self._jobs.set_status(chunk_result.job_id, "processing", 35, "Separando audio...")

        instrumental_bytes, vocals_bytes = self._separation.separate(
            chunk_result.file_bytes, chunk_result.filename, chunk_result.job_id
        )

        self._storage.upload_instrumental(chunk_result.job_id, "no_vocals.wav", instrumental_bytes)
        self._jobs.set_status(chunk_result.job_id, "processing", 70, "Transcribiendo...")

        extracted_lrc = self._transcription.transcribe_lrc(vocals_bytes, title or "", artist or "")
        preview = build_preview(extracted_lrc)

        song_record = SongRecord(
            job_id=chunk_result.job_id,
            title=title.strip() if title else Path(chunk_result.filename).stem,
            artist=artist.strip() if artist else "Artista desconocido",
            bpm=0,
            duration=duration,
            lrc_preview=preview,
            lrc=extracted_lrc,
            tags=normalize_tags(tags),
            video_url=f"/uploads/{chunk_result.job_id}/{chunk_result.filename}",
            instrumental_url=f"/files/{chunk_result.job_id}/no_vocals.wav",
        )

        self._songs.insert_song(song_record)
        song_public = to_public(song_record)
        self._jobs.set_status(chunk_result.job_id, "completed", 100, "Completado", song_public)

        return {
            "job_id": chunk_result.job_id,
            "download_url": song_record.instrumental_url or "",
            "song": song_public.model_dump(),
        }

    def get_status(self, job_id: str) -> dict[str, object]:
        status = self._jobs.get_status(job_id)
        if status:
            payload = JobStatus(
                job_id=status.job_id,
                status=status.status,
                progress=status.progress,
                message=status.message,
                song=status.song,
            )
            return payload.model_dump()

        song = self._songs.get_song(job_id)
        if song:
            song_public = to_public(song)
            payload = JobStatus(
                job_id=job_id,
                status="completed",
                progress=100,
                message="Completado",
                song=song_public,
            )
            return payload.model_dump()

        payload = JobStatus(
            job_id=job_id,
            status="processing",
            progress=50,
            message="Procesando...",
            song=None,
        )
        return payload.model_dump()
