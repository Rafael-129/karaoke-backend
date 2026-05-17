from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import BackgroundTasks
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
        background_tasks: BackgroundTasks | None = None,
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

        # 1. Create immediate placeholder in DB BEFORE background task
        placeholder_record = SongRecord(
            job_id=chunk_result.job_id,
            title=title.strip() if title else Path(chunk_result.filename).stem,
            artist=artist.strip() if artist else "Artista desconocido",
            tags=normalize_tags(tags),
            status="processing",
            video_url=f"/uploads/{chunk_result.job_id}/{chunk_result.filename}",
        )
        self._songs.insert_song(placeholder_record)

        # 2. Upload original file IMMEDIATELY so it's not 404 when clicking the card
        try:
            self._storage.upload_original(chunk_result.job_id, chunk_result.filename, chunk_result.file_bytes)
        except Exception as e:
            logger.error("Failed to upload original file for job %s: %s", chunk_result.job_id, e)

        if background_tasks is not None:
            self._jobs.set_status(chunk_result.job_id, "processing", 10, "En cola...")
            background_tasks.add_task(
                self._process_upload,
                job_id=chunk_result.job_id,
                file_bytes=chunk_result.file_bytes,
                filename=chunk_result.filename,
                title=title,
                artist=artist,
                tags=tags,
            )
            return {
                "job_id": chunk_result.job_id,
                "status": "processing",
                "message": "Upload complete, processing started.",
            }

        return self._process_upload(
            job_id=chunk_result.job_id,
            file_bytes=chunk_result.file_bytes,
            filename=chunk_result.filename,
            title=title,
            artist=artist,
            tags=tags,
        )

    def _process_upload(
        self,
        *,
        job_id: str,
        file_bytes: bytes,
        filename: str,
        title: str | None,
        artist: str | None,
        tags: str,
    ) -> dict[str, object]:
        try:
            # 2. Start fast processing (just extracting audio and transcribing)
            self._jobs.set_status(job_id, "processing", 15, "Preparando audio...")
            duration = self._conversion.probe_duration_label(file_bytes, filename)

            # Extract the raw WAV bytes of the original audio track from the video
            original_suffix = Path(filename).suffix.lower() or ".bin"
            temp_input = self._settings.data_dir / f"temp-extract-{job_id}{original_suffix}"
            temp_wav = self._settings.data_dir / f"temp-extract-{job_id}.wav"
            temp_input.write_bytes(file_bytes)
            
            self._jobs.set_status(job_id, "processing", 25, "Extrayendo audio...")
            try:
                self._conversion.convert_to_wav(temp_input, temp_wav)
                audio_bytes = temp_wav.read_bytes()
            except Exception as e:
                raise RuntimeError(
                    "ffmpeg is required to extract audio from video uploads. "
                    "Install ffmpeg in the Railway image (nixpacks aptPkgs) or upload a WAV/MP3 file."
                ) from e
            finally:
                temp_input.unlink(missing_ok=True)
                temp_wav.unlink(missing_ok=True)

            self._jobs.set_status(job_id, "processing", 45, "Transcribiendo...")
            extracted_lrc = self._transcription.transcribe_lrc(audio_bytes, title or "", artist or "")
            preview = build_preview(extracted_lrc)

            self._jobs.set_status(job_id, "processing", 65, "Separando voz...")
            instrumental_bytes, _ = self._separation.separate(file_bytes, filename, job_id)

            self._jobs.set_status(job_id, "processing", 80, "Subiendo instrumental...")
            self._storage.upload_instrumental(job_id, "no_vocals.mp3", instrumental_bytes)

            self._jobs.set_status(job_id, "processing", 90, "Guardando resultados...")

            # 3. Update record to completed without instrumental track initially
            song_record = SongRecord(
                job_id=job_id,
                title=title.strip() if title else Path(filename).stem,
                artist=artist.strip() if artist else "Artista desconocido",
                bpm=0,
                duration=duration,
                lrc_preview=preview,
                lrc=extracted_lrc,
                tags=normalize_tags(tags),
                video_url=f"/uploads/{job_id}/{filename}",
                instrumental_url=f"/files/{job_id}/no_vocals.mp3",
                status="completed",
            )

            self._songs.update_song(job_id, song_record)
            song_public = to_public(song_record)
            self._jobs.set_status(job_id, "completed", 100, "Completado", song_public)

            return {
                "job_id": job_id,
                "download_url": "",
                "song": song_public.model_dump(),
            }
        except Exception as exc:
            message = str(exc).strip() or "Error durante el procesamiento."
            logger.error("Job %s failed: %s", job_id, message)
            
            # Update DB to error status if possible
            try:
                error_record = self._songs.get_song(job_id)
                if error_record:
                    error_record.status = "error"
                    self._songs.update_song(job_id, error_record)
            except Exception:
                pass

            self._jobs.set_status(job_id, "error", 0, message)
            return {
                "job_id": job_id,
                "status": "error",
                "message": message,
            }

    def separate_instrumental_on_demand(self, job_id: str) -> str:
        """
        Runs Demucs on-demand for a completed song that doesn't have an instrumental yet,
        uploads it to storage, updates the DB, and returns the URL.
        """
        song = self._songs.get_song(job_id)
        if not song:
            raise ValueError("Song not found.")

        if song.instrumental_url:
            return song.instrumental_url

        # Retrieve the original file name from the video_url (e.g. /uploads/job_id/filename)
        original_filename = Path(song.video_url).name

        # Download original video/audio from storage to process it
        logger.info("Downloading original file for job %s to separate on-demand", job_id)
        file_bytes = self._storage.download_upload(job_id, original_filename)

        # Run Demucs audio separation
        logger.info("Starting audio separation for job %s (on-demand)", job_id)
        instrumental_bytes, _ = self._separation.separate(file_bytes, original_filename, job_id)

        # Upload the instrumental track
        logger.info("Uploading instrumental track for job %s", job_id)
        self._storage.upload_instrumental(job_id, "no_vocals.mp3", instrumental_bytes)

        # Update DB record with the new instrumental URL
        song.instrumental_url = f"/files/{job_id}/no_vocals.mp3"
        self._songs.update_song(job_id, song)

        logger.info("On-demand separation completed successfully for job %s", job_id)
        return song.instrumental_url

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
