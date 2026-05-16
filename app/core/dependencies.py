from __future__ import annotations

from functools import lru_cache

from supabase import create_client

from app.core.config import Settings, get_settings
from app.repositories.jobs_repository import JobsRepository
from app.repositories.songs_repository import SongsRepository
from app.services.audio.conversion_service import AudioConversionService
from app.services.audio.separation_service import AudioSeparationService
from app.services.audio.transcription_service import TranscriptionService
from app.services.jobs.jobs_service import JobsService
from app.services.storage.database_service import DatabaseService
from app.services.storage.storage_service import StorageService


@lru_cache(maxsize=1)
def get_supabase_client():
    settings = get_settings()
    key = settings.supabase_secret or settings.supabase_key
    return create_client(settings.supabase_url, key)


@lru_cache(maxsize=1)
def get_storage_service() -> StorageService:
    settings = get_settings()
    return StorageService(get_supabase_client(), settings.uploads_bucket, settings.outputs_bucket)


@lru_cache(maxsize=1)
def get_database_service() -> DatabaseService:
    return DatabaseService(get_supabase_client())


@lru_cache(maxsize=1)
def get_songs_repository() -> SongsRepository:
    return SongsRepository(get_database_service())


@lru_cache(maxsize=1)
def get_jobs_repository() -> JobsRepository:
    return JobsRepository()


@lru_cache(maxsize=1)
def get_audio_conversion_service() -> AudioConversionService:
    return AudioConversionService(get_settings())


@lru_cache(maxsize=1)
def get_audio_separation_service() -> AudioSeparationService:
    settings = get_settings()
    conversion_service = get_audio_conversion_service()
    return AudioSeparationService(settings, conversion_service)


@lru_cache(maxsize=1)
def get_transcription_service() -> TranscriptionService:
    return TranscriptionService(get_settings())


@lru_cache(maxsize=1)
def get_jobs_service() -> JobsService:
    settings = get_settings()
    return JobsService(
        settings=settings,
        storage_service=get_storage_service(),
        songs_repository=get_songs_repository(),
        jobs_repository=get_jobs_repository(),
        conversion_service=get_audio_conversion_service(),
        separation_service=get_audio_separation_service(),
        transcription_service=get_transcription_service(),
    )
