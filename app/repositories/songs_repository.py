from __future__ import annotations

from typing import Iterable

from app.models.song import SongRecord
from app.services.storage.database_service import DatabaseService

# Supabase access for songs table.


class SongsRepository:
    def __init__(self, database_service: DatabaseService) -> None:
        self._db = database_service

    def list_songs(self) -> list[SongRecord]:
        response = self._db.table("songs").select("*").order("created_at", desc=True).execute()
        songs = response.data or []
        return [SongRecord.model_validate(song) for song in songs]

    def get_song(self, job_id: str) -> SongRecord | None:
        response = (
            self._db.table("songs")
            .select("*")
            .eq("job_id", job_id)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None
        return SongRecord.model_validate(rows[0])

    def insert_song(self, song: SongRecord) -> None:
        payload = song.model_dump()
        self._db.table("songs").insert(payload).execute()

    def delete_song(self, job_id: str) -> SongRecord | None:
        song = self.get_song(job_id)
        if not song:
            return None
        self._db.table("songs").delete().eq("job_id", job_id).execute()
        return song

    def reset_catalog(self) -> None:
        self._db.table("songs").delete().neq("job_id", "").execute()
