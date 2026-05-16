from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_songs_repository, get_storage_service
from app.models.song import to_public

# Catalog endpoints.

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/catalog")
def list_catalog() -> list[dict[str, Any]]:
    try:
        songs = get_songs_repository().list_songs()
        return [to_public(song).model_dump() for song in songs]
    except Exception as exc:
        logger.error("Catalog error: %s", exc)
        return []


@router.get("/catalog/{song_id}")
def get_catalog_song(song_id: str) -> dict[str, Any]:
    song = get_songs_repository().get_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")
    return to_public(song).model_dump()


@router.delete("/catalog/{song_id}")
def delete_catalog_song(song_id: str) -> dict[str, str]:
    song = get_songs_repository().delete_song(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")

    get_storage_service().delete_song_files(song.video_url, song.instrumental_url)
    return {"status": "ok", "message": "Song deleted"}


@router.delete("/catalog")
def reset_catalog() -> dict[str, str]:
    try:
        get_songs_repository().reset_catalog()
        return {"status": "ok", "message": "Catalog cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error clearing catalog: {str(exc)}")
