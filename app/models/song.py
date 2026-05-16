from __future__ import annotations

from pydantic import BaseModel, Field

# Pydantic models for song data.


class SongRecord(BaseModel):
    job_id: str
    title: str
    artist: str | None = None
    bpm: int = 0
    duration: str | None = None
    lrc_preview: str | None = None
    lrc: str | None = None
    tags: list[str] = Field(default_factory=list)
    video_url: str | None = None
    instrumental_url: str | None = None


class SongPublic(BaseModel):
    id: str
    title: str
    artist: str | None = None
    bpm: int = 0
    duration: str | None = None
    lrcPreview: str | None = None
    lrc: str | None = None
    tags: list[str] = Field(default_factory=list)
    videoUrl: str | None = None
    instrumentalUrl: str | None = None


def to_public(song: SongRecord) -> SongPublic:
    return SongPublic(
        id=song.job_id,
        title=song.title,
        artist=song.artist,
        bpm=song.bpm,
        duration=song.duration,
        lrcPreview=song.lrc_preview,
        lrc=song.lrc,
        tags=song.tags,
        videoUrl=song.video_url,
        instrumentalUrl=song.instrumental_url,
    )
