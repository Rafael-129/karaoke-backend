from __future__ import annotations

import logging
from typing import Iterable

# Supabase storage access helpers.

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self, client, uploads_bucket: str, outputs_bucket: str) -> None:
        self._client = client
        self.uploads_bucket = uploads_bucket
        self.outputs_bucket = outputs_bucket

    def ensure_buckets(self) -> None:
        for bucket in [self.uploads_bucket, self.outputs_bucket]:
            try:
                self._client.storage.get_bucket(bucket)
            except Exception:
                self._client.storage.create_bucket(bucket)
                logger.info("Created storage bucket: %s", bucket)

    def upload_bytes(self, bucket: str, path: str, payload: bytes) -> None:
        self._client.storage.from_(bucket).upload(path, payload)

    def download_bytes(self, bucket: str, path: str) -> bytes:
        return self._client.storage.from_(bucket).download(path)

    def delete_paths(self, bucket: str, paths: Iterable[str]) -> None:
        if not paths:
            return
        self._client.storage.from_(bucket).remove(list(paths))

    def upload_original(self, job_id: str, filename: str, payload: bytes) -> str:
        file_path = f"{job_id}/{filename}"
        self.upload_bytes(self.uploads_bucket, file_path, payload)
        return file_path

    def upload_instrumental(self, job_id: str, filename: str, payload: bytes) -> str:
        file_path = f"{job_id}/{filename}"
        self.upload_bytes(self.outputs_bucket, file_path, payload)
        return file_path

    def download_upload(self, job_id: str, filename: str) -> bytes:
        file_path = f"{job_id}/{filename}"
        return self.download_bytes(self.uploads_bucket, file_path)

    def download_output(self, job_id: str, filename: str) -> bytes:
        file_path = f"{job_id}/{filename}"
        return self.download_bytes(self.outputs_bucket, file_path)

    def delete_song_files(self, video_url: str | None, instrumental_url: str | None) -> None:
        cleanup_targets: list[tuple[str, str]] = []

        if isinstance(video_url, str) and video_url.startswith("/uploads/"):
            uploads_path = video_url.removeprefix("/uploads/")
            if uploads_path:
                cleanup_targets.append((self.uploads_bucket, uploads_path))

        if isinstance(instrumental_url, str) and instrumental_url.startswith("/files/"):
            outputs_path = instrumental_url.removeprefix("/files/")
            if outputs_path:
                cleanup_targets.append((self.outputs_bucket, outputs_path))

        for bucket, path in cleanup_targets:
            try:
                self.delete_paths(bucket, [path])
            except Exception as exc:
                logger.warning("Cleanup failed for %s/%s: %s", bucket, path, exc)
