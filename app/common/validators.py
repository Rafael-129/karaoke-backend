from __future__ import annotations

# Input validation helpers.


class ValidationError(ValueError):
    pass


def require_file(file_bytes: bytes | None, filename: str | None) -> None:
    if not file_bytes or not filename:
        raise ValidationError("Missing file.")


def require_title_artist(title: str | None, artist: str | None) -> None:
    if not title or not artist:
        raise ValidationError("Title and artist required.")
