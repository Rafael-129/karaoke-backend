from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Centralized environment and path configuration.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_root: Path
    data_dir: Path
    chunks_dir: Path
    uploads_bucket: str
    outputs_bucket: str
    supabase_url: str
    supabase_key: str
    supabase_secret: str
    demucs_model: str
    demucs_device: str
    whisper_model: str
    whisper_device: str
    whisper_language: str
    whisper_beam_size: int
    whisper_best_of: int
    whisper_download_root: Path
    preload_models: bool
    app_title: str = "karaoke-backend"


def _require(value: str, name: str) -> str:
    if not value:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY or SUPABASE_SECRET must be set in environment variables"
        )
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _ensure_directories(settings: Settings) -> None:
    settings.data_dir.mkdir(exist_ok=True)
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    settings.whisper_download_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    app_root = Path(__file__).resolve().parents[1]
    data_dir = app_root / "data"
    chunks_dir = data_dir / "chunks"

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    supabase_secret = os.getenv("SUPABASE_SECRET", "").strip()

    if not supabase_url or (not supabase_key and not supabase_secret):
        _require("", "SUPABASE_URL")

    uploads_bucket = os.getenv("UPLOADS_BUCKET", "uploads").strip() or "uploads"
    outputs_bucket = os.getenv("OUTPUTS_BUCKET", "outputs").strip() or "outputs"

    demucs_model = os.getenv("DEMUCS_MODEL", "htdemucs_6s").strip() or "htdemucs_6s"
    demucs_device = os.getenv("DEMUCS_DEVICE", "cpu").strip() or "cpu"

    # Prefer a smaller Whisper model by default for low-memory deploys.
    whisper_model = os.getenv("WHISPER_MODEL", "tiny").strip() or "tiny"
    whisper_device = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
    whisper_language = os.getenv("WHISPER_LANGUAGE", "es").strip() or "es"
    whisper_beam_size = _int_env("WHISPER_BEAM_SIZE", 5)
    whisper_best_of = _int_env("WHISPER_BEST_OF", 5)
    whisper_download_root = Path(
        os.getenv("WHISPER_DOWNLOAD_ROOT", str(data_dir / "whisper-models"))
    )
    preload_models = _bool_env("PRELOAD_MODELS", False)

    settings = Settings(
        app_root=app_root,
        data_dir=data_dir,
        chunks_dir=chunks_dir,
        uploads_bucket=uploads_bucket,
        outputs_bucket=outputs_bucket,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        supabase_secret=supabase_secret,
        demucs_model=demucs_model,
        demucs_device=demucs_device,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_language=whisper_language,
        whisper_beam_size=whisper_beam_size,
        whisper_best_of=whisper_best_of,
        whisper_download_root=whisper_download_root,
        preload_models=preload_models,
    )
    _ensure_directories(settings)
    return settings
