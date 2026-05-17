from __future__ import annotations

import logging
from functools import lru_cache

import whisper
from demucs.pretrained import get_model as get_demucs_pretrained

from app.core.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_whisper_model(model_name: str, download_root: str, device: str = "cpu"):
    logger.info("Loading Whisper model '%s' (download_root=%s) on %s", model_name, download_root, device)
    # whisper.load_model handles download and returns a model instance.
    model = whisper.load_model(model_name, download_root=download_root)
    try:
        # Attempt to move model to device if supported; for CPU this is a no-op.
        model.to(device)
    except Exception:
        logger.debug("Could not call model.to(%s) for Whisper model", device)
    return model


@lru_cache(maxsize=1)
def get_demucs_model(model_name: str, device: str = "cpu"):
    logger.info("Loading Demucs model '%s' on %s", model_name, device)
    model = get_demucs_pretrained(model_name)
    try:
        model.to(device)
    except Exception:
        logger.debug("Could not move Demucs model to %s", device)
    model.eval()
    return model


def preload_models(settings: Settings) -> None:
    """Preload models according to settings. This will be a no-op if models
    are already loaded (because functions are cached with lru_cache).
    """
    settings.whisper_download_root.mkdir(parents=True, exist_ok=True)
    try:
        get_whisper_model(settings.whisper_model, str(settings.whisper_download_root), device=settings.whisper_device)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to preload Whisper model: %s", exc)

    try:
        get_demucs_model(settings.demucs_model, device=settings.demucs_device)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to preload Demucs model: %s", exc)
