from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path

import whisper

from app.core.config import Settings

# Whisper transcription utilities.

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_whisper_model(model_name: str, download_root: str):
    return whisper.load_model(model_name, download_root=download_root)


class TranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def transcribe_lrc(self, audio_bytes: bytes, title: str = "", artist: str = "") -> str:
        temp_path = self._settings.data_dir / f"temp-audio-{uuid.uuid4().hex}.wav"
        try:
            temp_path.write_bytes(audio_bytes)
            self._settings.whisper_download_root.mkdir(parents=True, exist_ok=True)
            model = _load_whisper_model(
                self._settings.whisper_model,
                str(self._settings.whisper_download_root),
            )

            context_prompt = "Transcribe la letra de una cancion en espanol. "
            if title or artist:
                context_prompt += f"La cancion se llama '{title}' y es de '{artist}'. "
            context_prompt += "Corrige palabras lo mejor posible sin resumir ni traducir."

            result = model.transcribe(
                str(temp_path),
                language=self._settings.whisper_language,
                task="transcribe",
                verbose=False,
                fp16=False,
                temperature=0.0,
                beam_size=self._settings.whisper_beam_size,
                best_of=self._settings.whisper_best_of,
                condition_on_previous_text=False,
                word_timestamps=True,
                initial_prompt=context_prompt,
            )

            lrc_lines = []
            for segment in result["segments"]:
                start = segment["start"]
                text = " ".join(segment["text"].strip().split())
                if not text:
                    continue

                minutes = int(start // 60)
                seconds = int(start % 60)
                centiseconds = int((start % 1) * 100)
                lrc_lines.append(f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}] {text}")

            return "\n".join(lrc_lines)
        except Exception as exc:
            logger.error("Whisper error: %s", exc)
            return ""
        finally:
            temp_path.unlink(missing_ok=True)
