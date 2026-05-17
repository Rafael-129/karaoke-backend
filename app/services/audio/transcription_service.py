from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.core.config import Settings
from app.services.audio.model_manager import get_whisper_model

# Whisper transcription utilities.

logger = logging.getLogger(__name__)
class TranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def transcribe_lrc(self, audio_bytes: bytes, title: str = "", artist: str = "") -> str:
        temp_path = self._settings.data_dir / f"temp-audio-{uuid.uuid4().hex}.wav"
        try:
            temp_path.write_bytes(audio_bytes)
            self._settings.whisper_download_root.mkdir(parents=True, exist_ok=True)
            model = get_whisper_model(
                self._settings.whisper_model,
                str(self._settings.whisper_download_root),
                device=self._settings.whisper_device,
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
                m = int(start // 60)
                s = int(start % 60)
                c = int((start % 1) * 100)
                
                line_text = ""
                if "words" in segment and segment["words"]:
                    for w in segment["words"]:
                        ws = w["start"]
                        wm = int(ws // 60)
                        ws_s = int(ws % 60)
                        wc = int((ws % 1) * 100)
                        word_clean = w["word"].strip()
                        if word_clean:
                            line_text += f"<{wm:02d}:{ws_s:02d}.{wc:02d}> {word_clean} "
                else:
                    line_text = segment["text"].strip()

                if line_text.strip():
                    lrc_lines.append(f"[{m:02d}:{s:02d}.{c:02d}] {line_text.strip()}")

            return "\n".join(lrc_lines)
        except Exception as exc:
            logger.error("Whisper error: %s", exc)
            return ""
        finally:
            temp_path.unlink(missing_ok=True)
