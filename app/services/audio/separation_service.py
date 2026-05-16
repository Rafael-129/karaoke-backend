from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from demucs.apply import apply_model
from demucs.pretrained import get_model

from app.core.config import Settings
from app.services.audio.conversion_service import AudioConversionService

# Demucs-based separation logic.


@lru_cache(maxsize=1)
def _load_demucs_model(model_name: str, device: str):
    model = get_model(model_name)
    model.to(device)
    model.eval()
    return model


class AudioSeparationService:
    def __init__(self, settings: Settings, conversion_service: AudioConversionService) -> None:
        self._settings = settings
        self._conversion = conversion_service

    def separate(self, file_bytes: bytes, original_filename: str, job_id: str) -> tuple[bytes, bytes]:
        model = _load_demucs_model(self._settings.demucs_model, self._settings.demucs_device)
        original_suffix = Path(original_filename).suffix.lower() or ".bin"
        temp_input = self._settings.data_dir / f"temp-input-{job_id}{original_suffix}"
        temp_wav = self._settings.data_dir / f"temp-input-{job_id}.wav"

        temp_input.write_bytes(file_bytes)

        try:
            try:
                self._conversion.convert_to_wav(temp_input, temp_wav)
            except Exception:
                temp_wav = temp_input

            mix = (
                self._conversion.load_audio_for_demucs(
                    temp_wav, model.audio_channels, model.samplerate
                )
                .unsqueeze(0)
            )
            sources = apply_model(model, mix, device=self._settings.demucs_device)

            try:
                sources_np = sources.detach().cpu().numpy()
            except Exception:
                sources_np = sources.cpu().numpy()

            batch_idx = 0
            non_vocals = []
            for idx, name in enumerate(model.sources):
                if name == "vocals":
                    continue
                non_vocals.append(sources_np[batch_idx, idx])

            if non_vocals:
                instrumental_np = np.sum(np.stack(non_vocals, axis=0), axis=0)
            else:
                if "vocals" in model.sources:
                    vocals_idx = model.sources.index("vocals")
                    vocals_np = sources_np[batch_idx, vocals_idx]
                    try:
                        mix_np = mix.detach().cpu().numpy()[batch_idx]
                    except Exception:
                        mix_np = mix.cpu().numpy()[batch_idx]
                    instrumental_np = mix_np - vocals_np
                else:
                    try:
                        instrumental_np = mix.detach().cpu().numpy()[batch_idx]
                    except Exception:
                        instrumental_np = mix.cpu().numpy()[batch_idx]

            max_val = float(np.max(np.abs(instrumental_np))) if instrumental_np.size else 0.0
            if max_val > 1.0:
                instrumental_np = instrumental_np / max_val

            instrumental_buffer = io.BytesIO()
            sf.write(instrumental_buffer, instrumental_np.T, model.samplerate, format="WAV")
            instrumental_bytes = instrumental_buffer.getvalue()

            vocals_bytes = file_bytes
            if "vocals" in model.sources:
                vocals_idx = model.sources.index("vocals")
                vocals_np = sources_np[batch_idx, vocals_idx]
                if vocals_np.ndim == 2:
                    vocals_np = vocals_np.mean(axis=0)
                vocals_max = float(np.max(np.abs(vocals_np))) if vocals_np.size else 0.0
                if vocals_max > 1.0:
                    vocals_np = vocals_np / vocals_max

                vocals_buffer = io.BytesIO()
                sf.write(vocals_buffer, vocals_np, model.samplerate, format="WAV")
                vocals_bytes = vocals_buffer.getvalue()

            return instrumental_bytes, vocals_bytes
        finally:
            temp_input.unlink(missing_ok=True)
            if temp_wav != temp_input:
                temp_wav.unlink(missing_ok=True)
