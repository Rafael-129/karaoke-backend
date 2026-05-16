from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import soundfile as sf
import torch
from demucs.audio import convert_audio

from app.common.formatters import format_duration
from app.core.config import Settings

# Audio conversion and probing utilities.


class AudioConversionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def convert_to_wav(self, input_path: Path, output_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed (code {result.returncode}):\n{result.stderr[-2000:]}"
            )

    def load_audio_for_demucs(
        self, wav_path: Path, target_channels: int, target_samplerate: int
    ) -> torch.Tensor:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        wav = convert_audio(wav, sr, target_samplerate, target_channels)
        return wav

    def probe_duration_label(self, file_bytes: bytes, filename: str) -> str:
        temp_path = self._settings.data_dir / f"temp-{uuid.uuid4().hex}-{filename}"
        try:
            temp_path.write_bytes(file_bytes)
            command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(temp_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return format_duration(float(result.stdout.strip()))
        except Exception:
            return "--:--"
        finally:
            temp_path.unlink(missing_ok=True)
