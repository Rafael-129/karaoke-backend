from __future__ import annotations

from functools import lru_cache
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import soundfile as sf
import whisper

from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
import numpy as np

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
CATALOG_FILE = DATA_DIR / "catalog.json"
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "es")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))
WHISPER_DOWNLOAD_ROOT = Path(os.getenv("WHISPER_DOWNLOAD_ROOT", str(DATA_DIR / "whisper-models")))

app = FastAPI(title="karaoke-backend")


@lru_cache(maxsize=1)
def load_demucs_model():
    model = get_model(DEMUCS_MODEL)
    model.to("cpu")
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_whisper_model():
    WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return whisper.load_model(WHISPER_MODEL, download_root=str(WHISPER_DOWNLOAD_ROOT))



def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"

    total_seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

    return f"{minutes}:{remaining_seconds:02d}"


def build_preview(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_FILE.exists():
        return []

    try:
        raw = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(raw, list):
        return []

    return [item for item in raw if isinstance(item, dict)]


def save_catalog(items: list[dict[str, Any]]) -> None:
    CATALOG_FILE.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clear_directory(directory: Path) -> int:
    removed = 0
    if not directory.exists():
        return removed

    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            removed += 1
        else:
            child.unlink(missing_ok=True)
            removed += 1

    return removed


def probe_duration_label(path: Path) -> str:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return format_duration(float(result.stdout.strip()))
    except Exception:
        return "--:--"


def normalize_tags(raw_tags: str) -> list[str]:
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags or ["subido"]


def extract_lyrics_with_timestamps(audio_path: Path) -> str:
    """Extract lyrics from audio using Whisper and generate LRC format with timestamps."""
    try:
        model = load_whisper_model()
        result = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            task="transcribe",
            verbose=False,
            fp16=False,
            temperature=0.0,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            condition_on_previous_text=False,
            initial_prompt=(
                "Transcribe la letra de una canción en español. "
                "Corrige palabras lo mejor posible sin resumir ni traducir."
            ),
        )
        
        lrc_lines = []
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = " ".join(segment["text"].strip().split())
            
            if not text:
                continue
            
            # Convert seconds to [MM:SS.CC] format
            minutes = int(start // 60)
            seconds = int(start % 60)
            centiseconds = int((start % 1) * 100)
            
            lrc_line = f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}] {text}"
            lrc_lines.append(lrc_line)
        
        return "\n".join(lrc_lines)
    except Exception as exc:
        print(f"[WHISPER ERROR] {str(exc)}", flush=True)
        return ""



@app.on_event("startup")
def on_startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CATALOG_FILE.exists():
        save_catalog([])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/catalog")
def list_catalog() -> list[dict[str, Any]]:
    return load_catalog()


@app.get("/catalog/{song_id}")
def get_catalog_song(song_id: str) -> dict[str, Any]:
    for item in load_catalog():
        if item.get("id") == song_id:
            return item

    raise HTTPException(status_code=404, detail="Song not found.")


@app.delete("/catalog")
def reset_catalog() -> dict[str, int | str]:
    removed_uploads = clear_directory(UPLOAD_DIR)
    removed_outputs = clear_directory(OUTPUT_DIR)
    save_catalog([])

    return {
        "status": "ok",
        "removed_uploads": removed_uploads,
        "removed_outputs": removed_outputs,
        "removed_catalog_items": 0,
    }


@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(...),
    lyrics: str = Form(""),
    tags: str = Form(""),
) -> dict[str, Any]:
    safe_filename = Path(file.filename).name if file.filename else f"upload-{uuid.uuid4().hex}.bin"

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}-{safe_filename}"

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    output_root = OUTPUT_DIR / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        model = load_demucs_model()
        mix = load_track(input_path, model.audio_channels, model.samplerate).unsqueeze(0)
        sources = apply_model(model, mix, device="cpu")

        # Debug: print available source names and shapes
        try:
            print(f"[DEMUCS] sources names={model.sources}", flush=True)
            print(f"[DEMUCS] sources shape={getattr(sources, 'shape', None)}", flush=True)
        except Exception:
            pass

        # Convert torch tensors to numpy for safer processing
        try:
            sources_np = sources.detach().cpu().numpy()
        except Exception:
            sources_np = sources.cpu().numpy()

        # sources_np shape: (batch, n_sources, channels, samples)
        batch_idx = 0

        # Build instrumental as sum of all non-vocals sources; if vocals bleed, fall back to mix - vocals
        non_vocals = []
        for idx, name in enumerate(model.sources):
            if name == "vocals":
                continue
            non_vocals.append(sources_np[batch_idx, idx])

        if non_vocals:
            instrumental_np = np.sum(np.stack(non_vocals, axis=0), axis=0)
        else:
            # If we couldn't find non-vocals sources, try subtracting vocals from the mix
            if "vocals" in model.sources:
                vocals_idx = model.sources.index("vocals")
                vocals_np = sources_np[batch_idx, vocals_idx]
                try:
                    mix_np = mix.detach().cpu().numpy()[batch_idx]
                except Exception:
                    mix_np = mix.cpu().numpy()[batch_idx]
                instrumental_np = mix_np - vocals_np
            else:
                # As a last resort, use the original mix
                try:
                    instrumental_np = mix.detach().cpu().numpy()[batch_idx]
                except Exception:
                    instrumental_np = mix.cpu().numpy()[batch_idx]

        vocals_path = output_root / "vocals.wav"
        if "vocals" in model.sources:
            vocals_idx = model.sources.index("vocals")
            vocals_np = sources_np[batch_idx, vocals_idx]
            if vocals_np.ndim == 2:
                vocals_np = vocals_np.mean(axis=0)
            try:
                vocals_max = float(np.max(np.abs(vocals_np))) if vocals_np.size else 0.0
            except Exception:
                vocals_max = 0.0
            if vocals_max > 1.0:
                vocals_np = vocals_np / vocals_max
            sf.write(str(vocals_path), vocals_np, model.samplerate)

        # Normalize to avoid clipping and ensure values are in [-1,1]
        max_val = float(np.max(np.abs(instrumental_np))) if instrumental_np.size else 0.0
        if max_val > 1.0:
            instrumental_np = instrumental_np / max_val

        final_path = output_root / "no_vocals.wav"
        sf.write(str(final_path), instrumental_np.T, model.samplerate)
    except Exception as exc:
        error_text = str(exc).strip()
        print(f"[DEMUCS ERROR] Full error:\n{error_text}", flush=True)
        snippet = error_text[:300] if error_text else "Demucs failed."
        raise HTTPException(status_code=500, detail=snippet) from exc

    duration = probe_duration_label(input_path)
    
    # Extract lyrics automatically using Whisper
    whisper_input_path = output_root / "vocals.wav"
    if not whisper_input_path.exists():
        whisper_input_path = input_path

    extracted_lrc = extract_lyrics_with_timestamps(whisper_input_path)
    preview = build_preview(extracted_lrc)
    
    catalog_item = {
        "id": job_id,
        "title": title.strip() or Path(safe_filename).stem,
        "artist": artist.strip() or "Artista desconocido",
        "bpm": 0,
        "duration": duration,
        "lrcPreview": preview,
        "tags": normalize_tags(tags),
        "lrc": extracted_lrc,
        "videoUrl": f"/uploads/{job_id}/{safe_filename}",
        "instrumentalUrl": f"/files/{job_id}/no_vocals.wav",
    }

    catalog = load_catalog()
    catalog.insert(0, catalog_item)
    save_catalog(catalog)

    return {
        "job_id": job_id,
        "download_url": catalog_item["instrumentalUrl"],
        "song": catalog_item,
    }


@app.get("/uploads/{job_id}/{filename}")
def get_upload_file(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / f"{job_id}-{safe_name}"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(file_path)


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / job_id / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(file_path)
