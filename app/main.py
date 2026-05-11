from __future__ import annotations

from functools import lru_cache
import io
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import soundfile as sf
import whisper
import numpy as np

from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from supabase import create_client

# Load environment variables
load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_SECRET = os.getenv("SUPABASE_SECRET", "").strip()

# Require SUPABASE_URL and at least one key (prefer the service role secret for server-side operations)
if not SUPABASE_URL or (not SUPABASE_KEY and not SUPABASE_SECRET):
    raise ValueError("SUPABASE_URL and SUPABASE_KEY or SUPABASE_SECRET must be set in environment variables")

# Use the service role (secret) key for backend/admin operations when available.
if SUPABASE_SECRET:
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET)
else:
    # Fallback to the publishable key (less privileged). Creation of buckets/policies may fail.
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Demucs and Whisper configuration
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "es")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))
WHISPER_DOWNLOAD_ROOT = Path(os.getenv("WHISPER_DOWNLOAD_ROOT", str(DATA_DIR / "whisper-models")))

app = FastAPI(title="karaoke-backend")

# Storage bucket names
UPLOADS_BUCKET = "uploads"
OUTPUTS_BUCKET = "outputs"


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


def probe_duration_label(file_bytes: bytes, filename: str) -> str:
    """Probe audio duration from bytes using ffprobe."""
    temp_path = DATA_DIR / f"temp-{uuid.uuid4().hex}-{filename}"
    try:
        temp_path.write_bytes(file_bytes)
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(temp_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return format_duration(float(result.stdout.strip()))
    except Exception:
        return "--:--"
    finally:
        temp_path.unlink(missing_ok=True)


def normalize_tags(raw_tags: str) -> list[str]:
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags or ["subido"]


def extract_lyrics_with_timestamps(audio_bytes: bytes, filename: str) -> str:
    """Extract lyrics from audio bytes using Whisper and generate LRC format with timestamps."""
    temp_path = DATA_DIR / f"temp-audio-{uuid.uuid4().hex}.wav"
    try:
        temp_path.write_bytes(audio_bytes)
        model = load_whisper_model()
        result = model.transcribe(
            str(temp_path),
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
    finally:
        temp_path.unlink(missing_ok=True)



@app.on_event("startup")
def on_startup() -> None:
    """Initialize Supabase buckets and database."""
    try:
        # Create storage buckets if they don't exist
        for bucket in [UPLOADS_BUCKET, OUTPUTS_BUCKET]:
            try:
                supabase.storage.get_bucket(bucket)
            except Exception:
                supabase.storage.create_bucket(bucket)
                print(f"✅ Created storage bucket: {bucket}", flush=True)
    except Exception as e:
        print(f"⚠️ Startup warning (buckets may already exist): {e}", flush=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/catalog")
def list_catalog() -> list[dict[str, Any]]:
    try:
        response = supabase.table("songs").select("*").order("created_at", desc=True).execute()
        songs = response.data if response.data else []
        # Convert database format to API format
        return [
            {
                "id": song["job_id"],
                "title": song["title"],
                "artist": song["artist"],
                "bpm": song["bpm"],
                "duration": song["duration"],
                "lrcPreview": song["lrc_preview"],
                "lrc": song["lrc"],
                "tags": song["tags"],
                "videoUrl": song["video_url"],
                "instrumentalUrl": song["instrumental_url"],
            }
            for song in songs
        ]
    except Exception as e:
        print(f"[CATALOG ERROR] {str(e)}", flush=True)
        return []


@app.get("/catalog/{song_id}")
def get_catalog_song(song_id: str) -> dict[str, Any]:
    try:
        response = supabase.table("songs").select("*").eq("job_id", song_id).single().execute()
        song = response.data
        return {
            "id": song["job_id"],
            "title": song["title"],
            "artist": song["artist"],
            "bpm": song["bpm"],
            "duration": song["duration"],
            "lrcPreview": song["lrc_preview"],
            "lrc": song["lrc"],
            "tags": song["tags"],
            "videoUrl": song["video_url"],
            "instrumentalUrl": song["instrumental_url"],
        }
    except Exception:
        raise HTTPException(status_code=404, detail="Song not found.")


@app.delete("/catalog")
def reset_catalog() -> dict[str, str]:
    try:
        supabase.table("songs").delete().neq("job_id", "").execute()
        return {"status": "ok", "message": "Catalog cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing catalog: {str(e)}")


@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(...),
    lyrics: str = Form(""),
    tags: str = Form(""),
) -> dict[str, Any]:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file.")

    job_id = uuid.uuid4().hex
    file_content = await file.read()
    await file.close()

    safe_filename = Path(file.filename).name if file.filename else "upload.bin"

    try:
        # Probe duration from file bytes
        duration = probe_duration_label(file_content, safe_filename)

        # Upload original file to Supabase Storage
        file_path = f"{job_id}/{safe_filename}"
        supabase.storage.from_(UPLOADS_BUCKET).upload(file_path, file_content)

        # Separate audio locally
        model = load_demucs_model()

        # Save file temporarily for processing
        temp_input = DATA_DIR / f"temp-input-{job_id}.bin"
        temp_input.write_bytes(file_content)

        try:
            mix = load_track(temp_input, model.audio_channels, model.samplerate).unsqueeze(0)
            sources = apply_model(model, mix, device="cpu")

            try:
                print(f"[DEMUCS] sources names={model.sources}", flush=True)
                print(f"[DEMUCS] sources shape={getattr(sources, 'shape', None)}", flush=True)
            except Exception:
                pass

            try:
                sources_np = sources.detach().cpu().numpy()
            except Exception:
                sources_np = sources.cpu().numpy()

            batch_idx = 0

            # Build instrumental
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

            # Normalize instrumental
            max_val = float(np.max(np.abs(instrumental_np))) if instrumental_np.size else 0.0
            if max_val > 1.0:
                instrumental_np = instrumental_np / max_val

            # Save instrumental to bytes
            instrumental_buffer = io.BytesIO()
            sf.write(instrumental_buffer, instrumental_np.T, model.samplerate, format="WAV")
            instrumental_bytes = instrumental_buffer.getvalue()

            # Upload instrumental to Storage
            instrumental_path = f"{job_id}/no_vocals.wav"
            supabase.storage.from_(OUTPUTS_BUCKET).upload(instrumental_path, instrumental_bytes)

            # Extract vocals for transcription
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
            else:
                vocals_bytes = file_content

            # Extract lyrics
            extracted_lrc = extract_lyrics_with_timestamps(vocals_bytes, "vocals.wav")
            preview = build_preview(extracted_lrc)

            # Save to Supabase database
            song_data = {
                "job_id": job_id,
                "title": title.strip() or Path(safe_filename).stem,
                "artist": artist.strip() or "Artista desconocido",
                "bpm": 0,
                "duration": duration,
                "lrc_preview": preview,
                "lrc": extracted_lrc,
                "tags": normalize_tags(tags),
                "video_url": f"/uploads/{job_id}/{safe_filename}",
                "instrumental_url": f"/files/{job_id}/no_vocals.wav",
            }

            supabase.table("songs").insert(song_data).execute()

            return {
                "job_id": job_id,
                "download_url": song_data["instrumental_url"],
                "song": {
                    "id": job_id,
                    "title": song_data["title"],
                    "artist": song_data["artist"],
                    "bpm": song_data["bpm"],
                    "duration": song_data["duration"],
                    "lrcPreview": song_data["lrc_preview"],
                    "lrc": song_data["lrc"],
                    "tags": song_data["tags"],
                    "videoUrl": song_data["video_url"],
                    "instrumentalUrl": song_data["instrumental_url"],
                },
            }

        finally:
            temp_input.unlink(missing_ok=True)

    except Exception as exc:
        error_text = str(exc).strip()
        print(f"[SEPARATE ERROR] Full error:\n{error_text}", flush=True)
        snippet = error_text[:300] if error_text else "Separation failed."
        raise HTTPException(status_code=500, detail=snippet) from exc


@app.get("/uploads/{job_id}/{filename}")
def get_upload_file(job_id: str, filename: str) -> StreamingResponse:
    try:
        safe_name = Path(filename).name
        file_path = f"{job_id}/{safe_name}"
        response = supabase.storage.from_(UPLOADS_BUCKET).download(file_path)
        return StreamingResponse(io.BytesIO(response), media_type="application/octet-stream")
    except Exception:
        raise HTTPException(status_code=404, detail="File not found.")


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str) -> StreamingResponse:
    try:
        safe_name = Path(filename).name
        file_path = f"{job_id}/{safe_name}"
        response = supabase.storage.from_(OUTPUTS_BUCKET).download(file_path)
        return StreamingResponse(io.BytesIO(response), media_type="audio/wav")
    except Exception:
        raise HTTPException(status_code=404, detail="File not found.")
