from __future__ import annotations

import asyncio
from functools import lru_cache
import io
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import soundfile as sf
import whisper
import numpy as np
import torch

from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model
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

# Chunk storage for parallel uploads
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)


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



def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert any audio/video file to WAV (PCM s16le stereo 44100Hz) via ffmpeg subprocess."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed (code {result.returncode}):\n{result.stderr[-2000:]}"
        )


def load_audio_for_demucs(wav_path: Path, target_channels: int, target_samplerate: int) -> torch.Tensor:
    """
    Load a WAV file with soundfile (no torchaudio/torchcodec needed) and
    convert it to the tensor format expected by demucs: float32 [C, T].
    Resampling and channel conversion are handled by demucs.audio.convert_audio.
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # soundfile returns [T, C] — transpose to [C, T]
    wav = torch.from_numpy(data.T)  # shape: [C, T]
    wav = convert_audio(wav, sr, target_samplerate, target_channels)
    return wav


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


@app.delete("/catalog/{song_id}")
def delete_catalog_song(song_id: str) -> dict[str, str]:
    try:
        song_response = (
            supabase.table("songs")
            .select("job_id,video_url,instrumental_url")
            .eq("job_id", song_id)
            .single()
            .execute()
        )
        song = song_response.data
    except Exception:
        raise HTTPException(status_code=404, detail="Song not found.")

    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")

    try:
        supabase.table("songs").delete().eq("job_id", song_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting song: {str(exc)}")

    storage_targets: list[tuple[str, str]] = []
    video_url = song.get("video_url")
    instrumental_url = song.get("instrumental_url")

    if isinstance(video_url, str) and video_url.startswith("/uploads/"):
        uploads_path = video_url.removeprefix("/uploads/")
        if uploads_path:
            storage_targets.append((UPLOADS_BUCKET, uploads_path))

    if isinstance(instrumental_url, str) and instrumental_url.startswith("/files/"):
        outputs_path = instrumental_url.removeprefix("/files/")
        if outputs_path:
            storage_targets.append((OUTPUTS_BUCKET, outputs_path))

    for bucket_name, file_path in storage_targets:
        try:
            supabase.storage.from_(bucket_name).remove([file_path])
        except Exception as cleanup_error:
            print(
                f"[CATALOG CLEANUP WARNING] Could not delete {bucket_name}/{file_path}: {cleanup_error}",
                flush=True,
            )

    return {"status": "ok", "message": "Song deleted"}


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
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    lyrics: str = Form(""),
    tags: str = Form(""),
    job_id: Optional[str] = Form(None),
    chunk_index: Optional[int] = Form(None),
    total_chunks: Optional[int] = Form(None),
) -> dict[str, Any]:
    """
    Handle file upload with optional chunked transfer.
    Chunks are reassembled before processing.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Missing file.")

    # Use provided job_id or generate one for single-file uploads
    if not job_id:
        job_id = uuid.uuid4().hex

    chunk_file_content = await file.read()
    await file.close()
    safe_filename = Path(file.filename).name if file.filename else "upload.bin"
    
    # Strip .chunk_N suffix if present (from chunked uploads)
    original_filename = re.sub(r"\.chunk_\d+$", "", safe_filename)
    if not original_filename:
        original_filename = "upload.bin"

    try:
        # Handle chunked upload
        if chunk_index is not None and total_chunks is not None:
            # Store chunk
            chunk_dir = CHUNKS_DIR / job_id
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / f"chunk_{chunk_index:06d}"
            chunk_path.write_bytes(chunk_file_content)
            print(f"[CHUNKS] Stored chunk {chunk_index}: {len(chunk_file_content)} bytes to {chunk_path}", flush=True)

            # Check if this is the last chunk
            is_final_chunk = chunk_index == total_chunks - 1
            
            if not is_final_chunk:
                return {
                    "job_id": job_id,
                    "status": "chunk_received",
                    "chunk": chunk_index,
                    "total": total_chunks,
                }

            # Final chunk received - validate and reassemble all chunks
            print(f"[CHUNKS] Final chunk received. Validating all {total_chunks} chunks for job {job_id}", flush=True)
            
            # Check all chunks exist (with retries for network delays)
            missing_chunks = []
            max_retries = 5
            retry_delay = 0.5  # seconds
            
            for attempt in range(max_retries):
                missing_chunks = []
                for i in range(total_chunks):
                    chunk_file = chunk_dir / f"chunk_{i:06d}"
                    if not chunk_file.exists():
                        missing_chunks.append(i)
                    else:
                        file_size = chunk_file.stat().st_size
                        print(f"[CHUNKS] Chunk {i} exists: {file_size} bytes", flush=True)
                
                if not missing_chunks:
                    break
                
                if attempt < max_retries - 1:
                    print(f"[CHUNKS] Missing chunks {missing_chunks}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})", flush=True)
                    await asyncio.sleep(retry_delay)
            
            if missing_chunks:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing chunks: {missing_chunks}. Received {total_chunks - len(missing_chunks)}/{total_chunks}."
                )
            
            # Reassemble all chunks in order
            print(f"[CHUNKS] Assembling {total_chunks} chunks for job {job_id}", flush=True)
            file_content = b""
            for i in range(total_chunks):
                chunk_file = chunk_dir / f"chunk_{i:06d}"
                chunk_data = chunk_file.read_bytes()
                file_content += chunk_data
                chunk_file.unlink()
                print(f"[CHUNKS] Added chunk {i}: {len(chunk_data)} bytes (total: {len(file_content)} bytes)", flush=True)
            
            chunk_dir.rmdir()
            print(f"[CHUNKS] Assembly complete. Final file size: {len(file_content)} bytes", flush=True)

            if not file_content:
                raise HTTPException(status_code=400, detail="Reassembled file is empty.")
        else:
            file_content = chunk_file_content

        # Validate metadata for processing
        if not title or not artist:
            raise HTTPException(status_code=400, detail="Title and artist required.")

        # Probe duration from file bytes
        duration = probe_duration_label(file_content, original_filename)

        # Upload original file to Supabase Storage
        file_path = f"{job_id}/{original_filename}"
        supabase.storage.from_(UPLOADS_BUCKET).upload(file_path, file_content)

        # Separate audio locally
        model = load_demucs_model()

        # Save file temporarily for processing
        original_suffix = Path(original_filename).suffix.lower() or ".bin"
        temp_input = DATA_DIR / f"temp-input-{job_id}{original_suffix}"
        temp_input.write_bytes(file_content)

        # Pre-convert to WAV so demucs/torchaudio can always read it
        temp_wav = DATA_DIR / f"temp-input-{job_id}.wav"
        try:
            convert_to_wav(temp_input, temp_wav)
        except Exception as conv_err:
            print(f"[CONVERT] ffmpeg conversion error: {conv_err}", flush=True)
            # Fallback: try passing the original file directly
            temp_wav = temp_input

        try:
            mix = load_audio_for_demucs(temp_wav, model.audio_channels, model.samplerate).unsqueeze(0)
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
                "title": title.strip() or Path(original_filename).stem,
                "artist": artist.strip() or "Artista desconocido",
                "bpm": 0,
                "duration": duration,
                "lrc_preview": preview,
                "lrc": extracted_lrc,
                "tags": normalize_tags(tags),
                "video_url": f"/uploads/{job_id}/{original_filename}",
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
            if temp_wav != temp_input:
                temp_wav.unlink(missing_ok=True)

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
