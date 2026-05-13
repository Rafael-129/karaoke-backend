from __future__ import annotations

import asyncio
from functools import lru_cache
import io
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import soundfile as sf
import whisper
import numpy as np
import torch
from typing import Any, Optional, Dict

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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
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


# Global status tracker for background jobs
jobs_status: Dict[str, Dict[str, Any]] = {}


def audio_to_mp3_bytes(audio_np: np.ndarray, samplerate: int) -> bytes:
    """Converts a numpy audio array to MP3 bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        sf.write(wav_path, audio_np.T, samplerate, format="WAV")
    
    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
            check=True,
            capture_output=True
        )
        with open(mp3_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[CONVERT ERROR] Failed to convert to mp3: {e}")
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        except Exception:
            pass


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


def extract_lyrics_with_timestamps(audio_bytes: bytes, filename: str, title: str = "", artist: str = "") -> str:
    """Extract lyrics from audio bytes using Whisper and generate LRC format with timestamps."""
    temp_path = DATA_DIR / f"temp-audio-{uuid.uuid4().hex}.wav"
    try:
        temp_path.write_bytes(audio_bytes)
        model = load_whisper_model()
        
        # Inyectamos contexto a la IA para mejorar la precisión con nombres propios y estilo
        context_prompt = "Transcribe la letra de una canción en español. "
        if title or artist:
            context_prompt += f"La canción se llama '{title}' y es de '{artist}'. "
        context_prompt += "Corrige palabras lo mejor posible sin resumir ni traducir."

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
            word_timestamps=True,
            initial_prompt=context_prompt,
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


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_status:
        # Check if it already exists in the database as a fallback
        try:
            res = supabase.table("songs").select("*").eq("job_id", job_id).execute()
            if res.data:
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "progress": 100,
                    "message": "Completado",
                    "song": res.data[0]
                }
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_status[job_id]


@app.post("/separate")
async def separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    lyrics: str = Form(""),
    tags: str = Form(""),
    job_id: Optional[str] = Form(None),
    chunk_index: Optional[int] = Form(None),
    total_chunks: Optional[int] = Form(None),
) -> dict[str, Any]:
    print(f"[DEBUG] /separate hit: job_id={job_id}, chunk={chunk_index}/{total_chunks}", flush=True)
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
            
            # Start background processing
            jobs_status[job_id] = {
                "job_id": job_id,
                "status": "processing",
                "progress": 10,
                "message": "Ensamblando archivos..."
            }
            
            background_tasks.add_task(
                process_audio_task,
                job_id=job_id,
                total_chunks=total_chunks,
                original_filename=original_filename,
                title=title,
                artist=artist,
                lyrics=lyrics,
                tags=tags
            )
            
            return {
                "job_id": job_id,
                "status": "processing",
                "message": "Procesamiento iniciado en segundo plano"
            }
        
        # Single file upload (not chunked)
        file_content = chunk_file_content
        background_tasks.add_task(
            process_audio_task,
            job_id=job_id,
            file_content=file_content,
            original_filename=original_filename,
            title=title,
            artist=artist,
            lyrics=lyrics,
            tags=tags
        )
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Procesamiento iniciado"
        }

    except Exception as e:
        print(f"[UPLOAD ERROR] {str(e)}", flush=True)
        jobs_status[job_id] = {"status": "error", "message": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


async def process_audio_task(
    job_id: str,
    total_chunks: Optional[int] = None,
    file_content: Optional[bytes] = None,
    original_filename: str = "upload.bin",
    title: Optional[str] = None,
    artist: Optional[str] = None,
    lyrics: str = "",
    tags: str = ""
):
    try:
        if total_chunks:
            chunk_dir = CHUNKS_DIR / job_id
            assembled_path = chunk_dir / "assembled.bin"
            with open(assembled_path, "wb") as f:
                for i in range(total_chunks):
                    chunk_file = chunk_dir / f"chunk_{i:06d}"
                    f.write(chunk_file.read_bytes())
                    chunk_file.unlink() # Cleanup chunks
            file_content = assembled_path.read_bytes()
            assembled_path.unlink() # Cleanup assembled file
            try:
                chunk_dir.rmdir() # Cleanup dir
            except Exception:
                pass
        
        if not file_content:
            jobs_status[job_id] = {"status": "error", "message": "No file content to process"}
            return

        # Start Processing
        jobs_status[job_id] = {"status": "processing", "progress": 20, "message": "Quitando la voz con IA..."}
        
        # Process audio with Demucs
        model = load_demucs_model()
        
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

            # Save instrumental to bytes as MP3 to avoid Payload Too Large errors
            print(f"[SEPARATE] Converting instrumental to MP3...")
            instrumental_bytes = audio_to_mp3_bytes(instrumental_np, model.samplerate)

            # Upload instrumental to Storage
            instrumental_path = f"{job_id}/no_vocals.mp3"
            print(f"[SEPARATE] Uploading instrumental ({len(instrumental_bytes)} bytes)...")
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
            jobs_status[job_id] = {"status": "processing", "progress": 70, "message": "Sincronizando letras con IA..."}
            extracted_lrc = extract_lyrics_with_timestamps(vocals_bytes, "vocals.wav", title or "", artist or "")
            preview = build_preview(extracted_lrc)

            # Save to Supabase database
            jobs_status[job_id] = {"status": "processing", "progress": 90, "message": "Guardando en el catálogo..."}
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
                "instrumental_url": f"/files/{job_id}/no_vocals.mp3",
            }

            supabase.table("songs").insert(song_data).execute()

            jobs_status[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "progress": 100,
                "message": "¡Todo listo!",
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
        jobs_status[job_id] = {"status": "error", "message": error_text[:300]}


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
        
        # Determine media type based on extension
        ext = Path(safe_name).suffix.lower()
        media_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
        
        return StreamingResponse(io.BytesIO(response), media_type=media_type)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found.")
