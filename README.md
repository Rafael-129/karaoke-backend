# karaoke-backend

FastAPI service that accepts audio/video uploads, runs Demucs to extract the
instrumental track, and returns a download URL.

## Requirements

- Python 3.10+
- ffmpeg installed on the host

If you deploy with Railway/Nixpacks, include `ffmpeg` in the image via `nixpacks.toml`.

## Local run

Create and activate a virtual environment, then install deps:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the API:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

- `GET /health` -> `{ "status": "ok" }`
- `POST /separate` -> returns `{ job_id, download_url }`
- `GET /files/{job_id}/{filename}` -> returns the processed file

## Notes

- Demucs and ffmpeg are required for processing.
- Video uploads need ffmpeg to extract audio before Whisper and Demucs run.
- Outputs are stored under `data/outputs/`.
