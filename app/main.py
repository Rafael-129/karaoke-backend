from __future__ import annotations

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.dependencies import get_storage_service
from app.core.logging import configure_logging
from app.middleware.errors import add_error_handlers
from app.middleware.logging import add_logging_middleware
import os


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    # Override temporary directory to E: drive because C: drive is 100% full (0.00 GB free).
    # This prevents Starlette's body parsing and SpooledTemporaryFile write failures.
    import tempfile
    temp_dir = settings.data_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_dir)

    # Configure torch/hub/cache directories. Prefer explicit env vars (e.g. set in
    # .env) so users can point caches to another drive (D:) with more space.
    torch_home = os.getenv("TORCH_HOME") or str(settings.data_dir / ".torch")
    xdg_cache = os.getenv("XDG_CACHE_HOME") or str(settings.data_dir / ".cache")

    # Ensure directories exist and then set env vars so downstream imports
    # (demucs/torch) write caches to these locations instead of the user's profile.
    try:
        os.makedirs(torch_home, exist_ok=True)
        os.makedirs(xdg_cache, exist_ok=True)
    except Exception:
        # Best effort; if creation fails, fall back to not overriding env vars.
        pass

    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)

    # Import routes after environment/cache dirs are configured so any imports
    # that trigger model downloads use the configured cache location.
    from app.routes import catalog, jobs, separate

    app = FastAPI(title=settings.app_title)
    add_logging_middleware(app)
    add_error_handlers(app)

    app.include_router(catalog.router)
    app.include_router(separate.router)
    app.include_router(jobs.router)

    @app.on_event("startup")
    def on_startup() -> None:
        get_storage_service().ensure_buckets()
        # Optionally preload ML models into memory (controlled by PRELOAD_MODELS).
        try:
            from app.services.audio.model_manager import preload_models

            if settings.preload_models:
                preload_models(settings)
        except Exception:
            # Keep startup resilient if model preload fails or Demucs/Whisper not installed.
            pass

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/internal/models")
    def internal_models() -> dict:
        """Returns configured model names and whether they're loaded in this process.

        Notes: This endpoint intentionally does NOT load models if they're not
        already cached. It inspects the lru_cache statistics to determine
        whether a model is present in the cache.
        """
        from app.services.audio import model_manager

        whisper_cfg = {
            "configured": settings.whisper_model,
            "device": settings.whisper_device,
            "loaded": bool(getattr(model_manager.get_whisper_model, "cache_info")().currsize),
        }

        demucs_cfg = {
            "configured": settings.demucs_model,
            "device": settings.demucs_device,
            "loaded": bool(getattr(model_manager.get_demucs_model, "cache_info")().currsize),
        }

        # If models are loaded, return a small identifying string; do NOT
        # call loaders if not loaded to avoid allocating memory here.
        if whisper_cfg["loaded"]:
            try:
                model = model_manager.get_whisper_model(settings.whisper_model, str(settings.whisper_download_root), device=settings.whisper_device)
                whisper_cfg["repr"] = type(model).__name__
            except Exception:
                whisper_cfg["repr"] = "(error reading instance)"

        if demucs_cfg["loaded"]:
            try:
                model = model_manager.get_demucs_model(settings.demucs_model, device=settings.demucs_device)
                demucs_cfg["repr"] = type(model).__name__
            except Exception:
                demucs_cfg["repr"] = "(error reading instance)"

        return {"whisper": whisper_cfg, "demucs": demucs_cfg}

    return app


app = create_app()
