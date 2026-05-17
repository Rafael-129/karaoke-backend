from __future__ import annotations

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.dependencies import get_storage_service
from app.core.logging import configure_logging
from app.middleware.errors import add_error_handlers
from app.middleware.logging import add_logging_middleware
from app.routes import catalog, jobs, separate


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    # Override temporary directory to E: drive because C: drive is 100% full (0.00 GB free).
    # This prevents Starlette's body parsing and SpooledTemporaryFile write failures.
    import tempfile
    temp_dir = settings.data_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_dir)

    app = FastAPI(title=settings.app_title)
    add_logging_middleware(app)
    add_error_handlers(app)

    app.include_router(catalog.router)
    app.include_router(separate.router)
    app.include_router(jobs.router)

    @app.on_event("startup")
    def on_startup() -> None:
        get_storage_service().ensure_buckets()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
