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
