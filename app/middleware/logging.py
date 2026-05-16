from __future__ import annotations

import logging
from time import perf_counter

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Simple request logging middleware.

logger = logging.getLogger("app.request")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def add_logging_middleware(app: FastAPI) -> None:
    app.add_middleware(LoggingMiddleware)
