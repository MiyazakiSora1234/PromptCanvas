"""Application factory. Start with: ``uvicorn app.main:create_app --factory`` (from ``backend/``)."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import api
from .catalog import Catalog, load_catalog
from .config import Settings
from .error_handlers import register_error_handlers
from .generator import DiffusersGenerator, ImageGenerator
from .limiter import ConcurrencyLimiter

logger = logging.getLogger("promptcanvas")

EXPOSED_HEADERS = ["X-Request-ID"]


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # huggingface_hub logs every HTTP request at INFO via httpx; too noisy during model download.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def _add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    generator: ImageGenerator = app.state.generator
    # The limiter's asyncio primitives must be created inside the running loop.
    app.state.limiter = ConcurrencyLimiter(
        max_waiting=settings.max_queue_size,
        timeout_seconds=settings.queue_timeout_seconds,
    )
    # Load once at startup, in the background so the UI can show "loading"
    # (first run downloads several GB). Daemon thread: don't block shutdown.
    threading.Thread(target=generator.load, name="model-loader", daemon=True).start()
    yield


def create_app(
    settings: Settings | None = None,
    generator: ImageGenerator | None = None,
    catalog: Catalog | None = None,
) -> FastAPI:
    settings = settings or Settings()
    _configure_logging(settings.log_level)
    catalog = catalog or load_catalog(settings)

    app = FastAPI(title="PromptCanvas API", version="0.2.0", lifespan=_lifespan)
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.generator = generator or DiffusersGenerator(settings, catalog)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            expose_headers=EXPOSED_HEADERS,
        )
    app.middleware("http")(_add_request_id)
    register_error_handlers(app)
    app.include_router(api.router)

    # Mounted last so it never shadows /api routes.
    if settings.serve_frontend:
        if settings.frontend_dir.is_dir():
            app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
        else:
            logger.warning(
                "Frontend build not found at %s; serving the API only. Run `npm run build` in frontend/ "
                "(or `make frontend-build`).",
                settings.frontend_dir,
            )

    return app
