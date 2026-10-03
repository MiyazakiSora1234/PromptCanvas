"""HTTP routes under ``/api``. Shared objects are read from ``app.state`` (set up in main.py)."""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any, cast

import anyio.to_thread
from fastapi import APIRouter, Depends, Request, Response

from .config import Settings
from .error_handlers import request_id
from .errors import classify_generation_error
from .generator import ImageGenerator
from .limiter import ConcurrencyLimiter
from .schemas import ConfigResponse, ErrorResponse, GenerateRequest, HealthResponse, QueueInfo, build_params

logger = logging.getLogger("promptcanvas")

router = APIRouter(prefix="/api")


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_generator(request: Request) -> ImageGenerator:
    return cast(ImageGenerator, request.app.state.generator)


def get_limiter(request: Request) -> ConcurrencyLimiter:
    return cast(ConcurrencyLimiter, request.app.state.limiter)


SettingsDep = Annotated[Settings, Depends(get_settings)]
GeneratorDep = Annotated[ImageGenerator, Depends(get_generator)]
LimiterDep = Annotated[ConcurrencyLimiter, Depends(get_limiter)]

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {code: {"model": ErrorResponse} for code in (422, 429, 500, 503)}


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep, generator: GeneratorDep, limiter: LimiterDep) -> HealthResponse:
    status = generator.status
    return HealthResponse(
        status=status.state.value,
        model_id=settings.model_id,
        device=status.device,
        dtype=status.dtype,
        message=status.message,
        queue=QueueInfo(running=limiter.running, waiting=limiter.waiting, max_waiting=limiter.max_waiting),
    )


@router.get("/config", response_model=ConfigResponse)
async def config(settings: SettingsDep) -> ConfigResponse:
    return ConfigResponse.from_settings(settings)


@router.post(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "生成されたPNG画像"}, **_ERROR_RESPONSES},
)
async def generate(
    body: GenerateRequest,
    request: Request,
    settings: SettingsDep,
    generator: GeneratorDep,
    limiter: LimiterDep,
) -> Response:
    params = build_params(body, settings)
    generator.status.ensure_ready()

    async with limiter.slot():
        started = time.perf_counter()
        try:
            result = await anyio.to_thread.run_sync(generator.generate, params)
        except Exception as exc:
            app_error = classify_generation_error(exc)
            if app_error is not exc:
                logger.exception("Generation failed as %s (request_id=%s)", app_error.code, request_id(request))
            raise app_error from None
        elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Prompts may be private: log only their size.
    logger.info(
        "Generated %dx%d steps=%d seed=%d in %dms (prompt_len=%d, request_id=%s)",
        params.width,
        params.height,
        params.num_inference_steps,
        result.seed,
        elapsed_ms,
        len(params.prompt),
        request_id(request),
    )
    return Response(
        content=result.png,
        media_type="image/png",
        headers={
            "X-Seed": str(result.seed),
            "X-Generation-Time-Ms": str(elapsed_ms),
            "Cache-Control": "no-store",
        },
    )
