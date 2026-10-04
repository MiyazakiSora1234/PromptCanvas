"""HTTP routes under ``/api``. Shared objects are read from ``app.state`` (set up in main.py)."""

from __future__ import annotations

import base64
import logging
import time
from typing import Annotated, Any, cast

import anyio.to_thread
from fastapi import APIRouter, Depends, Path, Request, Response

from .catalog import Catalog
from .config import Settings
from .error_handlers import request_id
from .errors import GenerationCancelledError, classify_generation_error
from .generator import ImageGenerator
from .jobs import CancelRegistry
from .limiter import ConcurrencyLimiter
from .schemas import (
    JOB_ID_PATTERN,
    CancelResponse,
    ConfigResponse,
    ErrorResponse,
    GeneratedImageModel,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    QueueInfo,
    build_params,
)

logger = logging.getLogger("promptcanvas")

router = APIRouter(prefix="/api")


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_catalog(request: Request) -> Catalog:
    return cast(Catalog, request.app.state.catalog)


def get_generator(request: Request) -> ImageGenerator:
    return cast(ImageGenerator, request.app.state.generator)


def get_limiter(request: Request) -> ConcurrencyLimiter:
    return cast(ConcurrencyLimiter, request.app.state.limiter)


def get_jobs(request: Request) -> CancelRegistry:
    return cast(CancelRegistry, request.app.state.jobs)


SettingsDep = Annotated[Settings, Depends(get_settings)]
CatalogDep = Annotated[Catalog, Depends(get_catalog)]
GeneratorDep = Annotated[ImageGenerator, Depends(get_generator)]
LimiterDep = Annotated[ConcurrencyLimiter, Depends(get_limiter)]
JobsDep = Annotated[CancelRegistry, Depends(get_jobs)]

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (409, 422, 429, 500, 503)
}


@router.get("/health", response_model=HealthResponse)
async def health(generator: GeneratorDep, limiter: LimiterDep) -> HealthResponse:
    status = generator.status
    return HealthResponse(
        status=status.state.value,
        model=status.model,
        cached_models=sorted(generator.cached_models),
        identity_cached=generator.identity_cached,
        translator_cached=generator.translator_cached,
        device=status.device,
        dtype=status.dtype,
        message=status.message,
        queue=QueueInfo(running=limiter.running, waiting=limiter.waiting, max_waiting=limiter.max_waiting),
    )


@router.get("/config", response_model=ConfigResponse)
async def config(settings: SettingsDep, catalog: CatalogDep) -> ConfigResponse:
    return ConfigResponse.build(settings, catalog)


@router.post("/jobs/{job_id}/cancel", response_model=CancelResponse)
async def cancel_job(job_id: Annotated[str, Path(pattern=JOB_ID_PATTERN)], jobs: JobsDep) -> CancelResponse:
    """Stop a running (or queued) generation. It ends within one denoising step with a ``cancelled`` error."""
    return CancelResponse(cancelled=jobs.cancel(job_id))


@router.post("/generate", response_model=GenerateResponse, responses=_ERROR_RESPONSES)
async def generate(
    body: GenerateRequest,
    request: Request,
    response: Response,
    settings: SettingsDep,
    catalog: CatalogDep,
    generator: GeneratorDep,
    limiter: LimiterDep,
    jobs: JobsDep,
) -> GenerateResponse:
    with jobs.register(body.job_id) as cancel:
        # Decoding an uploaded image is CPU work; keep it off the event loop.
        params = await anyio.to_thread.run_sync(build_params, body, settings, catalog)
        generator.status.ensure_ready()

        async with limiter.slot():
            if cancel.is_set():  # cancelled while waiting in the queue
                raise GenerationCancelledError()
            started = time.perf_counter()
            try:
                result = await anyio.to_thread.run_sync(generator.generate, params, cancel)
            except Exception as exc:
                app_error = classify_generation_error(exc)
                if app_error is not exc:
                    logger.exception("Generation failed as %s (request_id=%s)", app_error.code, request_id(request))
                elif isinstance(app_error, GenerationCancelledError):
                    logger.info("Generation cancelled (request_id=%s)", request_id(request))
                raise app_error from None
            elapsed_ms = int((time.perf_counter() - started) * 1000)

    # Prompts may be private: log only their size.
    logger.info(
        "Generated %d image(s) model=%s mode=%s %dx%d steps=%d scheduler=%s loras=%s in %dms "
        "(prompt_len=%d, request_id=%s)",
        len(result.images),
        params.model.id,
        "reference" if params.uses_reference else "img2img" if params.init_image is not None else "txt2img",
        params.width,
        params.height,
        params.num_inference_steps,
        params.scheduler,
        ",".join(lora.id for lora, _ in params.loras) or "-",
        elapsed_ms,
        len(params.prompt),
        request_id(request),
    )
    response.headers["Cache-Control"] = "no-store"
    return GenerateResponse(
        images=[
            GeneratedImageModel(seed=img.seed, mime_type=img.mime_type, data=base64.b64encode(img.data).decode())
            for img in result.images
        ],
        model=params.model.id,
        scheduler=params.scheduler,
        style=params.style,
        width=params.width,
        height=params.height,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        output_format=params.output_format,
        elapsed_ms=elapsed_ms,
        filtered_count=result.filtered_count,
        translated_prompt=result.translated_prompt,
        translated_negative_prompt=result.translated_negative_prompt,
    )
