"""FastAPI application. Start with: ``uvicorn app.main:create_app --factory`` (from ``backend/``)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio.to_thread
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .errors import (
    AppError,
    FieldError,
    InvalidInputError,
    ModelLoadingError,
    ModelUnavailableError,
    classify_generation_error,
)
from .generator import DiffusersGenerator, ImageGenerator, ModelState, ModelStatus
from .limiter import ConcurrencyLimiter
from .schemas import (
    ConfigResponse,
    ErrorDetail,
    ErrorResponse,
    FieldErrorModel,
    GenerateRequest,
    HealthResponse,
    QueueInfo,
    build_params,
)

logger = logging.getLogger("promptcanvas")

EXPOSED_HEADERS = ["X-Seed", "X-Generation-Time-Ms", "X-Request-ID"]

_VALIDATION_MESSAGES = {
    "missing": "必須項目です。",
    "string_type": "文字列で入力してください。",
    "int_type": "整数で入力してください。",
    "int_parsing": "整数で入力してください。",
    "int_from_float": "整数で入力してください。",
    "float_type": "数値で入力してください。",
    "float_parsing": "数値で入力してください。",
    "finite_number": "有限の数値で入力してください。",
    "extra_forbidden": "不明な項目です。",
    "json_invalid": "JSONの形式が正しくありません。",
    "model_attributes_type": "リクエスト本文はJSONオブジェクトで送信してください。",
    "dict_type": "リクエスト本文はJSONオブジェクトで送信してください。",
}


def _convert_validation_errors(errors: Any) -> list[FieldError]:
    result: list[FieldError] = []
    for err in errors:
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        field = loc[-1] if loc else "body"
        err_type = err.get("type", "")
        if err_type == "string_too_long":
            max_length = err.get("ctx", {}).get("max_length")
            message = f"{max_length}文字以内で入力してください。"
        else:
            message = _VALIDATION_MESSAGES.get(err_type, "値が正しくありません。")
        result.append(FieldError(field, message))
    return result


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(request: Request, exc: AppError) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            fields=[FieldErrorModel(field=f.field, message=f.message) for f in exc.fields],
            request_id=_request_id(request),
        )
    )
    headers = {"Retry-After": "10"} if exc.status_code in (429, 503) else None
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)


def _ensure_ready(status: ModelStatus) -> None:
    if status.state in (ModelState.NOT_LOADED, ModelState.LOADING):
        raise ModelLoadingError()
    if status.state is ModelState.FAILED:
        detail = f"（{status.message}）" if status.message else ""
        raise ModelUnavailableError(ModelUnavailableError.default_message + detail)


def create_app(settings: Settings | None = None, generator: ImageGenerator | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # huggingface_hub logs every HTTP request at INFO via httpx; too noisy during model download.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    gen: ImageGenerator = generator or DiffusersGenerator(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.limiter = ConcurrencyLimiter(
            max_waiting=settings.max_queue_size,
            timeout_seconds=settings.queue_timeout_seconds,
        )
        # Load once at startup, in the background so the UI can show "loading"
        # (first run downloads several GB). Daemon thread: don't block shutdown.
        threading.Thread(target=gen.load, name="model-loader", daemon=True).start()
        yield

    app = FastAPI(title="PromptCanvas API", version="0.1.0", lifespan=lifespan)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            expose_headers=EXPOSED_HEADERS,
        )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(request, InvalidInputError(fields=_convert_validation_errors(exc.errors())))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error = AppError("リクエストされたリソースが見つからないか、許可されていない操作です。")
        error.status_code = exc.status_code
        error.code = "not_found" if exc.status_code == 404 else "http_error"
        return _error_response(request, error)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error (request_id=%s)", _request_id(request))
        return _error_response(request, AppError())

    error_responses: dict[int | str, dict[str, Any]] = {
        code: {"model": ErrorResponse} for code in (422, 429, 500, 503)
    }

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        status = gen.status
        limiter: ConcurrencyLimiter = request.app.state.limiter
        return HealthResponse(
            status=status.state.value,
            model_id=settings.model_id,
            device=status.device,
            dtype=status.dtype,
            message=status.message,
            queue=QueueInfo(running=limiter.running, waiting=limiter.waiting, max_waiting=limiter.max_waiting),
        )

    @app.get("/api/config", response_model=ConfigResponse)
    async def config() -> ConfigResponse:
        return ConfigResponse.from_settings(settings)

    @app.post(
        "/api/generate",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}, "description": "生成されたPNG画像"}, **error_responses},
    )
    async def generate(body: GenerateRequest, request: Request) -> Response:
        params = build_params(body, settings)
        _ensure_ready(gen.status)
        limiter: ConcurrencyLimiter = request.app.state.limiter

        async with limiter.slot():
            started = time.perf_counter()
            try:
                result = await anyio.to_thread.run_sync(gen.generate, params)
            except Exception as exc:
                app_error = classify_generation_error(exc)
                if app_error is not exc:
                    logger.exception(
                        "Generation failed as %s (request_id=%s)", app_error.code, _request_id(request)
                    )
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
            _request_id(request),
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

    if settings.serve_frontend:
        if settings.frontend_dir.is_dir():
            app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
        else:
            logger.warning("Frontend directory not found: %s (serving API only)", settings.frontend_dir)

    return app
