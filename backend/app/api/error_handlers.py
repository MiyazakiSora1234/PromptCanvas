"""Exception handlers that turn every failure into the ``{"error": {...}}`` JSON shape."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import AppError, FieldError, HttpError, InvalidInputError
from .schemas import ErrorDetail, ErrorResponse, FieldErrorModel

logger = logging.getLogger("promptcanvas")

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


def request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def convert_validation_errors(errors: Iterable[Any]) -> list[FieldError]:
    """Map Pydantic's (English) validation errors to per-field Japanese messages."""
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


def error_response(request: Request, exc: AppError) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            fields=[FieldErrorModel(field=f.field, message=f.message) for f in exc.fields],
            request_id=request_id(request),
        )
    )
    headers = {"Retry-After": "10"} if exc.status_code in (429, 503) else None
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)


async def _handle_app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return error_response(request, exc)


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(request, InvalidInputError(fields=convert_validation_errors(exc.errors())))


async def _handle_http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return error_response(request, HttpError(exc.status_code))


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error (request_id=%s)", request_id(request))
    return error_response(request, AppError())


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)
    app.add_exception_handler(Exception, _handle_unexpected)
