"""Request/response models and server-side input validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from .config import SEED_MAX, SIZE_MULTIPLE, Settings
from .errors import FieldError, InvalidInputError

# Absolute cap so oversized payloads are rejected before any further work.
_HARD_TEXT_CAP = 10_000


class GenerateRequest(BaseModel):
    """Body of ``POST /api/generate``. Omitted optional fields fall back to server defaults."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(max_length=_HARD_TEXT_CAP)
    negative_prompt: str = Field(default="", max_length=_HARD_TEXT_CAP)
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = Field(default=None, description="未指定ならランダム")


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int | None


def build_params(req: GenerateRequest, settings: Settings) -> GenerationParams:
    """Apply defaults and validate ranges that depend on configuration."""
    errors: list[FieldError] = []

    prompt = req.prompt.strip()
    if not prompt:
        errors.append(FieldError("prompt", "プロンプトを入力してください。"))
    elif len(prompt) > settings.max_prompt_length:
        errors.append(FieldError("prompt", f"{settings.max_prompt_length}文字以内で入力してください。"))

    negative_prompt = req.negative_prompt.strip()
    if len(negative_prompt) > settings.max_prompt_length:
        errors.append(
            FieldError("negative_prompt", f"{settings.max_prompt_length}文字以内で入力してください。")
        )

    width = settings.default_width if req.width is None else req.width
    height = settings.default_height if req.height is None else req.height
    for name, value in (("width", width), ("height", height)):
        if not settings.min_image_size <= value <= settings.max_image_size:
            errors.append(
                FieldError(name, f"{settings.min_image_size}〜{settings.max_image_size}の範囲で指定してください。")
            )
        elif value % SIZE_MULTIPLE:
            errors.append(FieldError(name, f"{SIZE_MULTIPLE}の倍数で指定してください。"))

    steps = settings.default_steps if req.num_inference_steps is None else req.num_inference_steps
    if not 1 <= steps <= settings.max_steps:
        errors.append(FieldError("num_inference_steps", f"1〜{settings.max_steps}の範囲で指定してください。"))

    guidance = settings.default_guidance_scale if req.guidance_scale is None else req.guidance_scale
    if not math.isfinite(guidance) or not 0 <= guidance <= settings.max_guidance_scale:
        errors.append(
            FieldError("guidance_scale", f"0〜{settings.max_guidance_scale:g}の範囲で指定してください。")
        )

    if req.seed is not None and not 0 <= req.seed <= SEED_MAX:
        errors.append(FieldError("seed", f"0〜{SEED_MAX}の整数で指定してください。"))

    if errors:
        raise InvalidInputError(fields=errors)

    return GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=req.seed,
    )


# --- Response models (also used for OpenAPI docs) ---------------------------


class FieldErrorModel(BaseModel):
    field: str
    message: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    fields: list[FieldErrorModel] = []
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class QueueInfo(BaseModel):
    running: int
    waiting: int
    max_waiting: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="not_loaded | loading | ready | failed")
    model_id: str
    device: str | None
    dtype: str | None
    message: str | None
    queue: QueueInfo


class Limits(BaseModel):
    min_image_size: int
    max_image_size: int
    size_multiple: int
    min_steps: int
    max_steps: int
    min_guidance_scale: float
    max_guidance_scale: float
    max_prompt_length: int
    seed_max: int


class Defaults(BaseModel):
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float


class ConfigResponse(BaseModel):
    limits: Limits
    defaults: Defaults

    @classmethod
    def from_settings(cls, s: Settings) -> ConfigResponse:
        return cls(
            limits=Limits(
                min_image_size=s.min_image_size,
                max_image_size=s.max_image_size,
                size_multiple=SIZE_MULTIPLE,
                min_steps=1,
                max_steps=s.max_steps,
                min_guidance_scale=0,
                max_guidance_scale=s.max_guidance_scale,
                max_prompt_length=s.max_prompt_length,
                seed_max=SEED_MAX,
            ),
            defaults=Defaults(
                width=s.default_width,
                height=s.default_height,
                num_inference_steps=s.default_steps,
                guidance_scale=s.default_guidance_scale,
            ),
        )
