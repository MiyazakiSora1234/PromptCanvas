"""Request/response models and server-side input validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .catalog import Catalog, LoraEntry, ModelEntry
from .config import SEED_MAX, SIZE_MULTIPLE, Settings
from .errors import FieldError, InvalidInputError
from .imaging import OUTPUT_FORMATS, ImageDecodeError, OutputFormat, decode_base64_image
from .schedulers import SCHEDULERS

# Absolute caps so oversized payloads are rejected before any further work.
_HARD_TEXT_CAP = 10_000
_HARD_IMAGE_CAP = 70_000_000  # base64 characters (~50MB decoded)

MIN_STRENGTH, MAX_STRENGTH, DEFAULT_STRENGTH = 0.1, 1.0, 0.6
MIN_LORA_SCALE, MAX_LORA_SCALE = 0.0, 2.0
MIN_QUALITY, MAX_QUALITY, DEFAULT_QUALITY = 1, 100, 90


class LoraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(max_length=64)
    scale: float | None = None


class GenerateRequest(BaseModel):
    """Body of ``POST /api/generate``. Omitted optional fields fall back to the model's defaults."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    prompt: str = Field(max_length=_HARD_TEXT_CAP)
    negative_prompt: str = Field(default="", max_length=_HARD_TEXT_CAP)
    model: str | None = Field(default=None, max_length=64, description="catalog.json のモデル ID")
    scheduler: str | None = Field(default=None, max_length=64)
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = Field(default=None, description="未指定ならランダム。複数枚では 1 枚ごとに +1")
    num_images: int = 1
    output_format: OutputFormat = "png"
    quality: int = DEFAULT_QUALITY
    init_image: str | None = Field(
        default=None, max_length=_HARD_IMAGE_CAP, description="img2img の元画像（base64 または data URL）"
    )
    strength: float = DEFAULT_STRENGTH
    loras: list[LoraRequest] = Field(default_factory=list, max_length=10)


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    negative_prompt: str
    model: ModelEntry
    scheduler: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int | None
    num_images: int
    output_format: OutputFormat
    quality: int
    init_image: Image.Image | None
    strength: float
    loras: tuple[tuple[LoraEntry, float], ...]


def _in_range(value: float, low: float, high: float) -> bool:
    return math.isfinite(value) and low <= value <= high


def build_params(req: GenerateRequest, settings: Settings, catalog: Catalog) -> GenerationParams:
    """Apply per-model defaults and validate everything that depends on configuration."""
    errors: list[FieldError] = []

    def fail(field: str, message: str) -> None:
        errors.append(FieldError(field, message))

    prompt = req.prompt.strip()
    if not prompt:
        fail("prompt", "プロンプトを入力してください。")
    elif len(prompt) > settings.max_prompt_length:
        fail("prompt", f"{settings.max_prompt_length}文字以内で入力してください。")

    negative_prompt = req.negative_prompt.strip()
    if len(negative_prompt) > settings.max_prompt_length:
        fail("negative_prompt", f"{settings.max_prompt_length}文字以内で入力してください。")

    model = catalog.default if req.model is None else catalog.model(req.model)
    if model is None:
        fail("model", "選択できないモデルです。画面を再読み込みしてモデルを選び直してください。")
        model = catalog.default  # keep validating the rest against something sensible
    defaults = model.defaults

    scheduler = defaults.scheduler if req.scheduler is None else req.scheduler
    if scheduler not in SCHEDULERS:
        fail("scheduler", "選択できないサンプラーです。")

    width = defaults.width if req.width is None else req.width
    height = defaults.height if req.height is None else req.height
    for name, value in (("width", width), ("height", height)):
        if not settings.min_image_size <= value <= settings.max_image_size:
            fail(name, f"{settings.min_image_size}〜{settings.max_image_size}の範囲で指定してください。")
        elif value % SIZE_MULTIPLE:
            fail(name, f"{SIZE_MULTIPLE}の倍数で指定してください。")

    steps = defaults.num_inference_steps if req.num_inference_steps is None else req.num_inference_steps
    if not 1 <= steps <= settings.max_steps:
        fail("num_inference_steps", f"1〜{settings.max_steps}の範囲で指定してください。")

    guidance = defaults.guidance_scale if req.guidance_scale is None else req.guidance_scale
    if not _in_range(guidance, 0, settings.max_guidance_scale):
        fail("guidance_scale", f"0〜{settings.max_guidance_scale:g}の範囲で指定してください。")

    if req.seed is not None and not 0 <= req.seed <= SEED_MAX:
        fail("seed", f"0〜{SEED_MAX}の整数で指定してください。")

    if not 1 <= req.num_images <= settings.max_batch_size:
        fail("num_images", f"1〜{settings.max_batch_size}枚の範囲で指定してください。")

    if not MIN_QUALITY <= req.quality <= MAX_QUALITY:
        fail("quality", f"{MIN_QUALITY}〜{MAX_QUALITY}の範囲で指定してください。")

    init_image: Image.Image | None = None
    if req.init_image:
        if not _in_range(req.strength, MIN_STRENGTH, MAX_STRENGTH):
            fail("strength", f"{MIN_STRENGTH:g}〜{MAX_STRENGTH:g}の範囲で指定してください。")
        elif int(steps * req.strength) < 1:
            fail("strength", "ステップ数 × 変換強度が 1 以上になるようにしてください。")
        try:
            init_image = decode_base64_image(req.init_image, max_bytes=settings.max_init_image_mb * 1024 * 1024)
        except ImageDecodeError as exc:
            fail("init_image", str(exc))

    loras: list[tuple[LoraEntry, float]] = []
    if len(req.loras) > settings.max_loras:
        fail("loras", f"LoRA は{settings.max_loras}個まで選択できます。")
    seen: set[str] = set()
    for item in req.loras:
        lora = catalog.lora(item.id)
        if lora is None:
            fail("loras", "選択できない LoRA が含まれています。画面を再読み込みしてください。")
            continue
        if lora.id in seen:
            fail("loras", f"LoRA「{lora.label}」が重複しています。")
            continue
        seen.add(lora.id)
        if lora.family != model.family:
            fail("loras", f"LoRA「{lora.label}」は選択中のモデル（{model.label}）では使えません。")
            continue
        scale = lora.default_scale if item.scale is None else item.scale
        if not _in_range(scale, MIN_LORA_SCALE, MAX_LORA_SCALE):
            fail("loras", f"LoRA の強さは{MIN_LORA_SCALE:g}〜{MAX_LORA_SCALE:g}の範囲で指定してください。")
            continue
        loras.append((lora, scale))

    if errors:
        raise InvalidInputError(fields=errors)

    return GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        scheduler=scheduler,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=req.seed,
        num_images=req.num_images,
        output_format=req.output_format,
        quality=req.quality,
        init_image=init_image,
        strength=req.strength,
        loras=tuple(loras),
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


class GeneratedImageModel(BaseModel):
    seed: int
    mime_type: str
    data: str = Field(description="base64 エンコードされた画像")


class GenerateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    images: list[GeneratedImageModel]
    model: str
    scheduler: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    output_format: OutputFormat
    elapsed_ms: int
    filtered_count: int = Field(description="セーフティフィルタで除外された枚数")


class QueueInfo(BaseModel):
    running: int
    waiting: int
    max_waiting: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="not_loaded | loading | ready | failed")
    model: str | None = Field(description="読み込み済み（または読み込み中）のモデル ID")
    cached_models: list[str] = Field(description="ダウンロード済みで、すぐ読み込めるモデル ID")
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
    max_batch_size: int
    max_loras: int
    max_init_image_mb: int
    min_strength: float
    max_strength: float
    min_lora_scale: float
    max_lora_scale: float
    min_quality: int
    max_quality: int


class ModelDefaultsModel(BaseModel):
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    scheduler: str


class ModelOption(BaseModel):
    id: str
    label: str
    description: str
    family: str
    download_size_gb: float | None
    defaults: ModelDefaultsModel


class LoraOption(BaseModel):
    id: str
    label: str
    description: str
    family: str
    trigger_words: str
    default_scale: float


class Option(BaseModel):
    id: str
    label: str


class FormatOption(Option):
    lossy: bool
    extension: str


class Defaults(BaseModel):
    num_images: int
    output_format: OutputFormat
    quality: int
    strength: float


class ConfigResponse(BaseModel):
    limits: Limits
    default_model: str
    models: list[ModelOption]
    schedulers: list[Option]
    loras: list[LoraOption]
    output_formats: list[FormatOption]
    defaults: Defaults

    @classmethod
    def build(cls, s: Settings, catalog: Catalog) -> ConfigResponse:
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
                max_batch_size=s.max_batch_size,
                max_loras=s.max_loras,
                max_init_image_mb=s.max_init_image_mb,
                min_strength=MIN_STRENGTH,
                max_strength=MAX_STRENGTH,
                min_lora_scale=MIN_LORA_SCALE,
                max_lora_scale=MAX_LORA_SCALE,
                min_quality=MIN_QUALITY,
                max_quality=MAX_QUALITY,
            ),
            default_model=catalog.default_model,
            models=[
                ModelOption(
                    id=m.id,
                    label=m.label,
                    description=m.description,
                    family=m.family,
                    download_size_gb=m.download_size_gb,
                    defaults=ModelDefaultsModel(**m.defaults.model_dump()),
                )
                for m in catalog.models
            ],
            schedulers=[Option(id=k, label=v.label) for k, v in SCHEDULERS.items()],
            loras=[
                LoraOption(
                    id=lora.id,
                    label=lora.label,
                    description=lora.description,
                    family=lora.family,
                    trigger_words=lora.trigger_words,
                    default_scale=lora.default_scale,
                )
                for lora in catalog.loras
            ],
            output_formats=[
                FormatOption(id=k, label=v.label, lossy=v.lossy, extension=v.extension)
                for k, v in OUTPUT_FORMATS.items()
            ],
            defaults=Defaults(num_images=1, output_format="png", quality=DEFAULT_QUALITY, strength=DEFAULT_STRENGTH),
        )
