"""HTTP request/response models (also used for the OpenAPI docs)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from ..catalog import Catalog, PresetEntry
from ..config import SEED_MAX, SIZE_MULTIPLE, Settings
from ..generation.imaging import OUTPUT_FORMATS, OutputFormat
from ..generation.params import (
    DEFAULT_IDENTITY_STRENGTH,
    DEFAULT_POSE_STRENGTH,
    DEFAULT_QUALITY,
    DEFAULT_STRENGTH,
    MAX_CONTROL_STRENGTH,
    MAX_LORA_SCALE,
    MAX_QUALITY,
    MAX_STRENGTH,
    MIN_CONTROL_STRENGTH,
    MIN_LORA_SCALE,
    MIN_QUALITY,
    MIN_STRENGTH,
)
from ..generation.schedulers import SCHEDULERS
from ..generation.styles import STYLES

# Absolute caps so oversized payloads are rejected before any further work.
_HARD_TEXT_CAP = 10_000
_HARD_IMAGE_CAP = 70_000_000  # base64 characters (~50MB decoded)
JOB_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


# --- Requests ------------------------------------------------------------------


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
    style: str | None = Field(default=None, max_length=64, description="none | photo（省略時はモデルの既定）")
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
    face_images: list[Annotated[str, Field(max_length=_HARD_IMAGE_CAP)]] = Field(
        default_factory=list,
        max_length=10,
        description="顔を保つ人物の写真（同じ人物を複数枚可。InstantID、SDXL 系のみ）",
    )
    pose_image: str | None = Field(
        default=None, max_length=_HARD_IMAGE_CAP, description="ポーズ参考画像（OpenPose ControlNet、SDXL 系のみ）"
    )
    identity_strength: float = DEFAULT_IDENTITY_STRENGTH
    pose_strength: float = DEFAULT_POSE_STRENGTH
    job_id: str | None = Field(
        default=None,
        pattern=JOB_ID_PATTERN,
        description="クライアントが決めるランダムな ID。POST /api/jobs/{job_id}/cancel で中止できる",
    )


# --- Responses -----------------------------------------------------------------


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
    style: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    output_format: OutputFormat
    elapsed_ms: int
    filtered_count: int = Field(description="セーフティフィルタで除外された枚数")
    translated_prompt: str | None = Field(default=None, description="日本語を英訳したプロンプト（英訳なしは null）")
    translated_negative_prompt: str | None = Field(default=None, description="ネガティブプロンプトの英訳")


class CancelResponse(BaseModel):
    cancelled: bool = Field(description="false: 該当する生成がない（すでに終わっている）")


class QueueInfo(BaseModel):
    running: int
    waiting: int
    max_waiting: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="not_loaded | loading | ready | failed")
    model: str | None = Field(description="読み込み済み（または読み込み中）のモデル ID")
    cached_models: list[str] = Field(description="ダウンロード済みで、すぐ読み込めるモデル ID")
    identity_cached: bool = Field(description="顔・ポーズ参照に必要なモデルがダウンロード済みか")
    translator_cached: bool = Field(default=False, description="日本語プロンプトの翻訳モデルがダウンロード済みか")
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
    min_control_strength: float
    max_control_strength: float
    max_face_images: int


class IdentityOption(BaseModel):
    families: list[str]
    download_size_gb: float | None


class ModelDefaultsModel(BaseModel):
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    scheduler: str
    style: str


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


class StyleOption(Option):
    description: str


class FormatOption(Option):
    lossy: bool
    extension: str


class Defaults(BaseModel):
    num_images: int
    output_format: OutputFormat
    quality: int
    strength: float
    identity_strength: float
    pose_strength: float


class TranslationOption(BaseModel):
    download_size_gb: float | None


class ConfigResponse(BaseModel):
    limits: Limits
    default_model: str
    models: list[ModelOption]
    schedulers: list[Option]
    styles: list[StyleOption]
    loras: list[LoraOption]
    output_formats: list[FormatOption]
    identity: IdentityOption | None = Field(description="顔・ポーズ参照（null なら無効）")
    translation: TranslationOption | None = Field(default=None, description="日本語の自動英訳（null なら無効）")
    presets: list[PresetEntry] = Field(description="組み込みの設定プリセット")
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
                min_control_strength=MIN_CONTROL_STRENGTH,
                max_control_strength=MAX_CONTROL_STRENGTH,
                max_face_images=s.max_face_images,
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
            styles=[StyleOption(id=k, label=v.label, description=v.description) for k, v in STYLES.items()],
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
            identity=(
                IdentityOption(
                    families=list(catalog.identity.families), download_size_gb=catalog.identity.download_size_gb
                )
                if catalog.identity
                else None
            ),
            translation=(
                TranslationOption(download_size_gb=catalog.translator.download_size_gb) if catalog.translator else None
            ),
            presets=list(catalog.presets),
            defaults=Defaults(
                num_images=1,
                output_format="png",
                quality=DEFAULT_QUALITY,
                strength=DEFAULT_STRENGTH,
                identity_strength=DEFAULT_IDENTITY_STRENGTH,
                pose_strength=DEFAULT_POSE_STRENGTH,
            ),
        )
