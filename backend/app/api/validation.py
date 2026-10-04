"""Server-side validation: turns a ``GenerateRequest`` into ``GenerationParams`` (model defaults applied)."""

from __future__ import annotations

import math

from PIL import Image

from ..catalog import Catalog, LoraEntry
from ..config import SEED_MAX, SIZE_MULTIPLE, Settings
from ..errors import FieldError, InvalidInputError
from ..generation.imaging import ImageDecodeError, decode_base64_image
from ..generation.params import (
    MAX_CONTROL_STRENGTH,
    MAX_LORA_SCALE,
    MAX_QUALITY,
    MAX_STRENGTH,
    MIN_CONTROL_STRENGTH,
    MIN_LORA_SCALE,
    MIN_QUALITY,
    MIN_STRENGTH,
    GenerationParams,
)
from ..generation.schedulers import SCHEDULERS
from ..generation.styles import STYLES
from .schemas import GenerateRequest


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

    style = defaults.style if req.style is None else req.style
    if style not in STYLES:
        fail("style", "選択できないスタイルです。")

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

    max_bytes = settings.max_init_image_mb * 1024 * 1024

    def decode(field: str, data: str) -> Image.Image | None:
        try:
            return decode_base64_image(data, max_bytes=max_bytes)
        except ImageDecodeError as exc:
            fail(field, str(exc))
            return None

    init_image: Image.Image | None = None
    if req.init_image:
        if not _in_range(req.strength, MIN_STRENGTH, MAX_STRENGTH):
            fail("strength", f"{MIN_STRENGTH:g}〜{MAX_STRENGTH:g}の範囲で指定してください。")
        elif int(steps * req.strength) < 1:
            fail("strength", "ステップ数 × 変換強度が 1 以上になるようにしてください。")
        init_image = decode("init_image", req.init_image)

    face_images: list[Image.Image] = []
    pose_image: Image.Image | None = None
    if req.face_images or req.pose_image:
        identity = catalog.identity
        field = "face_images" if req.face_images else "pose_image"
        if identity is None:
            fail(field, "このサーバーでは顔・ポーズの参照機能が有効になっていません。")
        elif model.family not in identity.families:
            fail(field, f"顔・ポーズの参照は SDXL 系のモデルでのみ使えます（選択中: {model.label}）。")
        if req.init_image:
            fail(field, "img2img（元画像から生成）と顔・ポーズの参照は同時に使えません。どちらかを外してください。")
        for name, control in (("identity_strength", req.identity_strength), ("pose_strength", req.pose_strength)):
            if not _in_range(control, MIN_CONTROL_STRENGTH, MAX_CONTROL_STRENGTH):
                fail(name, f"{MIN_CONTROL_STRENGTH:g}〜{MAX_CONTROL_STRENGTH:g}の範囲で指定してください。")
        if len(req.face_images) > settings.max_face_images:
            fail("face_images", f"顔の写真は{settings.max_face_images}枚まで選べます。")
        for i, data in enumerate(req.face_images):
            try:
                face_images.append(decode_base64_image(data, max_bytes=max_bytes))
            except ImageDecodeError as exc:
                fail("face_images", f"{i + 1}枚目: {exc}")
        if req.pose_image:
            pose_image = decode("pose_image", req.pose_image)

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
        style=style,
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
        face_images=tuple(face_images),
        pose_image=pose_image,
        identity_strength=req.identity_strength,
        pose_strength=req.pose_strength,
    )
