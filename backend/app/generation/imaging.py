"""Decoding uploaded images (img2img) and encoding generated ones (PNG/JPEG/WebP)."""

from __future__ import annotations

import base64
import binascii
import io
import re
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError

OutputFormat = Literal["png", "jpeg", "webp"]


@dataclass(frozen=True)
class FormatSpec:
    label: str
    mime_type: str
    extension: str
    lossy: bool


OUTPUT_FORMATS: dict[OutputFormat, FormatSpec] = {
    "png": FormatSpec("PNG（劣化なし）", "image/png", "png", lossy=False),
    "jpeg": FormatSpec("JPEG（小さい・劣化あり）", "image/jpeg", "jpg", lossy=True),
    "webp": FormatSpec("WebP（小さい・高画質）", "image/webp", "webp", lossy=True),
}

# Reject huge images before decoding pixels (decompression-bomb protection).
MAX_INPUT_DIMENSION = 4096
_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.ASCII)


class ImageDecodeError(ValueError):
    """Message is user-facing."""


def decode_base64_image(data: str, *, max_bytes: int) -> Image.Image:
    """Decode a base64 string or data URL into an RGB image (EXIF orientation applied)."""
    payload = _DATA_URL.sub("", data.strip(), count=1)
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageDecodeError("画像データの形式が正しくありません。画像を選び直してください。") from None
    if len(raw) > max_bytes:
        raise ImageDecodeError(f"画像のファイルサイズは{max_bytes // (1024 * 1024)}MB以下にしてください。")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if max(img.size) > MAX_INPUT_DIMENSION:
                raise ImageDecodeError(f"画像の縦横は{MAX_INPUT_DIMENSION}px以下にしてください。")
            img.load()
            return ImageOps.exif_transpose(img).convert("RGB")
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise ImageDecodeError("画像を読み込めませんでした。PNG / JPEG / WebP の画像を指定してください。") from None


def fit_to(img: Image.Image, width: int, height: int) -> Image.Image:
    """Scale and center-crop to exactly width x height without distorting."""
    return ImageOps.fit(img, (width, height), method=Image.Resampling.LANCZOS)


def encode_image(img: Image.Image, fmt: OutputFormat, quality: int) -> bytes:
    buffer = io.BytesIO()
    if fmt == "png":
        img.save(buffer, format="PNG")
    elif fmt == "jpeg":
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buffer, format="WEBP", quality=quality, method=4)
    return buffer.getvalue()
