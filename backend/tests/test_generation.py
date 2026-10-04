from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.errors import (
    ConfigurationError,
)
from app.generation import model_cache
from app.generation.generator import batch_seeds
from app.generation.imaging import ImageDecodeError, decode_base64_image, encode_image, fit_to
from app.generation.runtime import resolve_device, resolve_dtype
from app.generation.schedulers import SCHEDULERS, build_scheduler
from app.generation.styles import STYLES, apply_style

from .conftest import image_data_url

# --- Device / dtype ---------------------------------------------------------


def test_resolve_device() -> None:
    assert resolve_device("auto", cuda_available=True, mps_available=False) == "cuda"
    assert resolve_device("auto", cuda_available=False, mps_available=True) == "mps"
    assert resolve_device("auto", cuda_available=False, mps_available=False) == "cpu"
    with pytest.raises(ConfigurationError):
        resolve_device("cuda", cuda_available=False, mps_available=False)


def test_resolve_dtype() -> None:
    assert resolve_dtype("auto", "cuda") == "float16"
    assert resolve_dtype("auto", "cpu") == "float32"
    with pytest.raises(ConfigurationError):
        resolve_dtype("float16", "cpu")


def test_batch_seeds_increment_and_wrap() -> None:
    assert batch_seeds(5, 3) == [5, 6, 7]
    assert batch_seeds(2**32 - 1, 2) == [2**32 - 1, 0]
    random_seeds = batch_seeds(None, 2)
    assert random_seeds[1] == (random_seeds[0] + 1) % 2**32


# --- Schedulers ------------------------------------------------------------------


class _FakeScheduler:
    def __init__(self, config: dict[str, Any], **options: Any) -> None:
        self.config = config
        self.options = options

    @classmethod
    def from_config(cls, config: dict[str, Any], **options: Any) -> _FakeScheduler:
        return cls(config, **options)


def test_default_scheduler_is_a_fresh_instance_of_the_original_class() -> None:
    original = _FakeScheduler({"beta_start": 0.1})
    rebuilt = build_scheduler("default", original)
    assert type(rebuilt) is _FakeScheduler
    assert rebuilt is not original
    assert rebuilt.config == {"beta_start": 0.1}


def test_scheduler_ids_are_slugs() -> None:
    assert all(k.replace("_", "").isalnum() and k.islower() for k in SCHEDULERS)


# --- Imaging ---------------------------------------------------------------------


def test_decode_data_url_and_plain_base64() -> None:
    url = image_data_url((40, 30))
    assert decode_base64_image(url, max_bytes=10_000_000).size == (40, 30)
    assert decode_base64_image(url.split(",", 1)[1], max_bytes=10_000_000).mode == "RGB"


@pytest.mark.parametrize("data", ["%%%", "aGVsbG8=", ""])
def test_decode_rejects_non_images(data: str) -> None:
    with pytest.raises(ImageDecodeError):
        decode_base64_image(data, max_bytes=10_000_000)


def test_decode_rejects_huge_dimensions_and_bytes() -> None:
    with pytest.raises(ImageDecodeError, match="4096"):
        decode_base64_image(image_data_url((5000, 10)), max_bytes=10_000_000)
    with pytest.raises(ImageDecodeError, match="MB"):
        decode_base64_image(image_data_url((400, 400), fmt="BMP"), max_bytes=1024 * 1024 // 4)


def test_fit_to_crops_without_distortion() -> None:
    assert fit_to(Image.new("RGB", (400, 100)), 64, 64).size == (64, 64)


@pytest.mark.parametrize(("fmt", "pil"), [("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP")])
def test_encode_image_formats(fmt: Any, pil: str) -> None:
    data = encode_image(Image.new("RGB", (16, 16), "blue"), fmt, 80)
    assert Image.open(io.BytesIO(data)).format == pil


# --- Model cache -----------------------------------------------------------------


def _fake_snapshot(root: Path, components: dict[str, Any], weights: list[str]) -> Path:
    root.mkdir(parents=True)
    index = root / "model_index.json"
    index.write_text(json.dumps(components), encoding="utf-8")
    for rel in weights:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"w")
    return index


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (["unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors"], True),
        (["vae/diffusion_pytorch_model.safetensors"], False),  # unet still downloading
    ],
)
def test_is_model_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights: list[str], expected: bool) -> None:
    components = {
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "safety_checker": [None, None],
    }
    index = _fake_snapshot(tmp_path / "snap", components, weights)
    monkeypatch.setattr(model_cache, "try_to_load_from_cache", lambda *a, **k: str(index))
    assert model_cache.is_model_cached("org/model") is expected


def test_is_model_cached_when_absent_or_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_cache, "try_to_load_from_cache", lambda *a, **k: None)
    assert model_cache.is_model_cached("org/missing") is False
    assert model_cache.is_model_cached(str(tmp_path)) is True  # a local directory is always available


# --- Styles ------------------------------------------------------------------------


@pytest.mark.parametrize("style_id", [s for s in STYLES if s != "none"])
def test_every_style_adds_positive_and_negative_wording(style_id: str) -> None:
    prompt, negative = apply_style(style_id, "a cat", "")
    spec = STYLES[style_id]
    assert prompt.startswith(spec.prefix) and prompt.endswith(spec.suffix) and "a cat" in prompt
    assert spec.prefix and spec.suffix and spec.negative and spec.description
    assert negative.startswith(spec.negative)
    assert "watermark" in negative  # common defects are always excluded


def test_style_none_leaves_prompts_untouched() -> None:
    assert apply_style("none", "a cat", "dog") == ("a cat", "dog")


def test_style_negative_keeps_user_negative_first() -> None:
    _, negative = apply_style("anime", "a cat", "hat")
    assert negative.startswith("hat, photo, realistic")
