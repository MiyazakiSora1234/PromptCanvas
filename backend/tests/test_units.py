from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app import model_cache
from app.catalog import Catalog, load_catalog
from app.config import BACKEND_DIR
from app.errors import (
    ConfigurationError,
    GenerationFailedError,
    GpuOutOfMemoryError,
    QueueTimeoutError,
    ServerBusyError,
    classify_generation_error,
    describe_load_error,
)
from app.generator import batch_seeds, resolve_device, resolve_dtype
from app.imaging import ImageDecodeError, decode_base64_image, encode_image, fit_to
from app.limiter import ConcurrencyLimiter
from app.schedulers import SCHEDULERS, build_scheduler
from app.styles import STYLES, apply_style

from .conftest import CATALOG, image_data_url, make_settings

# --- Limiter -----------------------------------------------------------------


def test_limiter_rejects_when_queue_full() -> None:
    async def scenario() -> None:
        limiter = ConcurrencyLimiter(max_waiting=0, timeout_seconds=1)
        async with limiter.slot():
            assert limiter.running == 1
            with pytest.raises(ServerBusyError):
                async with limiter.slot():
                    pass
        assert limiter.running == 0

    asyncio.run(scenario())


def test_limiter_times_out_waiting_request() -> None:
    async def scenario() -> None:
        limiter = ConcurrencyLimiter(max_waiting=1, timeout_seconds=0.05)
        async with limiter.slot():
            with pytest.raises(QueueTimeoutError):
                async with limiter.slot():
                    pass
        assert limiter.waiting == 0

    asyncio.run(scenario())


def test_limiter_runs_waiting_request_after_release() -> None:
    async def scenario() -> list[str]:
        limiter = ConcurrencyLimiter(max_waiting=1, timeout_seconds=1)
        order: list[str] = []

        async def job(name: str) -> None:
            async with limiter.slot():
                order.append(f"{name}-start")
                await asyncio.sleep(0.01)
                order.append(f"{name}-end")

        await asyncio.gather(job("a"), job("b"))
        return order

    assert asyncio.run(scenario()) == ["a-start", "a-end", "b-start", "b-end"]


# --- Error classification ---------------------------------------------------


class OutOfMemoryError(RuntimeError):
    """Same name as torch.cuda.OutOfMemoryError."""


def test_classify_out_of_memory() -> None:
    assert isinstance(classify_generation_error(OutOfMemoryError("x")), GpuOutOfMemoryError)
    assert isinstance(classify_generation_error(RuntimeError("MPS backend out of memory")), GpuOutOfMemoryError)
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = OutOfMemoryError("inner")
    assert isinstance(classify_generation_error(wrapped), GpuOutOfMemoryError)


def test_classify_other_errors_as_generic() -> None:
    assert isinstance(classify_generation_error(ValueError("boom")), GenerationFailedError)


class GatedRepoError(Exception):
    pass


def test_describe_load_error_hides_raw_text() -> None:
    message = describe_load_error(GatedRepoError("401 https://huggingface.co/... secret"))
    assert "HF_TOKEN" in message
    assert "secret" not in message
    assert "secret" not in describe_load_error(OSError("some secret path"))


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


# --- Settings / catalog ---------------------------------------------------------


def test_settings_reject_inconsistent_limits() -> None:
    with pytest.raises(ValidationError):
        make_settings(min_image_size=1024, max_image_size=512)
    with pytest.raises(ValidationError):
        make_settings(max_image_size=1020)


def test_bundled_catalog_is_valid() -> None:
    catalog = load_catalog(make_settings())
    assert catalog.default.id == "sd15"
    assert {m.family for m in catalog.models} == {"sd15", "sdxl"}
    assert all(lora.family in {m.family for m in catalog.models} for lora in catalog.loras)


def _catalog_dict(**changes: Any) -> dict[str, Any]:
    data = CATALOG.model_dump()
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "changes",
    [
        {"default_model": "missing"},
        {"models": [CATALOG.models[0].model_dump(), CATALOG.models[0].model_dump()]},
        {"models": [CATALOG.models[0].model_dump() | {"id": "Bad Id!"}]},
        {"models": [CATALOG.models[0].model_dump() | {"family": "sd3"}]},
        {
            "models": [
                CATALOG.models[0].model_dump()
                | {"defaults": CATALOG.models[0].defaults.model_dump() | {"scheduler": "nope"}}
            ]
        },
        {
            "models": [
                CATALOG.models[0].model_dump()
                | {"defaults": CATALOG.models[0].defaults.model_dump() | {"style": "nope"}}
            ]
        },
    ],
)
def test_catalog_rejects_invalid_entries(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Catalog.model_validate(_catalog_dict(**changes))


def _preset(**settings: Any) -> dict[str, Any]:
    return {"id": "p", "label": "P", "settings": {"model": "sdxl", **settings}}


@pytest.mark.parametrize(
    "preset",
    [
        _preset(model="nope"),
        _preset(scheduler="nope"),
        _preset(style="nope"),
        _preset(loras=[{"id": "style15", "scale": 1.0}]),  # SD1.5 LoRA in an SDXL preset
        _preset(loras=[{"id": "missing", "scale": 1.0}]),
        _preset(unknown_field=1),
    ],
)
def test_catalog_rejects_invalid_presets(preset: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Catalog.model_validate(_catalog_dict(presets=[preset]))


def test_preset_sizes_must_fit_settings(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    data = CATALOG.model_dump()
    data["presets"] = [_preset(width=900)]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="preset 'p'"):
        load_catalog(make_settings(catalog_path=path))


def test_bundled_realistic_human_preset() -> None:
    catalog = load_catalog(make_settings())
    preset = next(p for p in catalog.presets if p.id == "realistic-human")
    assert preset.settings.model == "realvis-xl"
    assert preset.settings.style == "photo"


def test_catalog_defaults_must_fit_settings(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG.model_dump()), encoding="utf-8")
    with pytest.raises(ValueError, match="sdxl"):
        load_catalog(make_settings(catalog_path=path, max_image_size=768))


def test_default_model_can_be_overridden(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG.model_dump()), encoding="utf-8")
    assert load_catalog(make_settings(catalog_path=path, default_model="sdxl")).default.id == "sdxl"
    with pytest.raises(ValueError, match="nope"):
        load_catalog(make_settings(catalog_path=path, default_model="nope"))


def test_bundled_catalog_path() -> None:
    assert make_settings().catalog_path == BACKEND_DIR / "catalog.json"


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
def test_is_model_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights: list[str], expected: bool
) -> None:
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


# --- Identity (InstantID) helpers -------------------------------------------------


def test_identity_files_cover_every_asset() -> None:
    assert CATALOG.identity is not None
    files = CATALOG.identity.files()
    assert ("test/instantid", "ip-adapter.bin", None) in files
    assert ("test/instantid", "ControlNetModel/diffusion_pytorch_model.safetensors", None) in files
    assert ("test/faces", "det.onnx", None) in files
    assert ("test/annotators", "body.pth", None) in files
    assert len(files) == 8
    with_sketch = type(CATALOG.identity).model_validate(
        {**CATALOG.identity.model_dump(), "sketch_controlnet": {"repo": "test/scribble"}}
    )
    assert ("test/scribble", "diffusion_pytorch_model.safetensors", None) in with_sketch.files()


def test_extract_lines_keeps_strokes_and_drops_shading_and_specks() -> None:
    pytest.importorskip("cv2")
    from PIL import ImageDraw

    from app.identity import extract_lines

    img = Image.new("RGB", (400, 400), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 300, 400, 400), fill=(200, 200, 200))  # soft shading
    draw.line((50, 50, 350, 250), fill=(40, 40, 40), width=3)  # a stroke
    draw.point((380, 20), fill=(0, 0, 0))  # a speck
    lines = np.asarray(extract_lines(img).convert("L"))
    assert lines[150, 200] == 255  # on the stroke
    assert lines[20, 380] == 0
    assert lines[350, 200] == 0  # inside the shaded area
    assert lines[10, 10] == 0


def test_draw_kps_renders_keypoints_at_their_positions() -> None:
    pytest.importorskip("cv2")
    from app.identity import draw_kps

    kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)
    img = draw_kps((100, 120), kps)
    assert img.size == (100, 120)
    pixels = np.asarray(img)
    assert pixels[40, 30].tolist() == [255, 0, 0]  # left eye, full-color dot
    assert pixels[5, 5].tolist() == [0, 0, 0]  # background stays black


def _person(seed: int, noise: float = 0.0, base: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base if base is not None else rng.normal(size=512)
    return (v + noise * rng.normal(size=512)).astype(np.float32)


def test_combined_embedding_keeps_direction_and_magnitude() -> None:
    from app.identity import combine_embeddings

    a = _person(1)
    b = _person(2, noise=0.3, base=a) * 1.2  # same person, different photo and scale
    combined = combine_embeddings([a, b])
    assert combined.dtype == np.float32
    expected_norm = (np.linalg.norm(a) + np.linalg.norm(b)) / 2
    assert abs(np.linalg.norm(combined) - expected_norm) < 1e-3 * expected_norm
    cos = float(np.dot(combined, a) / (np.linalg.norm(combined) * np.linalg.norm(a)))
    assert cos > 0.9


def test_find_outliers_flags_a_different_person() -> None:
    from app.identity import find_outliers

    alice = _person(1)
    photos = [_person(10 + i, noise=0.4, base=alice) for i in range(3)]
    assert find_outliers(photos) == []
    assert find_outliers([photos[0]]) == []  # a single photo has nothing to compare with
    bob = _person(99)
    assert find_outliers([*photos[:2], bob]) == [2]


def test_identity_tokens_follow_the_guidance_batch_layout() -> None:
    torch = pytest.importorskip("torch")
    from app.identity import _IdentityTokens

    tokens = _IdentityTokens()
    tokens.cond = torch.ones(1, 16, 8)
    tokens.uncond = torch.zeros(1, 16, 8)
    tokens.batch = 2
    with_cfg = tokens.for_batch(4)  # [uncond, uncond, cond, cond]
    assert with_cfg.shape == (4, 16, 8)
    assert with_cfg[:2].sum() == 0 and bool((with_cfg[2:] == 1).all())
    assert bool((tokens.for_batch(2) == 1).all())  # no CFG: cond only

    seen: dict[str, Any] = {}

    class FakeNet:
        def forward(self, *args: Any, **kwargs: Any) -> str:
            seen.update(kwargs)
            return "ok"

    net = FakeNet()
    tokens.patch(net)
    assert net.forward(sample=torch.zeros(4, 4), timestep=1, encoder_hidden_states="text") == "ok"
    assert seen["encoder_hidden_states"].shape == (4, 16, 8)


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


def test_hf_token_is_not_exposed_in_repr() -> None:
    settings = make_settings(HF_TOKEN="hf_supersecret")
    assert settings.hf_token is not None
    assert "hf_supersecret" not in repr(settings)


# --- Japanese prompt translation ---------------------------------------------------


def test_has_japanese() -> None:
    from app.translate import has_japanese

    assert has_japanese("猫") and has_japanese("ねこ") and has_japanese("ネコ") and has_japanese("ﾈｺ")
    assert not has_japanese("a cat, (smile:1.2), 1girl")


def test_translate_prompt_translates_only_japanese_parts() -> None:
    from app.translate import translate_prompt

    calls: list[str] = []

    def fake(ja: str) -> str:
        calls.append(ja)
        return {"黒髪ロング": "long black hair", "笑顔": "smile", "教室": "classroom"}[ja]

    result = translate_prompt("1girl, 黒髪ロング、(笑顔:1.2)，教室, highly detailed", fake)
    assert result == "1girl, long black hair, (smile:1.2), classroom, highly detailed"
    assert calls == ["黒髪ロング", "笑顔", "教室"]
    assert translate_prompt("", fake) == ""


def test_clean_answer() -> None:
    from app.translate import clean_answer

    assert clean_answer('"a cat on a sofa."\nExplanation: ...') == "a cat on a sofa"
    assert clean_answer("「watermark」") == "watermark"
    assert clean_answer("  ") == ""
