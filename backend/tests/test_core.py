from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.catalog import Catalog, load_catalog
from app.config import BACKEND_DIR
from app.errors import (
    GenerationFailedError,
    GpuOutOfMemoryError,
    classify_generation_error,
    describe_load_error,
)

from .conftest import CATALOG, make_settings

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


def test_hf_token_is_not_exposed_in_repr() -> None:
    settings = make_settings(HF_TOKEN="hf_supersecret")
    assert settings.hf_token is not None
    assert "hf_supersecret" not in repr(settings)
