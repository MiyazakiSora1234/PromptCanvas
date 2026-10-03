"""Allow-list of models and LoRAs users may pick (``backend/catalog.json``).

Users choose catalog IDs, never raw repository names, so the server only ever
downloads what the operator listed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import SIZE_MULTIPLE, Settings
from .schedulers import SCHEDULERS

_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"

ModelFamily = Literal["sd15", "sdxl"]


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID_PATTERN)
    label: str
    description: str = ""
    repo: str = Field(description="Hugging Face repo id or local directory")
    revision: str | None = None
    family: ModelFamily


class ModelDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    scheduler: str = "default"


class ModelEntry(_Entry):
    variant: str | None = None
    # Shown to users before a first-time download; informational only.
    download_size_gb: float | None = Field(default=None, gt=0)
    defaults: ModelDefaults


class LoraEntry(_Entry):
    weight_name: str | None = None
    trigger_words: str = ""
    default_scale: float = Field(default=1.0, ge=0, le=2)


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_model: str
    models: tuple[ModelEntry, ...] = Field(min_length=1)
    loras: tuple[LoraEntry, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        for kind, entries in (("model", self.models), ("lora", self.loras)):
            ids = [e.id for e in entries]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} ids in catalog")
        if self.default_model not in {m.id for m in self.models}:
            raise ValueError(f"default_model '{self.default_model}' is not in models")
        for m in self.models:
            if m.defaults.scheduler not in SCHEDULERS:
                raise ValueError(f"model '{m.id}': unknown scheduler '{m.defaults.scheduler}'")
        return self

    def model(self, model_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    def lora(self, lora_id: str) -> LoraEntry | None:
        return next((lora for lora in self.loras if lora.id == lora_id), None)

    @property
    def default(self) -> ModelEntry:
        model = self.model(self.default_model)
        assert model is not None  # guaranteed by the validator
        return model

    def check_against(self, settings: Settings) -> None:
        """Model defaults must be valid inputs under the configured limits."""
        for m in self.models:
            d = m.defaults
            for name, value in (("width", d.width), ("height", d.height)):
                if not settings.min_image_size <= value <= settings.max_image_size or value % SIZE_MULTIPLE:
                    raise ValueError(f"model '{m.id}': default {name} {value} is outside the configured limits")
            if not 1 <= d.num_inference_steps <= settings.max_steps:
                raise ValueError(f"model '{m.id}': default steps exceed PROMPTCANVAS_MAX_STEPS")
            if not 0 <= d.guidance_scale <= settings.max_guidance_scale:
                raise ValueError(f"model '{m.id}': default guidance_scale exceeds the configured maximum")


def load_catalog(settings: Settings) -> Catalog:
    path: Path = settings.catalog_path
    catalog = Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if settings.default_model:
        if catalog.model(settings.default_model) is None:
            raise ValueError(f"PROMPTCANVAS_DEFAULT_MODEL '{settings.default_model}' is not in {path}")
        catalog = catalog.model_copy(update={"default_model": settings.default_model})
    catalog.check_against(settings)
    return catalog
