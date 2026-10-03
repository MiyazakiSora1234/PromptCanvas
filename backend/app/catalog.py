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
from .styles import STYLES

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
    style: str = "none"


class _Asset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    revision: str | None = None


class VaeAsset(_Asset):
    """A replacement VAE in diffusers layout (config.json + diffusion_pytorch_model.safetensors)."""

    def files(self) -> list[tuple[str, str, str | None]]:
        return [
            (self.repo, "config.json", self.revision),
            (self.repo, "diffusion_pytorch_model.safetensors", self.revision),
        ]


class ModelEntry(_Entry):
    variant: str | None = None
    # Shown to users before a first-time download; informational only.
    download_size_gb: float | None = Field(default=None, gt=0)
    # Use this VAE instead of the model's own. The stock SDXL VAE overflows in float16, so
    # Diffusers silently decodes it in float32 (~3GB extra VRAM); a fp16-fixed VAE avoids that.
    vae: VaeAsset | None = None
    defaults: ModelDefaults


class LoraEntry(_Entry):
    weight_name: str | None = None
    trigger_words: str = ""
    default_scale: float = Field(default=1.0, ge=0, le=2)


class InstantIdAsset(_Asset):
    controlnet_subfolder: str = "ControlNetModel"
    adapter_weight_name: str = "ip-adapter.bin"


class FaceModelsAsset(_Asset):
    detection: str
    recognition: str


class ControlNetAsset(_Asset):
    subfolder: str | None = None


class PoseDetectorAsset(_Asset):
    weight_name: str


class IdentityConfig(BaseModel):
    """Face identity (InstantID) and pose reference (OpenPose ControlNet) assets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    families: tuple[ModelFamily, ...] = ("sdxl",)
    download_size_gb: float | None = Field(default=None, gt=0)
    # IP-Adapter scale = identity_strength x this. IdentityNet (face position) takes the full
    # strength; a weaker IP-Adapter keeps the likeness but avoids plastic, over-saturated skin.
    ip_adapter_ratio: float = Field(default=0.65, gt=0, le=1.5)
    instantid: InstantIdAsset
    face_models: FaceModelsAsset
    pose_controlnet: ControlNetAsset
    pose_detector: PoseDetectorAsset
    # Follows the outlines of a pose image in which no skeleton is found (line art, illustrations,
    # mannequins). None = such images are rejected.
    sketch_controlnet: ControlNetAsset | None = None
    # Outline guidance is stricter than a skeleton (it also copies body shape), so it gets
    # pose_strength x this.
    sketch_strength_ratio: float = Field(default=0.7, gt=0, le=1.5)
    # Outlines guide only the first part of denoising: enough to fix pose and silhouette, while the
    # drawing's own details (hoods, construction lines, joint circles) don't end up in the image.
    sketch_guidance_end: float = Field(default=0.5, gt=0, le=1.0)

    def files(self) -> list[tuple[str, str, str | None]]:
        """(repo, filename, revision) of every file needed, for downloads and cache checks."""
        i, f, d = self.instantid, self.face_models, self.pose_detector
        files = [
            (i.repo, f"{i.controlnet_subfolder}/config.json", i.revision),
            (i.repo, f"{i.controlnet_subfolder}/diffusion_pytorch_model.safetensors", i.revision),
            (i.repo, i.adapter_weight_name, i.revision),
            (f.repo, f.detection, f.revision),
            (f.repo, f.recognition, f.revision),
            (d.repo, d.weight_name, d.revision),
        ]
        for net in (self.pose_controlnet, self.sketch_controlnet):
            if net is not None:
                prefix = f"{net.subfolder}/" if net.subfolder else ""
                files.append((net.repo, f"{prefix}config.json", net.revision))
                files.append((net.repo, f"{prefix}diffusion_pytorch_model.safetensors", net.revision))
        return files


class PresetLora(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    scale: float = Field(ge=0, le=2)


class PresetSettings(BaseModel):
    """Generation settings a preset applies. Omitted fields keep the model's defaults.

    Seed and images are deliberately not part of presets. The prompt is optional: when
    omitted, applying the preset keeps whatever prompt the user has typed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model: str
    prompt: str | None = None
    style: str | None = None
    scheduler: str | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    num_images: int | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    quality: int | None = Field(default=None, ge=1, le=100)
    loras: tuple[PresetLora, ...] = ()
    identity_strength: float | None = Field(default=None, ge=0, le=1.5)
    pose_strength: float | None = Field(default=None, ge=0, le=1.5)


class PresetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID_PATTERN)
    label: str
    description: str = ""
    settings: PresetSettings


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_model: str
    models: tuple[ModelEntry, ...] = Field(min_length=1)
    loras: tuple[LoraEntry, ...] = ()
    identity: IdentityConfig | None = None
    presets: tuple[PresetEntry, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        for kind, entries in (("model", self.models), ("lora", self.loras), ("preset", self.presets)):
            ids = [e.id for e in entries]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} ids in catalog")
        if self.default_model not in {m.id for m in self.models}:
            raise ValueError(f"default_model '{self.default_model}' is not in models")
        for m in self.models:
            if m.defaults.scheduler not in SCHEDULERS:
                raise ValueError(f"model '{m.id}': unknown scheduler '{m.defaults.scheduler}'")
            if m.defaults.style not in STYLES:
                raise ValueError(f"model '{m.id}': unknown style '{m.defaults.style}'")
        for p in self.presets:
            s = p.settings
            model = self.model(s.model)
            if model is None:
                raise ValueError(f"preset '{p.id}': unknown model '{s.model}'")
            if s.scheduler is not None and s.scheduler not in SCHEDULERS:
                raise ValueError(f"preset '{p.id}': unknown scheduler '{s.scheduler}'")
            if s.style is not None and s.style not in STYLES:
                raise ValueError(f"preset '{p.id}': unknown style '{s.style}'")
            for item in s.loras:
                lora = self.lora(item.id)
                if lora is None or lora.family != model.family:
                    raise ValueError(f"preset '{p.id}': LoRA '{item.id}' is unknown or not for {model.family}")
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
        for p in self.presets:
            s = p.settings
            for dim, size in (("width", s.width), ("height", s.height)):
                if size is not None and (
                    not settings.min_image_size <= size <= settings.max_image_size or size % SIZE_MULTIPLE
                ):
                    raise ValueError(f"preset '{p.id}': {dim} {size} is outside the configured limits")
            if s.num_inference_steps is not None and not 1 <= s.num_inference_steps <= settings.max_steps:
                raise ValueError(f"preset '{p.id}': steps exceed PROMPTCANVAS_MAX_STEPS")
            if s.guidance_scale is not None and not 0 <= s.guidance_scale <= settings.max_guidance_scale:
                raise ValueError(f"preset '{p.id}': guidance_scale exceeds the configured maximum")
            if s.num_images is not None and not 1 <= s.num_images <= settings.max_batch_size:
                raise ValueError(f"preset '{p.id}': num_images exceeds PROMPTCANVAS_MAX_BATCH_SIZE")


def load_catalog(settings: Settings) -> Catalog:
    path: Path = settings.catalog_path
    catalog = Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if settings.default_model:
        if catalog.model(settings.default_model) is None:
            raise ValueError(f"PROMPTCANVAS_DEFAULT_MODEL '{settings.default_model}' is not in {path}")
        catalog = catalog.model_copy(update={"default_model": settings.default_model})
    catalog.check_against(settings)
    return catalog
