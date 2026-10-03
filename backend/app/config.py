"""Application settings, loaded from environment variables and ``backend/.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

# Stable Diffusion family models work on latents downsampled by 8.
SIZE_MULTIPLE = 8
SEED_MAX = 2**32 - 1

DeviceSetting = Literal["auto", "cuda", "mps", "cpu"]
DtypeSetting = Literal["auto", "float16", "bfloat16", "float32"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """All runtime configuration. Env vars use the ``PROMPTCANVAS_`` prefix (except ``HF_TOKEN``)."""

    model_config = SettingsConfigDict(
        env_prefix="PROMPTCANVAS_",
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Models ----------------------------------------------------------
    # Allow-list of selectable models and LoRAs (see catalog.json).
    catalog_path: Path = BACKEND_DIR / "catalog.json"
    # Catalog id loaded at startup; empty = the catalog's "default_model".
    default_model: str | None = None
    # Only needed for gated/private models. Never logged or returned in responses.
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "PROMPTCANVAS_HF_TOKEN"),
    )

    # --- Device ----------------------------------------------------------
    device: DeviceSetting = "auto"
    torch_dtype: DtypeSetting = "auto"
    enable_attention_slicing: bool = False
    enable_cpu_offload: bool = False

    # --- Input limits (per-model defaults live in catalog.json) ------------
    min_image_size: int = Field(default=256, ge=64)
    max_image_size: int = Field(default=1536, le=2048)
    max_steps: int = Field(default=50, ge=1, le=200)
    max_guidance_scale: float = Field(default=20.0, ge=0)
    max_prompt_length: int = Field(default=1000, ge=1, le=10_000)
    # Images per request; each one costs time and VRAM.
    max_batch_size: int = Field(default=4, ge=1, le=16)
    max_loras: int = Field(default=3, ge=0, le=10)
    # img2img upload limit (decoded bytes).
    max_init_image_mb: int = Field(default=10, ge=1, le=50)

    # --- Concurrency -----------------------------------------------------
    # One pipeline instance runs one generation at a time; these bound the queue behind it.
    max_queue_size: int = Field(default=4, ge=0, le=100)
    queue_timeout_seconds: float = Field(default=300.0, gt=0)

    # --- Server ----------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=list)
    serve_frontend: bool = True
    # Output of `npm run build` in frontend/.
    frontend_dir: Path = PROJECT_DIR / "frontend" / "dist"
    log_level: LogLevel = "INFO"

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.min_image_size > self.max_image_size:
            raise ValueError("min_image_size must be <= max_image_size")
        for name in ("min_image_size", "max_image_size"):
            if getattr(self, name) % SIZE_MULTIPLE:
                raise ValueError(f"{name} must be a multiple of {SIZE_MULTIPLE}")
        return self
