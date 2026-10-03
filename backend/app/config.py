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

    # --- Model -----------------------------------------------------------
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    model_revision: str | None = None
    model_variant: str | None = None
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

    # --- Input limits and defaults ----------------------------------------
    min_image_size: int = Field(default=256, ge=64)
    max_image_size: int = Field(default=1024, le=2048)
    default_width: int = 512
    default_height: int = 512
    max_steps: int = Field(default=50, ge=1, le=200)
    default_steps: int = Field(default=25, ge=1)
    max_guidance_scale: float = Field(default=20.0, ge=0)
    default_guidance_scale: float = Field(default=7.5, ge=0)
    max_prompt_length: int = Field(default=1000, ge=1, le=10_000)

    # --- Concurrency -----------------------------------------------------
    # One pipeline instance runs one generation at a time; these bound the queue behind it.
    max_queue_size: int = Field(default=4, ge=0, le=100)
    queue_timeout_seconds: float = Field(default=300.0, gt=0)

    # --- Server ----------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=list)
    serve_frontend: bool = True
    frontend_dir: Path = PROJECT_DIR / "frontend"
    log_level: LogLevel = "INFO"

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.min_image_size > self.max_image_size:
            raise ValueError("min_image_size must be <= max_image_size")
        for name in ("min_image_size", "max_image_size", "default_width", "default_height"):
            if getattr(self, name) % SIZE_MULTIPLE:
                raise ValueError(f"{name} must be a multiple of {SIZE_MULTIPLE}")
        for name in ("default_width", "default_height"):
            if not self.min_image_size <= getattr(self, name) <= self.max_image_size:
                raise ValueError(f"{name} must be within [min_image_size, max_image_size]")
        if self.default_steps > self.max_steps:
            raise ValueError("default_steps must be <= max_steps")
        if self.default_guidance_scale > self.max_guidance_scale:
            raise ValueError("default_guidance_scale must be <= max_guidance_scale")
        return self
