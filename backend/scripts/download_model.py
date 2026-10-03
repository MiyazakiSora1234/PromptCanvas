"""Pre-download the configured model into the Hugging Face cache.

Uses the same settings as the server (env vars / backend/.env), and fetches only the
files the Diffusers pipeline needs (not every checkpoint in the repository).
Usage (from backend/): python -m scripts.download_model
"""

from __future__ import annotations

from typing import Any

from diffusers import DiffusionPipeline

from app.config import Settings


def main() -> None:
    s = Settings()
    kwargs: dict[str, Any] = {}
    if s.model_revision:
        kwargs["revision"] = s.model_revision
    if s.model_variant:
        kwargs["variant"] = s.model_variant
    if s.hf_token is not None:
        kwargs["token"] = s.hf_token.get_secret_value()

    print(f"Downloading {s.model_id} ...")
    pipeline_cls: Any = DiffusionPipeline
    path = pipeline_cls.download(s.model_id, **kwargs)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
