"""Pre-download catalog models and LoRAs into the Hugging Face cache.

Uses the same settings as the server (env vars / backend/.env, catalog.json) and fetches
only the files the Diffusers pipelines need (not every checkpoint in each repository).

Usage (from backend/):
    python -m scripts.download_model              # everything in catalog.json
    python -m scripts.download_model sdxl pixel-art-xl
"""

from __future__ import annotations

import argparse
from typing import Any

from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download

from app.catalog import load_catalog
from app.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="catalog ids to download (default: all)")
    args = parser.parse_args()

    settings = Settings()
    catalog = load_catalog(settings)
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    wanted = set(args.ids)
    known = {m.id for m in catalog.models} | {lora.id for lora in catalog.loras}
    if unknown := wanted - known:
        parser.error(f"unknown catalog ids: {', '.join(sorted(unknown))}")

    pipeline_cls: Any = DiffusionPipeline  # diffusers is only partially typed
    for model in catalog.models:
        if wanted and model.id not in wanted:
            continue
        print(f"[model] {model.id}: {model.repo} ...", flush=True)
        kwargs: dict[str, Any] = {"token": token, "revision": model.revision}
        if model.variant:
            kwargs["variant"] = model.variant
        print(f"  -> {pipeline_cls.download(model.repo, **kwargs)}")

    for lora in catalog.loras:
        if wanted and lora.id not in wanted:
            continue
        print(f"[lora] {lora.id}: {lora.repo} ...", flush=True)
        if lora.weight_name:
            path = hf_hub_download(lora.repo, lora.weight_name, revision=lora.revision, token=token)
        else:
            path = snapshot_download(lora.repo, revision=lora.revision, token=token)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
