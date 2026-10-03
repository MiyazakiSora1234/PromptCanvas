"""Whether a Diffusers model is fully present in the local Hugging Face cache.

Used to warn users before they pick a model that would trigger a multi-GB download.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

logger = logging.getLogger(__name__)

# Pipeline components that carry weights (tokenizers, schedulers etc. are tiny configs).
_WEIGHT_COMPONENTS = {
    "unet",
    "transformer",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "safety_checker",
    "image_encoder",
}


def is_model_cached(repo: str, revision: str | None = None) -> bool:
    """True if every weight-bearing component listed in model_index.json has a weights file.

    Files only appear in the snapshot folder once completely downloaded (partial ones stay
    as ``*.incomplete`` blobs), so an interrupted download is reported as not cached.
    """
    local = Path(repo)
    if local.is_dir():
        return True
    try:
        index = try_to_load_from_cache(repo, "model_index.json", revision=revision)
        if not isinstance(index, str):
            return False
        root = Path(index).parent
        components = json.loads(Path(index).read_text(encoding="utf-8"))
        for name, spec in components.items():
            if name not in _WEIGHT_COMPONENTS or not isinstance(spec, list) or spec[0] is None:
                continue
            folder = root / name
            if not any(folder.glob("*.safetensors")) and not any(folder.glob("*.bin")):
                return False
        return True
    except Exception:  # cache layout surprises must never break the app
        logger.debug("Cache check failed for %s", repo, exc_info=True)
        return False
