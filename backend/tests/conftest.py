from __future__ import annotations

import base64
import io
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.catalog import Catalog
from app.config import Settings
from app.errors import GenerationCancelledError
from app.generator import GeneratedImage, GenerationResult, ModelState, ModelStatus, batch_seeds
from app.imaging import OUTPUT_FORMATS, encode_image
from app.main import create_app
from app.schemas import GenerationParams

CATALOG = Catalog.model_validate(
    {
        "default_model": "sd15",
        "models": [
            {
                "id": "sd15",
                "label": "SD 1.5",
                "repo": "test/sd15",
                "family": "sd15",
                "defaults": {"width": 512, "height": 512, "num_inference_steps": 25, "guidance_scale": 7.5},
            },
            {
                "id": "sdxl",
                "label": "SDXL",
                "repo": "test/sdxl",
                "family": "sdxl",
                "defaults": {
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.0,
                    "scheduler": "euler_a",
                    "style": "photo",
                },
            },
        ],
        "loras": [
            {"id": "pixel", "label": "Pixel", "repo": "test/pixel", "family": "sdxl", "default_scale": 0.8},
            {"id": "style15", "label": "Style15", "repo": "test/style15", "family": "sd15"},
        ],
        "presets": [
            {
                "id": "real",
                "label": "Real",
                "settings": {"model": "sdxl", "style": "photo", "width": 896, "height": 1152, "loras": []},
            }
        ],
        "identity": {
            "families": ["sdxl"],
            "download_size_gb": 6.6,
            "instantid": {"repo": "test/instantid"},
            "face_models": {"repo": "test/faces", "detection": "det.onnx", "recognition": "rec.onnx"},
            "pose_controlnet": {"repo": "test/openpose"},
            "pose_detector": {"repo": "test/annotators", "weight_name": "body.pth"},
        },
    }
)


class FakeGenerator:
    """Stands in for DiffusersGenerator so tests need neither torch nor a GPU."""

    def __init__(self, state: ModelState = ModelState.READY, error: Exception | None = None) -> None:
        self._status = ModelStatus(state, model="sd15", device="cpu", dtype="float32")
        self.error = error
        self.calls: list[GenerationParams] = []
        # Runs at the start of generate() (e.g. to cancel the job mid-flight).
        self.during: Callable[[], object] | None = None

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def cached_models(self) -> frozenset[str]:
        return frozenset({"sd15"})

    @property
    def identity_cached(self) -> bool:
        return False

    def load(self) -> None:
        pass

    def generate(self, params: GenerationParams, cancel: threading.Event | None = None) -> GenerationResult:
        self.calls.append(params)
        if self.during is not None:
            self.during()
        if cancel is not None and cancel.is_set():
            raise GenerationCancelledError()
        if self.error is not None:
            raise self.error
        img = Image.new("RGB", (params.width, params.height), "white")
        data = encode_image(img, params.output_format, params.quality)
        mime = OUTPUT_FORMATS[params.output_format].mime_type
        seed = 1234 if params.seed is None else params.seed
        return GenerationResult(images=[GeneratedImage(data, mime, s) for s in batch_seeds(seed, params.num_images)])


def make_settings(**overrides: Any) -> Settings:
    overrides.setdefault("serve_frontend", False)
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def image_data_url(size: tuple[int, int] = (64, 48), fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


@pytest.fixture
def make_client() -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(generator: FakeGenerator | None = None, catalog: Catalog = CATALOG, **overrides: Any) -> TestClient:
        app = create_app(make_settings(**overrides), generator or FakeGenerator(), catalog)
        client = TestClient(app)
        client.__enter__()  # run lifespan (creates the limiter)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
