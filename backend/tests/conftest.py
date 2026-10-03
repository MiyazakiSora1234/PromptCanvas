from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.generator import GenerationResult, ModelState, ModelStatus
from app.main import create_app
from app.schemas import GenerationParams


class FakeGenerator:
    """Stands in for DiffusersGenerator so tests need neither torch nor a GPU."""

    def __init__(self, state: ModelState = ModelState.READY, error: Exception | None = None) -> None:
        self._status = ModelStatus(state, device="cpu", dtype="float32")
        self.error = error
        self.calls: list[GenerationParams] = []

    @property
    def status(self) -> ModelStatus:
        return self._status

    def load(self) -> None:
        pass

    def generate(self, params: GenerationParams) -> GenerationResult:
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        buffer = io.BytesIO()
        Image.new("RGB", (params.width, params.height), "white").save(buffer, format="PNG")
        return GenerationResult(png=buffer.getvalue(), seed=params.seed if params.seed is not None else 1234)


def make_settings(**overrides: Any) -> Settings:
    overrides.setdefault("serve_frontend", False)
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture
def make_client() -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(generator: FakeGenerator | None = None, **overrides: Any) -> TestClient:
        app = create_app(make_settings(**overrides), generator or FakeGenerator())
        client = TestClient(app)
        client.__enter__()  # run lifespan (creates the limiter)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
