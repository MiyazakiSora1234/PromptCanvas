from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.errors import ContentFilteredError
from app.generator import ModelState, ModelStatus

from .conftest import FakeGenerator

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MakeClient = Callable[..., TestClient]


def test_health_reports_ready(make_client: MakeClient) -> None:
    res = make_client().get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["queue"] == {"running": 0, "waiting": 0, "max_waiting": 4}
    assert "X-Request-ID" in res.headers


def test_config_exposes_limits_and_defaults(make_client: MakeClient) -> None:
    body = make_client(max_steps=30, default_steps=20).get("/api/config").json()
    assert body["limits"]["max_steps"] == 30
    assert body["limits"]["size_multiple"] == 8
    assert body["defaults"]["num_inference_steps"] == 20


def test_generate_returns_png_with_defaults(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post("/api/generate", json={"prompt": "  a cat  "})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content.startswith(PNG_SIGNATURE)
    assert res.headers["X-Seed"] == "1234"
    params = gen.calls[0]
    assert params.prompt == "a cat"
    assert (params.width, params.height, params.num_inference_steps) == (512, 512, 25)


def test_generate_uses_given_parameters(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post(
        "/api/generate",
        json={
            "prompt": "a dog",
            "negative_prompt": "blurry",
            "width": 768,
            "height": 512,
            "num_inference_steps": 10,
            "guidance_scale": 5,
            "seed": 42,
        },
    )
    assert res.status_code == 200
    assert res.headers["X-Seed"] == "42"
    assert gen.calls[0].negative_prompt == "blurry"
    assert gen.calls[0].width == 768


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"prompt": "   "}, "prompt"),
        ({"prompt": "x" * 1001}, "prompt"),
        ({"prompt": "a", "width": 500}, "width"),
        ({"prompt": "a", "height": 2048}, "height"),
        ({"prompt": "a", "num_inference_steps": 0}, "num_inference_steps"),
        ({"prompt": "a", "num_inference_steps": 51}, "num_inference_steps"),
        ({"prompt": "a", "guidance_scale": -1}, "guidance_scale"),
        ({"prompt": "a", "seed": -1}, "seed"),
        ({"prompt": "a", "seed": 2**32}, "seed"),
        ({"prompt": "a", "width": "wide"}, "width"),
        ({"prompt": "a", "unknown": 1}, "unknown"),
        ({}, "prompt"),
    ],
)
def test_generate_rejects_invalid_input(make_client: MakeClient, payload: dict[str, object], field: str) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post("/api/generate", json=payload)
    assert res.status_code == 422
    error = res.json()["error"]
    assert error["code"] == "invalid_input"
    assert field in [f["field"] for f in error["fields"]]
    assert gen.calls == []


def test_generate_rejects_malformed_json(make_client: MakeClient) -> None:
    res = make_client().post(
        "/api/generate", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "invalid_input"


@pytest.mark.parametrize(
    ("state", "code"),
    [(ModelState.LOADING, "model_loading"), (ModelState.FAILED, "model_unavailable")],
)
def test_generate_when_model_not_ready(make_client: MakeClient, state: ModelState, code: str) -> None:
    res = make_client(FakeGenerator(state=state)).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == 503
    assert res.json()["error"]["code"] == code
    assert "Retry-After" in res.headers


def test_failed_model_message_is_shown(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    gen._status = ModelStatus(ModelState.FAILED, message="モデルが見つかりません。")
    res = make_client(gen).post("/api/generate", json={"prompt": "a"})
    assert "モデルが見つかりません。" in res.json()["error"]["message"]


def test_gpu_out_of_memory_is_reported(make_client: MakeClient) -> None:
    gen = FakeGenerator(error=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    res = make_client(gen).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "gpu_out_of_memory"


def test_content_filter_is_reported(make_client: MakeClient) -> None:
    res = make_client(FakeGenerator(error=ContentFilteredError())).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "content_filtered"


def test_unexpected_error_does_not_leak_details(make_client: MakeClient) -> None:
    secret = "C:\\secret\\path token=hf_abcdef"
    res = make_client(FakeGenerator(error=ValueError(secret))).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == 500
    error = res.json()["error"]
    assert error["code"] == "generation_failed"
    assert error["request_id"]
    assert secret not in res.text
    assert "hf_abcdef" not in res.text


def test_unknown_api_route_returns_json_error(make_client: MakeClient) -> None:
    res = make_client().get("/api/nope")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_frontend_build_is_served(make_client: MakeClient, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<title>PromptCanvas</title>", encoding="utf-8")
    client = make_client(serve_frontend=True, frontend_dir=tmp_path)
    res = client.get("/")
    assert res.status_code == 200
    assert "PromptCanvas" in res.text
    # API routes still win over the static mount.
    assert client.get("/api/health").json()["status"] == "ready"


def test_missing_frontend_build_serves_api_only(make_client: MakeClient, tmp_path: Path) -> None:
    client = make_client(serve_frontend=True, frontend_dir=tmp_path / "missing")
    assert client.get("/").status_code == 404
    assert client.get("/api/health").status_code == 200
