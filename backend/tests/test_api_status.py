from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import ContentFilteredError, LoraUnavailableError
from app.generation.models import ModelState

from .conftest import FakeGenerator, MakeClient

# --- health / config -----------------------------------------------------------


def test_health_reports_ready(make_client: MakeClient) -> None:
    res = make_client().get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["model"] == "sd15"
    assert body["cached_models"] == ["sd15"]
    assert body["identity_cached"] is False
    assert body["queue"] == {"running": 0, "waiting": 0, "max_waiting": 4}
    assert "X-Request-ID" in res.headers


def test_config_exposes_catalog_and_limits(make_client: MakeClient) -> None:
    body = make_client(max_steps=40, max_batch_size=2).get("/api/config").json()
    assert body["limits"]["max_steps"] == 40
    assert body["limits"]["max_batch_size"] == 2
    assert body["default_model"] == "sd15"
    assert [m["id"] for m in body["models"]] == ["sd15", "sdxl"]
    assert body["models"][1]["defaults"]["scheduler"] == "euler_a"
    assert {"default", "euler_a", "dpmpp_2m_karras"} <= {s["id"] for s in body["schedulers"]}
    style_ids = [s["id"] for s in body["styles"]]
    assert style_ids[:2] == ["none", "photo"]
    assert {"anime", "watercolor", "cinematic", "pixel"} <= set(style_ids)
    assert all(s["description"] for s in body["styles"])
    assert body["models"][1]["defaults"]["style"] == "photo"
    assert [lora["id"] for lora in body["loras"]] == ["pixel", "style15"]
    assert [f["id"] for f in body["output_formats"]] == ["png", "jpeg", "webp"]
    assert body["identity"] == {"families": ["sdxl"], "download_size_gb": 6.6}
    [preset] = body["presets"]
    assert preset["id"] == "real"
    assert preset["settings"]["model"] == "sdxl"
    assert preset["settings"]["width"] == 896
    assert body["limits"]["max_control_strength"] == 1.5
    # Repository names are server-side details, never exposed.
    assert "test/sdxl" not in str(body)


# --- model state / failures -----------------------------------------------------


def test_generate_while_model_loading(make_client: MakeClient) -> None:
    res = make_client(FakeGenerator(state=ModelState.LOADING)).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "model_loading"
    assert "Retry-After" in res.headers


def test_failed_model_does_not_block_retry(make_client: MakeClient) -> None:
    gen = FakeGenerator(state=ModelState.FAILED)
    res = make_client(gen).post("/api/generate", json={"prompt": "a", "model": "sdxl"})
    assert res.status_code == 200
    assert len(gen.calls) == 1


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), 503, "gpu_out_of_memory"),
        (ContentFilteredError(), 422, "content_filtered"),
        (LoraUnavailableError("LoRA「Pixel」を読み込めませんでした。"), 503, "lora_unavailable"),
    ],
)
def test_generation_errors_are_reported(make_client: MakeClient, error: Exception, status: int, code: str) -> None:
    res = make_client(FakeGenerator(error=error)).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == status
    assert res.json()["error"]["code"] == code


def test_unexpected_error_does_not_leak_details(make_client: MakeClient) -> None:
    secret = "C:\\secret\\path token=hf_abcdef"
    res = make_client(FakeGenerator(error=ValueError(secret))).post("/api/generate", json={"prompt": "a"})
    assert res.status_code == 500
    error = res.json()["error"]
    assert error["code"] == "generation_failed"
    assert error["request_id"]
    assert secret not in res.text
    assert "hf_abcdef" not in res.text


# --- routing / static files ------------------------------------------------------


def test_unknown_api_route_returns_json_error(make_client: MakeClient) -> None:
    res = make_client().get("/api/nope")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_frontend_build_is_served(make_client: MakeClient, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<title>PromptCanvas</title>", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    client = make_client(serve_frontend=True, frontend_dir=tmp_path)
    res = client.get("/")
    assert res.status_code == 200
    assert "PromptCanvas" in res.text
    # A rebuilt UI must show up on reload; hashed assets never change.
    assert res.headers["Cache-Control"] == "no-cache"
    assert "immutable" in client.get("/assets/index-abc123.js").headers["Cache-Control"]
    # API routes still win over the static mount.
    assert client.get("/api/health").json()["status"] == "ready"


def test_missing_frontend_build_serves_api_only(make_client: MakeClient, tmp_path: Path) -> None:
    client = make_client(serve_frontend=True, frontend_dir=tmp_path / "missing")
    assert client.get("/").status_code == 404
    assert client.get("/api/health").status_code == 200
