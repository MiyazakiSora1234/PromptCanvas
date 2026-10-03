from __future__ import annotations

import base64
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.errors import ContentFilteredError, LoraUnavailableError
from app.generator import ModelState

from .conftest import CATALOG, FakeGenerator, image_data_url

MakeClient = Callable[..., TestClient]


def _decode(item: dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(item["data"])))


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
    assert [lora["id"] for lora in body["loras"]] == ["pixel", "style15"]
    assert [f["id"] for f in body["output_formats"]] == ["png", "jpeg", "webp"]
    assert body["identity"] == {"families": ["sdxl"], "download_size_gb": 6.6}
    assert body["limits"]["max_control_strength"] == 1.5
    # Repository names are server-side details, never exposed.
    assert "test/sdxl" not in str(body)


# --- generate: basics -------------------------------------------------------------


def test_generate_returns_png_with_model_defaults(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post("/api/generate", json={"prompt": "  a cat  "})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "sd15"
    assert body["scheduler"] == "default"
    assert body["filtered_count"] == 0
    [image] = body["images"]
    assert image["mime_type"] == "image/png"
    assert image["seed"] == 1234
    assert _decode(image).size == (512, 512)
    params = gen.calls[0]
    assert params.prompt == "a cat"
    assert (params.width, params.height, params.num_inference_steps) == (512, 512, 25)


def test_selected_model_supplies_its_own_defaults(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    body = make_client(gen).post("/api/generate", json={"prompt": "a dog", "model": "sdxl"}).json()
    assert body["model"] == "sdxl"
    assert body["scheduler"] == "euler_a"
    assert (body["width"], body["height"], body["num_inference_steps"]) == (1024, 1024, 30)
    assert gen.calls[0].model.repo == "test/sdxl"


def test_generate_uses_given_parameters(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post(
        "/api/generate",
        json={
            "prompt": "a dog",
            "negative_prompt": "blurry",
            "scheduler": "dpmpp_2m_karras",
            "width": 768,
            "height": 512,
            "num_inference_steps": 10,
            "guidance_scale": 5,
            "seed": 42,
        },
    )
    assert res.status_code == 200
    assert res.json()["images"][0]["seed"] == 42
    params = gen.calls[0]
    assert (params.negative_prompt, params.scheduler, params.width) == ("blurry", "dpmpp_2m_karras", 768)


def test_batch_returns_one_image_per_seed(make_client: MakeClient) -> None:
    body = make_client().post("/api/generate", json={"prompt": "a", "num_images": 3, "seed": 10}).json()
    assert [img["seed"] for img in body["images"]] == [10, 11, 12]


@pytest.mark.parametrize(
    ("fmt", "mime", "pil_format"), [("jpeg", "image/jpeg", "JPEG"), ("webp", "image/webp", "WEBP")]
)
def test_output_formats(make_client: MakeClient, fmt: str, mime: str, pil_format: str) -> None:
    body = make_client().post("/api/generate", json={"prompt": "a", "output_format": fmt, "quality": 70}).json()
    image = body["images"][0]
    assert image["mime_type"] == mime
    assert _decode(image).format == pil_format
    assert body["output_format"] == fmt


# --- generate: img2img and LoRA -------------------------------------------------


def test_img2img_passes_decoded_image(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post(
        "/api/generate", json={"prompt": "a", "init_image": image_data_url((64, 48)), "strength": 0.5}
    )
    assert res.status_code == 200
    params = gen.calls[0]
    assert params.init_image is not None
    assert params.init_image.size == (64, 48)
    assert params.strength == 0.5


def test_loras_are_resolved_with_default_scale(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post(
        "/api/generate", json={"prompt": "a", "model": "sdxl", "loras": [{"id": "pixel"}]}
    )
    assert res.status_code == 200
    [(lora, scale)] = gen.calls[0].loras
    assert (lora.id, scale) == ("pixel", 0.8)


def test_face_and_pose_reference_are_passed_to_the_generator(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    res = make_client(gen).post(
        "/api/generate",
        json={
            "prompt": "a woman in a red dress, dancing",
            "model": "sdxl",
            "face_image": image_data_url((80, 80)),
            "pose_image": image_data_url((60, 90)),
            "identity_strength": 1.0,
            "pose_strength": 0.5,
        },
    )
    assert res.status_code == 200
    params = gen.calls[0]
    assert params.uses_reference
    assert params.face_image is not None and params.face_image.size == (80, 80)
    assert params.pose_image is not None and params.pose_image.size == (60, 90)
    assert (params.identity_strength, params.pose_strength) == (1.0, 0.5)


def test_reference_is_sdxl_only(make_client: MakeClient) -> None:
    res = make_client().post("/api/generate", json={"prompt": "a", "face_image": image_data_url()})
    [field] = res.json()["error"]["fields"]
    assert field["field"] == "face_image"
    assert "SDXL" in field["message"]


def test_reference_disabled_without_catalog_entry(make_client: MakeClient) -> None:
    catalog = CATALOG.model_copy(update={"identity": None})
    client = make_client(catalog=catalog)
    assert client.get("/api/config").json()["identity"] is None
    res = client.post("/api/generate", json={"prompt": "a", "model": "sdxl", "pose_image": image_data_url()})
    [field] = res.json()["error"]["fields"]
    assert field["field"] == "pose_image"
    assert "有効になっていません" in field["message"]


# --- validation -----------------------------------------------------------------


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
        ({"prompt": "a", "model": "nope"}, "model"),
        ({"prompt": "a", "model": "test/sd15"}, "model"),
        ({"prompt": "a", "scheduler": "nope"}, "scheduler"),
        ({"prompt": "a", "num_images": 0}, "num_images"),
        ({"prompt": "a", "num_images": 5}, "num_images"),
        ({"prompt": "a", "output_format": "gif"}, "output_format"),
        ({"prompt": "a", "quality": 0}, "quality"),
        ({"prompt": "a", "init_image": "not base64!"}, "init_image"),
        ({"prompt": "a", "init_image": base64.b64encode(b"hello").decode()}, "init_image"),
        ({"prompt": "a", "init_image": image_data_url(), "strength": 0}, "strength"),
        ({"prompt": "a", "init_image": image_data_url(), "strength": 1.5}, "strength"),
        ({"prompt": "a", "init_image": image_data_url(), "num_inference_steps": 1, "strength": 0.5}, "strength"),
        ({"prompt": "a", "loras": [{"id": "nope"}]}, "loras"),
        ({"prompt": "a", "loras": [{"id": "pixel"}]}, "loras"),  # SDXL LoRA on the SD1.5 model
        ({"prompt": "a", "loras": [{"id": "style15", "scale": 3}]}, "loras"),
        ({"prompt": "a", "loras": [{"id": "style15"}, {"id": "style15"}]}, "loras"),
        ({"prompt": "a", "model": "sdxl", "face_image": "not base64!"}, "face_image"),
        ({"prompt": "a", "model": "sdxl", "pose_image": base64.b64encode(b"x").decode()}, "pose_image"),
        ({"prompt": "a", "model": "sdxl", "face_image": image_data_url(), "identity_strength": 2}, "identity_strength"),
        ({"prompt": "a", "model": "sdxl", "pose_image": image_data_url(), "pose_strength": -0.1}, "pose_strength"),
        (
            {"prompt": "a", "model": "sdxl", "face_image": image_data_url(), "init_image": image_data_url()},
            "face_image",
        ),
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


def test_incompatible_lora_message_names_the_model(make_client: MakeClient) -> None:
    res = make_client().post("/api/generate", json={"prompt": "a", "loras": [{"id": "pixel"}]})
    [field] = res.json()["error"]["fields"]
    assert "Pixel" in field["message"] and "SD 1.5" in field["message"]


def test_too_many_loras(make_client: MakeClient) -> None:
    res = make_client(max_loras=1).post(
        "/api/generate", json={"prompt": "a", "loras": [{"id": "style15"}, {"id": "style15"}]}
    )
    assert "loras" in [f["field"] for f in res.json()["error"]["fields"]]


def test_oversized_init_image(make_client: MakeClient) -> None:
    big = image_data_url((1500, 1500), fmt="BMP")  # ~6.75MB uncompressed
    res = make_client(max_init_image_mb=1).post("/api/generate", json={"prompt": "a", "init_image": big})
    [field] = res.json()["error"]["fields"]
    assert field["field"] == "init_image"
    assert "1MB" in field["message"]


def test_generate_rejects_malformed_json(make_client: MakeClient) -> None:
    res = make_client().post("/api/generate", content=b"{not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "invalid_input"


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
