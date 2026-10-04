from __future__ import annotations

import base64

import pytest

from .conftest import FakeGenerator, MakeClient, image_data_url

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
        ({"prompt": "a", "style": "nope"}, "style"),
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
        ({"prompt": "a", "model": "sdxl", "face_images": [image_data_url(), "not base64!"]}, "face_images"),
        ({"prompt": "a", "model": "sdxl", "pose_image": base64.b64encode(b"x").decode()}, "pose_image"),
        (
            {"prompt": "a", "model": "sdxl", "face_images": [image_data_url()], "identity_strength": 2},
            "identity_strength",
        ),
        ({"prompt": "a", "model": "sdxl", "pose_image": image_data_url(), "pose_strength": -0.1}, "pose_strength"),
        (
            {"prompt": "a", "model": "sdxl", "face_images": [image_data_url()], "init_image": image_data_url()},
            "face_images",
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
