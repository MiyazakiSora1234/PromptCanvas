from __future__ import annotations

import base64
from typing import Any

import pytest

from app.generation.styles import apply_style

from .conftest import CATALOG, FakeGenerator, MakeClient, decode_image, image_data_url

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
    assert decode_image(image).size == (512, 512)
    params = gen.calls[0]
    assert params.prompt == "a cat"
    assert (params.width, params.height, params.num_inference_steps) == (512, 512, 25)


def test_selected_model_supplies_its_own_defaults(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    body = make_client(gen).post("/api/generate", json={"prompt": "a dog", "model": "sdxl"}).json()
    assert body["model"] == "sdxl"
    assert body["scheduler"] == "euler_a"
    assert body["style"] == "photo"
    assert (body["width"], body["height"], body["num_inference_steps"]) == (1024, 1024, 30)
    assert gen.calls[0].model.repo == "test/sdxl"


def test_photo_style_adds_realism_wording(make_client: MakeClient) -> None:
    gen = FakeGenerator()
    client = make_client(gen)
    client.post("/api/generate", json={"prompt": "a woman", "negative_prompt": "hat", "style": "photo"})
    params = gen.calls[0]
    # The generator adds the style's wording (after translating Japanese prompts).
    assert (params.prompt, params.negative_prompt, params.style) == ("a woman", "hat", "photo")
    prompt, negative = apply_style(params.style, params.prompt, params.negative_prompt)
    assert prompt.startswith("RAW photo, a woman, detailed skin texture")
    assert negative.startswith("hat, plastic skin, airbrushed")

    client.post("/api/generate", json={"prompt": "a woman", "model": "sdxl", "style": "none"})
    assert apply_style(gen.calls[1].style, gen.calls[1].prompt, gen.calls[1].negative_prompt) == ("a woman", "")


def test_translated_prompts_are_returned(make_client: MakeClient) -> None:
    client = make_client()
    body = client.post("/api/generate", json={"prompt": "猫, 1girl", "negative_prompt": "ぼやけた"}).json()
    assert body["translated_prompt"] == "EN(猫, 1girl)"
    assert body["translated_negative_prompt"] == "EN(ぼやけた)"
    english = client.post("/api/generate", json={"prompt": "a cat"}).json()
    assert english["translated_prompt"] is None
    assert client.get("/api/config").json()["translation"] is None  # the test catalog has no translator


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
    assert decode_image(image).format == pil_format
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
    res = make_client(gen).post("/api/generate", json={"prompt": "a", "model": "sdxl", "loras": [{"id": "pixel"}]})
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
            "face_images": [image_data_url((80, 80)), image_data_url((60, 60))],
            "pose_image": image_data_url((60, 90)),
            "identity_strength": 1.0,
            "pose_strength": 0.5,
        },
    )
    assert res.status_code == 200
    params = gen.calls[0]
    assert params.uses_reference
    assert [img.size for img in params.face_images] == [(80, 80), (60, 60)]
    assert params.pose_image is not None and params.pose_image.size == (60, 90)
    assert (params.identity_strength, params.pose_strength) == (1.0, 0.5)


def test_face_photo_errors_name_the_photo(make_client: MakeClient) -> None:
    res = make_client().post(
        "/api/generate",
        json={"prompt": "a", "model": "sdxl", "face_images": [image_data_url(), base64.b64encode(b"x").decode()]},
    )
    [field] = res.json()["error"]["fields"]
    assert field["field"] == "face_images"
    assert field["message"].startswith("2枚目: ")


def test_face_photo_limit(make_client: MakeClient) -> None:
    res = make_client(max_face_images=2).post(
        "/api/generate", json={"prompt": "a", "model": "sdxl", "face_images": [image_data_url()] * 3}
    )
    assert "2枚まで" in res.json()["error"]["fields"][0]["message"]
    assert make_client().get("/api/config").json()["limits"]["max_face_images"] == 5


def test_running_job_can_be_cancelled(make_client: MakeClient) -> None:
    generator = FakeGenerator()
    client = make_client(generator)
    cancelled: list[Any] = []
    generator.during = lambda: cancelled.append(client.post("/api/jobs/job-0001/cancel").json())
    res = client.post("/api/generate", json={"prompt": "a", "job_id": "job-0001"})
    assert cancelled == [{"cancelled": True}]
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "cancelled"
    # Finished jobs are forgotten; unknown ids are not an error.
    assert client.post("/api/jobs/job-0001/cancel").json() == {"cancelled": False}


def test_job_id_format_is_validated(make_client: MakeClient) -> None:
    client = make_client()
    res = client.post("/api/generate", json={"prompt": "a", "job_id": "x"})
    assert res.status_code == 422
    assert res.json()["error"]["fields"][0]["field"] == "job_id"
    assert client.post("/api/jobs/bad!id/cancel").status_code == 422


def test_reference_is_sdxl_only(make_client: MakeClient) -> None:
    res = make_client().post("/api/generate", json={"prompt": "a", "face_images": [image_data_url()]})
    [field] = res.json()["error"]["fields"]
    assert field["field"] == "face_images"
    assert "SDXL" in field["message"]


def test_reference_disabled_without_catalog_entry(make_client: MakeClient) -> None:
    catalog = CATALOG.model_copy(update={"identity": None})
    client = make_client(catalog=catalog)
    assert client.get("/api/config").json()["identity"] is None
    res = client.post("/api/generate", json={"prompt": "a", "model": "sdxl", "pose_image": image_data_url()})
    [field] = res.json()["error"]["fields"]
    assert field["field"] == "pose_image"
    assert "有効になっていません" in field["message"]
