from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from .conftest import CATALOG

# --- Identity (InstantID) helpers -------------------------------------------------


def test_identity_files_cover_every_asset() -> None:
    assert CATALOG.identity is not None
    files = CATALOG.identity.files()
    assert ("test/instantid", "ip-adapter.bin", None) in files
    assert ("test/instantid", "ControlNetModel/diffusion_pytorch_model.safetensors", None) in files
    assert ("test/faces", "det.onnx", None) in files
    assert ("test/annotators", "body.pth", None) in files
    assert len(files) == 8
    with_sketch = type(CATALOG.identity).model_validate(
        {**CATALOG.identity.model_dump(), "sketch_controlnet": {"repo": "test/scribble"}}
    )
    assert ("test/scribble", "diffusion_pytorch_model.safetensors", None) in with_sketch.files()


def test_extract_lines_keeps_strokes_and_drops_shading_and_specks() -> None:
    pytest.importorskip("cv2")
    from PIL import ImageDraw

    from app.generation.identity import extract_lines

    img = Image.new("RGB", (400, 400), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 300, 400, 400), fill=(200, 200, 200))  # soft shading
    draw.line((50, 50, 350, 250), fill=(40, 40, 40), width=3)  # a stroke
    draw.point((380, 20), fill=(0, 0, 0))  # a speck
    lines = np.asarray(extract_lines(img).convert("L"))
    assert lines[150, 200] == 255  # on the stroke
    assert lines[20, 380] == 0
    assert lines[350, 200] == 0  # inside the shaded area
    assert lines[10, 10] == 0


def test_draw_kps_renders_keypoints_at_their_positions() -> None:
    pytest.importorskip("cv2")
    from app.generation.identity import draw_kps

    kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)
    img = draw_kps((100, 120), kps)
    assert img.size == (100, 120)
    pixels = np.asarray(img)
    assert pixels[40, 30].tolist() == [255, 0, 0]  # left eye, full-color dot
    assert pixels[5, 5].tolist() == [0, 0, 0]  # background stays black


def _person(seed: int, noise: float = 0.0, base: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base if base is not None else rng.normal(size=512)
    return (v + noise * rng.normal(size=512)).astype(np.float32)


def test_combined_embedding_keeps_direction_and_magnitude() -> None:
    from app.generation.identity import combine_embeddings

    a = _person(1)
    b = _person(2, noise=0.3, base=a) * 1.2  # same person, different photo and scale
    combined = combine_embeddings([a, b])
    assert combined.dtype == np.float32
    expected_norm = (np.linalg.norm(a) + np.linalg.norm(b)) / 2
    assert abs(np.linalg.norm(combined) - expected_norm) < 1e-3 * expected_norm
    cos = float(np.dot(combined, a) / (np.linalg.norm(combined) * np.linalg.norm(a)))
    assert cos > 0.9


def test_find_outliers_flags_a_different_person() -> None:
    from app.generation.identity import find_outliers

    alice = _person(1)
    photos = [_person(10 + i, noise=0.4, base=alice) for i in range(3)]
    assert find_outliers(photos) == []
    assert find_outliers([photos[0]]) == []  # a single photo has nothing to compare with
    bob = _person(99)
    assert find_outliers([*photos[:2], bob]) == [2]


def test_identity_tokens_follow_the_guidance_batch_layout() -> None:
    torch = pytest.importorskip("torch")
    from app.generation.identity import _IdentityTokens

    tokens = _IdentityTokens()
    tokens.cond = torch.ones(1, 16, 8)
    tokens.uncond = torch.zeros(1, 16, 8)
    tokens.batch = 2
    with_cfg = tokens.for_batch(4)  # [uncond, uncond, cond, cond]
    assert with_cfg.shape == (4, 16, 8)
    assert with_cfg[:2].sum() == 0 and bool((with_cfg[2:] == 1).all())
    assert bool((tokens.for_batch(2) == 1).all())  # no CFG: cond only

    seen: dict[str, Any] = {}

    class FakeNet:
        def forward(self, *args: Any, **kwargs: Any) -> str:
            seen.update(kwargs)
            return "ok"

    net = FakeNet()
    tokens.patch(net)
    assert net.forward(sample=torch.zeros(4, 4), timestep=1, encoder_hidden_states="text") == "ok"
    assert seen["encoder_hidden_states"].shape == (4, 16, 8)
