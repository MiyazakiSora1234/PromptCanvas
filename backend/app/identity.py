"""Keep a person's face (InstantID) and copy a pose (OpenPose ControlNet) for SDXL models.

InstantID = an "IdentityNet" ControlNet driven by 5 facial keypoints + an IP-Adapter
fed with an ArcFace embedding of the reference face. Diffusers has no built-in InstantID
pipeline, so this module composes it from standard pieces:

* the adapter weights load through ``load_ip_adapter`` (same layout as IP-Adapter Plus);
* IdentityNet is a regular ``ControlNetModel`` whose cross-attention input is replaced
  with the projected face tokens instead of the text embeddings (``_IdentityTokens``).

Heavy dependencies (torch, diffusers, insightface, controlnet_aux) are imported lazily.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .catalog import IdentityConfig
from .errors import FieldError, InvalidInputError
from .imaging import fit_to

logger = logging.getLogger(__name__)

_NO_FACE = "顔を検出できませんでした。顔が正面に近く、はっきり写った写真を選んでください。"
_NO_FACE_IN_POSE = (
    "この画像から顔の位置を検出できませんでした。顔が写っている画像を選んでください"
    "（顔の位置はポーズ参考画像の人物に合わせます）。"
)
_FACE_CROPPED = "出力サイズに切り抜くと顔が見切れてしまいます。サイズの縦横比を元画像に合わせてください。"
_NO_POSE = "ポーズを検出できませんでした。人物の全身または上半身が写った画像を選んでください。"

# Colors / limbs of the InstantID keypoint image (eyes, nose, mouth corners).
_KPS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
_KPS_LIMBS = [(0, 2), (1, 2), (3, 2), (4, 2)]


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray  # (512,) ArcFace embedding (not normalized, as InstantID expects)
    kps: np.ndarray  # (5, 2) keypoints in image pixels
    area: float


# Cosine similarity of ArcFace embeddings: same person is typically > 0.4 even across angles and
# lighting, different people < 0.2. Below this, a photo is reported as likely someone else.
SAME_PERSON_MIN_SIMILARITY = 0.2


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v / (np.linalg.norm(v) + 1e-8))


def combine_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average several photos of one person: mean direction, rescaled to the mean magnitude
    (InstantID was trained on raw, un-normalized embeddings)."""
    direction = _unit(np.mean([_unit(e) for e in embeddings], axis=0))
    magnitude = float(np.mean([np.linalg.norm(e) for e in embeddings]))
    return (direction * magnitude).astype(np.float32)


def find_outliers(embeddings: list[np.ndarray], threshold: float = SAME_PERSON_MIN_SIMILARITY) -> list[int]:
    """Indexes of photos whose face doesn't match the rest (likely a different person)."""
    if len(embeddings) < 2:
        return []
    units = [_unit(e) for e in embeddings]
    outliers = []
    for i, u in enumerate(units):
        others = _unit(np.mean([v for j, v in enumerate(units) if j != i], axis=0))
        if float(np.dot(u, others)) < threshold:
            outliers.append(i)
    return outliers


def draw_kps(size: tuple[int, int], kps: np.ndarray) -> Image.Image:
    """Render facial keypoints the way InstantID's IdentityNet was trained on."""
    import cv2

    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    stick = 4
    for a, b in _KPS_LIMBS:
        x = kps[[a, b], 0]
        y = kps[[a, b], 1]
        length = float(np.hypot(x[0] - x[1], y[0] - y[1]))
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stick), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, np.asarray(polygon, dtype=np.int32), _KPS_COLORS[a])
    canvas = (canvas * 0.6).astype(np.uint8)
    for (x, y), color in zip(kps, _KPS_COLORS, strict=False):
        cv2.circle(canvas, (int(x), int(y)), 10, color, -1)
    return Image.fromarray(canvas)


class FaceAnalyzer:
    """SCRFD detection + ArcFace (glintr100) embedding from the antelopev2 pack, on CPU."""

    def __init__(self, detection_path: str, recognition_path: str) -> None:
        from insightface.model_zoo import get_model

        providers = ["CPUExecutionProvider"]  # ~0.2s per image; avoids CUDA/cuDNN version coupling
        self._det: Any = get_model(detection_path, providers=providers)
        self._det.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
        self._rec: Any = get_model(recognition_path, providers=providers)
        self._rec.prepare(ctx_id=-1)

    def largest_face(self, image: Image.Image) -> DetectedFace | None:
        """The biggest face in the image. Extreme close-ups (face filling the frame, common in
        selfies) defeat the detector, so if nothing is found it retries with a padded border."""
        rgb = np.asarray(image.convert("RGB"))
        found = self._detect(rgb, offset=0)
        if found is None:
            pad = max(rgb.shape[:2]) // 4
            found = self._detect(np.pad(rgb, ((pad, pad), (pad, pad), (0, 0))), offset=pad)
        return found

    def _detect(self, rgb: np.ndarray, *, offset: int) -> DetectedFace | None:
        from insightface.app.common import Face

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        bboxes, kpss = self._det.detect(bgr, max_num=0)
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas))
        face = Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
        self._rec.get(bgr, face)
        # Keypoints back in the original (unpadded) image's coordinates.
        return DetectedFace(np.asarray(face.embedding, dtype=np.float32), np.asarray(kpss[i]) - offset, float(areas[i]))


class _IdentityTokens:
    """Feeds IdentityNet the projected face tokens instead of the text embeddings."""

    def __init__(self) -> None:
        self.cond: Any = None  # (1, tokens, dim)
        self.uncond: Any = None
        self.batch = 1

    def for_batch(self, n: int) -> Any:
        import torch

        if n == 2 * self.batch:  # classifier-free guidance: [uncond x B, cond x B]
            return torch.cat([self.uncond.expand(self.batch, -1, -1), self.cond.expand(self.batch, -1, -1)])
        return self.cond.expand(n, -1, -1)

    def patch(self, controlnet: Any) -> None:
        original = controlnet.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            sample = kwargs["sample"] if "sample" in kwargs else args[0]
            tokens = self.for_batch(sample.shape[0])
            if "encoder_hidden_states" in kwargs:
                kwargs["encoder_hidden_states"] = tokens
            else:
                args = (*args[:2], tokens, *args[3:])
            return original(*args, **kwargs)

        controlnet.forward = forward


@dataclass
class ControlSetup:
    """What the ControlNet pipeline call needs for one request."""

    controlnets: list[Any]
    images: list[Image.Image]
    scales: list[float]
    ip_adapter_embeds: Any | None  # tensor (2|1, 1, 1, 512) or None


class IdentityConditioner:
    """Lazily loads and owns the InstantID / OpenPose assets."""

    def __init__(self, config: IdentityConfig, device: str, torch_dtype: Any, token: str | None) -> None:
        self._cfg = config
        self._device = device
        self._dtype = torch_dtype
        self._token = token
        self._faces: FaceAnalyzer | None = None
        self._pose_detector: Any = None
        self._identitynet: Any = None
        self._posenet: Any = None
        self._adapter_state: dict[str, Any] | None = None
        self._tokens = _IdentityTokens()

    # --- asset loading ------------------------------------------------------

    def _download(self, repo: str, filename: str, revision: str | None = None) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo, filename, revision=revision, token=self._token)

    def _face_analyzer(self) -> FaceAnalyzer:
        if self._faces is None:
            f = self._cfg.face_models
            self._faces = FaceAnalyzer(
                self._download(f.repo, f.detection, f.revision), self._download(f.repo, f.recognition, f.revision)
            )
        return self._faces

    def _pose(self) -> Any:
        if self._pose_detector is None:
            with warnings.catch_warnings():  # controlnet_aux imports many optional detectors noisily
                warnings.simplefilter("ignore")
                from controlnet_aux.open_pose import OpenposeDetector
                from controlnet_aux.open_pose.body import Body
            p = self._cfg.pose_detector
            body = Body(self._download(p.repo, p.weight_name, p.revision))
            # Body only (no hand/face models); OpenposeDetector.to() assumes all three exist.
            body.to(self._device)
            self._pose_detector = OpenposeDetector(body)
        return self._pose_detector

    def _controlnet(self, repo: str, subfolder: str | None, revision: str | None) -> Any:
        from diffusers import ControlNetModel

        cls: Any = ControlNetModel
        logger.info("Loading ControlNet %s/%s", repo, subfolder or "")
        # Kept in CPU RAM between requests (see park()); moved to the GPU only while generating.
        return cls.from_pretrained(
            repo, subfolder=subfolder, revision=revision, torch_dtype=self._dtype, token=self._token
        )

    def _identity_net(self) -> Any:
        if self._identitynet is None:
            i = self._cfg.instantid
            self._identitynet = self._controlnet(i.repo, i.controlnet_subfolder, i.revision)
            self._tokens.patch(self._identitynet)
        return self._identitynet

    def _pose_net(self) -> Any:
        if self._posenet is None:
            p = self._cfg.pose_controlnet
            self._posenet = self._controlnet(p.repo, p.subfolder, p.revision)
        return self._posenet

    def adapter_state(self) -> dict[str, Any]:
        if self._adapter_state is None:
            import torch

            i = self._cfg.instantid
            path = self._download(i.repo, i.adapter_weight_name, i.revision)
            # weights_only: never execute pickled code from a downloaded file.
            self._adapter_state = torch.load(path, map_location="cpu", weights_only=True)
        return self._adapter_state

    def release(self) -> None:
        """Drop everything (when switching to a model family InstantID can't use)."""
        self._identitynet = None
        self._posenet = None
        self._pose_detector = None

    def park(self) -> None:
        """Move the ControlNets (~2.4GB each) back to CPU RAM so plain generations keep their VRAM."""
        for net in (self._identitynet, self._posenet):
            if net is not None:
                net.to("cpu")

    # --- per request --------------------------------------------------------

    def prepare(
        self,
        unet: Any,
        *,
        face_images: tuple[Image.Image, ...],
        pose_image: Image.Image | None,
        width: int,
        height: int,
        identity_strength: float,
        pose_strength: float,
        num_images: int,
        guidance: bool,
    ) -> ControlSetup:
        """Detect the faces / pose and build ControlNet inputs. Raises InvalidInputError for unusable images.

        Several face photos are averaged into one identity. With a pose image, the face keypoints
        are taken from it too, so the face lands where the person in the pose image has theirs;
        otherwise from the first face photo's composition.
        """
        import torch

        layout_source = pose_image if pose_image is not None else face_images[0]
        layout = fit_to(layout_source, width, height)
        setup = ControlSetup([], [], [], None)

        if face_images:
            analyzer = self._face_analyzer()
            references = [analyzer.largest_face(img) for img in face_images]
            missing = [i + 1 for i, face in enumerate(references) if face is None]
            if missing:
                which = "、".join(f"{n}枚目" for n in missing)
                message = _NO_FACE if len(face_images) == 1 else f"{which}の写真から{_NO_FACE}"
                raise InvalidInputError(fields=[FieldError("face_images", message)])
            embeddings = [face.embedding for face in references if face is not None]
            outliers = find_outliers(embeddings)
            if outliers:
                which = "、".join(f"{i + 1}枚目" for i in outliers)
                raise InvalidInputError(
                    fields=[
                        FieldError(
                            "face_images",
                            f"{which}の顔が他の写真と大きく異なります（別人の可能性があります）。"
                            "同じ人物の写真だけを選んでください。",
                        )
                    ]
                )
            placed = analyzer.largest_face(layout)
            if placed is None:
                if pose_image is not None:
                    raise InvalidInputError(fields=[FieldError("pose_image", _NO_FACE_IN_POSE)])
                raise InvalidInputError(fields=[FieldError("face_images", _FACE_CROPPED)])

            combined = combine_embeddings(embeddings)
            embedding = torch.from_numpy(combined).to(self._device, self._dtype).reshape(1, 1, 512)
            projection = unet.encoder_hid_proj.image_projection_layers[0]
            with torch.inference_mode():
                self._tokens.cond = projection(embedding)
                self._tokens.uncond = projection(torch.zeros_like(embedding))
            self._tokens.batch = num_images

            embeds = embedding.unsqueeze(0)  # (1, num_ip_images=1, seq=1, 512)
            setup.ip_adapter_embeds = torch.cat([torch.zeros_like(embeds), embeds]) if guidance else embeds
            setup.controlnets.append(self._identity_net().to(self._device))
            setup.images.append(draw_kps((width, height), placed.kps))
            setup.scales.append(identity_strength)

        if pose_image is not None:
            skeleton = self._pose()(layout, detect_resolution=512, image_resolution=max(width, height))
            if not np.asarray(skeleton).any():
                raise InvalidInputError(fields=[FieldError("pose_image", _NO_POSE)])
            setup.controlnets.append(self._pose_net().to(self._device))
            setup.images.append(skeleton.resize((width, height), Image.Resampling.BILINEAR))
            setup.scales.append(pose_strength)

        return setup
