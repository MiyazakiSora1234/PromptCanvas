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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..catalog import IdentityConfig
from ..errors import FieldError, InvalidInputError
from .imaging import fit_to

logger = logging.getLogger(__name__)

_NO_FACE = "顔を検出できませんでした。顔が正面に近く、はっきり写った写真を選んでください。"
_FACE_CROPPED = "出力サイズに切り抜くと顔が見切れてしまいます。サイズの縦横比を元画像に合わせてください。"
_NO_POSE = "ポーズを検出できませんでした。人物の全身または上半身が写った画像を選んでください。"

# Colors / limbs of the InstantID keypoint image (eyes, nose, mouth corners).
_KPS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
_KPS_LIMBS = [(0, 2), (1, 2), (3, 2), (4, 2)]


@dataclass(frozen=True)
class DetectedFace:
    kps: np.ndarray  # (5, 2) keypoints in image pixels
    embedding: np.ndarray | None  # (512,) ArcFace embedding (not normalized, as InstantID expects)


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
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stick), int(angle), 0, 360, 1)
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

    def largest_face(self, image: Image.Image, *, embed: bool = True) -> DetectedFace | None:
        """The biggest face in the image (with its embedding unless `embed` is False). Extreme
        close-ups (face filling the frame, common in selfies) defeat the detector, so if nothing
        is found it retries with a padded border."""
        rgb = np.asarray(image.convert("RGB"))
        found = self._detect(rgb, offset=0, embed=embed)
        if found is None:
            pad = max(rgb.shape[:2]) // 4
            found = self._detect(np.pad(rgb, ((pad, pad), (pad, pad), (0, 0))), offset=pad, embed=embed)
        return found

    def _detect(self, rgb: np.ndarray, *, offset: int, embed: bool) -> DetectedFace | None:
        from insightface.app.common import Face

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        bboxes, kpss = self._det.detect(bgr, max_num=0)
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        i = int(np.argmax(areas))
        embedding = None
        if embed:
            face = Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            self._rec.get(bgr, face)
            embedding = np.asarray(face.embedding, dtype=np.float32)
        # Keypoints back in the original (unpadded) image's coordinates.
        return DetectedFace(np.asarray(kpss[i]) - offset, embedding)


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


def extract_lines(image: Image.Image) -> Image.Image:
    """White-on-black outlines of a drawing (or the dark edges of any image) for the sketch ControlNet."""
    import cv2

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Dark strokes relative to their surroundings, so soft shading and paper tone drop out.
    lines = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    # Remove specks (paper grain, JPEG noise).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    min_area = max(16, gray.size // 20000)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    lines = np.where(keep[labels], 255, 0).astype(np.uint8)
    # Strokes a few pixels wide at any resolution, like the scribbles the ControlNet was trained on.
    width = max(1, round(max(gray.shape) / 400))
    lines = cv2.dilate(lines, np.ones((width, width), np.uint8))
    return Image.fromarray(lines).convert("RGB")


# A pose image counts as "a person OpenPose understands" with at least this many body keypoints;
# otherwise (drawings, mannequins, illustrations) its outlines are used instead.
MIN_POSE_KEYPOINTS = 6


@dataclass
class ControlSetup:
    """What the ControlNet pipeline call needs for one request."""

    controlnets: list[Any] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    # Fraction of the denoising steps each ControlNet guides (1.0 = all of them).
    guidance_ends: list[float] = field(default_factory=list)
    ip_adapter_embeds: Any | None = None  # tensor (2|1, 1, 1, 512)
    # False when the face photos are used but no face position is known (e.g. a faceless
    # mannequin as the pose image): identity then comes from the IP-Adapter alone.
    identity_net: bool = False
    pose_mode: str | None = None  # "skeleton" | "sketch"

    def add(self, controlnet: Any, image: Image.Image, scale: float, guidance_end: float = 1.0) -> None:
        self.controlnets.append(controlnet)
        self.images.append(image)
        self.scales.append(scale)
        self.guidance_ends.append(guidance_end)


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
        self._sketchnet: Any = None
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

    def _sketch_net(self) -> Any:
        if self._sketchnet is None:
            s = self._cfg.sketch_controlnet
            assert s is not None
            self._sketchnet = self._controlnet(s.repo, s.subfolder, s.revision)
        return self._sketchnet

    def _skeleton(self, layout: Image.Image) -> Image.Image | None:
        """OpenPose skeleton of the person in `layout` (same size), or None if no clear person is found."""
        from controlnet_aux.open_pose import draw_poses

        # Detect once at 512px (what the body model expects), draw there, then scale to the output size.
        small = layout.convert("RGB")
        small.thumbnail((512, 512))
        poses = self._pose().detect_poses(np.asarray(small))
        if not any(sum(k is not None for k in p.body.keypoints) >= MIN_POSE_KEYPOINTS for p in poses):
            return None
        canvas = draw_poses(poses, small.height, small.width, draw_body=True, draw_hand=False, draw_face=False)
        return Image.fromarray(canvas).resize(layout.size, Image.Resampling.BILINEAR)

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
        self._sketchnet = None
        self._pose_detector = None

    def park(self) -> None:
        """Move the ControlNets (~2.4GB each) back to CPU RAM so plain generations keep their VRAM."""
        for net in (self._identitynet, self._posenet, self._sketchnet):
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

        The pose is copied as an OpenPose skeleton when a person is detected, else (drawings,
        mannequins, illustrations) by following the image's outlines with the sketch ControlNet.
        """
        import torch

        layout_source = pose_image if pose_image is not None else face_images[0]
        layout = fit_to(layout_source, width, height)
        setup = ControlSetup()

        if face_images:
            analyzer = self._face_analyzer()
            references = [analyzer.largest_face(img) for img in face_images]
            missing = [i + 1 for i, face in enumerate(references) if face is None]
            if missing:
                which = "、".join(f"{n}枚目" for n in missing)
                message = _NO_FACE if len(face_images) == 1 else f"{which}の写真から{_NO_FACE}"
                raise InvalidInputError(fields=[FieldError("face_images", message)])
            embeddings = [face.embedding for face in references if face is not None and face.embedding is not None]
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
            placed = analyzer.largest_face(layout, embed=False)  # only its position is needed
            if placed is None and pose_image is None:
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
            if placed is not None:
                setup.identity_net = True
                setup.add(
                    self._identity_net().to(self._device), draw_kps((width, height), placed.kps), identity_strength
                )
            else:
                # The pose image shows no face (a drawing, a mannequin, a back view): there is no face
                # position to pin, so the likeness comes from the IP-Adapter alone.
                logger.info("No face in the pose image; using the face photos without IdentityNet")

        if pose_image is not None:
            skeleton = self._skeleton(layout)
            if skeleton is not None:
                setup.pose_mode = "skeleton"
                setup.add(self._pose_net().to(self._device), skeleton, pose_strength)
            elif self._cfg.sketch_controlnet is not None:
                lines = extract_lines(layout)
                if not np.asarray(lines).any():
                    raise InvalidInputError(fields=[FieldError("pose_image", _NO_POSE)])
                setup.pose_mode = "sketch"
                ratio, end = self._cfg.sketch_strength_ratio, self._cfg.sketch_guidance_end
                setup.add(self._sketch_net().to(self._device), lines, pose_strength * ratio, end)
            else:
                raise InvalidInputError(fields=[FieldError("pose_image", _NO_POSE)])

        return setup
