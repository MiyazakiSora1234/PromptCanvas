"""Style presets: wording added before/after the user's prompt and to the negative prompt.

"photo" was chosen by comparing RealVisXL / SDXL portraits side by side: the added terms
bring out skin texture (pores, freckles) and the negatives suppress the airbrushed,
over-saturated look. The others follow the same pattern: a short medium/genre prefix,
a few quality cues as suffix, and negatives that push away competing media.
"""

from __future__ import annotations

from dataclasses import dataclass

# Added to every style's negatives: generic defects nobody wants.
_COMMON_NEGATIVE = "lowres, blurry, watermark, text, signature, jpeg artifacts"


@dataclass(frozen=True)
class StyleSpec:
    label: str
    description: str = ""
    # Prefix goes first: SDXL's text encoders keep only the first 77 tokens of a long prompt.
    prefix: str = ""
    suffix: str = ""
    negative: str = ""


STYLES: dict[str, StyleSpec] = {
    "none": StyleSpec("なし", "プロンプトをそのまま使います。"),
    "photo": StyleSpec(
        "リアルな写真（肌の質感）",
        "実写の質感。肌のきめ・毛穴を出し、つるつるの肌や CG っぽさを避けます。",
        prefix="RAW photo, ",
        suffix=", detailed skin texture, visible skin pores, natural skin, subtle film grain, 85mm lens",
        negative=(
            "plastic skin, airbrushed, smooth skin, waxy skin, cgi, 3d render, illustration, painting, "
            "oversaturated, overexposed, high contrast"
        ),
    ),
    "cinematic": StyleSpec(
        "映画のワンシーン",
        "映画のような光と色。浅い被写界深度とフィルムの粒子感。",
        prefix="cinematic film still, ",
        suffix=", dramatic lighting, shallow depth of field, color graded, anamorphic lens, film grain",
        negative="cartoon, anime, illustration, flat lighting, oversaturated, amateur photo",
    ),
    "anime": StyleSpec(
        "アニメ",
        "日本のアニメ調。くっきりした線とセル塗り。",
        prefix="anime style, ",
        suffix=", clean lineart, cel shading, vibrant colors, highly detailed, key visual",
        negative="photo, realistic, 3d render, western cartoon, bad anatomy, extra fingers, sketch",
    ),
    "illustration": StyleSpec(
        "デジタルイラスト",
        "書籍やゲームのような描き込みのあるイラスト。",
        prefix="digital illustration, ",
        suffix=", detailed, vibrant colors, artstation, professional artwork",
        negative="photo, realistic, 3d render, sketch, unfinished, bad anatomy",
    ),
    "watercolor": StyleSpec(
        "水彩画",
        "にじみと紙の質感がある柔らかい水彩。",
        prefix="watercolor painting, ",
        suffix=", soft washes, wet-on-wet, delicate brush strokes, paper texture, pastel colors",
        negative="photo, realistic, 3d render, digital art, harsh lines, oversaturated, dark",
    ),
    "oil": StyleSpec(
        "油絵",
        "厚塗りの筆跡が残る油絵。",
        prefix="oil painting, ",
        suffix=", visible brush strokes, impasto, rich colors, canvas texture, fine art",
        negative="photo, realistic, 3d render, digital art, smooth, flat colors",
    ),
    "sketch": StyleSpec(
        "鉛筆スケッチ",
        "鉛筆で描いたモノクロのスケッチ。",
        prefix="pencil sketch, ",
        suffix=", graphite, cross-hatching, detailed linework, paper texture, monochrome",
        negative="color, photo, realistic, 3d render, painting, colorful",
    ),
    "3d": StyleSpec(
        "3DCG",
        "なめらかな 3D レンダリング風。",
        prefix="3d render, ",
        suffix=", octane render, soft studio lighting, subsurface scattering, highly detailed, pbr materials",
        negative="photo, 2d, sketch, painting, flat, cartoon lineart",
    ),
    "pixel": StyleSpec(
        "ドット絵",
        "レトロゲームのようなドット絵。",
        prefix="pixel art, ",
        suffix=", 16-bit, limited color palette, crisp pixels, retro game sprite",
        negative="photo, realistic, smooth gradients, antialiasing, 3d render, painting",
    ),
    "fantasy": StyleSpec(
        "ファンタジーアート",
        "壮大で魔法的な雰囲気のコンセプトアート。",
        prefix="epic fantasy art, ",
        suffix=", magical atmosphere, intricate details, dramatic lighting, concept art, matte painting",
        negative="photo, modern, mundane, flat lighting, cartoon",
    ),
    "monochrome": StyleSpec(
        "モノクロ写真",
        "白黒のフィルム写真。陰影を強調。",
        prefix="black and white photo, ",
        suffix=", high contrast, dramatic shadows, film grain, fine art photography",
        negative="color, colorful, saturated, cartoon, illustration, 3d render",
    ),
}


def apply_style(style_id: str, prompt: str, negative_prompt: str) -> tuple[str, str]:
    spec = STYLES[style_id]
    if style_id == "none":
        return prompt, negative_prompt
    style_negative = f"{spec.negative}, {_COMMON_NEGATIVE}" if spec.negative else _COMMON_NEGATIVE
    styled_negative = ", ".join(part for part in (negative_prompt, style_negative) if part)
    return f"{spec.prefix}{prompt}{spec.suffix}", styled_negative
