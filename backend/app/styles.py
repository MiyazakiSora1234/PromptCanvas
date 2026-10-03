"""Style presets: wording added to the user's prompt / negative prompt.

"photo" was chosen by comparing RealVisXL / SDXL portraits side by side: the added terms
bring out skin texture (pores, freckles) and the negatives suppress the airbrushed,
over-saturated look.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleSpec:
    label: str
    prefix: str = ""
    suffix: str = ""
    negative: str = ""


STYLES: dict[str, StyleSpec] = {
    "none": StyleSpec("なし"),
    "photo": StyleSpec(
        "リアルな写真（肌の質感）",
        # Up front: SDXL's text encoders keep only the first 77 tokens of a long prompt.
        prefix="RAW photo, ",
        suffix=", detailed skin texture, visible skin pores, natural skin, subtle film grain, 85mm lens",
        negative=(
            "plastic skin, airbrushed, smooth skin, waxy skin, cgi, 3d render, illustration, painting, "
            "oversaturated, overexposed, high contrast, watermark, text"
        ),
    ),
}


def apply_style(style_id: str, prompt: str, negative_prompt: str) -> tuple[str, str]:
    spec = STYLES[style_id]
    styled_negative = ", ".join(part for part in (negative_prompt, spec.negative) if part)
    return f"{spec.prefix}{prompt}{spec.suffix}", styled_negative
