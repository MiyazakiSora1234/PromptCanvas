from __future__ import annotations

# --- Japanese prompt translation ---------------------------------------------------


def test_has_japanese() -> None:
    from app.generation.translate import has_japanese

    assert has_japanese("猫") and has_japanese("ねこ") and has_japanese("ネコ") and has_japanese("ﾈｺ")
    assert not has_japanese("a cat, (smile:1.2), 1girl")


def test_translate_prompt_translates_only_japanese_parts() -> None:
    from app.generation.translate import translate_prompt

    calls: list[str] = []

    def fake(ja: str) -> str:
        calls.append(ja)
        return {"黒髪ロング": "long black hair", "笑顔": "smile", "教室": "classroom"}[ja]

    result = translate_prompt("1girl, 黒髪ロング、(笑顔:1.2)，教室, highly detailed", fake)
    assert result == "1girl, long black hair, (smile:1.2), classroom, highly detailed"
    assert calls == ["黒髪ロング", "笑顔", "教室"]
    assert translate_prompt("", fake) == ""


def test_clean_answer() -> None:
    from app.generation.translate import clean_answer

    assert clean_answer('"a cat on a sofa."\nExplanation: ...') == "a cat on a sofa"
    assert clean_answer("「watermark」") == "watermark"
    assert clean_answer("  ") == ""
