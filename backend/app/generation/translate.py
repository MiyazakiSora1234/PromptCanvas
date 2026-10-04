"""Translate Japanese prompts into English with a small local LLM (Qwen2.5-1.5B-Instruct by default).

Stable Diffusion's text encoders only understand English. Only the comma-separated parts that
contain Japanese are translated; English words, tags (``1girl``) and weights (``(笑顔:1.2)`` ->
``(smile:1.2)``) are kept as typed. Runs locally: prompts never leave the machine.

``torch`` and ``transformers`` are imported lazily.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from ..catalog import TranslatorConfig
from ..errors import TranslationUnavailableError, describe_load_error, is_out_of_memory
from .model_cache import are_files_cached
from .runtime import Runtime

logger = logging.getLogger(__name__)

_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_SEGMENT_SPLIT = re.compile(r"\s*[,、，]\s*")
_WEIGHTED = re.compile(r"^\((.+):(\d+(?:\.\d+)?)\)$")

_SYSTEM = (
    "Translate Japanese text for an image generation prompt into concise English. "
    "Reply with the English only. Do not add anything that is not in the Japanese."
)
# A few examples steer the model away from literal mistakes (ワンピース is a dress, 透かし a watermark)
# and from padding the answer with extra words.
_EXAMPLES = [
    ("白いワンピースの女の子", "girl in a white dress"),
    ("透かし", "watermark"),
    ("夜の東京の街並み", "Tokyo cityscape at night"),
    ("手の指が多い", "extra fingers"),
    ("浮世絵風", "ukiyo-e style"),
    ("猫", "cat"),
]
_MAX_NEW_TOKENS = 80


def has_japanese(text: str) -> bool:
    return bool(_JAPANESE.search(text))


def translate_prompt(text: str, translate_phrase: Callable[[str], str]) -> str:
    """Translate the Japanese comma-separated parts of a prompt; keep everything else as typed."""
    parts: list[str] = []
    for segment in _SEGMENT_SPLIT.split(text.strip()):
        if not has_japanese(segment):
            parts.append(segment)
            continue
        weighted = _WEIGHTED.match(segment)
        if weighted:
            parts.append(f"({translate_phrase(weighted.group(1))}:{weighted.group(2)})")
        else:
            parts.append(translate_phrase(segment))
    return ", ".join(part for part in parts if part)


def clean_answer(answer: str) -> str:
    """First line of the model's reply, without quotes or a trailing period."""
    line = answer.strip().splitlines()[0] if answer.strip() else ""
    return line.strip().strip("\"'`「」").rstrip("。.").strip()


class PromptTranslator:
    """Lazily loads the LLM; keeps it in CPU RAM and moves it to the GPU only while translating."""

    def __init__(self, config: TranslatorConfig, device: str, torch_dtype: Any, token: str | None) -> None:
        self._cfg = config
        self._device = device
        self._dtype = torch_dtype
        self._token = token
        self._model: Any = None
        self._tokenizer: Any = None
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading prompt translator %s", self._cfg.repo)
        kwargs: dict[str, Any] = {"revision": self._cfg.revision, "token": self._token}
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.repo, **kwargs)
        model_cls: Any = AutoModelForCausalLM  # transformers is only partially typed
        self._model = model_cls.from_pretrained(self._cfg.repo, dtype=self._dtype, **kwargs).eval()

    def translate(self, text: str) -> str:
        """English version of `text` (unchanged if it has no Japanese)."""
        if not has_japanese(text):
            return text
        with self._lock:
            self._load()
            started = time.perf_counter()
            self._model.to(self._device)
            try:
                result = translate_prompt(text, self._phrase)
            finally:
                if self._device != "cpu":
                    self._model.to("cpu")
            # Prompts may be private: log only their size.
            logger.info("Translated a %d-char prompt in %.1fs", len(text), time.perf_counter() - started)
            return result

    def _phrase(self, japanese: str) -> str:
        if japanese in self._cache:
            return self._cache[japanese]
        import torch

        messages = [{"role": "system", "content": _SYSTEM}]
        for ja, en in _EXAMPLES:
            messages += [{"role": "user", "content": ja}, {"role": "assistant", "content": en}]
        messages.append({"role": "user", "content": japanese})
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.1
            )
        answer = self._tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        english = clean_answer(answer) or japanese
        if len(self._cache) >= 512:
            self._cache.clear()
        self._cache[japanese] = english
        return english


class PromptTranslation:
    """Translates a request's prompts when they contain Japanese; passes English through untouched."""

    def __init__(self, config: TranslatorConfig | None, runtime: Runtime) -> None:
        self._cfg = config
        self._runtime = runtime
        self._translator: PromptTranslator | None = None
        self.cached = False  # model on disk

    def scan_cache(self) -> None:
        if self._cfg is not None:
            self.cached = are_files_cached(self._cfg.files())

    def prompts(self, prompt: str, negative_prompt: str) -> tuple[str, str, bool]:
        """(prompt, negative prompt, whether anything was translated)."""
        if self._cfg is None or not (has_japanese(prompt) or has_japanese(negative_prompt)):
            return prompt, negative_prompt, False
        rt = self._runtime
        rt.ensure()
        if self._translator is None:
            self._translator = PromptTranslator(self._cfg, str(rt.device), rt.dtype, rt.hf_token)
        try:
            prompt = self._translator.translate(prompt)
            negative_prompt = self._translator.translate(negative_prompt)
        except Exception as exc:
            if is_out_of_memory(exc):
                rt.release_memory()
                raise
            logger.exception("Failed to translate the prompt. %s", describe_load_error(exc))
            raise TranslationUnavailableError() from None
        self.cached = True
        return prompt, negative_prompt, True
