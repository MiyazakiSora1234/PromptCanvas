"""Domain errors with user-facing messages, plus classification of raw library exceptions.

Messages here are shown to end users, so they must never contain exception text,
paths, tokens or other internal details. Details go to the server log only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldError:
    field: str
    message: str


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    default_message: str = (
        "サーバー内部でエラーが発生しました。時間をおいて再度お試しください。"
        "問題が続く場合は問い合わせIDを添えて管理者に連絡してください。"
    )

    def __init__(self, message: str | None = None, *, fields: list[FieldError] | None = None) -> None:
        self.message = message or self.default_message
        self.fields = fields or []
        super().__init__(self.message)


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"
    default_message = "入力内容に誤りがあります。各項目のメッセージを確認して修正してください。"


class ModelLoadingError(AppError):
    status_code = 503
    code = "model_loading"
    default_message = (
        "モデルを読み込み中です。しばらく待ってから再度お試しください"
        "（初回はモデルのダウンロードに数分以上かかることがあります）。"
    )


class ModelUnavailableError(AppError):
    status_code = 503
    code = "model_unavailable"
    default_message = "モデルを利用できないため画像を生成できません。サーバーのログと設定を確認してください。"


class LoraUnavailableError(AppError):
    status_code = 503
    code = "lora_unavailable"
    default_message = "LoRA を読み込めませんでした。LoRA を外すか、サーバーのログと設定を確認してください。"


class ReferenceUnavailableError(AppError):
    status_code = 503
    code = "reference_unavailable"
    default_message = "顔・ポーズ参照用のモデルを読み込めませんでした。サーバーのログと設定を確認してください。"


class TranslationUnavailableError(AppError):
    status_code = 503
    code = "translation_unavailable"
    default_message = (
        "日本語のプロンプトを英語に翻訳できませんでした（翻訳モデルを読み込めません）。"
        "英語で入力するか、サーバーのログと設定を確認してください。"
    )


class ServerBusyError(AppError):
    status_code = 429
    code = "server_busy"
    default_message = "現在ほかの生成リクエストで混み合っています。少し時間をおいて再度お試しください。"


class QueueTimeoutError(AppError):
    status_code = 503
    code = "queue_timeout"
    default_message = "順番待ちの時間が上限を超えました。少し時間をおいて再度お試しください。"


class GpuOutOfMemoryError(AppError):
    status_code = 503
    code = "gpu_out_of_memory"
    default_message = "GPUメモリが不足しました。画像サイズ（幅・高さ）やステップ数を小さくして再度お試しください。"


class ContentFilteredError(AppError):
    status_code = 422
    code = "content_filtered"
    default_message = (
        "生成した画像がセーフティフィルタによりブロックされました。プロンプトを変更して再度お試しください。"
    )


class GenerationCancelledError(AppError):
    status_code = 409
    code = "cancelled"
    default_message = "生成を中止しました。"


class GenerationFailedError(AppError):
    status_code = 500
    code = "generation_failed"
    default_message = (
        "画像の生成に失敗しました。設定を変えて再度お試しください。"
        "問題が続く場合は問い合わせIDを添えて管理者に連絡してください。"
    )


class HttpError(AppError):
    """Framework-level HTTP errors (unknown route, wrong method, ...)."""

    default_message = "リクエストされたリソースが見つからないか、許可されていない操作です。"

    def __init__(self, status_code: int) -> None:
        super().__init__()
        self.status_code = status_code
        self.code = "not_found" if status_code == 404 else "http_error"


class ConfigurationError(Exception):
    """Invalid device/dtype configuration detected while loading. Message is safe to show."""


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def is_out_of_memory(exc: BaseException) -> bool:
    """True for CUDA/MPS/CPU allocation failures, without importing torch."""
    for err in _exception_chain(exc):
        if "OutOfMemoryError" in _class_names(err) or isinstance(err, MemoryError):
            return True
        if "out of memory" in str(err).lower():
            return True
    return False


def classify_generation_error(exc: BaseException) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if is_out_of_memory(exc):
        return GpuOutOfMemoryError()
    return GenerationFailedError()


def describe_load_error(exc: BaseException) -> str:
    """Map a model-loading failure to an operator-facing hint that contains no raw exception text."""
    for err in _exception_chain(exc):
        if isinstance(err, ConfigurationError):
            return str(err)
        names = _class_names(err)
        if "GatedRepoError" in names:
            return (
                "モデルへのアクセス権がありません。Hugging Faceでモデルの利用規約に同意し、"
                "環境変数 HF_TOKEN にトークンを設定してください。"
            )
        if "RepositoryNotFoundError" in names or "RevisionNotFoundError" in names:
            return "モデルが見つかりません。PROMPTCANVAS_MODEL_ID / PROMPTCANVAS_MODEL_REVISION を確認してください。"
        if "EntryNotFoundError" in names or "LocalEntryNotFoundError" in names:
            return (
                "モデルファイルを取得できませんでした。ネットワーク接続、または "
                "PROMPTCANVAS_MODEL_VARIANT の指定（例: fp16 版が存在するか）を確認してください。"
            )
        if names & {"ConnectionError", "ConnectTimeout", "ReadTimeout", "OfflineModeIsEnabled"}:
            return "モデルをダウンロードできませんでした。ネットワーク接続を確認してください。"
    if is_out_of_memory(exc):
        return (
            "モデルをメモリに載せられませんでした。PROMPTCANVAS_ENABLE_CPU_OFFLOAD=true を試すか、"
            "より小さいモデルを使用してください。"
        )
    return "モデルの読み込みに失敗しました。詳細はサーバーログを確認してください。"
