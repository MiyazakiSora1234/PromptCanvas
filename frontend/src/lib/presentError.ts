import { ApiError, NetworkError } from "../api/client";
import { hasAdvancedFieldError, type FieldErrors } from "./validation";

export interface ErrorPresentation {
  message: string;
  fieldErrors: FieldErrors;
  /** Open "詳細設定" so the user can see / change the relevant inputs. */
  openAdvanced: boolean;
  /** The model state may have changed; re-poll /api/health. */
  recheckHealth: boolean;
}

/** Turn any error thrown while generating into what the UI should show. */
export function presentError(err: unknown): ErrorPresentation {
  if (err instanceof NetworkError) {
    return {
      message:
        "サーバーに接続できませんでした。サーバーが起動しているか、ネットワーク接続を確認してから再度お試しください。",
      fieldErrors: {},
      openAdvanced: false,
      recheckHealth: true,
    };
  }
  if (err instanceof ApiError) {
    const fieldErrors: FieldErrors = Object.fromEntries(err.fields.map((f) => [f.field, f.message]));
    // Only unexpected failures need an inquiry ID; busy/loading/out-of-memory have a clear remedy.
    const showRequestId = err.requestId !== null && err.status === 500;
    return {
      message: showRequestId ? `${err.message}（問い合わせID: ${err.requestId}）` : err.message,
      fieldErrors,
      openAdvanced: err.code === "gpu_out_of_memory" || hasAdvancedFieldError(fieldErrors),
      recheckHealth: ["model_loading", "model_unavailable", "lora_unavailable"].includes(err.code),
    };
  }
  return {
    message: "予期しないエラーが発生しました。ページを再読み込みしてから再度お試しください。",
    fieldErrors: {},
    openAdvanced: false,
    recheckHealth: false,
  };
}
