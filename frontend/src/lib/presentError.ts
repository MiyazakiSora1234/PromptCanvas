import { ApiError, NetworkError } from "../api/client";
import { firstErrorTab, type TabId } from "./formTabs";
import type { FieldErrors } from "./validation";

export interface ErrorPresentation {
  message: string;
  fieldErrors: FieldErrors;
  /** Settings tab to show so the user can see / change the relevant inputs (null: stay). */
  showTab: TabId | null;
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
      showTab: null,
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
      // Out of memory: the remedy (smaller size / fewer steps) is on the "詳細" tab.
      showTab: firstErrorTab(fieldErrors) ?? (err.code === "gpu_out_of_memory" ? "advanced" : null),
      recheckHealth: ["model_loading", "model_unavailable", "lora_unavailable"].includes(err.code),
    };
  }
  return {
    message: "予期しないエラーが発生しました。ページを再読み込みしてから再度お試しください。",
    fieldErrors: {},
    showTab: null,
    recheckHealth: false,
  };
}
