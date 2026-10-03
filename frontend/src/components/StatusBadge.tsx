import type { ModelOption } from "../api/types";
import type { ServerStatus } from "../hooks/useServerStatus";

type Tone = "neutral" | "ok" | "warn" | "error";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "border-slate-300 text-slate-500 dark:border-slate-600 dark:text-slate-400",
  ok: "border-emerald-600 text-emerald-700 dark:border-emerald-400 dark:text-emerald-300",
  warn: "border-amber-600 text-amber-700 dark:border-amber-400 dark:text-amber-300",
  error: "border-red-600 text-red-700 dark:border-red-400 dark:text-red-300",
};

function describe(status: ServerStatus, models: ModelOption[]): { text: string; tone: Tone; title?: string } {
  switch (status.kind) {
    case "unknown":
      return { text: "サーバーに接続中…", tone: "neutral" };
    case "offline":
      return { text: "サーバーに接続できません", tone: "error" };
    case "online": {
      const { health } = status;
      const model = models.find((m) => m.id === health.model)?.label ?? health.model ?? "モデル";
      if (health.status === "ready") {
        const device = health.device === "cuda" ? "GPU (CUDA)" : health.device === "mps" ? "GPU (MPS)" : "CPU";
        const slow = health.device === "cpu" ? "・生成に時間がかかります" : "";
        return { text: `準備完了：${model}・${device}${slow}`, tone: "ok", title: health.dtype ?? undefined };
      }
      if (health.status === "failed") {
        return { text: `${model} の読み込みに失敗しました`, tone: "error", title: health.message ?? undefined };
      }
      return { text: `${model} を読み込み中…（初回はダウンロードに時間がかかります）`, tone: "warn" };
    }
  }
}

export function StatusBadge({ status, models }: { status: ServerStatus; models: ModelOption[] }) {
  const { text, tone, title } = describe(status, models);
  return (
    <p
      role="status"
      aria-live="polite"
      title={title}
      className={`rounded-full border px-3 py-1 text-sm ${TONE_CLASSES[tone]}`}
    >
      {text}
    </p>
  );
}
