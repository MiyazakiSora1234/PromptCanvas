import { StatusBadge } from "./components/StatusBadge";
import { Alert } from "./components/ui";
import { Workspace } from "./components/Workspace";
import { useAppConfig } from "./hooks/useAppConfig";
import { cachedModels, isModelLoading, loadedModel, useServerStatus } from "./hooks/useServerStatus";

export default function App() {
  const configState = useAppConfig();
  const { status, refresh } = useServerStatus();
  const config = configState.kind === "ready" ? configState.config : null;
  const failedHealth = status.kind === "online" && status.health.status === "failed" ? status.health : null;
  const failedLabel = config?.models.find((m) => m.id === failedHealth?.model)?.label ?? failedHealth?.model;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 pt-5 pb-3">
        <h1 className="text-2xl font-bold">PromptCanvas</h1>
        <StatusBadge status={status} models={config?.models ?? []} />
      </header>

      {failedHealth && (
        <div className="mx-auto mb-3 max-w-6xl px-4">
          <Alert>
            モデル「{failedLabel}」を利用できません。{failedHealth.message ?? "サーバーログを確認してください。"}
            別のモデルを選ぶか、設定を見直してから再度お試しください。
          </Alert>
        </div>
      )}

      {config ? (
        <Workspace
          config={config}
          modelLoading={isModelLoading(status)}
          loadedModel={loadedModel(status)}
          cachedModels={cachedModels(status)}
          identityCached={status.kind === "online" ? status.health.identity_cached : null}
          translatorCached={status.kind === "online" ? status.health.translator_cached : null}
          onRecheckHealth={refresh}
        />
      ) : (
        <p className="mx-auto max-w-6xl px-4 text-slate-500 dark:text-slate-400">
          {configState.kind === "loading"
            ? "読み込み中…"
            : "サーバーに接続できません。サーバーが起動しているか確認してください（自動で再接続します）。"}
        </p>
      )}
    </div>
  );
}
