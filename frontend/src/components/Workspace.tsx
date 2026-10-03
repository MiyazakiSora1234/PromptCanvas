import { useState } from "react";
import type { AppConfig } from "../api/types";
import { useImageGeneration } from "../hooks/useImageGeneration";
import { presentError } from "../lib/presentError";
import {
  hasAdvancedFieldError,
  initialFormValues,
  validateForm,
  type FieldErrors,
  type FieldName,
  type FormValues,
} from "../lib/validation";
import { GenerateForm } from "./GenerateForm";
import { ResultPanel } from "./ResultPanel";

interface WorkspaceProps {
  config: AppConfig;
  /** Model is loading or failed: generation is disabled. */
  modelBlocked: boolean;
  modelLoading: boolean;
  /** Ask the parent to re-check /api/health (e.g. after a "model loading" error). */
  onRecheckHealth: () => void;
}

export function Workspace({ config, modelBlocked, modelLoading, onRecheckHealth }: WorkspaceProps) {
  const [values, setValues] = useState<FormValues>(() => initialFormValues(config.defaults));
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [message, setMessage] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [focusRequest, setFocusRequest] = useState(0);
  const { busy, startedAt, result, generate } = useImageGeneration();

  const showErrors = (text: string, errors: FieldErrors, openAdvanced: boolean) => {
    setMessage(text);
    setFieldErrors(errors);
    if (openAdvanced) setAdvancedOpen(true);
    if (Object.keys(errors).length > 0) setFocusRequest((n) => n + 1);
  };

  const handleChange = (name: FieldName, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }));
    // Editing a field clears its error.
    setFieldErrors((prev) => {
      if (!(name in prev)) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  const handleSubmit = async () => {
    if (busy) return;
    setMessage(null);
    setFieldErrors({});

    const validation = validateForm(values, config.limits);
    if (!validation.ok) {
      showErrors("入力内容を確認してください。", validation.errors, hasAdvancedFieldError(validation.errors));
      return;
    }

    try {
      await generate(validation.payload);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const presented = presentError(err);
      showErrors(presented.message, presented.fieldErrors, presented.openAdvanced);
      if (presented.recheckHealth) onRecheckHealth();
    }
  };

  const handleReuseSeed = (seed: number) => {
    handleChange("seed", String(seed));
    setAdvancedOpen(true);
  };

  const submitLabel = busy ? "生成中…" : modelLoading ? "モデル準備中…" : "画像を生成";

  return (
    <main className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-4 pb-8 md:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
      <GenerateForm
        limits={config.limits}
        values={values}
        fieldErrors={fieldErrors}
        onChange={handleChange}
        advancedOpen={advancedOpen}
        onAdvancedOpenChange={setAdvancedOpen}
        onSubmit={() => void handleSubmit()}
        submitDisabled={busy || modelBlocked}
        submitLabel={submitLabel}
        message={message}
        focusRequest={focusRequest}
      />
      <ResultPanel startedAt={startedAt} result={result} onReuseSeed={handleReuseSeed} />
    </main>
  );
}
