import type { AppConfig } from "../api/types";
import type { FormState } from "../lib/validation";
import { SettingsSummary } from "./settings/SettingsSummary";
import { buttonClass, primaryButtonClass } from "./styles";
import { Alert } from "./ui";

interface GeneratePanelProps {
  config: AppConfig;
  state: FormState;
  /** id of the settings <form> this button submits. */
  formId: string;
  disabled: boolean;
  label: string;
  message: string | null;
  /** Set while a generation is running: shows the stop button. */
  onCancel: (() => void) | null;
  cancelling: boolean;
}

/** Summary + generate button, placed right under the result image (linked to the settings form via `form`). */
export function GeneratePanel({
  config,
  state,
  formId,
  disabled,
  label,
  message,
  onCancel,
  cancelling,
}: GeneratePanelProps) {
  return (
    <div className="flex flex-col gap-2">
      <SettingsSummary config={config} state={state} />
      <div className="flex gap-2">
        <button
          type="submit"
          form={formId}
          disabled={disabled}
          className={`${primaryButtonClass} min-w-0 flex-1 py-3 text-base`}
        >
          {label}
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={cancelling}
            className={`${buttonClass} shrink-0 px-5 font-semibold text-red-700 dark:text-red-300`}
          >
            中止
          </button>
        )}
      </div>
      {message && <Alert>{message}</Alert>}
    </div>
  );
}
