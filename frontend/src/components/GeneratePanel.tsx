import type { AppConfig } from "../api/types";
import type { FormState } from "../lib/validation";
import { SettingsSummary } from "./settings/SettingsSummary";
import { primaryButtonClass } from "./styles";
import { Alert } from "./ui";

interface GeneratePanelProps {
  config: AppConfig;
  state: FormState;
  /** id of the settings <form> this button submits. */
  formId: string;
  disabled: boolean;
  label: string;
  message: string | null;
}

/** Summary + generate button, placed right under the result image (linked to the settings form via `form`). */
export function GeneratePanel({ config, state, formId, disabled, label, message }: GeneratePanelProps) {
  return (
    <div className="flex flex-col gap-2">
      <SettingsSummary config={config} state={state} />
      <button type="submit" form={formId} disabled={disabled} className={`${primaryButtonClass} w-full py-3 text-base`}>
        {label}
      </button>
      {message && <Alert>{message}</Alert>}
    </div>
  );
}
