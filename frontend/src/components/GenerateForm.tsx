import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";
import type { AppConfig } from "../api/types";
import type { PresetStore } from "../hooks/usePresets";
import { TAB_LABELS, TAB_ORDER, tabsWithErrors, type TabId } from "../lib/formTabs";
import {
  compatibleLoras,
  findModel,
  FIELD_NAMES,
  supportsReference,
  type FieldErrors,
  type FormState,
  type FormValues,
  type LoraSelection,
} from "../lib/validation";
import { makeControlProps } from "./formControl";
import { InitImagePicker } from "./InitImagePicker";
import { LoraPicker } from "./LoraPicker";
import { PresetBar } from "./PresetBar";
import { ReferencePicker } from "./ReferencePicker";
import { AdvancedSettings } from "./settings/AdvancedSettings";
import { BasicSettings } from "./settings/BasicSettings";
import { SettingsSummary } from "./settings/SettingsSummary";
import { inputClass, primaryButtonClass } from "./styles";
import { TabPanel, Tabs, type TabItem } from "./Tabs";
import { Alert, Field, Panel } from "./ui";

interface GenerateFormProps {
  config: AppConfig;
  /** Downloaded model ids; null while unknown. */
  cachedModels: string[] | null;
  state: FormState;
  fieldErrors: FieldErrors;
  onValueChange: (name: keyof FormValues, value: string) => void;
  onModelChange: (modelId: string) => void;
  onLorasChange: (loras: LoraSelection[]) => void;
  onInitImageFile: (file: File) => void;
  onInitImageClear: () => void;
  onReferenceFile: (kind: "face" | "pose", file: File) => void;
  onReferenceClear: (kind: "face" | "pose") => void;
  /** Face/pose assets downloaded; null while unknown. */
  identityCached: boolean | null;
  presets: PresetStore;
  onApplyPreset: (id: string) => string;
  onSavePreset: (name: string) => string;
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  onSubmit: () => void;
  busy: boolean;
  submitDisabled: boolean;
  submitLabel: string;
  message: string | null;
  /** Incremented by the parent to move focus to the first field with an error. */
  focusRequest: number;
}

const TABS_ID = "settings";

export function GenerateForm({
  config,
  cachedModels,
  state,
  fieldErrors,
  onValueChange,
  onModelChange,
  onLorasChange,
  onInitImageFile,
  onInitImageClear,
  onReferenceFile,
  onReferenceClear,
  identityCached,
  presets,
  onApplyPreset,
  onSavePreset,
  activeTab,
  onTabChange,
  onSubmit,
  busy,
  submitDisabled,
  submitLabel,
  message,
  focusRequest,
}: GenerateFormProps) {
  const formRef = useRef<HTMLFormElement>(null);
  const { limits } = config;
  const { values } = state;
  const model = findModel(config, values.model);
  const needsDownload = (id: string) => cachedModels !== null && !cachedModels.includes(id);
  const control = makeControlProps(values, fieldErrors, onValueChange);

  // Runs after the parent switched to the tab holding the first error, so the field is visible.
  useEffect(() => {
    if (focusRequest === 0) return;
    const first = FIELD_NAMES.find((name) => fieldErrors[name]);
    const el = first ? document.getElementById(first === "loras" ? "loras-error" : first) : null;
    if (el instanceof HTMLElement) el.focus();
    // Only when explicitly requested, not on every error change (that would steal focus while typing).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusRequest]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLFormElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!submitDisabled) formRef.current?.requestSubmit();
    }
  };

  const errorTabs = tabsWithErrors(fieldErrors);
  const referenceCount = [state.initImage, state.faceImage, state.poseImage].filter(Boolean).length;
  const tabs: TabItem<TabId>[] = TAB_ORDER.map((id) => ({
    id,
    label: TAB_LABELS[id],
    badge:
      id === "images" && referenceCount > 0
        ? "●"
        : id === "lora" && state.loras.length > 0
          ? `(${state.loras.length})`
          : undefined,
    hasError: errorTabs.has(id),
  }));
  const promptLength = values.prompt.trim().length;

  return (
    <Panel>
      <form ref={formRef} noValidate onSubmit={handleSubmit} onKeyDown={handleKeyDown} className="flex flex-col gap-4">
        <Field
          id="prompt"
          label={
            <>
              プロンプト{" "}
              <span className="ml-1 rounded bg-indigo-600 px-1.5 py-px align-middle text-[0.7rem] text-white">必須</span>
            </>
          }
          error={fieldErrors.prompt}
        >
          <textarea
            {...control("prompt")}
            rows={4}
            maxLength={limits.max_prompt_length}
            placeholder="例: a watercolor painting of a lighthouse at sunset, soft light, highly detailed"
            className={`${inputClass} resize-y`}
          />
          <div className="flex justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
            <span>英語での入力を推奨します。Ctrl+Enter でも生成できます。</span>
            <span className={promptLength > limits.max_prompt_length ? "font-semibold text-red-600" : ""}>
              {promptLength} / {limits.max_prompt_length}
            </span>
          </div>
        </Field>

        <div className="flex flex-col gap-3">
          <Tabs idPrefix={TABS_ID} label="生成の設定" items={tabs} active={activeTab} onChange={onTabChange} />

          <TabPanel idPrefix={TABS_ID} id="basic" active={activeTab === "basic"}>
            <BasicSettings
              config={config}
              model={model}
              values={values}
              fieldErrors={fieldErrors}
              control={control}
              needsDownload={needsDownload}
              busy={busy}
              onModelChange={onModelChange}
              onValueChange={onValueChange}
            />
          </TabPanel>

          <TabPanel idPrefix={TABS_ID} id="images" active={activeTab === "images"}>
            <InitImagePicker
              limits={limits}
              image={state.initImage}
              strength={values.strength}
              steps={values.num_inference_steps}
              imageError={fieldErrors.init_image}
              strengthError={fieldErrors.strength}
              disabled={busy}
              unavailableReason={
                state.faceImage || state.poseImage
                  ? "顔・ポーズの参照と同時には使えません。参照画像を外すと選べます。"
                  : undefined
              }
              onFile={onInitImageFile}
              onClear={onInitImageClear}
              onStrengthChange={(v) => onValueChange("strength", v)}
            />
            <ReferencePicker
              config={config}
              state={state}
              supported={supportsReference(config, model)}
              modelLabel={model.label}
              assetsCached={identityCached}
              fieldErrors={fieldErrors}
              disabled={busy}
              onFaceFile={(file) => onReferenceFile("face", file)}
              onPoseFile={(file) => onReferenceFile("pose", file)}
              onClearFace={() => onReferenceClear("face")}
              onClearPose={() => onReferenceClear("pose")}
              onValueChange={onValueChange}
            />
          </TabPanel>

          <TabPanel idPrefix={TABS_ID} id="lora" active={activeTab === "lora"}>
            <LoraPicker
              limits={limits}
              options={compatibleLoras(config, model)}
              modelLabel={model.label}
              selected={state.loras}
              error={fieldErrors.loras}
              onChange={onLorasChange}
            />
          </TabPanel>

          <TabPanel idPrefix={TABS_ID} id="advanced" active={activeTab === "advanced"}>
            <AdvancedSettings
              limits={limits}
              schedulers={config.schedulers}
              model={model}
              values={values}
              fieldErrors={fieldErrors}
              control={control}
              onValueChange={onValueChange}
            />
          </TabPanel>

          <TabPanel idPrefix={TABS_ID} id="presets" active={activeTab === "presets"}>
            <PresetBar config={config} presets={presets} disabled={busy} onApply={onApplyPreset} onSave={onSavePreset} />
          </TabPanel>
        </div>

        <div className="flex flex-col gap-2 border-t border-slate-200 pt-3 dark:border-slate-700">
          <SettingsSummary config={config} state={state} />
          <button type="submit" disabled={submitDisabled} className={`${primaryButtonClass} w-full py-3 text-base`}>
            {submitLabel}
          </button>
          {message && <Alert>{message}</Alert>}
        </div>
      </form>
    </Panel>
  );
}
