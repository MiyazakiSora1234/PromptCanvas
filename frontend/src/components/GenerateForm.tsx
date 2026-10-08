import { useEffect, useRef, type FormEvent, type KeyboardEvent } from "react";
import type { AppConfig } from "../api/types";
import type { PresetStore } from "../hooks/usePresets";
import { TAB_LABELS, TAB_ORDER, tabsWithErrors, type TabId } from "../lib/formTabs";
import { HELP } from "../lib/help";
import { hasJapanese } from "../lib/japanese";
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
import { inputClass } from "./styles";
import { TabPanel, Tabs, type TabItem } from "./Tabs";
import { Field, Panel } from "./ui";

interface GenerateFormProps {
  /** The generate button outside this panel submits the form via this id. */
  formId: string;
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
  onFaceFiles: (files: File[]) => void;
  onRemoveFace: (index: number) => void;
  onPoseFiles: (files: File[]) => void;
  onRemovePose: (index: number) => void;
  /** Face/pose assets downloaded; null while unknown. */
  identityCached: boolean | null;
  translatorCached: boolean | null;
  presets: PresetStore;
  onApplyPreset: (id: string) => string;
  onSavePreset: (name: string, includePrompt: boolean) => string;
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  onSubmit: () => void;
  busy: boolean;
  submitDisabled: boolean;
  /** Incremented by the parent to move focus to the first field with an error. */
  focusRequest: number;
}

const TABS_ID = "settings";

/**
 * The settings panel: prompt and tab names stay put, the tab content scrolls inside the
 * panel when it is taller than the window (on wide screens; narrow screens scroll the page).
 */
export function GenerateForm({
  formId,
  config,
  cachedModels,
  state,
  fieldErrors,
  onValueChange,
  onModelChange,
  onLorasChange,
  onInitImageFile,
  onInitImageClear,
  onFaceFiles,
  onRemoveFace,
  onPoseFiles,
  onRemovePose,
  identityCached,
  translatorCached,
  presets,
  onApplyPreset,
  onSavePreset,
  activeTab,
  onTabChange,
  onSubmit,
  busy,
  submitDisabled,
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
    if (el instanceof HTMLElement) {
      el.focus();
      el.scrollIntoView?.({ block: "nearest" });
    }
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
  const referenceCount = (state.initImage ? 1 : 0) + state.faceImages.length + state.poseImages.length;
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
    <Panel className="flex flex-col md:sticky md:top-4 md:max-h-[calc(100vh-6rem)]">
      <form
        ref={formRef}
        id={formId}
        noValidate
        onSubmit={handleSubmit}
        onKeyDown={handleKeyDown}
        className="flex min-h-0 flex-1 flex-col gap-4"
      >
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
            <span>
              {HELP.prompt}
              {config.translation && hasJapanese(values.prompt) && (
                <span className="block text-indigo-700 dark:text-indigo-300">
                  日本語は英語に翻訳してから生成します（訳は結果に表示）。
                  {translatorCached === false &&
                    `初回は翻訳モデル${config.translation.download_size_gb ? `（約${config.translation.download_size_gb}GB）` : ""}をダウンロードします。`}
                </span>
              )}
            </span>
            <span className={promptLength > limits.max_prompt_length ? "font-semibold text-red-600" : ""}>
              {promptLength} / {limits.max_prompt_length}
            </span>
          </div>
        </Field>

        <div className="flex min-h-0 flex-1 flex-col gap-3">
          <Tabs idPrefix={TABS_ID} label="生成の設定" items={tabs} active={activeTab} onChange={onTabChange} />

          <div data-testid="settings-scroll" className="min-h-0 flex-1 md:overflow-y-auto md:pr-1">
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
                  state.faceImages.length > 0 || state.poseImages.length > 0
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
                onFaceFiles={onFaceFiles}
                onRemoveFace={onRemoveFace}
                onPoseFiles={onPoseFiles}
                onRemovePose={onRemovePose}
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
              <PresetBar
                config={config}
                presets={presets}
                disabled={busy}
                onApply={onApplyPreset}
                onSave={onSavePreset}
              />
            </TabPanel>
          </div>
        </div>
      </form>
    </Panel>
  );
}
