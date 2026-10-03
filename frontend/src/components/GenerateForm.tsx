import { useEffect, useRef, type ChangeEvent, type FormEvent, type KeyboardEvent } from "react";
import type { Limits } from "../api/types";
import { FIELD_NAMES, type FieldErrors, type FieldName, type FormValues } from "../lib/validation";
import { buttonClass, inputClass, primaryButtonClass } from "./styles";
import { Alert, Field, Panel } from "./ui";

const SIZE_PRESETS: ReadonlyArray<readonly [number, number]> = [
  [512, 512],
  [768, 512],
  [512, 768],
  [768, 768],
  [1024, 1024],
];

interface GenerateFormProps {
  limits: Limits;
  values: FormValues;
  fieldErrors: FieldErrors;
  onChange: (name: FieldName, value: string) => void;
  advancedOpen: boolean;
  onAdvancedOpenChange: (open: boolean) => void;
  onSubmit: () => void;
  submitDisabled: boolean;
  submitLabel: string;
  message: string | null;
  /** Incremented by the parent to move focus to the first field with an error. */
  focusRequest: number;
}

export function GenerateForm({
  limits,
  values,
  fieldErrors,
  onChange,
  advancedOpen,
  onAdvancedOpenChange,
  onSubmit,
  submitDisabled,
  submitLabel,
  message,
  focusRequest,
}: GenerateFormProps) {
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (focusRequest === 0) return;
    const first = FIELD_NAMES.find((name) => fieldErrors[name]);
    const el = first ? formRef.current?.elements.namedItem(first) : null;
    if (el instanceof HTMLElement) el.focus();
    // Only when explicitly requested, not on every error change (that would steal focus while typing).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusRequest]);

  const controlProps = (name: FieldName) => ({
    id: name,
    name,
    value: values[name],
    onChange: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => onChange(name, e.target.value),
    "aria-invalid": fieldErrors[name] ? true : undefined,
    "aria-describedby": fieldErrors[name] ? `${name}-error` : undefined,
    className: inputClass,
  });

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

  const promptLength = values.prompt.trim().length;
  const presets = SIZE_PRESETS.filter(
    ([w, h]) => Math.max(w, h) <= limits.max_image_size && Math.min(w, h) >= limits.min_image_size,
  );

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
            {...controlProps("prompt")}
            rows={5}
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

        <details open={advancedOpen} onToggle={(e) => onAdvancedOpenChange(e.currentTarget.open)}>
          <summary className="mb-3 cursor-pointer font-semibold">詳細設定</summary>
          <div className="flex flex-col gap-4">
            <Field id="negative_prompt" label="ネガティブプロンプト" error={fieldErrors.negative_prompt}>
              <textarea
                {...controlProps("negative_prompt")}
                rows={2}
                maxLength={limits.max_prompt_length}
                placeholder="例: blurry, low quality, watermark"
                className={`${inputClass} resize-y`}
              />
            </Field>

            <fieldset className="flex flex-col gap-2">
              <legend className="mb-1 text-sm font-semibold">画像サイズ（px）</legend>
              <div className="flex flex-wrap gap-1.5">
                {presets.map(([w, h]) => {
                  const selected = values.width === String(w) && values.height === String(h);
                  return (
                    <button
                      key={`${w}x${h}`}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => {
                        onChange("width", String(w));
                        onChange("height", String(h));
                      }}
                      className={`${buttonClass} px-2 py-1 text-xs aria-pressed:border-indigo-600 aria-pressed:text-indigo-700 dark:aria-pressed:text-indigo-300`}
                    >
                      {w}×{h}
                    </button>
                  );
                })}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Field id="width" label="幅" error={fieldErrors.width}>
                  <input
                    {...controlProps("width")}
                    type="number"
                    inputMode="numeric"
                    min={limits.min_image_size}
                    max={limits.max_image_size}
                    step={limits.size_multiple}
                  />
                </Field>
                <Field id="height" label="高さ" error={fieldErrors.height}>
                  <input
                    {...controlProps("height")}
                    type="number"
                    inputMode="numeric"
                    min={limits.min_image_size}
                    max={limits.max_image_size}
                    step={limits.size_multiple}
                  />
                </Field>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {limits.size_multiple}の倍数で指定してください。大きいほど時間とGPUメモリを消費します。
              </p>
            </fieldset>

            <div className="grid grid-cols-2 gap-3">
              <Field id="num_inference_steps" label="ステップ数" error={fieldErrors.num_inference_steps}>
                <input
                  {...controlProps("num_inference_steps")}
                  type="number"
                  inputMode="numeric"
                  min={limits.min_steps}
                  max={limits.max_steps}
                  step={1}
                />
              </Field>
              <Field id="guidance_scale" label="ガイダンススケール" error={fieldErrors.guidance_scale}>
                <input
                  {...controlProps("guidance_scale")}
                  type="number"
                  inputMode="decimal"
                  min={limits.min_guidance_scale}
                  max={limits.max_guidance_scale}
                  step={0.5}
                />
              </Field>
            </div>

            <Field
              id="seed"
              label="シード値"
              error={fieldErrors.seed}
              hint="同じシードと設定で同じ画像を再現できます。"
            >
              <div className="flex gap-2">
                <input
                  {...controlProps("seed")}
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={limits.seed_max}
                  step={1}
                  placeholder="空欄でランダム"
                  className={`${inputClass} min-w-0 flex-1`}
                />
                <button type="button" className={buttonClass} onClick={() => onChange("seed", "")}>
                  ランダム
                </button>
              </div>
            </Field>
          </div>
        </details>

        <button type="submit" disabled={submitDisabled} className={`${primaryButtonClass} w-full py-3 text-base`}>
          {submitLabel}
        </button>
        {message && <Alert>{message}</Alert>}
      </form>
    </Panel>
  );
}
