import { useEffect, useRef, type ChangeEvent, type FormEvent, type KeyboardEvent } from "react";
import type { AppConfig } from "../api/types";
import {
  compatibleLoras,
  findModel,
  FIELD_NAMES,
  type FieldErrors,
  type FormState,
  type FormValues,
  type LoraSelection,
} from "../lib/validation";
import { InitImagePicker } from "./InitImagePicker";
import { LoraPicker } from "./LoraPicker";
import { buttonClass, inputClass, primaryButtonClass } from "./styles";
import { Alert, Field, Panel } from "./ui";

const SIZE_PRESETS: ReadonlyArray<readonly [number, number]> = [
  [512, 512],
  [768, 512],
  [512, 768],
  [1024, 1024],
  [1216, 832],
  [832, 1216],
];

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
  advancedOpen: boolean;
  onAdvancedOpenChange: (open: boolean) => void;
  onSubmit: () => void;
  busy: boolean;
  submitDisabled: boolean;
  submitLabel: string;
  message: string | null;
  /** Incremented by the parent to move focus to the first field with an error. */
  focusRequest: number;
}

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
  advancedOpen,
  onAdvancedOpenChange,
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
  const format = config.output_formats.find((f) => f.id === values.output_format);
  const needsDownload = (id: string) => cachedModels !== null && !cachedModels.includes(id);

  useEffect(() => {
    if (focusRequest === 0) return;
    const first = FIELD_NAMES.find((name) => fieldErrors[name]);
    const el = first ? document.getElementById(first === "loras" ? "loras-error" : first) : null;
    if (el instanceof HTMLElement) el.focus();
    // Only when explicitly requested, not on every error change (that would steal focus while typing).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusRequest]);

  const controlProps = (name: keyof FormValues) => ({
    id: name,
    name,
    value: values[name],
    onChange: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
      onValueChange(name, e.target.value),
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
  const batchSizes = Array.from({ length: limits.max_batch_size }, (_, i) => i + 1);

  return (
    <Panel>
      <form ref={formRef} noValidate onSubmit={handleSubmit} onKeyDown={handleKeyDown} className="flex flex-col gap-4">
        <Field id="model" label="モデル" error={fieldErrors.model} hint={model.description}>
          <select {...controlProps("model")} onChange={(e) => onModelChange(e.target.value)} disabled={busy}>
            {config.models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
                {needsDownload(m.id) ? "（要ダウンロード）" : ""}
              </option>
            ))}
          </select>
          {needsDownload(model.id) && (
            <p role="note" className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
              このモデルはまだダウンロードされていません。初回の生成時に
              {model.download_size_gb ? `約${model.download_size_gb}GB の` : ""}
              ダウンロードが行われ、回線速度によって数分〜数十分かかります（サーバーで{" "}
              <code>make download-model</code> を実行しておくと待たずに使えます）。
            </p>
          )}
        </Field>

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

        <InitImagePicker
          limits={limits}
          image={state.initImage}
          strength={values.strength}
          steps={values.num_inference_steps}
          imageError={fieldErrors.init_image}
          strengthError={fieldErrors.strength}
          disabled={busy}
          onFile={onInitImageFile}
          onClear={onInitImageClear}
          onStrengthChange={(v) => onValueChange("strength", v)}
        />

        <div className="grid grid-cols-2 gap-3">
          <fieldset className="flex min-w-0 flex-col gap-1">
            <legend className="mb-1 text-sm font-semibold">枚数</legend>
            <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="枚数">
              {batchSizes.map((n) => (
                <label key={n} className="cursor-pointer">
                  <input
                    type="radio"
                    name="num_images"
                    value={n}
                    checked={values.num_images === String(n)}
                    onChange={() => onValueChange("num_images", String(n))}
                    className="peer sr-only"
                  />
                  <span
                    className={`${buttonClass} px-3 py-1 text-sm peer-checked:border-indigo-600 peer-checked:bg-indigo-50 peer-checked:text-indigo-700 peer-focus-visible:outline-2 peer-focus-visible:outline-indigo-600 dark:peer-checked:bg-indigo-950 dark:peer-checked:text-indigo-300`}
                  >
                    {n}
                  </span>
                </label>
              ))}
            </div>
            {fieldErrors.num_images && (
              <p className="text-xs text-red-700 dark:text-red-300">{fieldErrors.num_images}</p>
            )}
          </fieldset>
          <Field id="output_format" label="画像形式" error={fieldErrors.output_format}>
            <select {...controlProps("output_format")}>
              {config.output_formats.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </Field>
        </div>

        <LoraPicker
          limits={limits}
          options={compatibleLoras(config, model)}
          modelLabel={model.label}
          selected={state.loras}
          error={fieldErrors.loras}
          onChange={onLorasChange}
        />

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

            <Field
              id="scheduler"
              label="サンプラー"
              error={fieldErrors.scheduler}
              hint="ノイズを取り除く手順です。速度や仕上がりが変わり、モデルとの相性があります。"
            >
              <select {...controlProps("scheduler")}>
                {config.schedulers.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                    {s.id === model.defaults.scheduler ? "（このモデルの推奨）" : ""}
                  </option>
                ))}
              </select>
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
                        onValueChange("width", String(w));
                        onValueChange("height", String(h));
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
                {limits.size_multiple}の倍数で指定してください。このモデルの基本サイズは {model.defaults.width}×
                {model.defaults.height} です。大きいほど時間と GPU メモリを消費します。
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
              hint="同じシードと設定で同じ画像を再現できます。複数枚のときは 1 枚ごとに +1 されます。"
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
                <button type="button" className={buttonClass} onClick={() => onValueChange("seed", "")}>
                  ランダム
                </button>
              </div>
            </Field>

            {format?.lossy && (
              <div className="flex flex-col gap-1">
                <label htmlFor="quality" className="flex justify-between text-sm font-semibold">
                  <span>画質（{format.id.toUpperCase()}）</span>
                  <span className="tabular-nums">{values.quality}</span>
                </label>
                <input
                  id="quality"
                  name="quality"
                  type="range"
                  min={limits.min_quality}
                  max={limits.max_quality}
                  step={1}
                  value={values.quality}
                  aria-invalid={fieldErrors.quality ? true : undefined}
                  onChange={(e) => onValueChange("quality", e.target.value)}
                  className="accent-indigo-600"
                />
                <p className="text-xs text-slate-500 dark:text-slate-400">高いほどきれいで、ファイルが大きくなります。</p>
                {fieldErrors.quality && (
                  <p className="text-xs text-red-700 dark:text-red-300">{fieldErrors.quality}</p>
                )}
              </div>
            )}
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
