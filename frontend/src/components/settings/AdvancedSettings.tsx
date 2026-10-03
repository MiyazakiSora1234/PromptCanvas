import type { Limits, ModelOption, Option } from "../../api/types";
import type { FieldErrors, FormValues } from "../../lib/validation";
import type { makeControlProps } from "../formControl";
import { HELP } from "../../lib/help";
import { buttonClass, inputClass } from "../styles";
import { Field } from "../ui";

const SIZE_PRESETS: ReadonlyArray<readonly [number, number]> = [
  [512, 512],
  [768, 512],
  [512, 768],
  [1024, 1024],
  [896, 1152],
  [1216, 832],
  [832, 1216],
];

interface AdvancedSettingsProps {
  limits: Limits;
  schedulers: Option[];
  model: ModelOption;
  values: FormValues;
  fieldErrors: FieldErrors;
  control: ReturnType<typeof makeControlProps>;
  onValueChange: (name: keyof FormValues, value: string) => void;
}

/** "詳細" tab: sampling parameters, size and seed. */
export function AdvancedSettings({
  limits,
  schedulers,
  model,
  values,
  fieldErrors,
  control,
  onValueChange,
}: AdvancedSettingsProps) {
  const sizePresets = SIZE_PRESETS.filter(
    ([w, h]) => Math.max(w, h) <= limits.max_image_size && Math.min(w, h) >= limits.min_image_size,
  );

  return (
    <>
      <Field id="negative_prompt" label="ネガティブプロンプト" error={fieldErrors.negative_prompt} hint={HELP.negativePrompt}>
        <textarea
          {...control("negative_prompt")}
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
        hint={HELP.scheduler}
      >
        <select {...control("scheduler")}>
          {schedulers.map((s) => (
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
          {sizePresets.map(([w, h]) => {
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
              {...control("width")}
              type="number"
              inputMode="numeric"
              min={limits.min_image_size}
              max={limits.max_image_size}
              step={limits.size_multiple}
            />
          </Field>
          <Field id="height" label="高さ" error={fieldErrors.height}>
            <input
              {...control("height")}
              type="number"
              inputMode="numeric"
              min={limits.min_image_size}
              max={limits.max_image_size}
              step={limits.size_multiple}
            />
          </Field>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {HELP.size}（このモデルの基本: {model.defaults.width}×{model.defaults.height}）
        </p>
      </fieldset>

      <div className="grid grid-cols-2 gap-3">
        <Field id="num_inference_steps" label="ステップ数" error={fieldErrors.num_inference_steps} hint={HELP.steps}>
          <input
            {...control("num_inference_steps")}
            type="number"
            inputMode="numeric"
            min={limits.min_steps}
            max={limits.max_steps}
            step={1}
          />
        </Field>
        <Field id="guidance_scale" label="ガイダンススケール" error={fieldErrors.guidance_scale} hint={HELP.guidance}>
          <input
            {...control("guidance_scale")}
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
        hint={HELP.seed}
      >
        <div className="flex gap-2">
          <input
            {...control("seed")}
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
    </>
  );
}
