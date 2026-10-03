import type { AppConfig, ModelOption } from "../../api/types";
import type { FieldErrors, FormValues } from "../../lib/validation";
import type { makeControlProps } from "../formControl";
import { StrengthSlider } from "../StrengthSlider";
import { buttonClass } from "../styles";
import { Field } from "../ui";

interface BasicSettingsProps {
  config: AppConfig;
  model: ModelOption;
  values: FormValues;
  fieldErrors: FieldErrors;
  control: ReturnType<typeof makeControlProps>;
  needsDownload: (modelId: string) => boolean;
  busy: boolean;
  onModelChange: (modelId: string) => void;
  onValueChange: (name: keyof FormValues, value: string) => void;
}

/** "基本" tab: what to generate with and how many / in which format. */
export function BasicSettings({
  config,
  model,
  values,
  fieldErrors,
  control,
  needsDownload,
  busy,
  onModelChange,
  onValueChange,
}: BasicSettingsProps) {
  const { limits } = config;
  const format = config.output_formats.find((f) => f.id === values.output_format);
  const batchSizes = Array.from({ length: limits.max_batch_size }, (_, i) => i + 1);

  return (
    <>
      <Field id="model" label="モデル" error={fieldErrors.model} hint={model.description}>
        <select {...control("model")} onChange={(e) => onModelChange(e.target.value)} disabled={busy}>
          {config.models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
              {needsDownload(m.id) ? "（要ダウンロード）" : ""}
            </option>
          ))}
        </select>
        {needsDownload(model.id) && (
          <p
            role="note"
            className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300"
          >
            このモデルはまだダウンロードされていません。初回の生成時に
            {model.download_size_gb ? `約${model.download_size_gb}GB の` : ""}
            ダウンロードが行われ、回線速度によって数分〜数十分かかります（サーバーで{" "}
            <code>make download-model</code> を実行しておくと待たずに使えます）。
          </p>
        )}
      </Field>

      <Field
        id="style"
        label="スタイル"
        error={fieldErrors.style}
        hint={
          values.style === "photo"
            ? "肌のきめ・毛穴などの質感を出す語句と、つるつるした肌・CG っぽさを避ける語句をプロンプトに自動で加えます。"
            : undefined
        }
      >
        <select {...control("style")}>
          {config.styles.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
              {s.id === model.defaults.style && s.id !== "none" ? "（このモデルの推奨）" : ""}
            </option>
          ))}
        </select>
      </Field>

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
          {fieldErrors.num_images && <p className="text-xs text-red-700 dark:text-red-300">{fieldErrors.num_images}</p>}
        </fieldset>
        <Field id="output_format" label="画像形式" error={fieldErrors.output_format}>
          <select {...control("output_format")}>
            {config.output_formats.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {format?.lossy && (
        <StrengthSlider
          id="quality"
          label={`画質（${format.id.toUpperCase()}）`}
          value={values.quality}
          min={limits.min_quality}
          max={limits.max_quality}
          step={1}
          decimals={0}
          error={fieldErrors.quality}
          hint="高いほどきれいで、ファイルが大きくなります。"
          onChange={(v) => onValueChange("quality", v)}
        />
      )}
    </>
  );
}
