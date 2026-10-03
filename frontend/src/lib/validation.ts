// Client-side validation. Mirrors backend/app/schemas.py#build_params; the server
// re-validates everything, this only gives faster feedback.
import type { Defaults, GenerateRequest, Limits } from "../api/types";

export const FIELD_NAMES = [
  "prompt",
  "negative_prompt",
  "width",
  "height",
  "num_inference_steps",
  "guidance_scale",
  "seed",
] as const;

export type FieldName = (typeof FIELD_NAMES)[number];

/** Raw input strings, as typed by the user. */
export type FormValues = Record<FieldName, string>;

/** Keyed by field name; may also contain server-side names such as "body". */
export type FieldErrors = Partial<Record<string, string>>;

export type ValidationResult = { ok: true; payload: GenerateRequest } | { ok: false; errors: FieldErrors };

export function initialFormValues(defaults: Defaults): FormValues {
  return {
    prompt: "",
    negative_prompt: "",
    width: String(defaults.width),
    height: String(defaults.height),
    num_inference_steps: String(defaults.num_inference_steps),
    guidance_scale: String(defaults.guidance_scale),
    seed: "",
  };
}

function parseInteger(raw: string): number | null {
  const text = raw.trim();
  return /^-?\d+$/.test(text) ? Number(text) : null;
}

export function validateForm(values: FormValues, limits: Limits): ValidationResult {
  const errors: FieldErrors = {};

  const prompt = values.prompt.trim();
  if (!prompt) errors.prompt = "プロンプトを入力してください。";
  else if (prompt.length > limits.max_prompt_length)
    errors.prompt = `${limits.max_prompt_length}文字以内で入力してください。`;

  const negativePrompt = values.negative_prompt.trim();
  if (negativePrompt.length > limits.max_prompt_length)
    errors.negative_prompt = `${limits.max_prompt_length}文字以内で入力してください。`;

  const size = (key: "width" | "height"): number => {
    const n = parseInteger(values[key]);
    if (n === null) errors[key] = "整数で入力してください。";
    else if (n < limits.min_image_size || n > limits.max_image_size)
      errors[key] = `${limits.min_image_size}〜${limits.max_image_size}の範囲で指定してください。`;
    else if (n % limits.size_multiple !== 0) errors[key] = `${limits.size_multiple}の倍数で指定してください。`;
    return n ?? 0;
  };
  const width = size("width");
  const height = size("height");

  const steps = parseInteger(values.num_inference_steps);
  if (steps === null) errors.num_inference_steps = "整数で入力してください。";
  else if (steps < limits.min_steps || steps > limits.max_steps)
    errors.num_inference_steps = `${limits.min_steps}〜${limits.max_steps}の範囲で指定してください。`;

  const guidanceText = values.guidance_scale.trim();
  const guidance = guidanceText === "" ? Number.NaN : Number(guidanceText);
  if (!Number.isFinite(guidance)) errors.guidance_scale = "数値で入力してください。";
  else if (guidance < limits.min_guidance_scale || guidance > limits.max_guidance_scale)
    errors.guidance_scale = `${limits.min_guidance_scale}〜${limits.max_guidance_scale}の範囲で指定してください。`;

  let seed: number | null = null;
  if (values.seed.trim() !== "") {
    seed = parseInteger(values.seed);
    if (seed === null || seed < 0 || seed > limits.seed_max) {
      errors.seed = `0〜${limits.seed_max}の整数で指定するか、空欄にしてください。`;
    }
  }

  if (Object.keys(errors).length > 0) return { ok: false, errors };
  return {
    ok: true,
    payload: {
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      num_inference_steps: steps ?? 0,
      guidance_scale: guidance,
      seed,
    },
  };
}

/** True when an error belongs to a field inside the collapsible "詳細設定" section. */
export function hasAdvancedFieldError(errors: FieldErrors): boolean {
  return Object.keys(errors).some((name) => name !== "prompt");
}
