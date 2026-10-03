// @ts-check
// Client-side input validation. Mirrors backend/app/schemas.py#build_params;
// the server re-validates everything, this only gives faster feedback.

/**
 * @typedef {object} Limits
 * @property {number} min_image_size
 * @property {number} max_image_size
 * @property {number} size_multiple
 * @property {number} min_steps
 * @property {number} max_steps
 * @property {number} min_guidance_scale
 * @property {number} max_guidance_scale
 * @property {number} max_prompt_length
 * @property {number} seed_max
 *
 * @typedef {object} Defaults
 * @property {number} width
 * @property {number} height
 * @property {number} num_inference_steps
 * @property {number} guidance_scale
 *
 * @typedef {{ limits: Limits, defaults: Defaults }} AppConfig
 *
 * @typedef {object} FormValues   Raw strings as typed by the user.
 * @property {string} prompt
 * @property {string} negative_prompt
 * @property {string} width
 * @property {string} height
 * @property {string} num_inference_steps
 * @property {string} guidance_scale
 * @property {string} seed
 *
 * @typedef {object} GeneratePayload
 * @property {string} prompt
 * @property {string} negative_prompt
 * @property {number} width
 * @property {number} height
 * @property {number} num_inference_steps
 * @property {number} guidance_scale
 * @property {number | null} seed
 */

/** Used when /api/config cannot be reached. Must match the backend defaults. */
/** @type {AppConfig} */
export const FALLBACK_CONFIG = {
  limits: {
    min_image_size: 256,
    max_image_size: 1024,
    size_multiple: 8,
    min_steps: 1,
    max_steps: 50,
    min_guidance_scale: 0,
    max_guidance_scale: 20,
    max_prompt_length: 1000,
    seed_max: 4294967295,
  },
  defaults: { width: 512, height: 512, num_inference_steps: 25, guidance_scale: 7.5 },
};

/**
 * @param {string} raw
 * @returns {number | null} null when empty or not an integer
 */
function parseInteger(raw) {
  const text = raw.trim();
  if (!/^-?\d+$/.test(text)) return null;
  return Number(text);
}

/**
 * @param {FormValues} values
 * @param {Limits} limits
 * @returns {{ payload: GeneratePayload | null, errors: Record<string, string> }}
 */
export function validateForm(values, limits) {
  /** @type {Record<string, string>} */
  const errors = {};

  const prompt = values.prompt.trim();
  if (!prompt) errors.prompt = "プロンプトを入力してください。";
  else if (prompt.length > limits.max_prompt_length)
    errors.prompt = `${limits.max_prompt_length}文字以内で入力してください。`;

  const negativePrompt = values.negative_prompt.trim();
  if (negativePrompt.length > limits.max_prompt_length)
    errors.negative_prompt = `${limits.max_prompt_length}文字以内で入力してください。`;

  /** @param {"width" | "height"} key */
  const size = (key) => {
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
  const guidance = guidanceText === "" ? NaN : Number(guidanceText);
  if (!Number.isFinite(guidance)) errors.guidance_scale = "数値で入力してください。";
  else if (guidance < limits.min_guidance_scale || guidance > limits.max_guidance_scale)
    errors.guidance_scale = `${limits.min_guidance_scale}〜${limits.max_guidance_scale}の範囲で指定してください。`;

  /** @type {number | null} */
  let seed = null;
  if (values.seed.trim() !== "") {
    seed = parseInteger(values.seed);
    if (seed === null || seed < 0 || seed > limits.seed_max) {
      errors.seed = `0〜${limits.seed_max}の整数で指定するか、空欄にしてください。`;
      seed = null;
    }
  }

  if (Object.keys(errors).length > 0) return { payload: null, errors };
  return {
    payload: {
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      num_inference_steps: /** @type {number} */ (steps),
      guidance_scale: guidance,
      seed,
    },
    errors,
  };
}
