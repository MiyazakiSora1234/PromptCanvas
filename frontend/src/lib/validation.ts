// Client-side validation. Mirrors backend/app/schemas.py#build_params; the server
// re-validates everything, this only gives faster feedback.
import type { AppConfig, GenerateRequest, LoraOption, ModelOption, OutputFormat } from "../api/types";
import type { PickedImage } from "./images";

/** Inputs in on-screen order: the prompt, then tab by tab (used to focus the first invalid one). */
export const FIELD_NAMES = [
  "prompt",
  // 基本
  "model",
  "style",
  "num_images",
  "output_format",
  "quality",
  // 画像参照
  "init_image",
  "strength",
  "face_images",
  "pose_image",
  "identity_strength",
  "pose_strength",
  // LoRA
  "loras",
  // 詳細
  "negative_prompt",
  "scheduler",
  "width",
  "height",
  "num_inference_steps",
  "guidance_scale",
  "seed",
] as const;

export type FieldName = (typeof FIELD_NAMES)[number];

/** Raw input strings, as typed by the user. */
export type FormValues = Record<Exclude<FieldName, "init_image" | "face_images" | "pose_image" | "loras">, string>;

export interface LoraSelection {
  id: string;
  scale: number;
}

export interface FormState {
  values: FormValues;
  loras: LoraSelection[];
  initImage: PickedImage | null;
  /** Photos of the person whose face to keep (InstantID); averaged on the server. */
  faceImages: PickedImage[];
  /** Pose to copy (OpenPose ControlNet). */
  poseImage: PickedImage | null;
}

/** Whether the face/pose reference feature can be used with this model. */
export function supportsReference(config: AppConfig, model: ModelOption): boolean {
  return config.identity !== null && config.identity.families.includes(model.family);
}

/** Keyed by field name; may also contain server-side names such as "body". */
export type FieldErrors = Partial<Record<string, string>>;

export type ValidationResult = { ok: true; payload: GenerateRequest } | { ok: false; errors: FieldErrors };

export function findModel(config: AppConfig, id: string): ModelOption {
  return config.models.find((m) => m.id === id) ?? config.models[0]!;
}

export function compatibleLoras(config: AppConfig, model: ModelOption): LoraOption[] {
  return config.loras.filter((lora) => lora.family === model.family);
}

/** The values a model starts with; also applied when the user switches models. */
export function modelDefaultValues(model: ModelOption): Pick<
  FormValues,
  "model" | "style" | "scheduler" | "width" | "height" | "num_inference_steps" | "guidance_scale"
> {
  return {
    model: model.id,
    style: model.defaults.style,
    scheduler: model.defaults.scheduler,
    width: String(model.defaults.width),
    height: String(model.defaults.height),
    num_inference_steps: String(model.defaults.num_inference_steps),
    guidance_scale: String(model.defaults.guidance_scale),
  };
}

export function initialFormState(config: AppConfig): FormState {
  return {
    values: {
      ...modelDefaultValues(findModel(config, config.default_model)),
      prompt: "",
      negative_prompt: "",
      seed: "",
      num_images: String(config.defaults.num_images),
      output_format: config.defaults.output_format,
      quality: String(config.defaults.quality),
      strength: String(config.defaults.strength),
      identity_strength: String(config.defaults.identity_strength),
      pose_strength: String(config.defaults.pose_strength),
    },
    loras: [],
    initImage: null,
    faceImages: [],
    poseImage: null,
  };
}

function parseInteger(raw: string): number | null {
  const text = raw.trim();
  return /^-?\d+$/.test(text) ? Number(text) : null;
}

function parseNumber(raw: string): number {
  const text = raw.trim();
  return text === "" ? Number.NaN : Number(text);
}

export function validateForm(state: FormState, config: AppConfig): ValidationResult {
  const { values, initImage } = state;
  const { limits } = config;
  const errors: FieldErrors = {};

  const prompt = values.prompt.trim();
  if (!prompt) errors.prompt = "プロンプトを入力してください。";
  else if (prompt.length > limits.max_prompt_length)
    errors.prompt = `${limits.max_prompt_length}文字以内で入力してください。`;

  const negativePrompt = values.negative_prompt.trim();
  if (negativePrompt.length > limits.max_prompt_length)
    errors.negative_prompt = `${limits.max_prompt_length}文字以内で入力してください。`;

  const model = config.models.find((m) => m.id === values.model);
  if (!model) errors.model = "モデルを選んでください。";

  if (!config.schedulers.some((s) => s.id === values.scheduler)) errors.scheduler = "サンプラーを選んでください。";
  if (!config.styles.some((s) => s.id === values.style)) errors.style = "スタイルを選んでください。";

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

  const guidance = parseNumber(values.guidance_scale);
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

  const numImages = parseInteger(values.num_images);
  if (numImages === null || numImages < 1 || numImages > limits.max_batch_size)
    errors.num_images = `1〜${limits.max_batch_size}枚の範囲で指定してください。`;

  const format = config.output_formats.find((f) => f.id === values.output_format);
  if (!format) errors.output_format = "画像形式を選んでください。";

  const quality = parseInteger(values.quality);
  if (format?.lossy && (quality === null || quality < limits.min_quality || quality > limits.max_quality))
    errors.quality = `${limits.min_quality}〜${limits.max_quality}の整数で指定してください。`;

  const strength = parseNumber(values.strength);
  if (initImage) {
    if (!Number.isFinite(strength) || strength < limits.min_strength || strength > limits.max_strength)
      errors.strength = `${limits.min_strength}〜${limits.max_strength}の範囲で指定してください。`;
    else if (steps !== null && Math.floor(steps * strength) < 1)
      errors.strength = "ステップ数 × 変換強度が 1 以上になるようにしてください。";
  }

  const { faceImages, poseImage } = state;
  const hasFaces = faceImages.length > 0;
  const identityStrength = parseNumber(values.identity_strength);
  const poseStrength = parseNumber(values.pose_strength);
  if (hasFaces || poseImage) {
    const field = hasFaces ? "face_images" : "pose_image";
    if (model && !supportsReference(config, model))
      errors[field] = `顔・ポーズの参照は SDXL 系のモデルでのみ使えます（選択中: ${model.label}）。`;
    if (initImage) errors[field] = "img2img（元画像から生成）と顔・ポーズの参照は同時に使えません。どちらかを外してください。";
    if (faceImages.length > limits.max_face_images)
      errors.face_images = `顔の写真は${limits.max_face_images}枚まで選べます。`;
    const range = (value: number) =>
      Number.isFinite(value) && value >= limits.min_control_strength && value <= limits.max_control_strength;
    const message = `${limits.min_control_strength}〜${limits.max_control_strength}の範囲で指定してください。`;
    if (hasFaces && !range(identityStrength)) errors.identity_strength = message;
    if (poseImage && !range(poseStrength)) errors.pose_strength = message;
  }

  if (state.loras.length > limits.max_loras) errors.loras = `LoRA は${limits.max_loras}個まで選択できます。`;
  for (const selection of state.loras) {
    const lora = config.loras.find((l) => l.id === selection.id);
    if (!lora || (model && lora.family !== model.family)) {
      errors.loras = `LoRA「${lora?.label ?? selection.id}」は選択中のモデルでは使えません。`;
    } else if (
      !Number.isFinite(selection.scale) ||
      selection.scale < limits.min_lora_scale ||
      selection.scale > limits.max_lora_scale
    ) {
      errors.loras = `LoRA の強さは${limits.min_lora_scale}〜${limits.max_lora_scale}の範囲で指定してください。`;
    }
  }

  if (Object.keys(errors).length > 0) return { ok: false, errors };
  return {
    ok: true,
    payload: {
      prompt,
      negative_prompt: negativePrompt,
      model: values.model,
      scheduler: values.scheduler,
      style: values.style,
      width,
      height,
      num_inference_steps: steps ?? 0,
      guidance_scale: guidance,
      seed,
      num_images: numImages ?? 1,
      output_format: values.output_format as OutputFormat,
      quality: quality ?? config.defaults.quality,
      init_image: initImage?.dataUrl ?? null,
      strength: Number.isFinite(strength) ? strength : config.defaults.strength,
      loras: state.loras.map(({ id, scale }) => ({ id, scale })),
      face_images: faceImages.map((img) => img.dataUrl),
      pose_image: poseImage?.dataUrl ?? null,
      identity_strength: Number.isFinite(identityStrength) ? identityStrength : config.defaults.identity_strength,
      pose_strength: Number.isFinite(poseStrength) ? poseStrength : config.defaults.pose_strength,
    },
  };
}
