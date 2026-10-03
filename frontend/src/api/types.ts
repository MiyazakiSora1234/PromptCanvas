// Mirrors the response models in backend/app/schemas.py.

export type ModelState = "not_loaded" | "loading" | "ready" | "failed";
export type OutputFormat = "png" | "jpeg" | "webp";

export interface Health {
  status: ModelState;
  /** Catalog id of the model that is loaded / being loaded / failed. */
  model: string | null;
  /** Models already downloaded (selecting others triggers a multi-GB download). */
  cached_models: string[];
  /** Face/pose reference assets downloaded. */
  identity_cached: boolean;
  device: string | null;
  dtype: string | null;
  message: string | null;
  queue: { running: number; waiting: number; max_waiting: number };
}

export interface Limits {
  min_image_size: number;
  max_image_size: number;
  size_multiple: number;
  min_steps: number;
  max_steps: number;
  min_guidance_scale: number;
  max_guidance_scale: number;
  max_prompt_length: number;
  seed_max: number;
  max_batch_size: number;
  max_loras: number;
  max_init_image_mb: number;
  min_strength: number;
  max_strength: number;
  min_lora_scale: number;
  max_lora_scale: number;
  min_quality: number;
  max_quality: number;
  min_control_strength: number;
  max_control_strength: number;
  max_face_images: number;
}

export interface IdentityOption {
  /** Model families that support face/pose references (InstantID is SDXL-only). */
  families: string[];
  download_size_gb: number | null;
}

export interface ModelDefaults {
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
  scheduler: string;
  style: string;
}

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  family: string;
  download_size_gb: number | null;
  defaults: ModelDefaults;
}

export interface LoraOption {
  id: string;
  label: string;
  description: string;
  family: string;
  trigger_words: string;
  default_scale: number;
}

export interface Option {
  id: string;
  label: string;
}

export interface StyleOption extends Option {
  description: string;
}

export interface FormatOption extends Option {
  id: OutputFormat;
  lossy: boolean;
  extension: string;
}

/** Generation settings stored in a preset. Omitted fields mean "the model's default". */
export interface PresetSettings {
  model: string;
  style?: string | null;
  scheduler?: string | null;
  width?: number | null;
  height?: number | null;
  num_inference_steps?: number | null;
  guidance_scale?: number | null;
  negative_prompt?: string | null;
  num_images?: number | null;
  output_format?: OutputFormat | null;
  quality?: number | null;
  loras?: LoraRequest[];
  identity_strength?: number | null;
  pose_strength?: number | null;
}

export interface PresetEntry {
  id: string;
  label: string;
  description: string;
  settings: PresetSettings;
}

export interface AppConfig {
  limits: Limits;
  default_model: string;
  models: ModelOption[];
  schedulers: Option[];
  styles: StyleOption[];
  loras: LoraOption[];
  output_formats: FormatOption[];
  /** null when the server has face/pose references disabled. */
  identity: IdentityOption | null;
  /** Built-in presets shipped with the server. */
  presets: PresetEntry[];
  defaults: {
    num_images: number;
    output_format: OutputFormat;
    quality: number;
    strength: number;
    identity_strength: number;
    pose_strength: number;
  };
}

export interface LoraRequest {
  id: string;
  scale: number;
}

export interface GenerateRequest {
  prompt: string;
  negative_prompt: string;
  model: string;
  scheduler: string;
  style: string;
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
  seed: number | null;
  num_images: number;
  output_format: OutputFormat;
  quality: number;
  init_image: string | null;
  strength: number;
  loras: LoraRequest[];
  /** Photos of one person; their face features are averaged. */
  face_images: string[];
  pose_image: string | null;
  identity_strength: number;
  pose_strength: number;
}

export interface GeneratedImageData {
  seed: number;
  mime_type: string;
  /** base64 */
  data: string;
}

export interface GenerateResponse {
  images: GeneratedImageData[];
  model: string;
  scheduler: string;
  style: string;
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
  output_format: OutputFormat;
  elapsed_ms: number;
  filtered_count: number;
}

export interface FieldErrorItem {
  field: string;
  message: string;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  fields: FieldErrorItem[];
  request_id: string | null;
}
