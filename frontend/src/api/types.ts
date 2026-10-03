// Mirrors the response models in backend/app/schemas.py.

export type ModelState = "not_loaded" | "loading" | "ready" | "failed";

export interface Health {
  status: ModelState;
  model_id: string;
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
}

export interface Defaults {
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
}

export interface AppConfig {
  limits: Limits;
  defaults: Defaults;
}

export interface GenerateRequest {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  num_inference_steps: number;
  guidance_scale: number;
  seed: number | null;
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

export interface GeneratedImage {
  blob: Blob;
  seed: number;
  elapsedMs: number | null;
}
