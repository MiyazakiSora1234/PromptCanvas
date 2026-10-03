import type { AppConfig, GenerateRequest, Health } from "../api/types";

export const CONFIG: AppConfig = {
  limits: {
    min_image_size: 256,
    max_image_size: 1536,
    size_multiple: 8,
    min_steps: 1,
    max_steps: 50,
    min_guidance_scale: 0,
    max_guidance_scale: 20,
    max_prompt_length: 1000,
    seed_max: 4294967295,
    max_batch_size: 4,
    max_loras: 3,
    max_init_image_mb: 10,
    min_strength: 0.1,
    max_strength: 1,
    min_lora_scale: 0,
    max_lora_scale: 2,
    min_quality: 1,
    max_quality: 100,
    min_control_strength: 0,
    max_control_strength: 1.5,
  },
  default_model: "sd15",
  models: [
    {
      id: "sd15",
      label: "SD 1.5",
      description: "軽量",
      family: "sd15",
      download_size_gb: 5.2,
      defaults: {
        width: 512,
        height: 512,
        num_inference_steps: 25,
        guidance_scale: 7.5,
        scheduler: "default",
        style: "none",
      },
    },
    {
      id: "sdxl",
      label: "SDXL",
      description: "高画質",
      family: "sdxl",
      download_size_gb: 6.9,
      defaults: {
        width: 1024,
        height: 1024,
        num_inference_steps: 30,
        guidance_scale: 7,
        scheduler: "euler_a",
        style: "photo",
      },
    },
  ],
  schedulers: [
    { id: "default", label: "モデル既定" },
    { id: "euler_a", label: "Euler a" },
    { id: "dpmpp_2m_karras", label: "DPM++ 2M Karras" },
  ],
  styles: [
    { id: "none", label: "なし" },
    { id: "photo", label: "リアルな写真（肌の質感）" },
  ],
  loras: [
    {
      id: "pixel",
      label: "Pixel Art XL",
      description: "ドット絵風",
      family: "sdxl",
      trigger_words: "pixel art",
      default_scale: 1,
    },
  ],
  output_formats: [
    { id: "png", label: "PNG", lossy: false, extension: "png" },
    { id: "jpeg", label: "JPEG", lossy: true, extension: "jpg" },
    { id: "webp", label: "WebP", lossy: true, extension: "webp" },
  ],
  identity: { families: ["sdxl"], download_size_gb: 6.6 },
  defaults: { num_images: 1, output_format: "png", quality: 90, strength: 0.6, identity_strength: 0.8, pose_strength: 0.9 },
};

export function health(status: Health["status"] = "ready", model = "sd15"): Health {
  return {
    status,
    model,
    cached_models: ["sd15"],
    identity_cached: true,
    device: "cuda",
    dtype: "float16",
    message: status === "failed" ? "モデルが見つかりません。" : null,
    queue: { running: 0, waiting: 0, max_waiting: 4 },
  };
}

const MIME: Record<string, string> = { png: "image/png", jpeg: "image/jpeg", webp: "image/webp" };

/** A successful /api/generate response echoing the request, one image per requested count. */
export function generateResponse(req: GenerateRequest, elapsedMs = 2500): Response {
  const base = req.seed ?? 1234;
  return Response.json({
    images: Array.from({ length: req.num_images }, (_, i) => ({
      seed: base + i,
      mime_type: MIME[req.output_format],
      data: btoa("fake-image-bytes"),
    })),
    model: req.model,
    scheduler: req.scheduler,
    style: req.style,
    width: req.width,
    height: req.height,
    num_inference_steps: req.num_inference_steps,
    guidance_scale: req.guidance_scale,
    output_format: req.output_format,
    elapsed_ms: elapsedMs,
    filtered_count: 0,
  });
}

export function errorResponse(
  status: number,
  code: string,
  message: string,
  fields: Array<{ field: string; message: string }> = [],
): Response {
  return Response.json({ error: { code, message, fields, request_id: "req123" } }, { status });
}
