import type { AppConfig, Health } from "../api/types";

export const CONFIG: AppConfig = {
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

export function health(status: Health["status"] = "ready"): Health {
  return {
    status,
    model_id: "test/model",
    device: "cuda",
    dtype: "float16",
    message: status === "failed" ? "モデルが見つかりません。" : null,
    queue: { running: 0, waiting: 0, max_waiting: 4 },
  };
}

const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

export function pngResponse(seed: number, elapsedMs = 2500): Response {
  return new Response(PNG_SIGNATURE, {
    status: 200,
    headers: { "Content-Type": "image/png", "X-Seed": String(seed), "X-Generation-Time-Ms": String(elapsedMs) },
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
