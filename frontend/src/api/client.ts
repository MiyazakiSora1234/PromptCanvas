import type { ApiErrorBody, AppConfig, FieldErrorItem, GenerateRequest, GeneratedImage, Health } from "./types";

// Empty = same origin (FastAPI serves the build; Vite proxies /api in development).
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

/** The server answered with an error. `message` is safe to show to the user. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly fields: FieldErrorItem[];
  readonly requestId: string | null;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
    this.fields = body.fields;
    this.requestId = body.request_id;
  }
}

/** The server could not be reached at all. */
export class NetworkError extends Error {
  constructor() {
    super("サーバーに接続できませんでした。");
    this.name = "NetworkError";
  }
}

function isErrorBody(value: unknown): value is { error: ApiErrorBody } {
  if (typeof value !== "object" || value === null || !("error" in value)) return false;
  const error = (value as { error: unknown }).error;
  return typeof error === "object" && error !== null && "code" in error && "message" in error;
}

async function toApiError(res: Response): Promise<ApiError> {
  try {
    const body: unknown = await res.json();
    if (isErrorBody(body)) {
      const { error } = body;
      return new ApiError(res.status, { ...error, fields: error.fields ?? [], request_id: error.request_id ?? null });
    }
  } catch {
    // Not JSON (e.g. an HTML page from a proxy); fall through to a generic message.
  }
  const message =
    res.status === 502 || res.status === 504
      ? "サーバーから応答がありませんでした。生成に時間がかかりすぎた可能性があります。サイズやステップ数を減らして再度お試しください。"
      : `予期しないエラーが発生しました（HTTP ${res.status}）。時間をおいて再度お試しください。`;
  return new ApiError(res.status, { code: "unexpected_response", message, fields: [], request_id: null });
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new NetworkError();
  }
  if (!res.ok) throw await toApiError(res);
  return res;
}

export async function fetchConfig(signal?: AbortSignal): Promise<AppConfig> {
  const res = await request("/api/config", { signal });
  return (await res.json()) as AppConfig;
}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const res = await request("/api/health", { signal, cache: "no-store" });
  return (await res.json()) as Health;
}

export async function generateImage(payload: GenerateRequest, signal?: AbortSignal): Promise<GeneratedImage> {
  const res = await request("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  const blob = await res.blob();
  if (blob.type !== "image/png") {
    throw new ApiError(res.status, {
      code: "unexpected_response",
      message: "サーバーから画像以外の応答が返されました。時間をおいて再度お試しください。",
      fields: [],
      request_id: null,
    });
  }
  const seed = res.headers.get("X-Seed");
  const elapsed = res.headers.get("X-Generation-Time-Ms");
  return {
    blob,
    seed: seed !== null ? Number(seed) : (payload.seed ?? Number.NaN),
    elapsedMs: elapsed !== null ? Number(elapsed) : null,
  };
}
