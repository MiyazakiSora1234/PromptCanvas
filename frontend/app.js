// @ts-check
import { FALLBACK_CONFIG, validateForm } from "./validation.js";

/** @typedef {import("./validation.js").AppConfig} AppConfig */
/** @typedef {import("./validation.js").GeneratePayload} GeneratePayload */
/**
 * @typedef {object} ApiError
 * @property {string} code
 * @property {string} message
 * @property {{ field: string, message: string }[]} fields
 * @property {string | null} request_id
 *
 * @typedef {object} Health
 * @property {"not_loaded" | "loading" | "ready" | "failed"} status
 * @property {string} model_id
 * @property {string | null} device
 * @property {string | null} dtype
 * @property {string | null} message
 */

const FIELD_NAMES = /** @type {const} */ ([
  "prompt",
  "negative_prompt",
  "width",
  "height",
  "num_inference_steps",
  "guidance_scale",
  "seed",
]);

const apiBase = (document.querySelector('meta[name="api-base"]')?.getAttribute("content") ?? "").replace(/\/$/, "");

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @param {{ new (): T }} type
 * @returns {T}
 */
function byId(id, type) {
  const el = document.getElementById(id);
  if (!(el instanceof type)) throw new Error(`#${id} is missing`);
  return el;
}

const els = {
  form: byId("generate-form", HTMLFormElement),
  prompt: byId("prompt", HTMLTextAreaElement),
  promptCount: byId("prompt-count", HTMLElement),
  negativePrompt: byId("negative_prompt", HTMLTextAreaElement),
  width: byId("width", HTMLInputElement),
  height: byId("height", HTMLInputElement),
  steps: byId("num_inference_steps", HTMLInputElement),
  guidance: byId("guidance_scale", HTMLInputElement),
  seed: byId("seed", HTMLInputElement),
  randomSeed: byId("random-seed", HTMLButtonElement),
  presets: byId("size-presets", HTMLElement),
  advanced: byId("advanced", HTMLDetailsElement),
  submit: byId("submit-button", HTMLButtonElement),
  message: byId("form-message", HTMLElement),
  status: byId("model-status", HTMLElement),
  frame: byId("result-frame", HTMLElement),
  placeholder: byId("placeholder", HTMLElement),
  progress: byId("progress", HTMLElement),
  elapsed: byId("elapsed", HTMLElement),
  image: byId("result-image", HTMLImageElement),
  meta: byId("result-meta", HTMLElement),
  download: byId("download-link", HTMLAnchorElement),
  reuseSeed: byId("reuse-seed", HTMLButtonElement),
};

const state = {
  /** @type {AppConfig} */
  config: FALLBACK_CONFIG,
  busy: false,
  /** @type {Health["status"] | "unknown" | "offline"} */
  modelState: "unknown",
  /** @type {string | null} */
  objectUrl: null,
  /** @type {number | null} */
  lastSeed: null,
  /** @type {number | undefined} */
  healthTimer: undefined,
};

// ---------------------------------------------------------------------------
// Messages and field errors

/**
 * @param {string} text
 * @param {"error" | "info"} [kind]
 */
function showMessage(text, kind = "error") {
  els.message.textContent = text;
  els.message.dataset.kind = kind;
  els.message.hidden = false;
}

function clearMessages() {
  els.message.hidden = true;
  els.message.textContent = "";
  for (const name of FIELD_NAMES) setFieldError(name, "");
}

/**
 * @param {string} name
 * @param {string} text
 */
function setFieldError(name, text) {
  const slot = els.form.querySelector(`[data-error-for="${name}"]`);
  if (slot) slot.textContent = text;
  const input = els.form.elements.namedItem(name);
  if (input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement) {
    if (text) input.setAttribute("aria-invalid", "true");
    else input.removeAttribute("aria-invalid");
  }
}

/** @param {Record<string, string>} errors */
function showFieldErrors(errors) {
  let first = true;
  for (const [name, text] of Object.entries(errors)) {
    setFieldError(name, text);
    if (name !== "prompt") els.advanced.open = true;
    const input = els.form.elements.namedItem(name);
    if (first && (input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement)) {
      input.focus();
      first = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Config and model status

function applyConfig() {
  const { limits, defaults } = state.config;
  for (const input of [els.width, els.height]) {
    input.min = String(limits.min_image_size);
    input.max = String(limits.max_image_size);
    input.step = String(limits.size_multiple);
  }
  els.width.value = String(defaults.width);
  els.height.value = String(defaults.height);
  els.steps.min = String(limits.min_steps);
  els.steps.max = String(limits.max_steps);
  els.steps.value = String(defaults.num_inference_steps);
  els.guidance.min = String(limits.min_guidance_scale);
  els.guidance.max = String(limits.max_guidance_scale);
  els.guidance.value = String(defaults.guidance_scale);
  els.seed.max = String(limits.seed_max);
  els.prompt.maxLength = limits.max_prompt_length;
  els.negativePrompt.maxLength = limits.max_prompt_length;

  for (const button of els.presets.querySelectorAll("button")) {
    const w = Number(button.dataset.width);
    const h = Number(button.dataset.height);
    button.hidden = Math.max(w, h) > limits.max_image_size || Math.min(w, h) < limits.min_image_size;
  }
  updatePromptCount();
  updatePresetSelection();
}

async function loadConfig() {
  try {
    const res = await fetch(`${apiBase}/api/config`);
    if (res.ok) state.config = /** @type {AppConfig} */ (await res.json());
  } catch {
    // Keep fallback values; health polling reports the connection problem.
  }
  applyConfig();
}

/** @param {Health["status"] | "offline"} status @param {Health | null} health */
function renderStatus(status, health) {
  state.modelState = status;
  els.status.dataset.state = status;
  if (status === "ready" && health) {
    const device = health.device === "cuda" ? "GPU (CUDA)" : health.device === "mps" ? "GPU (MPS)" : "CPU";
    const slow = health.device === "cpu" ? "・生成に時間がかかります" : "";
    els.status.textContent = `準備完了：${device}${slow}`;
    els.status.title = `${health.model_id} / ${health.dtype ?? ""}`;
  } else if (status === "loading" || status === "not_loaded") {
    els.status.textContent = "モデル読み込み中…（初回はダウンロードに時間がかかります）";
  } else if (status === "failed") {
    els.status.textContent = "モデルの読み込みに失敗しました";
    els.status.title = health?.message ?? "";
    showMessage(
      `モデルを利用できません。${health?.message ?? "サーバーログを確認してください。"} 設定を見直してサーバーを再起動してください。`,
    );
  } else {
    els.status.textContent = "サーバーに接続できません";
  }
  updateSubmitState();
}

async function refreshHealth() {
  window.clearTimeout(state.healthTimer);
  /** @type {number | null} */
  let nextPoll = null;
  try {
    const res = await fetch(`${apiBase}/api/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    const health = /** @type {Health} */ (await res.json());
    renderStatus(health.status, health);
    if (health.status === "loading" || health.status === "not_loaded") nextPoll = 3000;
  } catch {
    renderStatus("offline", null);
    nextPoll = 5000;
  }
  if (nextPoll !== null) state.healthTimer = window.setTimeout(refreshHealth, nextPoll);
}

function updateSubmitState() {
  const blocked = state.modelState === "loading" || state.modelState === "not_loaded" || state.modelState === "failed";
  els.submit.disabled = state.busy || blocked;
  els.submit.textContent = state.busy ? "生成中…" : blocked && state.modelState !== "failed" ? "モデル準備中…" : "画像を生成";
}

// ---------------------------------------------------------------------------
// Form helpers

function updatePromptCount() {
  const max = state.config.limits.max_prompt_length;
  const length = els.prompt.value.trim().length;
  els.promptCount.textContent = `${length} / ${max}`;
  els.promptCount.dataset.over = String(length > max);
}

function updatePresetSelection() {
  for (const button of els.presets.querySelectorAll("button")) {
    const selected = button.dataset.width === els.width.value && button.dataset.height === els.height.value;
    button.setAttribute("aria-pressed", String(selected));
  }
}

/** @returns {import("./validation.js").FormValues} */
function readForm() {
  return {
    prompt: els.prompt.value,
    negative_prompt: els.negativePrompt.value,
    width: els.width.value,
    height: els.height.value,
    num_inference_steps: els.steps.value,
    guidance_scale: els.guidance.value,
    seed: els.seed.value,
  };
}

// ---------------------------------------------------------------------------
// Generation

/** @param {boolean} busy */
function setBusy(busy) {
  state.busy = busy;
  els.form.setAttribute("aria-busy", String(busy));
  els.frame.setAttribute("aria-busy", String(busy));
  els.progress.hidden = !busy;
  els.placeholder.hidden = busy || !els.image.hidden;
  els.reuseSeed.disabled = busy;
  if (busy) els.frame.dataset.state = "busy";
  updateSubmitState();
}

/**
 * @param {Response} res
 * @returns {Promise<void>}
 */
async function handleErrorResponse(res) {
  /** @type {ApiError | null} */
  let error = null;
  try {
    const body = await res.json();
    if (body && typeof body === "object" && body.error) error = body.error;
  } catch {
    // Non-JSON error (e.g. a proxy page); fall through to a generic message.
  }

  if (!error) {
    const text =
      res.status === 502 || res.status === 504
        ? "サーバーから応答がありませんでした。生成に時間がかかりすぎた可能性があります。サイズやステップ数を減らして再度お試しください。"
        : `予期しないエラーが発生しました（HTTP ${res.status}）。時間をおいて再度お試しください。`;
    showMessage(text);
    return;
  }

  const suffix = error.request_id && res.status >= 500 && error.code !== "gpu_out_of_memory" ? `（問い合わせID: ${error.request_id}）` : "";
  showMessage(error.message + suffix);
  if (error.fields.length > 0) {
    showFieldErrors(Object.fromEntries(error.fields.map((f) => [f.field, f.message])));
  }
  if (error.code === "model_loading") refreshHealth();
  if (error.code === "gpu_out_of_memory") els.advanced.open = true;
}

/**
 * @param {Blob} blob
 * @param {GeneratePayload} payload
 * @param {number} seed
 * @param {number | null} elapsedMs
 */
function showResult(blob, payload, seed, elapsedMs) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  const url = URL.createObjectURL(blob);
  state.objectUrl = url;
  state.lastSeed = seed;

  els.image.src = url;
  els.image.alt = `生成画像: ${payload.prompt}`;
  els.image.hidden = false;
  els.placeholder.hidden = true;
  els.frame.dataset.state = "done";

  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  els.download.href = url;
  els.download.download = `promptcanvas_${stamp}_seed${seed}.png`;
  els.download.hidden = false;
  els.reuseSeed.hidden = false;

  /** @type {[string, string][]} */
  const rows = [
    ["シード", String(seed)],
    ["サイズ", `${payload.width}×${payload.height}`],
    ["ステップ", String(payload.num_inference_steps)],
    ["ガイダンス", String(payload.guidance_scale)],
  ];
  if (elapsedMs !== null) rows.push(["生成時間", `${(elapsedMs / 1000).toFixed(1)} 秒`]);
  els.meta.replaceChildren(
    ...rows.flatMap(([k, v]) => {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      return [dt, dd];
    }),
  );
  els.meta.hidden = false;
}

/** @param {SubmitEvent | Event} event */
async function onSubmit(event) {
  event.preventDefault();
  if (state.busy) return; // ignore double submits
  clearMessages();

  const { payload, errors } = validateForm(readForm(), state.config.limits);
  if (!payload) {
    showFieldErrors(errors);
    showMessage("入力内容を確認してください。");
    return;
  }

  setBusy(true);
  const started = performance.now();
  const timer = window.setInterval(() => {
    els.elapsed.textContent = ((performance.now() - started) / 1000).toFixed(1);
  }, 100);
  els.elapsed.textContent = "0.0";

  try {
    const res = await fetch(`${apiBase}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      await handleErrorResponse(res);
      return;
    }
    const blob = await res.blob();
    if (blob.type !== "image/png") {
      showMessage("サーバーから画像以外の応答が返されました。時間をおいて再度お試しください。");
      return;
    }
    const seed = Number(res.headers.get("X-Seed") ?? payload.seed ?? NaN);
    const timeHeader = res.headers.get("X-Generation-Time-Ms");
    showResult(blob, payload, seed, timeHeader === null ? null : Number(timeHeader));
  } catch {
    showMessage("サーバーに接続できませんでした。サーバーが起動しているか、ネットワーク接続を確認してから再度お試しください。");
    refreshHealth();
  } finally {
    window.clearInterval(timer);
    setBusy(false);
    if (els.image.hidden) els.frame.dataset.state = "empty";
    else els.frame.dataset.state = "done";
  }
}

// ---------------------------------------------------------------------------
// Wiring

els.form.addEventListener("submit", onSubmit);
els.prompt.addEventListener("input", updatePromptCount);
els.form.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    if (!els.submit.disabled) els.form.requestSubmit();
  }
});
els.presets.addEventListener("click", (event) => {
  const button = event.target instanceof Element ? event.target.closest("button") : null;
  if (!button) return;
  els.width.value = button.dataset.width ?? "";
  els.height.value = button.dataset.height ?? "";
  setFieldError("width", "");
  setFieldError("height", "");
  updatePresetSelection();
});
els.width.addEventListener("input", updatePresetSelection);
els.height.addEventListener("input", updatePresetSelection);
els.randomSeed.addEventListener("click", () => {
  els.seed.value = "";
  setFieldError("seed", "");
});
els.reuseSeed.addEventListener("click", () => {
  if (state.lastSeed === null) return;
  els.seed.value = String(state.lastSeed);
  els.advanced.open = true;
  els.seed.focus();
});
window.addEventListener("beforeunload", () => {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
});

applyConfig();
loadConfig();
refreshHealth();
