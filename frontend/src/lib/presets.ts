// Settings presets: built-in ones come from the server (/api/config), the user's own are
// kept in this browser's localStorage. The prompt is optional; seed and images are never part of a preset.
import type { AppConfig, OutputFormat, PresetSettings } from "../api/types";
import { findModel, modelDefaultValues, supportsReference, type FormState, type FormValues } from "./validation";

export interface Preset {
  id: string;
  label: string;
  description: string;
  builtIn: boolean;
  settings: PresetSettings;
}

const STORAGE_KEY = "promptcanvas.presets.v1";
export const MAX_PRESET_NAME = 40;

function finite(raw: string): number | null {
  const n = Number(raw.trim());
  return raw.trim() !== "" && Number.isFinite(n) ? n : null;
}

/** Snapshot the current form as preset settings (unparseable numbers are left out = model default). */
export function captureSettings(state: FormState, { includePrompt = true } = {}): PresetSettings {
  const v = state.values;
  const prompt = v.prompt.trim();
  return {
    model: v.model,
    ...(includePrompt && prompt ? { prompt } : {}),
    style: v.style,
    scheduler: v.scheduler,
    width: finite(v.width),
    height: finite(v.height),
    num_inference_steps: finite(v.num_inference_steps),
    guidance_scale: finite(v.guidance_scale),
    negative_prompt: v.negative_prompt,
    num_images: finite(v.num_images),
    output_format: v.output_format as OutputFormat,
    quality: finite(v.quality),
    loras: state.loras.map(({ id, scale }) => ({ id, scale })),
    identity_strength: finite(v.identity_strength),
    pose_strength: finite(v.pose_strength),
  };
}

type ApplyResult = { ok: true; state: FormState; warnings: string[] } | { ok: false; error: string };

/**
 * Apply a preset on top of the current form: start from the preset model's defaults, then
 * override with the preset's values. Anything this server doesn't offer (a removed model,
 * LoRA, sampler...) is skipped with a warning instead of failing.
 */
export function applyPreset(state: FormState, settings: PresetSettings, config: AppConfig): ApplyResult {
  const model = config.models.find((m) => m.id === settings.model);
  if (!model) return { ok: false, error: `このプリセットのモデル「${settings.model}」はこのサーバーにありません。` };

  const warnings: string[] = [];
  const values: FormValues = { ...state.values, ...modelDefaultValues(model) };
  const set = (name: keyof FormValues, value: string | number | null | undefined) => {
    if (value !== null && value !== undefined) values[name] = String(value);
  };

  if (settings.style && !config.styles.some((s) => s.id === settings.style)) {
    warnings.push(`スタイル「${settings.style}」は使えないため、モデルの既定にしました。`);
  } else set("style", settings.style);
  if (settings.scheduler && !config.schedulers.some((s) => s.id === settings.scheduler)) {
    warnings.push(`サンプラー「${settings.scheduler}」は使えないため、モデルの既定にしました。`);
  } else set("scheduler", settings.scheduler);
  if (settings.output_format && !config.output_formats.some((f) => f.id === settings.output_format)) {
    warnings.push(`画像形式「${settings.output_format}」は使えないため、そのままにしました。`);
  } else set("output_format", settings.output_format);
  if (settings.prompt) set("prompt", settings.prompt); // a preset without a prompt keeps the current one
  set("width", settings.width);
  set("height", settings.height);
  set("num_inference_steps", settings.num_inference_steps);
  set("guidance_scale", settings.guidance_scale);
  set("negative_prompt", settings.negative_prompt);
  set("num_images", settings.num_images);
  set("quality", settings.quality);
  set("identity_strength", settings.identity_strength);
  set("pose_strength", settings.pose_strength);

  const loras = (settings.loras ?? []).filter((item) => {
    const lora = config.loras.find((l) => l.id === item.id);
    if (lora && lora.family === model.family) return true;
    warnings.push(`LoRA「${lora?.label ?? item.id}」は使えないため外しました。`);
    return false;
  });

  // Face/pose references only work with some model families.
  const keepRefs = supportsReference(config, model);
  if (!keepRefs && (state.faceImages.length > 0 || state.poseImage)) {
    warnings.push("このモデルでは顔・ポーズの参照が使えないため、参照画像を外しました。");
  }

  return {
    ok: true,
    warnings,
    state: {
      values,
      loras,
      initImage: state.initImage,
      faceImages: keepRefs ? state.faceImages : [],
      poseImage: keepRefs ? state.poseImage : null,
    },
  };
}

export function builtInPresets(config: AppConfig): Preset[] {
  return config.presets.map((p) => ({ ...p, builtIn: true }));
}

function isPreset(value: unknown): value is Preset {
  if (typeof value !== "object" || value === null) return false;
  const p = value as Partial<Preset>;
  return (
    typeof p.id === "string" &&
    typeof p.label === "string" &&
    typeof p.settings === "object" &&
    p.settings !== null &&
    typeof p.settings.model === "string"
  );
}

/** The user's presets from localStorage; corrupted or unavailable storage yields an empty list. */
export function loadUserPresets(storage: Storage | undefined = globalThis.localStorage): Preset[] {
  try {
    const parsed: unknown = JSON.parse(storage?.getItem(STORAGE_KEY) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter(isPreset).map((p) => ({ ...p, description: p.description ?? "", builtIn: false }))
      : [];
  } catch {
    return [];
  }
}

/** Returns false if the browser refused to store (private mode, quota). */
export function saveUserPresets(presets: Preset[], storage: Storage | undefined = globalThis.localStorage): boolean {
  try {
    const stored = presets.map(({ id, label, description, settings }) => ({ id, label, description, settings }));
    storage?.setItem(STORAGE_KEY, JSON.stringify(stored));
    return storage !== undefined;
  } catch {
    return false;
  }
}

export function describeSettings(settings: PresetSettings, config: AppConfig): string {
  const model = findModel(config, settings.model);
  const parts = [config.models.some((m) => m.id === settings.model) ? model.label : settings.model];
  if (settings.width && settings.height) parts.push(`${settings.width}×${settings.height}`);
  if (settings.num_inference_steps) parts.push(`${settings.num_inference_steps}ステップ`);
  if (settings.guidance_scale !== null && settings.guidance_scale !== undefined)
    parts.push(`ガイダンス ${settings.guidance_scale}`);
  if (settings.loras?.length) parts.push(`LoRA ${settings.loras.length}個`);
  const summary = parts.join("・");
  if (!settings.prompt) return summary;
  const prompt = settings.prompt.length > 40 ? `${settings.prompt.slice(0, 40)}…` : settings.prompt;
  return `${summary}／プロンプト: ${prompt}`;
}
