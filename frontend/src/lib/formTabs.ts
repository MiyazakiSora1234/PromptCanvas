// Which settings tab each form field lives on (the prompt is always visible, outside the tabs).
import { FIELD_NAMES, type FieldErrors, type FieldName } from "./validation";

export type TabId = "basic" | "images" | "lora" | "advanced" | "presets";

export const TAB_ORDER: readonly TabId[] = ["basic", "images", "lora", "advanced", "presets"];

export const TAB_LABELS: Record<TabId, string> = {
  basic: "基本",
  images: "画像参照",
  lora: "LoRA",
  advanced: "詳細",
  presets: "プリセット",
};

const FIELD_TAB: Partial<Record<FieldName, TabId>> = {
  model: "basic",
  style: "basic",
  num_images: "basic",
  output_format: "basic",
  quality: "basic",
  init_image: "images",
  strength: "images",
  face_image: "images",
  pose_image: "images",
  identity_strength: "images",
  pose_strength: "images",
  loras: "lora",
  negative_prompt: "advanced",
  scheduler: "advanced",
  width: "advanced",
  height: "advanced",
  num_inference_steps: "advanced",
  guidance_scale: "advanced",
  seed: "advanced",
};

export function tabOfField(field: string): TabId | null {
  return FIELD_TAB[field as FieldName] ?? null;
}

/** Tabs containing at least one error (for the red marker on the tab). */
export function tabsWithErrors(errors: FieldErrors): Set<TabId> {
  const tabs = new Set<TabId>();
  for (const field of Object.keys(errors)) {
    const tab = tabOfField(field);
    if (tab) tabs.add(tab);
  }
  return tabs;
}

/** The tab of the first invalid field in on-screen order, or null if it's outside the tabs (prompt). */
export function firstErrorTab(errors: FieldErrors): TabId | null {
  const first = FIELD_NAMES.find((name) => errors[name]);
  return first ? tabOfField(first) : null;
}
