import { describe, expect, it } from "vitest";
import { CONFIG } from "../test/fixtures";
import type { PickedImage } from "./images";
import { applyPreset, captureSettings, loadUserPresets, saveUserPresets, type Preset } from "./presets";
import { initialFormState } from "./validation";

const IMAGE: PickedImage = { dataUrl: "data:,", name: "a.png", width: 10, height: 10, bytes: 1 };

describe("captureSettings / applyPreset", () => {
  it("round-trips the form settings but not prompt, seed or images", () => {
    const base = initialFormState(CONFIG);
    const form = {
      ...base,
      values: {
        ...base.values,
        model: "sdxl",
        style: "photo",
        width: "768",
        guidance_scale: "4",
        negative_prompt: "blurry",
        prompt: "a cat",
        seed: "42",
      },
      loras: [{ id: "pixel", scale: 0.5 }],
    };
    const settings = captureSettings(form);
    expect(settings).toMatchObject({ model: "sdxl", style: "photo", width: 768, guidance_scale: 4, negative_prompt: "blurry" });
    expect(settings).not.toHaveProperty("prompt");
    expect(settings).not.toHaveProperty("seed");

    const fresh = { ...initialFormState(CONFIG), values: { ...initialFormState(CONFIG).values, prompt: "keep me" } };
    const applied = applyPreset(fresh, settings, CONFIG);
    expect(applied.ok).toBe(true);
    if (!applied.ok) return;
    expect(applied.state.values).toMatchObject({ model: "sdxl", style: "photo", width: "768", guidance_scale: "4" });
    expect(applied.state.values.prompt).toBe("keep me");
    expect(applied.state.loras).toEqual([{ id: "pixel", scale: 0.5 }]);
    expect(applied.warnings).toEqual([]);
  });

  it("fills unspecified fields from the preset model's defaults", () => {
    const applied = applyPreset(initialFormState(CONFIG), { model: "sdxl" }, CONFIG);
    expect(applied.ok && applied.state.values).toMatchObject({ width: "1024", scheduler: "euler_a", style: "photo" });
  });

  it("skips things this server doesn't offer, with warnings", () => {
    const applied = applyPreset(
      { ...initialFormState(CONFIG), faceImage: IMAGE },
      { model: "sd15", scheduler: "gone", loras: [{ id: "pixel", scale: 1 }] },
      CONFIG,
    );
    expect(applied.ok).toBe(true);
    if (!applied.ok) return;
    expect(applied.state.values.scheduler).toBe("default");
    expect(applied.state.loras).toEqual([]);
    expect(applied.state.faceImage).toBeNull();
    expect(applied.warnings).toHaveLength(3);
  });

  it("refuses a preset whose model is gone", () => {
    const applied = applyPreset(initialFormState(CONFIG), { model: "deleted" }, CONFIG);
    expect(applied).toEqual({ ok: false, error: expect.stringContaining("deleted") });
  });
});

describe("user preset storage", () => {
  const preset: Preset = { id: "user-1", label: "Mine", description: "", builtIn: false, settings: { model: "sd15" } };

  it("saves and loads", () => {
    expect(saveUserPresets([preset])).toBe(true);
    expect(loadUserPresets()).toEqual([preset]);
  });

  it("ignores corrupted data", () => {
    localStorage.setItem("promptcanvas.presets.v1", "{not json");
    expect(loadUserPresets()).toEqual([]);
    localStorage.setItem("promptcanvas.presets.v1", JSON.stringify([{ id: 1 }, preset]));
    expect(loadUserPresets()).toEqual([preset]);
  });

  it("reports failure when storage refuses writes", () => {
    const broken = { setItem: () => { throw new Error("quota"); } } as unknown as Storage;
    expect(saveUserPresets([preset], broken)).toBe(false);
  });
});
