import { describe, expect, it } from "vitest";
import { CONFIG } from "../test/fixtures";
import type { PickedImage } from "./images";
import { initialFormState, validateForm, type FormState, type FormValues } from "./validation";

const IMAGE: PickedImage = { dataUrl: "data:image/png;base64,AAAA", name: "a.png", width: 64, height: 64, bytes: 3 };

function state(
  overrides: Partial<FormValues> = {},
  extra: Partial<Pick<FormState, "loras" | "initImage" | "faceImages" | "poseImages">> = {},
): FormState {
  const base = initialFormState(CONFIG);
  return {
    values: { ...base.values, prompt: "a cat", ...overrides },
    loras: extra.loras ?? [],
    initImage: extra.initImage ?? null,
    faceImages: extra.faceImages ?? [],
    poseImages: extra.poseImages ?? [],
  };
}

describe("validateForm", () => {
  it("builds the request payload with the default model's settings", () => {
    const result = validateForm(state({ prompt: "  a cat  ", seed: "42" }), CONFIG);
    expect(result).toEqual({
      ok: true,
      count: 1,
      payload: {
        prompt: "a cat",
        negative_prompt: "",
        model: "sd15",
        scheduler: "default",
        style: "none",
        width: 512,
        height: 512,
        num_inference_steps: 25,
        guidance_scale: 7.5,
        seed: 42,
        num_images: 1,
        output_format: "png",
        quality: 90,
        init_image: null,
        strength: 0.6,
        loras: [],
        face_images: [],
        pose_image: null,
        identity_strength: 0.8,
        pose_strength: 0.9,
      },
    });
  });

  it("includes face/pose references for SDXL models", () => {
    const result = validateForm(
      state({ model: "sdxl", identity_strength: "1.1" }, { faceImages: [IMAGE, IMAGE], poseImages: [IMAGE] }),
      CONFIG,
    );
    expect(result.ok && result.payload).toMatchObject({
      face_images: [IMAGE.dataUrl, IMAGE.dataUrl],
      pose_image: IMAGE.dataUrl,
      identity_strength: 1.1,
      pose_strength: 0.9,
    });
  });

  it("rejects references on unsupported models, with img2img, or with bad strengths", () => {
    const sd15 = validateForm(state({}, { faceImages: [IMAGE] }), CONFIG);
    expect(!sd15.ok && sd15.errors.face_images).toContain("SDXL");
    const withInit = validateForm(state({ model: "sdxl" }, { poseImages: [IMAGE], initImage: IMAGE }), CONFIG);
    expect(!withInit.ok && withInit.errors.pose_image).toContain("同時に使えません");
    const strong = validateForm(state({ model: "sdxl", pose_strength: "2" }, { poseImages: [IMAGE] }), CONFIG);
    expect(!strong.ok && strong.errors.pose_strength).toBeTruthy();
    // Several poses multiply the image count, which is capped in total.
    const many = validateForm(state({ model: "sdxl", num_images: "40" }, { poseImages: [IMAGE, IMAGE, IMAGE] }), CONFIG);
    expect(!many.ok && many.errors.num_images).toContain("合計100枚");
    const tooManyPoses = validateForm(state({ model: "sdxl" }, { poseImages: Array(11).fill(IMAGE) }), CONFIG);
    expect(!tooManyPoses.ok && tooManyPoses.errors.pose_image).toContain("10枚まで");
    const noConfig = validateForm(state({ model: "sdxl" }, { faceImages: [IMAGE] }), { ...CONFIG, identity: null });
    expect(noConfig.ok).toBe(false);
  });

  it("includes img2img and LoRA settings", () => {
    const result = validateForm(
      state({ model: "sdxl", strength: "0.4" }, { initImage: IMAGE, loras: [{ id: "pixel", scale: 0.7 }] }),
      CONFIG,
    );
    expect(result.ok && result.payload).toMatchObject({
      model: "sdxl",
      init_image: IMAGE.dataUrl,
      strength: 0.4,
      loras: [{ id: "pixel", scale: 0.7 }],
    });
  });

  it.each<[Partial<FormValues>, string]>([
    [{ prompt: "   " }, "prompt"],
    [{ prompt: "x".repeat(1001) }, "prompt"],
    [{ negative_prompt: "x".repeat(1001) }, "negative_prompt"],
    [{ model: "nope" }, "model"],
    [{ scheduler: "nope" }, "scheduler"],
    [{ width: "500" }, "width"],
    [{ width: "2048" }, "width"],
    [{ height: "" }, "height"],
    [{ height: "512.5" }, "height"],
    [{ num_inference_steps: "0" }, "num_inference_steps"],
    [{ num_inference_steps: "51" }, "num_inference_steps"],
    [{ guidance_scale: "-1" }, "guidance_scale"],
    [{ guidance_scale: "abc" }, "guidance_scale"],
    [{ seed: "-1" }, "seed"],
    [{ seed: "4294967296" }, "seed"],
    [{ num_images: "0" }, "num_images"],
    [{ num_images: "101" }, "num_images"],
    [{ num_images: "2.5" }, "num_images"],
    [{ output_format: "gif" }, "output_format"],
    [{ output_format: "jpeg", quality: "0" }, "quality"],
  ])("rejects %o", (overrides, field) => {
    const result = validateForm(state(overrides), CONFIG);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors[field]).toBeTruthy();
  });

  it("ignores quality for lossless PNG", () => {
    expect(validateForm(state({ output_format: "png", quality: "0" }), CONFIG).ok).toBe(true);
  });

  it("validates strength only for img2img", () => {
    expect(validateForm(state({ strength: "5" }), CONFIG).ok).toBe(true);
    const bad = validateForm(state({ strength: "5" }, { initImage: IMAGE }), CONFIG);
    expect(!bad.ok && bad.errors.strength).toBeTruthy();
    const tooFewSteps = validateForm(state({ strength: "0.5", num_inference_steps: "1" }, { initImage: IMAGE }), CONFIG);
    expect(!tooFewSteps.ok && tooFewSteps.errors.strength).toContain("1 以上");
  });

  it("rejects LoRAs for another model family and out-of-range scales", () => {
    const wrongFamily = validateForm(state({}, { loras: [{ id: "pixel", scale: 1 }] }), CONFIG);
    expect(!wrongFamily.ok && wrongFamily.errors.loras).toContain("Pixel Art XL");
    const badScale = validateForm(state({ model: "sdxl" }, { loras: [{ id: "pixel", scale: 3 }] }), CONFIG);
    expect(!badScale.ok && badScale.errors.loras).toBeTruthy();
  });
});
