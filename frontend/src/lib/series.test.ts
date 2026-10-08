import { describe, expect, it } from "vitest";
import type { GenerateRequest } from "../api/types";
import { CONFIG } from "../test/fixtures";
import type { PickedImage } from "./images";
import { buildSeries } from "./series";

const BASE: GenerateRequest = {
  prompt: "a cat",
  negative_prompt: "",
  model: "sdxl",
  scheduler: "default",
  style: "none",
  width: 832,
  height: 1216,
  num_inference_steps: 30,
  guidance_scale: 7,
  seed: 10,
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
};

const pose = (name: string, width: number, height: number): PickedImage => ({
  dataUrl: `data:image/png;base64,${name}`,
  name,
  width,
  height,
  bytes: 1,
});

describe("buildSeries", () => {
  it("sends one request per image with continuing seeds", () => {
    const items = buildSeries(BASE, 3, [], CONFIG.limits);
    expect(items.map((i) => [i.request.seed, i.request.num_images, i.poseNumber])).toEqual([
      [10, 1, null],
      [11, 1, null],
      [12, 1, null],
    ]);
    expect(buildSeries({ ...BASE, seed: null }, 2, [], CONFIG.limits).map((i) => i.request.seed)).toEqual([null, null]);
  });

  it("makes `count` images for each pose, each in its own aspect ratio", () => {
    const portrait = pose("P", 900, 1300);
    const landscape = pose("L", 1600, 900);
    const items = buildSeries(BASE, 2, [portrait, landscape], CONFIG.limits);

    expect(items.map((i) => [i.poseNumber, i.request.pose_image, i.request.seed])).toEqual([
      [1, portrait.dataUrl, 10],
      [1, portrait.dataUrl, 11],
      [2, landscape.dataUrl, 12],
      [2, landscape.dataUrl, 13],
    ]);
    // The first pose keeps the form's size; the landscape one gets a landscape size of similar area.
    expect(items[0]!.request).toMatchObject({ width: 832, height: 1216 });
    const { width, height } = items[2]!.request;
    expect(width).toBeGreaterThan(height);
    expect(Math.abs(width * height - 832 * 1216) / (832 * 1216)).toBeLessThan(0.05);
  });

  it("wraps the seed at 2^32", () => {
    const items = buildSeries({ ...BASE, seed: 2 ** 32 - 1 }, 2, [], CONFIG.limits);
    expect(items.map((i) => i.request.seed)).toEqual([2 ** 32 - 1, 0]);
  });
});
