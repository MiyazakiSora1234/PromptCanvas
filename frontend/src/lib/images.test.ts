import { describe, expect, it } from "vitest";
import { CONFIG } from "../test/fixtures";
import { base64ToBlob, buildFileName, sizeForAspect } from "./images";

describe("sizeForAspect", () => {
  const sd15 = { width: 512, height: 512 };
  const sdxl = { width: 1024, height: 1024 };

  it("keeps the model's pixel count and the image's aspect ratio", () => {
    expect(sizeForAspect(1920, 1080, sdxl, CONFIG.limits)).toEqual({ width: 1368, height: 768 });
    expect(sizeForAspect(600, 600, sd15, CONFIG.limits)).toEqual({ width: 512, height: 512 });
  });

  it("snaps to the size grid and clamps to the limits", () => {
    const { width, height } = sizeForAspect(4000, 500, sd15, CONFIG.limits);
    expect(width % 8).toBe(0);
    expect(height).toBe(CONFIG.limits.min_image_size);
  });
});

describe("base64ToBlob", () => {
  it("decodes bytes with the given type", async () => {
    const blob = base64ToBlob(btoa("abc"), "image/webp");
    expect(blob.type).toBe("image/webp");
    expect(await blob.text()).toBe("abc");
  });
});

describe("buildFileName", () => {
  it("uses timestamp, seed and extension", () => {
    expect(buildFileName(42, "jpg", new Date("2026-10-03T12:34:56.789Z"))).toBe(
      "promptcanvas_20261003T123456_seed42.jpg",
    );
  });
});
