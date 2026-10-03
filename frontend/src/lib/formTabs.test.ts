import { describe, expect, it } from "vitest";
import { firstErrorTab, tabOfField, tabsWithErrors } from "./formTabs";
import { FIELD_NAMES } from "./validation";

describe("form tabs", () => {
  it("assigns every field except the prompt to a tab", () => {
    const unassigned = FIELD_NAMES.filter((f) => tabOfField(f) === null);
    expect(unassigned).toEqual(["prompt"]);
  });

  it("finds the tab of the first error in on-screen order", () => {
    expect(firstErrorTab({ seed: "x", loras: "y" })).toBe("lora");
    expect(firstErrorTab({ width: "x" })).toBe("advanced");
    expect(firstErrorTab({ prompt: "x", width: "y" })).toBeNull(); // prompt is outside the tabs
    expect(firstErrorTab({})).toBeNull();
  });

  it("collects every tab with an error, ignoring unknown fields", () => {
    expect(tabsWithErrors({ face_images: "x", quality: "y", body: "z" })).toEqual(new Set(["images", "basic"]));
  });
});
