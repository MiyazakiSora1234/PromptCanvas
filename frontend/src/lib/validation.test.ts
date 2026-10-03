import { describe, expect, it } from "vitest";
import { CONFIG } from "../test/fixtures";
import { initialFormValues, validateForm, type FormValues } from "./validation";

const valid = (overrides: Partial<FormValues> = {}): FormValues => ({
  ...initialFormValues(CONFIG.defaults),
  prompt: "a cat",
  ...overrides,
});

describe("validateForm", () => {
  it("builds the request payload from valid input", () => {
    const result = validateForm(valid({ prompt: "  a cat  ", seed: "42" }), CONFIG.limits);
    expect(result).toEqual({
      ok: true,
      payload: {
        prompt: "a cat",
        negative_prompt: "",
        width: 512,
        height: 512,
        num_inference_steps: 25,
        guidance_scale: 7.5,
        seed: 42,
      },
    });
  });

  it("treats an empty seed as random", () => {
    const result = validateForm(valid({ seed: " " }), CONFIG.limits);
    expect(result.ok && result.payload.seed).toBeNull();
  });

  it.each<[Partial<FormValues>, string]>([
    [{ prompt: "   " }, "prompt"],
    [{ prompt: "x".repeat(1001) }, "prompt"],
    [{ negative_prompt: "x".repeat(1001) }, "negative_prompt"],
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
  ])("rejects %o", (overrides, field) => {
    const result = validateForm(valid(overrides), CONFIG.limits);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors[field]).toBeTruthy();
  });
});
