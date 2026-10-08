import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// jsdom has no object URLs.
let objectUrls = 0;
URL.createObjectURL = vi.fn(() => `blob:mock-image-${++objectUrls}`);
URL.revokeObjectURL = vi.fn();

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
});
