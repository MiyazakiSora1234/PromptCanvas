import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// jsdom has no object URLs.
URL.createObjectURL = vi.fn(() => "blob:mock-image");
URL.revokeObjectURL = vi.fn();

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
