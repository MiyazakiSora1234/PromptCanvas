import { describe, expect, it } from "vitest";
import { ApiError, NetworkError } from "../api/client";
import { presentError } from "./presentError";

const apiError = (status: number, code: string, fields: Array<{ field: string; message: string }> = []) =>
  new ApiError(status, { code, message: "msg", fields, request_id: "req123" });

describe("presentError", () => {
  it("tells the user to check the server when it is unreachable", () => {
    const p = presentError(new NetworkError());
    expect(p.message).toContain("サーバーに接続できませんでした");
    expect(p.recheckHealth).toBe(true);
  });

  it("opens advanced settings for out-of-memory without an inquiry ID", () => {
    const p = presentError(apiError(503, "gpu_out_of_memory"));
    expect(p.openAdvanced).toBe(true);
    expect(p.message).toBe("msg");
  });

  it("appends the request ID to server errors", () => {
    expect(presentError(apiError(500, "generation_failed")).message).toBe("msg（問い合わせID: req123）");
  });

  it("maps field errors and opens advanced settings for non-prompt fields", () => {
    const p = presentError(apiError(422, "invalid_input", [{ field: "width", message: "8の倍数" }]));
    expect(p.fieldErrors).toEqual({ width: "8の倍数" });
    expect(p.openAdvanced).toBe(true);
    expect(p.message).toBe("msg");
  });

  it("re-checks health when the model is not ready", () => {
    expect(presentError(apiError(503, "model_loading")).recheckHealth).toBe(true);
  });
});
