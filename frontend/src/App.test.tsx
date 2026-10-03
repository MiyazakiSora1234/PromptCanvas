import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { GenerateRequest, Health } from "./api/types";
import App from "./App";
import { CONFIG, errorResponse, health, pngResponse } from "./test/fixtures";

type GenerateHandler = (body: GenerateRequest) => Response | Promise<Response>;

function mockServer({ status = health(), generate }: { status?: Health; generate?: GenerateHandler } = {}) {
  const generateCalls: GenerateRequest[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/api/config")) return Response.json(CONFIG);
    if (url.endsWith("/api/health")) return Response.json(status);
    if (url.endsWith("/api/generate")) {
      const body = JSON.parse(String(init?.body)) as GenerateRequest;
      generateCalls.push(body);
      return generate ? generate(body) : pngResponse(body.seed ?? 1234);
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { generateCalls };
}

async function renderReady() {
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("準備完了：GPU (CUDA)");
  return user;
}

describe("App", () => {
  it("generates an image and offers it as a PNG download", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.type(screen.getByLabelText(/^プロンプト/), "  a lighthouse  ");
    await user.type(screen.getByLabelText("シード値"), "42");
    await user.click(screen.getByRole("button", { name: "画像を生成" }));

    const image = await screen.findByRole("img", { name: "生成画像: a lighthouse" });
    expect(image).toHaveAttribute("src", "blob:mock-image");
    const download = screen.getByRole("link", { name: "PNGをダウンロード" });
    expect(download).toHaveAttribute("href", "blob:mock-image");
    expect(download.getAttribute("download")).toMatch(/^promptcanvas_\d{8}T\d{6}_seed42\.png$/);
    expect(screen.getByText("2.5 秒")).toBeInTheDocument();
    expect(generateCalls).toEqual([
      {
        prompt: "a lighthouse",
        negative_prompt: "",
        width: 512,
        height: 512,
        num_inference_steps: 25,
        guidance_scale: 7.5,
        seed: 42,
      },
    ]);
  });

  it("validates input before sending", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.type(screen.getByLabelText(/^プロンプト/), "a cat");
    const width = screen.getByLabelText("幅");
    await user.clear(width);
    await user.type(width, "500");
    await user.click(screen.getByRole("button", { name: "画像を生成" }));

    expect(await screen.findByText("8の倍数で指定してください。")).toBeInTheDocument();
    expect(width).toHaveAttribute("aria-invalid", "true");
    expect(width).toHaveFocus();
    expect(screen.getByRole("alert")).toHaveTextContent("入力内容を確認してください。");
    expect(generateCalls).toHaveLength(0);
  });

  it("requires a prompt", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.click(screen.getByRole("button", { name: "画像を生成" }));

    expect(await screen.findByText("プロンプトを入力してください。")).toBeInTheDocument();
    expect(generateCalls).toHaveLength(0);
  });

  it("shows the server's guidance when generation fails", async () => {
    mockServer({
      generate: () => errorResponse(503, "gpu_out_of_memory", "GPUメモリが不足しました。サイズを小さくしてください。"),
    });
    const user = await renderReady();

    await user.type(screen.getByLabelText(/^プロンプト/), "a cat");
    await user.click(screen.getByRole("button", { name: "画像を生成" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("GPUメモリが不足しました。サイズを小さくしてください。");
    expect(screen.getByRole("button", { name: "画像を生成" })).toBeEnabled();
  });

  it("sends only one request while generating, even on repeated submits", async () => {
    let finish: (res: Response) => void = () => {};
    const { generateCalls } = mockServer({
      generate: () => new Promise<Response>((resolve) => (finish = resolve)),
    });
    const user = await renderReady();

    await user.type(screen.getByLabelText(/^プロンプト/), "a cat");
    await user.click(screen.getByRole("button", { name: "画像を生成" }));
    const busyButton = await screen.findByRole("button", { name: "生成中…" });
    expect(busyButton).toBeDisabled();
    expect(screen.getByText(/生成中…\s*\d+\.\d\s*秒/)).toBeInTheDocument();

    await user.click(busyButton);
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(generateCalls).toHaveLength(1);

    finish(pngResponse(7));
    await screen.findByRole("img");
    await waitFor(() => expect(screen.getByRole("button", { name: "画像を生成" })).toBeEnabled());
  });

  it("disables generation while the model is loading", async () => {
    mockServer({ status: health("loading") });
    render(<App />);

    expect(await screen.findByRole("button", { name: "モデル準備中…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("モデル読み込み中");
  });

  it("explains a failed model load", async () => {
    mockServer({ status: health("failed") });
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("モデルが見つかりません。");
    expect(screen.getByRole("button", { name: "画像を生成" })).toBeDisabled();
  });
});
