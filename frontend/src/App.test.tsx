import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { GenerateRequest, Health } from "./api/types";
import App from "./App";
import { CONFIG, errorResponse, generateResponse, health } from "./test/fixtures";

// jsdom cannot decode images, so stub the file reader used for img2img.
vi.mock("./lib/images", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./lib/images")>()),
  readImageFile: vi.fn(async (file: File) => ({
    dataUrl: "data:image/png;base64,AAAA",
    name: file.name,
    width: 1920,
    height: 1080,
    bytes: file.size,
  })),
}));

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
      return generate ? generate(body) : generateResponse(body);
    }
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { generateCalls };
}

async function renderReady() {
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("準備完了：SD 1.5・GPU (CUDA)");
  return user;
}

const promptBox = () => screen.getByLabelText(/^プロンプト/);
const generateButton = () => screen.getByRole("button", { name: /生成/ });
const tab = (name: string) => screen.getByRole("tab", { name: new RegExp(`^${name}`) });
const openTab = (user: ReturnType<typeof userEvent.setup>, name: string) => user.click(tab(name));

describe("App", () => {
  it("generates an image and offers it as a download", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.type(promptBox(), "  a lighthouse  ");
    await openTab(user, "詳細");
    await user.type(screen.getByLabelText("シード値"), "42");
    await user.click(screen.getByRole("button", { name: "画像を生成" }));

    const image = await screen.findByRole("img", { name: "生成画像: a lighthouse" });
    expect(image).toHaveAttribute("src", "blob:mock-image");
    const download = screen.getByRole("link", { name: "PNGをダウンロード" });
    expect(download.getAttribute("download")).toMatch(/^promptcanvas_\d{8}T\d{6}_seed42\.png$/);
    expect(screen.getByText("2.5 秒")).toBeInTheDocument();
    expect(generateCalls).toEqual([
      {
        prompt: "a lighthouse",
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
    ]);
  });

  it("validates input before sending", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.type(promptBox(), "a cat");
    await openTab(user, "詳細");
    const width = screen.getByLabelText("幅");
    await user.clear(width);
    await user.type(width, "500");
    await openTab(user, "基本");
    await user.click(generateButton());

    // The form switches to the tab holding the error, marks it, and focuses the field.
    expect(await screen.findByText("8の倍数で指定してください。")).toBeVisible();
    expect(tab("詳細")).toHaveAttribute("aria-selected", "true");
    expect(within(tab("詳細")).getByRole("img", { name: "入力エラーあり" })).toBeInTheDocument();
    expect(width).toHaveAttribute("aria-invalid", "true");
    expect(width).toHaveFocus();
    expect(screen.getByRole("alert")).toHaveTextContent("入力内容を確認してください。");
    expect(generateCalls).toHaveLength(0);
  });

  it("requires a prompt", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.click(generateButton());

    expect(await screen.findByText("プロンプトを入力してください。")).toBeInTheDocument();
    expect(generateCalls).toHaveLength(0);
  });

  it("shows the server's guidance when generation fails", async () => {
    mockServer({
      generate: () => errorResponse(503, "gpu_out_of_memory", "GPUメモリが不足しました。サイズを小さくしてください。"),
    });
    const user = await renderReady();

    await user.type(promptBox(), "a cat");
    await user.click(generateButton());

    expect(await screen.findByRole("alert")).toHaveTextContent("GPUメモリが不足しました。サイズを小さくしてください。");
    expect(generateButton()).toBeEnabled();
  });

  it("sends only one request while generating, even on repeated submits", async () => {
    let finish: (res: Response) => void = () => {};
    const { generateCalls } = mockServer({
      generate: (body) => new Promise<Response>((resolve) => (finish = () => resolve(generateResponse(body)))),
    });
    const user = await renderReady();

    await user.type(promptBox(), "a cat");
    await user.click(generateButton());
    const busyButton = await screen.findByRole("button", { name: "生成中…" });
    expect(busyButton).toBeDisabled();
    expect(screen.getByText(/生成中…\s*\d+\.\d\s*秒/)).toBeInTheDocument();

    await user.click(busyButton);
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(generateCalls).toHaveLength(1);

    finish(new Response());
    await screen.findByRole("img", { name: /生成画像/ });
    await waitFor(() => expect(screen.getByRole("button", { name: "画像を生成" })).toBeEnabled());
  });

  it("switches models: applies their defaults and offers only compatible LoRAs", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await openTab(user, "LoRA");
    expect(screen.getByText(/で使える LoRA はありません/)).toBeInTheDocument();
    await openTab(user, "基本");
    await user.selectOptions(screen.getByLabelText("モデル"), "sdxl");

    expect(screen.getByLabelText("幅")).toHaveValue(1024);
    expect(screen.getByLabelText("サンプラー")).toHaveValue("euler_a");
    expect(screen.getByLabelText("スタイル")).toHaveValue("photo");
    expect(screen.getByText(/肌の質感を出す語句を自動で足します/)).toBeInTheDocument();
    await openTab(user, "LoRA");
    await user.click(screen.getByRole("checkbox", { name: /Pixel Art XL/ }));
    expect(tab("LoRA")).toHaveTextContent("(1)");
    expect(screen.getByText(/pixel art/)).toBeInTheDocument();

    await user.type(promptBox(), "pixel art, a cat");
    await user.click(screen.getByRole("button", { name: "モデルを切り替えて生成" }));
    await screen.findByRole("img", { name: /生成画像/ });

    expect(generateCalls[0]).toMatchObject({
      model: "sdxl",
      scheduler: "euler_a",
      style: "photo",
      width: 1024,
      num_inference_steps: 30,
      loras: [{ id: "pixel", scale: 1 }],
    });

    // Switching back drops the SDXL-only LoRA.
    await openTab(user, "基本");
    await user.selectOptions(screen.getByLabelText("モデル"), "sd15");
    await openTab(user, "LoRA");
    expect(screen.queryByRole("checkbox", { name: /Pixel Art XL/ })).not.toBeInTheDocument();
    expect(tab("LoRA")).not.toHaveTextContent("(1)");
  });

  it("warns before picking a model that still has to be downloaded", async () => {
    mockServer();
    const user = await renderReady();

    expect(screen.queryByRole("note")).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "SDXL（要ダウンロード）" })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("モデル"), "sdxl");
    expect(screen.getByRole("note")).toHaveTextContent("約6.9GB のダウンロード");
  });

  it("generates a batch and lets the user pick which image to download", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.type(promptBox(), "a cat");
    await openTab(user, "詳細");
    await user.type(screen.getByLabelText("シード値"), "10");
    await openTab(user, "基本");
    await user.click(within(screen.getByRole("radiogroup", { name: "枚数" })).getByLabelText("4"));
    await user.click(generateButton());

    const thumbnails = await screen.findByRole("listbox", { name: "生成した画像" });
    expect(within(thumbnails).getAllByRole("option")).toHaveLength(4);
    expect(generateCalls[0]?.num_images).toBe(4);

    await user.click(screen.getByRole("option", { name: "3枚目（シード 12）" }));
    expect(screen.getByRole("option", { name: "3枚目（シード 12）" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("link", { name: "PNGをダウンロード" }).getAttribute("download")).toMatch(/_seed12\.png$/);
  });

  it("outputs JPEG with a quality setting", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await user.selectOptions(screen.getByLabelText("画像形式"), "jpeg");
    expect(screen.getByLabelText(/画質/)).toHaveValue("90");
    await user.type(promptBox(), "a cat");
    await user.click(generateButton());

    const download = await screen.findByRole("link", { name: "JPEGをダウンロード" });
    expect(download.getAttribute("download")).toMatch(/\.jpg$/);
    expect(generateCalls[0]).toMatchObject({ output_format: "jpeg", quality: 90 });
  });

  it("sends an uploaded image for img2img, sized to its aspect ratio", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    const file = new File(["x"], "photo.png", { type: "image/png" });
    await openTab(user, "画像参照");
    await user.upload(screen.getByLabelText("元画像ファイル"), file);
    expect(tab("画像参照")).toHaveTextContent("●");

    expect(await screen.findByAltText("元画像のプレビュー")).toBeInTheDocument();
    expect(screen.getByText("photo.png")).toBeInTheDocument();
    expect(screen.getByLabelText("幅")).toHaveValue(680);
    expect(screen.getByLabelText("高さ")).toHaveValue(384);

    await user.type(promptBox(), "oil painting");
    await user.click(generateButton());
    await screen.findByRole("img", { name: /生成画像/ });
    expect(generateCalls[0]).toMatchObject({
      init_image: "data:image/png;base64,AAAA",
      strength: 0.6,
      width: 680,
      height: 384,
    });

    await user.click(screen.getByRole("button", { name: "元画像を外す" }));
    expect(screen.queryByAltText("元画像のプレビュー")).not.toBeInTheDocument();
    expect(screen.getByLabelText("幅")).toHaveValue(512);
  });

  it("keeps a face and copies a pose with SDXL models", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    // SD 1.5 can't use references.
    await openTab(user, "画像参照");
    expect(screen.getByText(/「SD 1.5」では使えません/)).toBeInTheDocument();
    expect(screen.queryByLabelText("顔の写真ファイル")).not.toBeInTheDocument();

    await openTab(user, "基本");
    await user.selectOptions(screen.getByLabelText("モデル"), "sdxl");
    await openTab(user, "画像参照");
    const face = (name: string) => new File(["f"], name, { type: "image/png" });
    // Several photos of the same person, picked in two goes; the 4th exceeds the limit of 3.
    await user.upload(screen.getByLabelText("顔の写真ファイル"), [face("a.png"), face("b.png")]);
    await user.upload(screen.getByLabelText("顔の写真ファイル"), [face("c.png"), face("d.png")]);
    await user.upload(screen.getByLabelText("ポーズ参考画像ファイル"), new File(["p"], "pose.png", { type: "image/png" }));

    expect(await screen.findByAltText("顔の写真 3枚目")).toBeInTheDocument();
    expect(screen.queryByAltText("顔の写真 4枚目")).not.toBeInTheDocument();
    expect(screen.getByText(/顔の写真は3枚までです/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上限の3枚に達しました" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "顔の写真 2枚目を外す" }));
    expect(screen.queryByAltText("顔の写真 3枚目")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "顔の写真を追加（あと1枚）" })).toBeEnabled();
    expect(screen.getByAltText("ポーズ参考画像のプレビュー")).toBeInTheDocument();
    expect(screen.getByLabelText(/顔の再現度/)).toHaveValue("0.8");
    // img2img can't be combined with references.
    expect(screen.getByText(/顔・ポーズの参照と同時には使えません/)).toBeInTheDocument();
    // Output follows the pose image's aspect ratio at SDXL's pixel count.
    expect(screen.getByLabelText("幅")).toHaveValue(1368);
    expect(screen.getByLabelText("高さ")).toHaveValue(768);

    await user.type(promptBox(), "a woman in a red dress");
    await user.click(generateButton());
    await screen.findByRole("img", { name: /生成画像/ });

    expect(generateCalls[0]).toMatchObject({
      model: "sdxl",
      face_images: ["data:image/png;base64,AAAA", "data:image/png;base64,AAAA"],
      pose_image: "data:image/png;base64,AAAA",
      identity_strength: 0.8,
      pose_strength: 0.9,
      init_image: null,
    });
    expect(screen.getByText("2枚・再現度 0.8")).toBeInTheDocument();
    expect(screen.getByLabelText("現在の設定")).toHaveTextContent("顔の参照 2枚・ポーズ参照");

    // Switching to a model without reference support drops the images.
    await openTab(user, "基本");
    await user.selectOptions(screen.getByLabelText("モデル"), "sd15");
    await user.selectOptions(screen.getByLabelText("モデル"), "sdxl");
    await openTab(user, "画像参照");
    expect(screen.queryByAltText("顔の写真 1枚目")).not.toBeInTheDocument();
    expect(tab("画像参照")).not.toHaveTextContent("●");
  });

  it("applies the built-in realistic-human preset", async () => {
    const { generateCalls } = mockServer();
    const user = await renderReady();

    await openTab(user, "プリセット");
    await user.selectOptions(screen.getByLabelText("使うプリセット"), "realistic-human");
    expect(screen.getByText("人物写真向け")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "適用" }));
    expect(await screen.findByText("「リアルな人間」を適用しました。")).toBeInTheDocument();
    // The summary shows the result without opening the other tabs.
    expect(screen.getByLabelText("現在の設定")).toHaveTextContent("SDXL・リアルな写真・896×1152・30ステップ");

    expect(screen.getByLabelText("モデル")).toHaveValue("sdxl");
    expect(screen.getByLabelText("スタイル")).toHaveValue("photo");
    expect(screen.getByLabelText("幅")).toHaveValue(896);
    expect(screen.getByLabelText("高さ")).toHaveValue(1152);
    // The built-in preset can't be deleted.
    expect(screen.getByRole("button", { name: "「リアルな人間」を削除" })).toBeDisabled();

    await user.type(promptBox(), "a man");
    await user.click(generateButton());
    await screen.findByRole("img", { name: /生成画像/ });
    expect(generateCalls[0]).toMatchObject({
      model: "sdxl",
      style: "photo",
      scheduler: "dpmpp_2m_karras",
      width: 896,
      height: 1152,
      guidance_scale: 3.5,
      negative_prompt: "bad hands",
      prompt: "a man",
    });
  });

  it("saves the current settings, keeps them after a reload, and deletes them", async () => {
    mockServer();
    const user = await renderReady();

    await user.selectOptions(screen.getByLabelText("画像形式"), "webp");
    await openTab(user, "詳細");
    const steps = screen.getByLabelText("ステップ数");
    await user.clear(steps);
    await user.type(steps, "40");
    await openTab(user, "プリセット");
    await user.type(screen.getByLabelText("保存する名前"), "速い WebP{Enter}");
    expect(await screen.findByText("「速い WebP」を保存しました。")).toBeInTheDocument();

    // Reload the app: the preset comes back from localStorage.
    cleanup();
    mockServer();
    await renderReady();
    await openTab(user, "プリセット");
    await user.selectOptions(screen.getByLabelText("使うプリセット"), screen.getByRole("option", { name: "速い WebP" }));
    await user.click(screen.getByRole("button", { name: "適用" }));
    expect(screen.getByLabelText("画像形式")).toHaveValue("webp");
    expect(screen.getByLabelText("ステップ数")).toHaveValue(40);

    await user.click(screen.getByRole("button", { name: "「速い WebP」を削除" }));
    expect(await screen.findByText("「速い WebP」を削除しました。")).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "速い WebP" })).not.toBeInTheDocument();
  });

  it("warns that face/pose references need a first-time download", async () => {
    mockServer({ status: { ...health(), identity_cached: false } });
    const user = await renderReady();

    await user.selectOptions(screen.getByLabelText("モデル"), "sdxl");
    expect(screen.getByText(/約6.6GB の追加モデルのダウンロード/)).toBeInTheDocument();
  });

  it("disables generation while the model is loading", async () => {
    mockServer({ status: health("loading") });
    render(<App />);

    expect(await screen.findByRole("button", { name: "モデル準備中…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("SD 1.5 を読み込み中");
  });

  it("explains a failed model load but still allows trying again", async () => {
    mockServer({ status: health("failed") });
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("モデル「SD 1.5」を利用できません。モデルが見つかりません。");
    expect(screen.getByRole("button", { name: "画像を生成" })).toBeEnabled();
  });
});
