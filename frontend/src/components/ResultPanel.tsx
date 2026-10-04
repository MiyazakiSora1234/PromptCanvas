import { useEffect, useState, type ReactNode } from "react";
import type { AppConfig } from "../api/types";
import type { GenerationResult } from "../hooks/useImageGeneration";
import { buttonClass } from "./styles";
import { Panel } from "./ui";

function ProgressLabel({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(id);
  }, []);
  const seconds = (Math.max(0, now - startedAt) / 1000).toFixed(1);
  return <span>{`生成中… ${seconds} 秒`}</span>;
}

interface ResultPanelProps {
  config: AppConfig;
  startedAt: number | null;
  result: GenerationResult | null;
  onReuseSeed: (seed: number) => void;
  /** Generate controls, shown right under the image (above downloads and details). */
  children?: ReactNode;
}

/**
 * Image, then the generate button, then thumbnails / download / details. On wide screens the
 * image is capped to the window height so the button stays visible; details scroll in the panel.
 */
export function ResultPanel({ config, startedAt, result, onReuseSeed, children }: ResultPanelProps) {
  const busy = startedAt !== null;
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [shownResult, setShownResult] = useState(result);
  // A new result starts with its first image selected.
  if (shownResult !== result) {
    setShownResult(result);
    setSelectedIndex(0);
  }
  const selected = result?.images[selectedIndex] ?? result?.images[0];

  const meta: Array<[string, string]> = [];
  if (result && selected) {
    const r = result.response;
    const label = (list: { id: string; label: string }[], id: string) => list.find((x) => x.id === id)?.label ?? id;
    if (r.translated_prompt) meta.push(["英訳", r.translated_prompt]);
    if (r.translated_negative_prompt) meta.push(["ネガティブ英訳", r.translated_negative_prompt]);
    meta.push(
      ["シード", String(selected.seed)],
      ["モデル", label(config.models, r.model)],
      ["サンプラー", label(config.schedulers, r.scheduler)],
      ["スタイル", label(config.styles, r.style)],
      ["サイズ", `${r.width}×${r.height}`],
      ["ステップ", String(r.num_inference_steps)],
      ["ガイダンス", String(r.guidance_scale)],
      ["形式", r.output_format.toUpperCase()],
    );
    if (result.request.init_image) meta.push(["img2img", `変換強度 ${result.request.strength}`]);
    if (result.request.face_images.length > 0)
      meta.push(["顔の参照", `${result.request.face_images.length}枚・再現度 ${result.request.identity_strength}`]);
    if (result.request.pose_image) meta.push(["ポーズ参照", `強さ ${result.request.pose_strength}`]);
    if (result.request.loras.length > 0)
      meta.push(["LoRA", result.request.loras.map((l) => `${label(config.loras, l.id)} (${l.scale})`).join(", ")]);
    meta.push(["生成時間", `${(r.elapsed_ms / 1000).toFixed(1)} 秒`]);
  }

  return (
    // *:shrink-0 — children keep their size; the panel scrolls instead of squashing the image.
    <Panel className="flex flex-col gap-3 *:shrink-0 md:max-h-[calc(100vh-6rem)] md:overflow-y-auto">
      <h2 className="text-lg font-semibold">生成結果</h2>
      <div
        aria-busy={busy}
        className={`relative grid min-h-[min(360px,calc(100vh-19rem))] place-items-center overflow-hidden rounded-lg border bg-slate-50 dark:bg-slate-900 ${
          result
            ? "border-solid border-slate-200 dark:border-slate-700"
            : "border-dashed border-slate-300 dark:border-slate-600"
        }`}
      >
        {selected && result ? (
          <img
            src={selected.url}
            alt={`生成画像: ${result.request.prompt}`}
            className={`block h-auto max-h-[60vh] max-w-full md:max-h-[calc(100vh-19rem)] ${busy ? "opacity-30" : ""}`}
          />
        ) : (
          !busy && <p className="p-4 text-center text-slate-500 dark:text-slate-400">生成した画像がここに表示されます。</p>
        )}
        {busy && (
          <div className="absolute inset-0 flex items-center justify-center gap-3 font-semibold">
            <span
              aria-hidden="true"
              className="size-6 animate-spin rounded-full border-[3px] border-slate-300 border-t-indigo-600 motion-reduce:animate-[spin_3s_linear_infinite]"
            />
            <ProgressLabel startedAt={startedAt} />
          </div>
        )}
      </div>

      {children}

      {result && result.images.length > 1 && (
        <div className="flex flex-wrap gap-2" role="listbox" aria-label="生成した画像">
          {result.images.map((img, i) => (
            <button
              key={img.url}
              type="button"
              role="option"
              aria-selected={img === selected}
              aria-label={`${i + 1}枚目（シード ${img.seed}）`}
              onClick={() => setSelectedIndex(i)}
              className="overflow-hidden rounded-md border-2 border-transparent aria-selected:border-indigo-600"
            >
              <img src={img.url} alt="" className="block size-20 object-cover" />
            </button>
          ))}
        </div>
      )}

      {result && result.response.filtered_count > 0 && (
        <p className="text-sm text-amber-700 dark:text-amber-300">
          {result.response.filtered_count}枚がセーフティフィルタにより除外されました。
        </p>
      )}

      {result && selected && (
        <>
          <div className="flex flex-wrap gap-2">
            <a href={selected.url} download={selected.fileName} className={buttonClass}>
              {result.response.output_format.toUpperCase()}をダウンロード
            </a>
            <button type="button" disabled={busy} className={buttonClass} onClick={() => onReuseSeed(selected.seed)}>
              このシードを再利用
            </button>
          </div>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-sm">
            {meta.map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
                <dd className="break-all">{value}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </Panel>
  );
}
