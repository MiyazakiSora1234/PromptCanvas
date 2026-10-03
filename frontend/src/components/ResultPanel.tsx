import { useEffect, useState } from "react";
import type { GenerationResult } from "../hooks/useImageGeneration";
import { buttonClass, primaryButtonClass } from "./styles";
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
  startedAt: number | null;
  result: GenerationResult | null;
  onReuseSeed: (seed: number) => void;
}

export function ResultPanel({ startedAt, result, onReuseSeed }: ResultPanelProps) {
  const busy = startedAt !== null;

  const meta: Array<[string, string]> = result
    ? [
        ["シード", String(result.seed)],
        ["サイズ", `${result.params.width}×${result.params.height}`],
        ["ステップ", String(result.params.num_inference_steps)],
        ["ガイダンス", String(result.params.guidance_scale)],
        ...(result.elapsedMs !== null
          ? [["生成時間", `${(result.elapsedMs / 1000).toFixed(1)} 秒`] as [string, string]]
          : []),
      ]
    : [];

  return (
    <Panel>
      <h2 className="mb-3 text-lg font-semibold">生成結果</h2>
      <div
        aria-busy={busy}
        className={`relative grid min-h-[360px] place-items-center overflow-hidden rounded-lg border bg-slate-50 dark:bg-slate-900 ${
          result ? "border-solid border-slate-200 dark:border-slate-700" : "border-dashed border-slate-300 dark:border-slate-600"
        }`}
      >
        {result ? (
          <img
            src={result.url}
            alt={`生成画像: ${result.params.prompt}`}
            className={`block h-auto max-h-[75vh] max-w-full ${busy ? "opacity-30" : ""}`}
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

      {result && (
        <>
          <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-sm">
            {meta.map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
                <dd className="break-all">{value}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-3 flex flex-wrap gap-2">
            <a href={result.url} download={result.fileName} className={primaryButtonClass}>
              PNGをダウンロード
            </a>
            <button type="button" disabled={busy} className={buttonClass} onClick={() => onReuseSeed(result.seed)}>
              このシードを再利用
            </button>
          </div>
        </>
      )}
    </Panel>
  );
}
