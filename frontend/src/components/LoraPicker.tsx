import type { Limits, LoraOption } from "../api/types";
import { HELP } from "../lib/help";
import type { LoraSelection } from "../lib/validation";

interface LoraPickerProps {
  limits: Limits;
  options: LoraOption[];
  modelLabel: string;
  selected: LoraSelection[];
  error?: string;
  onChange: (selected: LoraSelection[]) => void;
}

export function LoraPicker({ limits, options, modelLabel, selected, error, onChange }: LoraPickerProps) {
  if (limits.max_loras === 0) return null;

  const toggle = (lora: LoraOption, checked: boolean) =>
    onChange(
      checked
        ? [...selected, { id: lora.id, scale: lora.default_scale }]
        : selected.filter((s) => s.id !== lora.id),
    );
  const setScale = (id: string, scale: number) =>
    onChange(selected.map((s) => (s.id === id ? { ...s, scale } : s)));
  const full = selected.length >= limits.max_loras;

  return (
    <fieldset className="flex flex-col gap-2" aria-describedby={error ? "loras-error" : undefined}>
      <legend className="mb-1 text-sm font-semibold">LoRA（画風・概念の追加）</legend>
      <p className="text-xs text-slate-500 dark:text-slate-400">{HELP.loras}</p>
      {options.length === 0 ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">「{modelLabel}」で使える LoRA はありません。</p>
      ) : (
        options.map((lora) => {
          const selection = selected.find((s) => s.id === lora.id);
          return (
            <div key={lora.id} className="rounded-md border border-slate-200 p-2 dark:border-slate-700">
              <label className="flex items-start gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={selection !== undefined}
                  disabled={!selection && full}
                  onChange={(e) => toggle(lora, e.target.checked)}
                  className="mt-1 accent-indigo-600"
                />
                <span>
                  <span className="font-semibold">{lora.label}</span>
                  {lora.description && (
                    <span className="block text-xs text-slate-500 dark:text-slate-400">{lora.description}</span>
                  )}
                  {lora.trigger_words && (
                    <span className="block text-xs text-slate-500 dark:text-slate-400">
                      プロンプトに「<code>{lora.trigger_words}</code>」を含めると効果が出やすくなります。
                    </span>
                  )}
                </span>
              </label>
              {selection && (
                <div className="mt-2 flex items-center gap-2 pl-6 text-xs">
                  <label htmlFor={`lora-scale-${lora.id}`}>強さ</label>
                  <input
                    id={`lora-scale-${lora.id}`}
                    type="range"
                    min={limits.min_lora_scale}
                    max={limits.max_lora_scale}
                    step={0.05}
                    value={selection.scale}
                    onChange={(e) => setScale(lora.id, Number(e.target.value))}
                    className="flex-1 accent-indigo-600"
                  />
                  <span className="w-10 text-right tabular-nums">{selection.scale.toFixed(2)}</span>
                </div>
              )}
              {selection && <p className="pl-6 text-xs text-slate-500 dark:text-slate-400">{HELP.loraScale}</p>}
            </div>
          );
        })
      )}
      {error && (
        <p id="loras-error" tabIndex={-1} className="text-xs text-red-700 dark:text-red-300">
          {error}
        </p>
      )}
    </fieldset>
  );
}
