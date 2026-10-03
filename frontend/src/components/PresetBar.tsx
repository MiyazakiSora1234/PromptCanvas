import { useState } from "react";
import type { AppConfig } from "../api/types";
import type { PresetStore } from "../hooks/usePresets";
import { HELP } from "../lib/help";
import { describeSettings, MAX_PRESET_NAME } from "../lib/presets";
import { buttonClass, inputClass } from "./styles";

interface PresetBarProps {
  config: AppConfig;
  presets: PresetStore;
  disabled: boolean;
  /** Apply the preset with this id; returns a message to show. */
  onApply: (id: string) => string;
  /** Save the current settings under this name; returns a message to show. */
  onSave: (name: string) => string;
}

/** Pick / apply / save / delete settings presets. */
export function PresetBar({ config, presets, disabled, onApply, onSave }: PresetBarProps) {
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const selected = selectedId ? presets.find(selectedId) : undefined;

  const handleSave = () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setStatus("保存する名前を入力してください。");
      return;
    }
    setStatus(onSave(trimmed));
    setName("");
  };

  const handleDelete = () => {
    if (!selected || selected.builtIn) return;
    setStatus(
      presets.remove(selected.id)
        ? `「${selected.label}」を削除しました。`
        : "削除できませんでした（ブラウザの保存領域が使えません）。",
    );
    setSelectedId("");
  };

  return (
    <section aria-labelledby="preset-heading" className="flex flex-col gap-2">
      <h2 id="preset-heading" className="text-sm font-semibold">
        設定プリセット
      </h2>

      <div className="flex gap-2">
        <label htmlFor="preset" className="sr-only">
          使うプリセット
        </label>
        <select
          id="preset"
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          className={`${inputClass} min-w-0 flex-1`}
        >
          <option value="">プリセットを選ぶ…</option>
          {presets.builtIn.length > 0 && (
            <optgroup label="おすすめ">
              {presets.builtIn.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </optgroup>
          )}
          {presets.user.length > 0 && (
            <optgroup label="保存した設定">
              {presets.user.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          type="button"
          className={buttonClass}
          disabled={!selected || disabled}
          onClick={() => selected && setStatus(onApply(selected.id))}
        >
          適用
        </button>
        <button
          type="button"
          className={buttonClass}
          disabled={!selected || selected.builtIn || disabled}
          onClick={handleDelete}
          aria-label={selected ? `「${selected.label}」を削除` : "削除"}
        >
          削除
        </button>
      </div>
      {selected && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {selected.description || describeSettings(selected.settings, config)}
        </p>
      )}

      <div className="flex gap-2">
        <label htmlFor="preset-name" className="sr-only">
          保存する名前
        </label>
        <input
          id="preset-name"
          type="text"
          value={name}
          maxLength={MAX_PRESET_NAME}
          placeholder="今の設定に名前を付けて保存"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault(); // don't submit the generate form
              handleSave();
            }
          }}
          className={`${inputClass} min-w-0 flex-1`}
        />
        <button type="button" className={buttonClass} disabled={disabled} onClick={handleSave}>
          保存
        </button>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400">{HELP.presets} 保存先はこのブラウザです。</p>
      {status && (
        <p role="status" className="text-xs font-medium text-indigo-700 dark:text-indigo-300">
          {status}
        </p>
      )}
    </section>
  );
}
