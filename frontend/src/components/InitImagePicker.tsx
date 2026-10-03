import { useRef } from "react";
import type { Limits } from "../api/types";
import type { InitImage } from "../lib/images";
import { buttonClass } from "./styles";

interface InitImagePickerProps {
  limits: Limits;
  image: InitImage | null;
  strength: string;
  steps: string;
  imageError?: string;
  strengthError?: string;
  disabled: boolean;
  onFile: (file: File) => void;
  onClear: () => void;
  onStrengthChange: (value: string) => void;
}

export function InitImagePicker({
  limits,
  image,
  strength,
  steps,
  imageError,
  strengthError,
  disabled,
  onFile,
  onClear,
  onStrengthChange,
}: InitImagePickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const effectiveSteps = Math.floor(Number(steps) * Number(strength));

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-1 text-sm font-semibold">元画像から生成（img2img・任意）</legend>
      <input
        ref={inputRef}
        id="init_image"
        name="init_image"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="sr-only"
        aria-label="元画像ファイル"
        aria-invalid={imageError ? true : undefined}
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = ""; // allow picking the same file again
        }}
      />
      {image ? (
        <div className="flex items-center gap-3">
          <img
            src={image.dataUrl}
            alt="元画像のプレビュー"
            className="size-16 rounded border border-slate-300 object-cover dark:border-slate-600"
          />
          <div className="min-w-0 flex-1 text-xs text-slate-500 dark:text-slate-400">
            <p className="truncate font-medium text-slate-700 dark:text-slate-200">{image.name}</p>
            <p>
              {image.width}×{image.height}px・{(image.bytes / 1024 / 1024).toFixed(1)}MB
            </p>
          </div>
          <button type="button" className={buttonClass} disabled={disabled} onClick={() => inputRef.current?.click()}>
            変更
          </button>
          <button type="button" className={buttonClass} disabled={disabled} onClick={onClear}>
            外す
          </button>
        </div>
      ) : (
        <button
          type="button"
          className={`${buttonClass} border-dashed py-3`}
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          画像を選ぶ（PNG / JPEG / WebP、{limits.max_init_image_mb}MBまで）
        </button>
      )}
      {imageError && <p className="text-xs text-red-700 dark:text-red-300">{imageError}</p>}

      {image && (
        <div className="flex flex-col gap-1">
          <label htmlFor="strength" className="flex justify-between text-sm font-semibold">
            <span>変換強度</span>
            <span className="tabular-nums">{Number(strength).toFixed(2)}</span>
          </label>
          <input
            id="strength"
            name="strength"
            type="range"
            min={limits.min_strength}
            max={limits.max_strength}
            step={0.05}
            value={strength}
            aria-invalid={strengthError ? true : undefined}
            onChange={(e) => onStrengthChange(e.target.value)}
            className="accent-indigo-600"
          />
          <p className="text-xs text-slate-500 dark:text-slate-400">
            小さいほど元画像に近く、大きいほど自由に描き直します（実際のステップ数 ≈{" "}
            {Number.isFinite(effectiveSteps) ? effectiveSteps : "-"}）。出力サイズは元画像の縦横比に合わせています。
          </p>
          {strengthError && <p className="text-xs text-red-700 dark:text-red-300">{strengthError}</p>}
        </div>
      )}
    </fieldset>
  );
}
