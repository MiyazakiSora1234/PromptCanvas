import { useRef } from "react";
import type { PickedImage } from "../lib/images";
import { buttonClass } from "./styles";

interface ImagePickerProps {
  /** Used as the file input's id/name, so field errors can focus it. */
  id: string;
  label: string;
  image: PickedImage | null;
  maxMb: number;
  error?: string;
  disabled: boolean;
  onFile: (file: File) => void;
  onClear: () => void;
}

/** File picker with a thumbnail preview, used for img2img and the face/pose references. */
export function ImagePicker({ id, label, image, maxMb, error, disabled, onFile, onClear }: ImagePickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="flex flex-col gap-1">
      <input
        ref={inputRef}
        id={id}
        name={id}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="sr-only"
        aria-label={`${label}ファイル`}
        aria-invalid={error ? true : undefined}
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
            alt={`${label}のプレビュー`}
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
          <button type="button" className={buttonClass} disabled={disabled} onClick={onClear} aria-label={`${label}を外す`}>
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
          {label}を選ぶ（PNG / JPEG / WebP、{maxMb}MBまで）
        </button>
      )}
      {error && <p className="text-xs text-red-700 dark:text-red-300">{error}</p>}
    </div>
  );
}
