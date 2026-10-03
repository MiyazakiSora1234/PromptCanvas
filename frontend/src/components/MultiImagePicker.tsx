import { useRef } from "react";
import type { PickedImage } from "../lib/images";
import { buttonClass } from "./styles";

interface MultiImagePickerProps {
  /** Used as the file input's id/name, so field errors can focus it. */
  id: string;
  label: string;
  images: PickedImage[];
  max: number;
  maxMb: number;
  error?: string;
  disabled: boolean;
  onFiles: (files: File[]) => void;
  onRemove: (index: number) => void;
}

/** Pick several images (e.g. face photos of one person) with thumbnails and per-image removal. */
export function MultiImagePicker({ id, label, images, max, maxMb, error, disabled, onFiles, onRemove }: MultiImagePickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const remaining = max - images.length;

  return (
    <div className="flex flex-col gap-2">
      <input
        ref={inputRef}
        id={id}
        name={id}
        type="file"
        multiple
        accept="image/png,image/jpeg,image/webp"
        className="sr-only"
        aria-label={`${label}ファイル`}
        aria-invalid={error ? true : undefined}
        disabled={disabled || remaining <= 0}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length > 0) onFiles(files);
          e.target.value = ""; // allow picking the same files again
        }}
      />
      {images.length > 0 && (
        <ul className="flex flex-wrap gap-2" aria-label={`選択中の${label}`}>
          {images.map((img, i) => (
            <li key={`${img.name}-${i}`} className="relative">
              <img
                src={img.dataUrl}
                alt={`${label} ${i + 1}枚目`}
                title={`${img.name}（${img.width}×${img.height}px）`}
                className="size-16 rounded border border-slate-300 object-cover dark:border-slate-600"
              />
              <button
                type="button"
                disabled={disabled}
                onClick={() => onRemove(i)}
                aria-label={`${label} ${i + 1}枚目を外す`}
                className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-slate-700 text-xs text-white hover:enabled:bg-red-600 disabled:opacity-50"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      <button
        type="button"
        className={`${buttonClass} border-dashed py-2`}
        disabled={disabled || remaining <= 0}
        onClick={() => inputRef.current?.click()}
      >
        {images.length === 0
          ? `${label}を選ぶ（複数可・${max}枚まで・各${maxMb}MBまで）`
          : remaining > 0
            ? `${label}を追加（あと${remaining}枚）`
            : `上限の${max}枚に達しました`}
      </button>
      {error && <p className="text-xs text-red-700 dark:text-red-300">{error}</p>}
    </div>
  );
}
