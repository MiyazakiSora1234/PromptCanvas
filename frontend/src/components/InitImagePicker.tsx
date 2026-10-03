import type { Limits } from "../api/types";
import { HELP } from "../lib/help";
import type { PickedImage } from "../lib/images";
import { ImagePicker } from "./ImagePicker";
import { StrengthSlider } from "./StrengthSlider";

interface InitImagePickerProps {
  limits: Limits;
  image: PickedImage | null;
  strength: string;
  steps: string;
  imageError?: string;
  strengthError?: string;
  disabled: boolean;
  /** Set when face/pose references are in use (the two modes can't be combined). */
  unavailableReason?: string;
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
  unavailableReason,
  onFile,
  onClear,
  onStrengthChange,
}: InitImagePickerProps) {
  const effectiveSteps = Math.floor(Number(steps) * Number(strength));

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-1 text-sm font-semibold">元画像から生成（img2img・任意）</legend>
      <p className="text-xs text-slate-500 dark:text-slate-400">{HELP.initImage}</p>
      {unavailableReason && !image ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">{unavailableReason}</p>
      ) : (
        <ImagePicker
          id="init_image"
          label="元画像"
          image={image}
          maxMb={limits.max_init_image_mb}
          error={imageError}
          disabled={disabled}
          onFile={onFile}
          onClear={onClear}
        />
      )}
      {image && (
        <StrengthSlider
          id="strength"
          label="変換強度"
          value={strength}
          min={limits.min_strength}
          max={limits.max_strength}
          error={strengthError}
          onChange={onStrengthChange}
          hint={`${HELP.strength}（実際のステップ数 ≈ ${Number.isFinite(effectiveSteps) ? effectiveSteps : "-"}）`}
        />
      )}
    </fieldset>
  );
}
