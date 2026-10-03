import type { AppConfig } from "../api/types";
import { HELP } from "../lib/help";
import type { FieldErrors, FormState, FormValues } from "../lib/validation";
import { ImagePicker } from "./ImagePicker";
import { MultiImagePicker } from "./MultiImagePicker";
import { StrengthSlider } from "./StrengthSlider";

interface ReferencePickerProps {
  config: AppConfig;
  state: FormState;
  /** Model supports references (SDXL family). */
  supported: boolean;
  modelLabel: string;
  /** null while unknown; false shows the first-use download warning. */
  assetsCached: boolean | null;
  fieldErrors: FieldErrors;
  disabled: boolean;
  onFaceFiles: (files: File[]) => void;
  onRemoveFace: (index: number) => void;
  onPoseFile: (file: File) => void;
  onClearPose: () => void;
  onValueChange: (name: keyof FormValues, value: string) => void;
}

/** "Keep this face, use that pose" — InstantID + OpenPose ControlNet. */
export function ReferencePicker({
  config,
  state,
  supported,
  modelLabel,
  assetsCached,
  fieldErrors,
  disabled,
  onFaceFiles,
  onRemoveFace,
  onPoseFile,
  onClearPose,
  onValueChange,
}: ReferencePickerProps) {
  if (!config.identity) return null;
  const { limits } = config;
  const inUse = state.faceImages.length > 0 || state.poseImage !== null;

  return (
    <fieldset className="flex flex-col gap-3 rounded-md border border-slate-200 p-3 dark:border-slate-700">
      <legend className="px-1 text-sm font-semibold">顔・ポーズの参照（任意）</legend>

      {!supported ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          「{modelLabel}」では使えません。SDXL 系のモデル（SDXL / RealVisXL / Animagine XL）を選んでください。
        </p>
      ) : state.initImage !== null && !inUse ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          img2img（元画像から生成）と同時には使えません。元画像を外すと選べます。
        </p>
      ) : (
        <>
          {assetsCached === false && (
            <p className="rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
              初回の使用時に
              {config.identity.download_size_gb ? `約${config.identity.download_size_gb}GB の` : ""}
              追加モデルのダウンロードがあり、数分〜数十分かかります。
            </p>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold">顔の写真</span>
            <p className="text-xs text-slate-500 dark:text-slate-400">{HELP.faceImages}</p>
            <MultiImagePicker
              id="face_images"
              label="顔の写真"
              images={state.faceImages}
              max={limits.max_face_images}
              maxMb={limits.max_init_image_mb}
              error={fieldErrors.face_images}
              disabled={disabled}
              onFiles={onFaceFiles}
              onRemove={onRemoveFace}
            />
          </div>
          {state.faceImages.length > 0 && (
            <StrengthSlider
              id="identity_strength"
              label="顔の再現度"
              value={state.values.identity_strength}
              min={limits.min_control_strength}
              max={limits.max_control_strength}
              error={fieldErrors.identity_strength}
              onChange={(v) => onValueChange("identity_strength", v)}
              hint={HELP.identityStrength}
            />
          )}

          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold">ポーズ参考画像</span>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {HELP.poseImage}
              {state.faceImages.length > 0 ? "顔の位置もこの人物に合わせるため、顔が写っている必要があります。" : ""}
            </p>
            <ImagePicker
              id="pose_image"
              label="ポーズ参考画像"
              image={state.poseImage}
              maxMb={limits.max_init_image_mb}
              error={fieldErrors.pose_image}
              disabled={disabled}
              onFile={onPoseFile}
              onClear={onClearPose}
            />
          </div>
          {state.poseImage && (
            <StrengthSlider
              id="pose_strength"
              label="ポーズの強さ"
              value={state.values.pose_strength}
              min={limits.min_control_strength}
              max={limits.max_control_strength}
              error={fieldErrors.pose_strength}
              onChange={(v) => onValueChange("pose_strength", v)}
              hint={HELP.poseStrength}
            />
          )}

          <p className="text-xs text-slate-500 dark:text-slate-400">
            ※ 実在の人物の写真は本人の同意を得たものだけを使い、なりすましや権利侵害に使わないでください。
          </p>
        </>
      )}
    </fieldset>
  );
}
