import type { AppConfig } from "../api/types";
import type { FieldErrors, FormState, FormValues } from "../lib/validation";
import { ImagePicker } from "./ImagePicker";
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
  onFaceFile: (file: File) => void;
  onPoseFile: (file: File) => void;
  onClearFace: () => void;
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
  onFaceFile,
  onPoseFile,
  onClearFace,
  onClearPose,
  onValueChange,
}: ReferencePickerProps) {
  if (!config.identity) return null;
  const { limits } = config;
  const inUse = state.faceImage !== null || state.poseImage !== null;

  return (
    <fieldset className="flex flex-col gap-3 rounded-md border border-slate-200 p-3 dark:border-slate-700">
      <legend className="px-1 text-sm font-semibold">顔・ポーズの参照（任意）</legend>
      <p className="text-xs text-slate-500 dark:text-slate-400">
        顔の写真を選ぶと、その人の顔のまま、服装や場面はプロンプトの内容で生成します。ポーズ参考画像を選ぶと、その人物と同じ姿勢になります。
      </p>

      {!supported ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          「{modelLabel}」では使えません。SDXL 系のモデル（SDXL / Animagine XL）を選んでください。
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
            <ImagePicker
              id="face_image"
              label="顔の写真"
              image={state.faceImage}
              maxMb={limits.max_init_image_mb}
              error={fieldErrors.face_image}
              disabled={disabled}
              onFile={onFaceFile}
              onClear={onClearFace}
            />
            <p className="text-xs text-slate-500 dark:text-slate-400">
              正面に近い向きで、顔が大きくはっきり写った写真が最も似ます。写真の人物向けで、イラストの顔は検出できないことがあります。
            </p>
          </div>
          {state.faceImage && (
            <StrengthSlider
              id="identity_strength"
              label="顔の再現度"
              value={state.values.identity_strength}
              min={limits.min_control_strength}
              max={limits.max_control_strength}
              error={fieldErrors.identity_strength}
              onChange={(v) => onValueChange("identity_strength", v)}
              hint="上げるほど元の顔に近づきます。色が濃すぎる・プロンプトが効かないときは下げてください。"
            />
          )}

          <div className="flex flex-col gap-1">
            <span className="text-sm font-semibold">ポーズ参考画像</span>
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
            <p className="text-xs text-slate-500 dark:text-slate-400">
              人物の全身または上半身が写った写真。顔の写真と併用する場合は、この画像の人物の顔も写っている必要があります（顔の位置を合わせるため）。出力サイズはこの画像の縦横比に合わせます。
            </p>
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
            />
          )}

          <p className="text-xs text-slate-500 dark:text-slate-400">
            実在の人物の写真は、本人の同意を得たものだけを使ってください。生成した画像で他人になりすましたり、名誉や権利を侵害したりしないでください。
          </p>
        </>
      )}
    </fieldset>
  );
}
