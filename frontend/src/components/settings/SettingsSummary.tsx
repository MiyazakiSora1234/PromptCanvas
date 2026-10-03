import type { AppConfig } from "../../api/types";
import { findModel, type FormState } from "../../lib/validation";

/** One-line overview of what will be generated, so settings in closed tabs aren't forgotten. */
export function SettingsSummary({ config, state }: { config: AppConfig; state: FormState }) {
  const { values } = state;
  const model = findModel(config, values.model);
  const style = config.styles.find((s) => s.id === values.style);
  const parts = [
    model.label.replace(/（.*）$/, ""),
    style && style.id !== "none" ? style.label.replace(/（.*）$/, "") : null,
    `${values.width}×${values.height}`,
    `${values.num_inference_steps}ステップ`,
    `ガイダンス ${values.guidance_scale}`,
    values.num_images !== "1" ? `${values.num_images}枚` : null,
    values.output_format.toUpperCase(),
    state.loras.length > 0 ? `LoRA ${state.loras.length}個` : null,
    state.initImage ? "img2img" : null,
    state.faceImages.length > 0 ? `顔の参照 ${state.faceImages.length}枚` : null,
    state.poseImage ? "ポーズ参照" : null,
    values.seed.trim() ? `シード ${values.seed.trim()}` : null,
  ].filter((p): p is string => p !== null);

  return (
    <p aria-label="現在の設定" className="text-xs text-slate-500 dark:text-slate-400">
      {parts.join("・")}
    </p>
  );
}
