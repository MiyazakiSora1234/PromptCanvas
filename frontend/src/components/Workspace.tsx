import { useState } from "react";
import type { AppConfig } from "../api/types";
import { useImageGeneration } from "../hooks/useImageGeneration";
import { usePresets } from "../hooks/usePresets";
import { applyPreset, captureSettings } from "../lib/presets";
import { buildSeries } from "../lib/series";
import { ImageFileError, readImageFile, sizeForAspect } from "../lib/images";
import { presentError } from "../lib/presentError";
import { firstErrorTab, type TabId } from "../lib/formTabs";
import {
  MAX_POSE_IMAGES,
  findModel,
  initialFormState,
  modelDefaultValues,
  supportsReference,
  validateForm,
  type FieldErrors,
  type FormState,
  type FormValues,
  type LoraSelection,
} from "../lib/validation";
import { GenerateForm } from "./GenerateForm";
import { GeneratePanel } from "./GeneratePanel";
import { ResultPanel } from "./ResultPanel";

interface WorkspaceProps {
  config: AppConfig;
  /** A model is being loaded: generation is disabled. */
  modelLoading: boolean;
  /** Catalog id of the model currently on the GPU, if known. */
  loadedModel: string | null;
  /** Downloaded model ids; null while unknown. */
  cachedModels: string[] | null;
  /** Face/pose reference assets downloaded; null while unknown. */
  identityCached: boolean | null;
  /** Japanese prompt translator downloaded; null while unknown. */
  translatorCached: boolean | null;
  /** Ask the parent to re-check /api/health (e.g. after a "model loading" error or a model switch). */
  onRecheckHealth: () => void;
}

const FORM_ID = "generate-form";

function withoutKeys(errors: FieldErrors, keys: string[]): FieldErrors {
  if (!keys.some((k) => k in errors)) return errors;
  const next = { ...errors };
  for (const k of keys) delete next[k];
  return next;
}

export function Workspace({
  config,
  modelLoading,
  loadedModel,
  cachedModels,
  identityCached,
  translatorCached,
  onRecheckHealth,
}: WorkspaceProps) {
  const [form, setForm] = useState<FormState>(() => initialFormState(config));
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [message, setMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("basic");
  const [focusRequest, setFocusRequest] = useState(0);
  const { busy, progress, result, generate, cancel, cancelling } = useImageGeneration();
  const presets = usePresets(config);

  /** Show a message and field errors, switching to the tab that holds the first one. */
  const showErrors = (text: string, errors: FieldErrors, tab: TabId | null) => {
    setMessage(text);
    setFieldErrors(errors);
    if (tab) setActiveTab(tab);
    if (Object.keys(errors).length > 0) setFocusRequest((n) => n + 1);
  };

  const handleValueChange = (name: keyof FormValues, value: string) => {
    setForm((prev) => ({ ...prev, values: { ...prev.values, [name]: value } }));
    setFieldErrors((prev) => withoutKeys(prev, [name])); // editing a field clears its error
  };

  /**
   * Output size: the aspect ratio of the image that defines the composition (img2img source,
   * else pose reference, else first face photo) at the model's native pixel count; otherwise model defaults.
   */
  const sizeFor = (state: Pick<FormState, "initImage" | "poseImages" | "faceImages">, modelId: string) => {
    const model = findModel(config, modelId);
    const layout = state.initImage ?? state.poseImages[0] ?? state.faceImages[0];
    const size = layout
      ? sizeForAspect(layout.width, layout.height, model.defaults, config.limits)
      : { width: model.defaults.width, height: model.defaults.height };
    return { width: String(size.width), height: String(size.height) };
  };

  const handleModelChange = (modelId: string) => {
    const model = findModel(config, modelId);
    setForm((prev) => {
      // Face/pose references only work with some model families (InstantID is SDXL-only).
      const keepRefs = supportsReference(config, model);
      const next = { ...prev, faceImages: keepRefs ? prev.faceImages : [], poseImages: keepRefs ? prev.poseImages : [] };
      return {
        ...next,
        // Each model has its own native size, step count, guidance and recommended sampler.
        values: { ...prev.values, ...modelDefaultValues(model), ...sizeFor(next, model.id) },
        // LoRAs only work with the model family they were trained for.
        loras: prev.loras.filter((s) => config.loras.find((l) => l.id === s.id)?.family === model.family),
      };
    });
    setFieldErrors((prev) =>
      withoutKeys(prev, [
        "model",
        "style",
        "scheduler",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "loras",
        "face_images",
        "pose_image",
      ]),
    );
  };

  /** Read the img2img source and resize the output to fit it. */
  const pickInitImage = async (file: File) => {
    try {
      const image = await readImageFile(file, config.limits.max_init_image_mb);
      setForm((prev) => {
        const next = { ...prev, initImage: image };
        return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
      });
      setFieldErrors((prev) => withoutKeys(prev, ["init_image", "width", "height"]));
    } catch (err) {
      const text = err instanceof ImageFileError ? err.message : "画像を読み込めませんでした。";
      setFieldErrors((prev) => ({ ...prev, init_image: text }));
    }
  };

  const clearInitImage = () => {
    setForm((prev) => {
      const next = { ...prev, initImage: null };
      return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
    });
    setFieldErrors((prev) => withoutKeys(prev, ["init_image", "strength"]));
  };

  /** Add face photos (of one person) or pose images, up to `max`; the output size follows the first. */
  const addImages = async (slot: "faceImages" | "poseImages", field: string, label: string, max: number, files: File[]) => {
    const room = max - form[slot].length;
    const accepted = files.slice(0, Math.max(0, room));
    const results = await Promise.allSettled(
      accepted.map((file) => readImageFile(file, config.limits.max_init_image_mb)),
    );
    const images = results.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []));
    const problems = results.flatMap((r, i) =>
      r.status === "rejected"
        ? [`${accepted[i]!.name}: ${r.reason instanceof ImageFileError ? r.reason.message : "読み込めませんでした。"}`]
        : [],
    );
    if (files.length > accepted.length) problems.push(`${label}は${max}枚までです。超えた分は追加していません。`);

    if (images.length > 0) {
      setForm((prev) => {
        const next = { ...prev, [slot]: [...prev[slot], ...images].slice(0, max) };
        return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
      });
    }
    setFieldErrors((prev) => {
      const cleared = withoutKeys(prev, [field, "width", "height", "num_images"]);
      return problems.length > 0 ? { ...cleared, [field]: problems.join(" ") } : cleared;
    });
  };

  const removeImage = (slot: "faceImages" | "poseImages", fields: string[], index: number) => {
    setForm((prev) => {
      const next = { ...prev, [slot]: prev[slot].filter((_, i) => i !== index) };
      return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
    });
    setFieldErrors((prev) => withoutKeys(prev, [...fields, "num_images"]));
  };

  const handleApplyPreset = (id: string): string => {
    const preset = presets.find(id);
    if (!preset) return "プリセットが見つかりません。";
    const applied = applyPreset(form, preset.settings, config);
    if (!applied.ok) return applied.error;
    let next = applied.state;
    const notes = [...applied.warnings];
    // With a reference / img2img image, the output keeps that image's aspect ratio.
    if (next.initImage ?? next.poseImages[0] ?? next.faceImages[0]) {
      next = { ...next, values: { ...next.values, ...sizeFor(next, next.values.model) } };
      notes.push("出力サイズは選択中の画像の縦横比に合わせました。");
    }
    setForm(next);
    setFieldErrors({});
    setMessage(null);
    return [`「${preset.label}」を適用しました。`, ...notes].join(" ");
  };

  const handleSavePreset = (name: string, includePrompt: boolean): string => {
    const saved = presets.save(name, captureSettings(form, { includePrompt }));
    if (!saved.ok) return "保存できませんでした（このブラウザでは設定を保存できない状態です）。";
    return saved.overwritten ? `「${name}」を上書き保存しました。` : `「${name}」を保存しました。`;
  };

  const handleLorasChange = (loras: LoraSelection[]) => {
    setForm((prev) => ({ ...prev, loras }));
    setFieldErrors((prev) => withoutKeys(prev, ["loras"]));
  };

  const handleSubmit = async () => {
    if (busy) return;
    setMessage(null);
    setFieldErrors({});

    const validation = validateForm(form, config);
    if (!validation.ok) {
      showErrors("入力内容を確認してください。", validation.errors, firstErrorTab(validation.errors));
      return;
    }

    // Switching models loads new weights (and downloads them the first time); show that in the header.
    const switching = loadedModel !== null && validation.payload.model !== loadedModel;
    if (switching) window.setTimeout(onRecheckHealth, 500);

    try {
      await generate(buildSeries(validation.payload, validation.count, form.poseImages, config.limits));
      if (switching) onRecheckHealth();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setMessage("生成を中止しました。"); // stopped before the server knew the job
        return;
      }
      const presented = presentError(err);
      showErrors(presented.message, presented.fieldErrors, presented.showTab);
      if (presented.recheckHealth || switching) onRecheckHealth();
    }
  };

  const handleReuseSeed = (seed: number) => {
    handleValueChange("seed", String(seed));
    setActiveTab("advanced"); // where the seed field lives
  };

  const switchingModel = loadedModel !== null && form.values.model !== loadedModel;
  const count = Number(form.values.num_images);
  const poses = form.poseImages.length;
  const action =
    poses > 1 && Number.isInteger(count)
      ? `${poses}ポーズ × ${count}枚（計${poses * count}枚）を生成`
      : Number.isInteger(count) && count > 1
        ? `${count}枚を生成`
        : "画像を生成";
  const submitLabel = busy
    ? cancelling
      ? "中止しています…"
      : modelLoading
        ? "モデルを読み込み中…"
        : "生成中…"
    : modelLoading
      ? "モデル準備中…"
      : switchingModel
        ? `モデルを切り替えて${action.replace("画像を", "")}`
        : action;

  return (
    <main className="mx-auto grid max-w-6xl grid-cols-1 items-start gap-4 px-4 pb-8 md:grid-cols-[minmax(0,440px)_minmax(0,1fr)]">
      <GenerateForm
        formId={FORM_ID}
        config={config}
        cachedModels={cachedModels}
        state={form}
        fieldErrors={fieldErrors}
        onValueChange={handleValueChange}
        onModelChange={handleModelChange}
        onLorasChange={handleLorasChange}
        onInitImageFile={(file) => void pickInitImage(file)}
        onInitImageClear={clearInitImage}
        onFaceFiles={(files) => void addImages("faceImages", "face_images", "顔の写真", config.limits.max_face_images, files)}
        onRemoveFace={(i) => removeImage("faceImages", ["face_images", "identity_strength"], i)}
        onPoseFiles={(files) => void addImages("poseImages", "pose_image", "ポーズ参考画像", MAX_POSE_IMAGES, files)}
        onRemovePose={(i) => removeImage("poseImages", ["pose_image", "pose_strength"], i)}
        identityCached={identityCached}
        translatorCached={translatorCached}
        presets={presets}
        onApplyPreset={handleApplyPreset}
        onSavePreset={handleSavePreset}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onSubmit={() => void handleSubmit()}
        busy={busy}
        submitDisabled={busy || modelLoading}
        focusRequest={focusRequest}
      />
      {/* The result with the generate button right under the image; stays in view while settings scroll. */}
      <div className="min-w-0 md:sticky md:top-4">
        <ResultPanel config={config} progress={progress} result={result} onReuseSeed={handleReuseSeed}>
          <GeneratePanel
            config={config}
            state={form}
            formId={FORM_ID}
            disabled={busy || modelLoading}
            label={submitLabel}
            message={message}
            onCancel={busy ? () => void cancel() : null}
            cancelling={cancelling}
          />
        </ResultPanel>
      </div>
    </main>
  );
}
