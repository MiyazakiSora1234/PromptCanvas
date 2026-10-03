import { useState } from "react";
import type { AppConfig } from "../api/types";
import { useImageGeneration } from "../hooks/useImageGeneration";
import { usePresets } from "../hooks/usePresets";
import { applyPreset, captureSettings } from "../lib/presets";
import { ImageFileError, readImageFile, sizeForAspect } from "../lib/images";
import { presentError } from "../lib/presentError";
import { firstErrorTab, type TabId } from "../lib/formTabs";
import {
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
  modelBlocked: boolean;
  modelLoading: boolean;
  /** Catalog id of the model currently on the GPU, if known. */
  loadedModel: string | null;
  /** Downloaded model ids; null while unknown. */
  cachedModels: string[] | null;
  /** Face/pose reference assets downloaded; null while unknown. */
  identityCached: boolean | null;
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
  modelBlocked,
  modelLoading,
  loadedModel,
  cachedModels,
  identityCached,
  onRecheckHealth,
}: WorkspaceProps) {
  const [form, setForm] = useState<FormState>(() => initialFormState(config));
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [message, setMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("basic");
  const [focusRequest, setFocusRequest] = useState(0);
  const { busy, startedAt, result, generate, cancel, cancelling } = useImageGeneration();
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
  const sizeFor = (state: Pick<FormState, "initImage" | "poseImage" | "faceImages">, modelId: string) => {
    const model = findModel(config, modelId);
    const layout = state.initImage ?? state.poseImage ?? state.faceImages[0];
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
      const next = { ...prev, faceImages: keepRefs ? prev.faceImages : [], poseImage: keepRefs ? prev.poseImage : null };
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

  /** Read a picked file into one of the image slots and resize the output to fit it. */
  const pickImage = async (slot: "initImage" | "poseImage", field: string, file: File) => {
    try {
      const image = await readImageFile(file, config.limits.max_init_image_mb);
      setForm((prev) => {
        const next = { ...prev, [slot]: image };
        return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
      });
      setFieldErrors((prev) => withoutKeys(prev, [field, "width", "height"]));
    } catch (err) {
      const text = err instanceof ImageFileError ? err.message : "画像を読み込めませんでした。";
      setFieldErrors((prev) => ({ ...prev, [field]: text }));
    }
  };

  const clearImage = (slot: "initImage" | "poseImage", fields: string[]) => {
    setForm((prev) => {
      const next = { ...prev, [slot]: null };
      return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
    });
    setFieldErrors((prev) => withoutKeys(prev, fields));
  };

  /** Add face photos (of the same person), up to the server's limit. */
  const addFaces = async (files: File[]) => {
    const max = config.limits.max_face_images;
    const room = max - form.faceImages.length;
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
    if (files.length > accepted.length) problems.push(`顔の写真は${max}枚までです。超えた分は追加していません。`);

    if (images.length > 0) {
      setForm((prev) => {
        const next = { ...prev, faceImages: [...prev.faceImages, ...images].slice(0, max) };
        return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
      });
    }
    setFieldErrors((prev) => {
      const cleared = withoutKeys(prev, ["face_images", "width", "height"]);
      return problems.length > 0 ? { ...cleared, face_images: problems.join(" ") } : cleared;
    });
  };

  const removeFace = (index: number) => {
    setForm((prev) => {
      const next = { ...prev, faceImages: prev.faceImages.filter((_, i) => i !== index) };
      return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
    });
    setFieldErrors((prev) => withoutKeys(prev, ["face_images", "identity_strength"]));
  };

  const handleApplyPreset = (id: string): string => {
    const preset = presets.find(id);
    if (!preset) return "プリセットが見つかりません。";
    const applied = applyPreset(form, preset.settings, config);
    if (!applied.ok) return applied.error;
    let next = applied.state;
    const notes = [...applied.warnings];
    // With a reference / img2img image, the output keeps that image's aspect ratio.
    if (next.initImage ?? next.poseImage ?? next.faceImages[0]) {
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
      await generate(validation.payload);
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
  const submitLabel = busy
    ? cancelling
      ? "中止しています…"
      : modelLoading
      ? "モデルを読み込み中…"
      : "生成中…"
    : modelLoading
      ? "モデル準備中…"
      : switchingModel
        ? "モデルを切り替えて生成"
        : "画像を生成";

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
        onInitImageFile={(file) => void pickImage("initImage", "init_image", file)}
        onInitImageClear={() => clearImage("initImage", ["init_image", "strength"])}
        onFaceFiles={(files) => void addFaces(files)}
        onRemoveFace={removeFace}
        onPoseFile={(file) => void pickImage("poseImage", "pose_image", file)}
        onClearPose={() => clearImage("poseImage", ["pose_image", "pose_strength"])}
        identityCached={identityCached}
        presets={presets}
        onApplyPreset={handleApplyPreset}
        onSavePreset={handleSavePreset}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onSubmit={() => void handleSubmit()}
        busy={busy}
        submitDisabled={busy || modelBlocked}
        focusRequest={focusRequest}
      />
      {/* The result with the generate button right under the image; stays in view while settings scroll. */}
      <div className="min-w-0 md:sticky md:top-4">
        <ResultPanel config={config} startedAt={startedAt} result={result} onReuseSeed={handleReuseSeed}>
          <GeneratePanel
            config={config}
            state={form}
            formId={FORM_ID}
            disabled={busy || modelBlocked}
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
