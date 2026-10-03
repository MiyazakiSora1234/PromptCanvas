import { useState } from "react";
import type { AppConfig } from "../api/types";
import { useImageGeneration } from "../hooks/useImageGeneration";
import { ImageFileError, readImageFile, sizeForAspect } from "../lib/images";
import { presentError } from "../lib/presentError";
import {
  findModel,
  hasAdvancedFieldError,
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
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [focusRequest, setFocusRequest] = useState(0);
  const { busy, startedAt, result, generate } = useImageGeneration();

  const showErrors = (text: string, errors: FieldErrors, openAdvanced: boolean) => {
    setMessage(text);
    setFieldErrors(errors);
    if (openAdvanced) setAdvancedOpen(true);
    if (Object.keys(errors).length > 0) setFocusRequest((n) => n + 1);
  };

  const handleValueChange = (name: keyof FormValues, value: string) => {
    setForm((prev) => ({ ...prev, values: { ...prev.values, [name]: value } }));
    setFieldErrors((prev) => withoutKeys(prev, [name])); // editing a field clears its error
  };

  /**
   * Output size: the aspect ratio of the image that defines the composition (img2img source,
   * else pose reference, else face photo) at the model's native pixel count; otherwise model defaults.
   */
  const sizeFor = (state: Pick<FormState, "initImage" | "poseImage" | "faceImage">, modelId: string) => {
    const model = findModel(config, modelId);
    const layout = state.initImage ?? state.poseImage ?? state.faceImage;
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
      const next = { ...prev, faceImage: keepRefs ? prev.faceImage : null, poseImage: keepRefs ? prev.poseImage : null };
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
        "face_image",
        "pose_image",
      ]),
    );
  };

  /** Read a picked file into one of the image slots and resize the output to fit it. */
  const pickImage = async (slot: "initImage" | "faceImage" | "poseImage", field: string, file: File) => {
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

  const clearImage = (slot: "initImage" | "faceImage" | "poseImage", fields: string[]) => {
    setForm((prev) => {
      const next = { ...prev, [slot]: null };
      return { ...next, values: { ...prev.values, ...sizeFor(next, prev.values.model) } };
    });
    setFieldErrors((prev) => withoutKeys(prev, fields));
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
      showErrors("入力内容を確認してください。", validation.errors, hasAdvancedFieldError(validation.errors));
      return;
    }

    // Switching models loads new weights (and downloads them the first time); show that in the header.
    const switching = loadedModel !== null && validation.payload.model !== loadedModel;
    if (switching) window.setTimeout(onRecheckHealth, 500);

    try {
      await generate(validation.payload);
      if (switching) onRecheckHealth();
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const presented = presentError(err);
      showErrors(presented.message, presented.fieldErrors, presented.openAdvanced);
      if (presented.recheckHealth || switching) onRecheckHealth();
    }
  };

  const handleReuseSeed = (seed: number) => {
    handleValueChange("seed", String(seed));
    setAdvancedOpen(true);
  };

  const switchingModel = loadedModel !== null && form.values.model !== loadedModel;
  const submitLabel = busy
    ? modelLoading
      ? "モデルを読み込み中…"
      : "生成中…"
    : modelLoading
      ? "モデル準備中…"
      : switchingModel
        ? "モデルを切り替えて生成"
        : "画像を生成";

  return (
    <main className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-4 pb-8 md:grid-cols-[minmax(0,440px)_minmax(0,1fr)]">
      <GenerateForm
        config={config}
        cachedModels={cachedModels}
        state={form}
        fieldErrors={fieldErrors}
        onValueChange={handleValueChange}
        onModelChange={handleModelChange}
        onLorasChange={handleLorasChange}
        onInitImageFile={(file) => void pickImage("initImage", "init_image", file)}
        onInitImageClear={() => clearImage("initImage", ["init_image", "strength"])}
        onReferenceFile={(kind, file) =>
          void pickImage(kind === "face" ? "faceImage" : "poseImage", `${kind}_image`, file)
        }
        onReferenceClear={(kind) =>
          clearImage(kind === "face" ? "faceImage" : "poseImage", [`${kind}_image`, `${kind === "face" ? "identity" : "pose"}_strength`])
        }
        identityCached={identityCached}
        advancedOpen={advancedOpen}
        onAdvancedOpenChange={setAdvancedOpen}
        onSubmit={() => void handleSubmit()}
        busy={busy}
        submitDisabled={busy || modelBlocked}
        submitLabel={submitLabel}
        message={message}
        focusRequest={focusRequest}
      />
      <ResultPanel config={config} startedAt={startedAt} result={result} onReuseSeed={handleReuseSeed} />
    </main>
  );
}
