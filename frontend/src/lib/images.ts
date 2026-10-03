import type { Limits, ModelDefaults } from "../api/types";

/** An image the user picked (img2img source, face or pose reference), already read into memory. */
export interface PickedImage {
  dataUrl: string;
  name: string;
  width: number;
  height: number;
  bytes: number;
}

export function base64ToBlob(data: string, mimeType: string): Blob {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mimeType });
}

export function buildFileName(seed: number, extension: string, now = new Date()): string {
  const stamp = now.toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  return `promptcanvas_${stamp}_seed${seed}.${extension}`;
}

/**
 * Output size that keeps the input image's aspect ratio at roughly the model's
 * native pixel count (what it was trained on), snapped to the allowed grid.
 */
export function sizeForAspect(
  imageWidth: number,
  imageHeight: number,
  defaults: Pick<ModelDefaults, "width" | "height">,
  limits: Limits,
): { width: number; height: number } {
  const aspect = imageWidth / imageHeight;
  const area = defaults.width * defaults.height;
  const snap = (v: number) =>
    Math.min(
      limits.max_image_size,
      Math.max(limits.min_image_size, Math.round(v / limits.size_multiple) * limits.size_multiple),
    );
  const width = Math.sqrt(area * aspect);
  return { width: snap(width), height: snap(width / aspect) };
}

export class ImageFileError extends Error {}

/** Read a user-selected file for img2img. Rejects with a user-facing message. */
export function readImageFile(file: File, maxMb: number): Promise<PickedImage> {
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    return Promise.reject(new ImageFileError("PNG / JPEG / WebP の画像を選んでください。"));
  }
  if (file.size > maxMb * 1024 * 1024) {
    return Promise.reject(new ImageFileError(`画像のファイルサイズは${maxMb}MB以下にしてください。`));
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new ImageFileError("画像を読み込めませんでした。別の画像を選んでください。"));
    reader.onload = () => {
      const dataUrl = String(reader.result);
      const img = new Image();
      img.onload = () =>
        resolve({ dataUrl, name: file.name, width: img.naturalWidth, height: img.naturalHeight, bytes: file.size });
      img.onerror = () => reject(new ImageFileError("画像を読み込めませんでした。別の画像を選んでください。"));
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}
