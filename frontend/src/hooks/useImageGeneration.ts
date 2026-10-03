import { useCallback, useEffect, useRef, useState } from "react";
import { generateImages } from "../api/client";
import type { GenerateRequest, GenerateResponse } from "../api/types";
import { base64ToBlob, buildFileName } from "../lib/images";

export interface ResultImage {
  /** Object URL; revoked automatically when the result is replaced or on unmount. */
  url: string;
  fileName: string;
  seed: number;
}

export interface GenerationResult {
  images: ResultImage[];
  response: Omit<GenerateResponse, "images">;
  request: GenerateRequest;
}

const EXTENSIONS: Record<string, string> = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };

/**
 * Runs one generation at a time. A second call while one is in flight is ignored,
 * so double clicks / repeated Ctrl+Enter never send duplicate requests.
 * `generate` rejects with ApiError / NetworkError on failure.
 */
export function useImageGeneration() {
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const inFlight = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);

  // Abort a pending request if the component goes away.
  useEffect(() => () => controllerRef.current?.abort(), []);

  // Free the previous images' memory whenever they are replaced.
  useEffect(() => {
    const urls = result?.images.map((img) => img.url) ?? [];
    return () => urls.forEach((url) => URL.revokeObjectURL(url));
  }, [result]);

  const generate = useCallback(async (request: GenerateRequest): Promise<void> => {
    if (inFlight.current) return;
    inFlight.current = true;
    const controller = new AbortController();
    controllerRef.current = controller;
    setStartedAt(performance.now());
    try {
      const { images, ...response } = await generateImages(request, controller.signal);
      const now = new Date();
      setResult({
        images: images.map((img) => ({
          url: URL.createObjectURL(base64ToBlob(img.data, img.mime_type)),
          fileName: buildFileName(img.seed, EXTENSIONS[img.mime_type] ?? "img", now),
          seed: img.seed,
        })),
        response,
        request,
      });
    } finally {
      inFlight.current = false;
      controllerRef.current = null;
      setStartedAt(null);
    }
  }, []);

  return { busy: startedAt !== null, startedAt, result, generate };
}
