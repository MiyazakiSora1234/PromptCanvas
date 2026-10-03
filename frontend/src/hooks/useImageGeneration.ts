import { useCallback, useEffect, useRef, useState } from "react";
import { generateImage } from "../api/client";
import type { GenerateRequest } from "../api/types";

export interface GenerationResult {
  /** Object URL of the PNG; revoked automatically when replaced or on unmount. */
  url: string;
  fileName: string;
  seed: number;
  elapsedMs: number | null;
  params: GenerateRequest;
}

export function buildFileName(seed: number, now = new Date()): string {
  const stamp = now.toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  return `promptcanvas_${stamp}_seed${seed}.png`;
}

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

  // Free the previous image's memory whenever it is replaced.
  useEffect(() => {
    const url = result?.url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [result]);

  const generate = useCallback(async (params: GenerateRequest): Promise<void> => {
    if (inFlight.current) return;
    inFlight.current = true;
    const controller = new AbortController();
    controllerRef.current = controller;
    setStartedAt(performance.now());
    try {
      const image = await generateImage(params, controller.signal);
      setResult({
        url: URL.createObjectURL(image.blob),
        fileName: buildFileName(image.seed),
        seed: image.seed,
        elapsedMs: image.elapsedMs,
        params,
      });
    } finally {
      inFlight.current = false;
      controllerRef.current = null;
      setStartedAt(null);
    }
  }, []);

  return { busy: startedAt !== null, startedAt, result, generate };
}
