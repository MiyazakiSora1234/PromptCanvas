import { useCallback, useEffect, useRef, useState } from "react";
import { cancelJob, generateImages } from "../api/client";
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
 * `generate` rejects with ApiError / NetworkError on failure; `cancel` stops the running one.
 */
export function useImageGeneration() {
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const inFlight = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);
  const jobRef = useRef<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

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
    jobRef.current = newJobId();
    setCancelling(false);
    setStartedAt(performance.now());
    try {
      const { images, ...response } = await generateImages({ ...request, job_id: jobRef.current }, controller.signal);
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
      jobRef.current = null;
      setCancelling(false);
      setStartedAt(null);
    }
  }, []);

  /**
   * Stop the running generation. The server ends it within one step and the pending `generate`
   * rejects with ApiError code "cancelled". If the server doesn't know the job yet (or can't be
   * reached), the request is abandoned locally instead (`generate` rejects with AbortError).
   */
  const cancel = useCallback(async (): Promise<void> => {
    const jobId = jobRef.current;
    if (!jobId || cancelling) return;
    setCancelling(true);
    const stopped = await cancelJob(jobId).catch(() => false);
    if (!stopped && jobRef.current === jobId) controllerRef.current?.abort();
  }, [cancelling]);

  return { busy: startedAt !== null, startedAt, result, generate, cancel, cancelling };
}

/** Random id for the cancel endpoint (crypto.randomUUID needs a secure context; LAN http isn't one). */
function newJobId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
