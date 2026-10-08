import { useCallback, useEffect, useRef, useState } from "react";
import { cancelJob, generateImages } from "../api/client";
import type { GenerateRequest, GenerateResponse } from "../api/types";
import { base64ToBlob, buildFileName } from "../lib/images";
import type { SeriesItem } from "../lib/series";

export interface ResultImage {
  /** Object URL; revoked when a new series replaces the results or on unmount. */
  url: string;
  blob: Blob;
  fileName: string;
  seed: number;
  /** The request that produced this image (settings, timing, translation). */
  response: Omit<GenerateResponse, "images">;
  request: GenerateRequest;
  /** 1-based pose image used (null without pose references). */
  poseNumber: number | null;
}

export interface GenerationResult {
  images: ResultImage[];
  /** Images removed by the safety checker. */
  filteredCount: number;
}

export interface Progress {
  /** 1-based number of the image being generated. */
  current: number;
  total: number;
  startedAt: number;
}

const EXTENSIONS: Record<string, string> = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };

/**
 * Sends a series of requests (see buildSeries) one at a time — barely slower than batching on the
 * GPU, but every finished image shows up at once and stopping keeps them. A second call while one
 * is in flight is ignored, so double clicks / repeated Ctrl+Enter never send duplicates.
 *
 * `generate` rejects with ApiError / NetworkError when a request fails (the images so far are
 * kept); `cancel` stops the series.
 */
export function useImageGeneration() {
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const inFlight = useRef(false);
  const stopRequested = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);
  const jobRef = useRef<string | null>(null);
  const urlsRef = useRef<string[]>([]);

  // Abort a pending request and free the images if the component goes away.
  useEffect(
    () => () => {
      controllerRef.current?.abort();
      urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    },
    [],
  );

  const generate = useCallback(async (series: SeriesItem[]): Promise<void> => {
    if (inFlight.current) return;
    inFlight.current = true;
    stopRequested.current = false;
    setCancelling(false);
    try {
      for (let current = 1; current <= series.length && !stopRequested.current; current++) {
        const { request, poseNumber } = series[current - 1]!;
        const controller = new AbortController();
        controllerRef.current = controller;
        jobRef.current = newJobId();
        setProgress({ current, total: series.length, startedAt: performance.now() });

        const { images, ...response } = await generateImages({ ...request, job_id: jobRef.current }, controller.signal);
        const now = new Date();
        const added = images.map((img) => {
          const blob = base64ToBlob(img.data, img.mime_type);
          return {
            url: URL.createObjectURL(blob),
            blob,
            fileName: buildFileName(img.seed, EXTENSIONS[img.mime_type] ?? "img", now),
            seed: img.seed,
            response,
            request,
            poseNumber,
          };
        });
        if (current === 1) {
          // The first finished image replaces the previous series (shown dimmed until now).
          urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
          urlsRef.current = [];
        }
        urlsRef.current.push(...added.map((img) => img.url));
        setResult((prev) => ({
          images: current === 1 || !prev ? added : [...prev.images, ...added],
          filteredCount: (current === 1 || !prev ? 0 : prev.filteredCount) + response.filtered_count,
        }));
      }
    } finally {
      inFlight.current = false;
      controllerRef.current = null;
      jobRef.current = null;
      setCancelling(false);
      setProgress(null);
    }
  }, []);

  /**
   * Stop the series. The running request ends within one step on the server and the pending
   * `generate` rejects with ApiError code "cancelled". If the server doesn't know the job yet (or
   * can't be reached), the request is abandoned locally instead (`generate` rejects with AbortError).
   */
  const cancel = useCallback(async (): Promise<void> => {
    if (!inFlight.current || cancelling) return;
    stopRequested.current = true;
    setCancelling(true);
    const jobId = jobRef.current;
    if (!jobId) return;
    const stopped = await cancelJob(jobId).catch(() => false);
    if (!stopped && jobRef.current === jobId) controllerRef.current?.abort();
  }, [cancelling]);

  return { busy: progress !== null, progress, result, generate, cancel, cancelling };
}

/** Random id for the cancel endpoint (crypto.randomUUID needs a secure context; LAN http isn't one). */
function newJobId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
