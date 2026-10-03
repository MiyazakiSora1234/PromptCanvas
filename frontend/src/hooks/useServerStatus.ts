import { useCallback, useEffect, useState } from "react";
import { fetchHealth } from "../api/client";
import type { Health } from "../api/types";

export type ServerStatus = { kind: "unknown" } | { kind: "offline" } | { kind: "online"; health: Health };

const LOADING_POLL_MS = 3000;
const OFFLINE_POLL_MS = 5000;

/**
 * Polls /api/health while the model is loading or the server is unreachable,
 * and stops once it is ready or has failed. `refresh()` forces a new check.
 */
export function useServerStatus(): { status: ServerStatus; refresh: () => void } {
  const [status, setStatus] = useState<ServerStatus>({ kind: "unknown" });
  const [generation, setGeneration] = useState(0);
  const refresh = useCallback(() => setGeneration((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;

    const poll = async () => {
      let nextPollMs: number | null = null;
      try {
        const health = await fetchHealth(controller.signal);
        setStatus({ kind: "online", health });
        if (health.status === "loading" || health.status === "not_loaded") nextPollMs = LOADING_POLL_MS;
      } catch {
        if (controller.signal.aborted) return;
        setStatus({ kind: "offline" });
        nextPollMs = OFFLINE_POLL_MS;
      }
      if (nextPollMs !== null) timer = window.setTimeout(poll, nextPollMs);
    };
    void poll();

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [generation]);

  return { status, refresh };
}

export function isModelLoading(status: ServerStatus): boolean {
  return status.kind === "online" && (status.health.status === "loading" || status.health.status === "not_loaded");
}

/**
 * The button is disabled only while a model is loading. A failed model does not block:
 * the user can retry or pick another model. Unknown/offline: let the request report it.
 */
export function isModelBlocked(status: ServerStatus): boolean {
  return isModelLoading(status);
}

/** Downloaded model ids, or null while unknown (don't warn about downloads we can't confirm). */
export function cachedModels(status: ServerStatus): string[] | null {
  return status.kind === "online" ? status.health.cached_models : null;
}

/** Catalog id of the model on the GPU, or null if none is ready. */
export function loadedModel(status: ServerStatus): string | null {
  return status.kind === "online" && status.health.status === "ready" ? status.health.model : null;
}
