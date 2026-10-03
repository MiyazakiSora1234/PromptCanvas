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

/** Generation is pointless (and the button disabled) while the model is not usable. */
export function isModelBlocked(status: ServerStatus): boolean {
  if (status.kind !== "online") return false; // unknown/offline: let the request report the problem
  return status.health.status !== "ready";
}

export function isModelLoading(status: ServerStatus): boolean {
  return status.kind === "online" && (status.health.status === "loading" || status.health.status === "not_loaded");
}
