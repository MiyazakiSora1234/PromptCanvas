import { useEffect, useState } from "react";
import { fetchConfig } from "../api/client";
import type { AppConfig } from "../api/types";

export type ConfigState = { kind: "loading" } | { kind: "ready"; config: AppConfig } | { kind: "unreachable" };

/** Loads input limits/defaults from the server, retrying until it answers. */
export function useAppConfig(retryMs = 5000): ConfigState {
  const [state, setState] = useState<ConfigState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;

    const load = async () => {
      try {
        const config = await fetchConfig(controller.signal);
        setState({ kind: "ready", config });
      } catch {
        if (controller.signal.aborted) return;
        setState({ kind: "unreachable" });
        timer = window.setTimeout(load, retryMs);
      }
    };
    void load();

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [retryMs]);

  return state;
}
