import { useCallback, useMemo, useState } from "react";
import type { AppConfig, PresetSettings } from "../api/types";
import { builtInPresets, loadUserPresets, saveUserPresets, type Preset } from "../lib/presets";

export interface PresetStore {
  builtIn: Preset[];
  user: Preset[];
  find: (id: string) => Preset | undefined;
  /** Save under `label`; an existing user preset with the same name is overwritten. Returns false if storage failed. */
  save: (label: string, settings: PresetSettings) => { ok: boolean; overwritten: boolean };
  remove: (id: string) => boolean;
}

export function usePresets(config: AppConfig): PresetStore {
  const builtIn = useMemo(() => builtInPresets(config), [config]);
  const [user, setUser] = useState<Preset[]>(() => loadUserPresets());

  const find = useCallback((id: string) => [...builtIn, ...user].find((p) => p.id === id), [builtIn, user]);

  const save = useCallback(
    (label: string, settings: PresetSettings) => {
      const existing = user.find((p) => p.label === label);
      const preset: Preset = {
        id: existing?.id ?? `user-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        label,
        description: "",
        builtIn: false,
        settings,
      };
      const next = existing ? user.map((p) => (p.id === existing.id ? preset : p)) : [...user, preset];
      const ok = saveUserPresets(next);
      if (ok) setUser(next);
      return { ok, overwritten: existing !== undefined };
    },
    [user],
  );

  const remove = useCallback(
    (id: string) => {
      const next = user.filter((p) => p.id !== id);
      const ok = saveUserPresets(next);
      if (ok) setUser(next);
      return ok;
    },
    [user],
  );

  return { builtIn, user, find, save, remove };
}
