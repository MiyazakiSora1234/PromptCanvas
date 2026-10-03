import { useRef, type KeyboardEvent, type ReactNode } from "react";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  /** Short marker after the label, e.g. "●" or "(2)". */
  badge?: string;
  hasError?: boolean;
}

interface TabsProps<T extends string> {
  idPrefix: string;
  label: string;
  items: TabItem<T>[];
  active: T;
  onChange: (id: T) => void;
}

const tabId = (prefix: string, id: string) => `${prefix}-tab-${id}`;
const panelId = (prefix: string, id: string) => `${prefix}-panel-${id}`;

/** WAI-ARIA tabs: arrow keys / Home / End move between tabs. */
export function Tabs<T extends string>({ idPrefix, label, items, active, onChange }: TabsProps<T>) {
  const refs = useRef(new Map<T, HTMLButtonElement>());

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const index = items.findIndex((t) => t.id === active);
    const next =
      e.key === "ArrowRight"
        ? (index + 1) % items.length
        : e.key === "ArrowLeft"
          ? (index - 1 + items.length) % items.length
          : e.key === "Home"
            ? 0
            : e.key === "End"
              ? items.length - 1
              : null;
    if (next === null) return;
    e.preventDefault();
    const target = items[next]!;
    onChange(target.id);
    refs.current.get(target.id)?.focus();
  };

  return (
    <div
      role="tablist"
      aria-label={label}
      onKeyDown={handleKeyDown}
      className="flex gap-1 overflow-x-auto overflow-y-hidden border-b border-slate-200 dark:border-slate-700"
    >
      {items.map((item) => {
        const selected = item.id === active;
        return (
          <button
            key={item.id}
            ref={(el) => {
              if (el) refs.current.set(item.id, el);
              else refs.current.delete(item.id);
            }}
            type="button"
            role="tab"
            id={tabId(idPrefix, item.id)}
            aria-controls={panelId(idPrefix, item.id)}
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.id)}
            className={`-mb-px flex shrink-0 items-center gap-1 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium ${
              selected
                ? "border-indigo-600 text-indigo-700 dark:border-indigo-400 dark:text-indigo-300"
                : "border-transparent text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100"
            }`}
          >
            {item.label}
            {item.badge && <span className="text-xs text-indigo-600 dark:text-indigo-300">{item.badge}</span>}
            {item.hasError && (
              <span className="size-2 rounded-full bg-red-600" aria-label="入力エラーあり" role="img" />
            )}
          </button>
        );
      })}
    </div>
  );
}

export function TabPanel({
  idPrefix,
  id,
  active,
  children,
}: {
  idPrefix: string;
  id: string;
  active: boolean;
  children: ReactNode;
}) {
  return (
    <div
      role="tabpanel"
      id={panelId(idPrefix, id)}
      aria-labelledby={tabId(idPrefix, id)}
      hidden={!active}
      className="flex flex-col gap-4 pt-1"
    >
      {children}
    </div>
  );
}
