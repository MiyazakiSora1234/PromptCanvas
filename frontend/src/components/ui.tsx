import type { ReactNode } from "react";

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section
      className={`min-w-0 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800 ${className}`}
    >
      {children}
    </section>
  );
}

export function Alert({ children }: { children: ReactNode }) {
  return (
    <div
      role="alert"
      className="rounded-md border border-red-600 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-400 dark:bg-red-950/40 dark:text-red-300"
    >
      {children}
    </div>
  );
}

interface FieldProps {
  id: string;
  label: ReactNode;
  error?: string;
  hint?: ReactNode;
  children: ReactNode;
}

/** Label + control + hint + error message. Controls link the error via aria-describedby={`${id}-error`}. */
export function Field({ id, label, error, hint, children }: FieldProps) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={id} className="text-sm font-semibold">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
      {error && (
        <p id={`${id}-error`} className="text-xs text-red-700 dark:text-red-300">
          {error}
        </p>
      )}
    </div>
  );
}
