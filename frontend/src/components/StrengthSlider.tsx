import type { ReactNode } from "react";

interface StrengthSliderProps {
  id: string;
  label: string;
  value: string;
  min: number;
  max: number;
  step?: number;
  /** Digits shown in the value readout. */
  decimals?: number;
  hint?: ReactNode;
  error?: string;
  onChange: (value: string) => void;
}

export function StrengthSlider({
  id,
  label,
  value,
  min,
  max,
  step = 0.05,
  decimals = 2,
  hint,
  error,
  onChange,
}: StrengthSliderProps) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="flex justify-between text-sm font-semibold">
        <span>{label}</span>
        <span className="tabular-nums">{Number(value).toFixed(decimals)}</span>
      </label>
      <input
        id={id}
        name={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-invalid={error ? true : undefined}
        onChange={(e) => onChange(e.target.value)}
        className="accent-indigo-600"
      />
      {hint && <p className="text-xs text-slate-500 dark:text-slate-400">{hint}</p>}
      {error && <p className="text-xs text-red-700 dark:text-red-300">{error}</p>}
    </div>
  );
}
