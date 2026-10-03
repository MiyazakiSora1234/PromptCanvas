import type { ChangeEvent } from "react";
import type { FieldErrors, FormValues } from "../lib/validation";
import { inputClass } from "./styles";

/** Props wiring a text input / textarea / select to a form value, with error a11y attributes. */
export function makeControlProps(
  values: FormValues,
  errors: FieldErrors,
  onValueChange: (name: keyof FormValues, value: string) => void,
) {
  return (name: keyof FormValues) => ({
    id: name,
    name,
    value: values[name],
    onChange: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
      onValueChange(name, e.target.value),
    "aria-invalid": errors[name] ? true : undefined,
    "aria-describedby": errors[name] ? `${name}-error` : undefined,
    className: inputClass,
  });
}
