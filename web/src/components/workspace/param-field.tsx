"use client";

import { Switch } from "@/components/ui/choice";
import { Field } from "@/components/ui/field";
import { Select } from "@/components/ui/select";
import type { HyperParamInfo } from "@/lib/types";

const NONE = "__none__";
const show = (v: unknown) => (v === null || v === undefined ? "" : String(v));

/** One hyperparameter, rendered from registry metadata (kind, range, choices). */
export function ParamField({ hp, value, onChange, disabled, note }: {
  hp: HyperParamInfo; value: unknown; onChange: (v: unknown) => void; disabled?: boolean; note?: string;
}) {
  const hint = [hp.description, note].filter(Boolean).join(" ") || undefined;
  if (hp.kind === "bool") {
    return (
      <div className="flex items-center justify-between gap-3 rounded-control bg-surface-2 px-3.5 py-2.5">
        <div>
          <p className="text-[14px]">{hp.name}</p>
          {hint && <p className="text-[12px] text-fg-subtle">{hint}</p>}
        </div>
        <Switch checked={Boolean(value)} onChange={onChange} label={hp.name} />
      </div>
    );
  }
  if (hp.kind === "choice") {
    return (
      <Select label={hp.name} value={value === null || value === undefined ? NONE : String(value)} disabled={disabled}
        onChange={(e) => onChange(e.target.value === NONE ? null : e.target.value)} hint={hint}>
        {hp.choices.map((c) => <option key={String(c)} value={c === null ? NONE : String(c)}>{c === null ? "None" : String(c)}</option>)}
      </Select>
    );
  }
  const range = hp.min !== null || hp.max !== null ? `${hp.min ?? "…"} to ${hp.max ?? "…"}` : undefined;
  return (
    <Field label={hp.name} type="number" disabled={disabled} suffix={range} value={show(value)}
      placeholder={hp.kind === "optional_int" ? "No limit" : undefined}
      step={hp.kind === "float" ? "any" : 1} min={hp.min ?? undefined} max={hp.max ?? undefined}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(hp.kind === "optional_int" ? null : hp.default);
        const n = Number(raw);
        if (Number.isFinite(n)) onChange(hp.kind === "float" ? n : Math.round(n));
      }}
      hint={hint} />
  );
}
