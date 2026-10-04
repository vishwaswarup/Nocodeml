"use client";

import { clsx } from "clsx";
import { motion } from "motion/react";
import { useId } from "react";

export type ChoiceOption<T extends string> = { value: T; label: string; description?: string; recommended?: boolean };

/** Segmented pill control for short option sets (e.g. imputation strategy). */
export function Segmented<T extends string>({
  options, value, onChange, label,
}: { options: ChoiceOption<T>[]; value: T; onChange: (v: T) => void; label: string }) {
  const id = useId();
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap gap-1 rounded-full bg-surface-2 p-1">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={clsx(
              "relative h-8 rounded-full px-3.5 text-[13px] font-medium transition-colors",
              on ? "text-on-light" : "text-fg-muted hover:text-fg",
            )}
          >
            {on && (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 rounded-full bg-fg"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Larger option rows with explanations, for decisions that need a reason. */
export function ChoiceList<T extends string>({
  options, value, onChange, label,
}: { options: ChoiceOption<T>[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-col gap-2">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={clsx(
              "flex items-start gap-3 rounded-control px-4 py-3 text-left ring-1 transition-[background-color,box-shadow]",
              on ? "bg-surface-2 ring-fg/70" : "bg-surface ring-line hover:ring-line-strong",
            )}
          >
            <span
              className={clsx(
                "mt-0.5 inline-flex size-[18px] shrink-0 items-center justify-center rounded-full ring-1 transition-colors",
                on ? "bg-fg ring-fg" : "ring-fg-subtle",
              )}
              aria-hidden
            >
              {on && <span className="size-1.5 rounded-full bg-on-light" />}
            </span>
            <span className="flex-1">
              <span className="flex items-center gap-2 text-[15px] text-fg">
                {o.label}
                {o.recommended && (
                  <span className="rounded-full bg-ember/12 px-2 py-0.5 text-[11px] font-medium text-ember">Recommended</span>
                )}
              </span>
              {o.description && <span className="mt-0.5 block text-[13px] leading-snug text-fg-muted">{o.description}</span>}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={clsx("relative h-6 w-10 rounded-full transition-colors", checked ? "bg-fg" : "bg-surface-3")}
    >
      <motion.span
        className={clsx("absolute top-1 size-4 rounded-full", checked ? "bg-on-light" : "bg-fg-muted")}
        animate={{ left: checked ? 20 : 4 }}
        transition={{ type: "spring", stiffness: 600, damping: 35 }}
      />
    </button>
  );
}
