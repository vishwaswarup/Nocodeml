"use client";

import { clsx } from "clsx";
import { Check } from "lucide-react";
import { motion } from "motion/react";

export type SectionState = "done" | "current" | "stale" | "todo";
export type SectionItem = { n: number; label: string; state: SectionState };

/** The workspace's left rail: pipeline stages 0-7 with their state. */
export function SectionRail({ items, onSelect }: { items: SectionItem[]; onSelect?: (n: number) => void }) {
  return (
    <nav aria-label="Pipeline sections" className="flex flex-col gap-0.5">
      {items.map((s) => {
        const current = s.state === "current";
        return (
          <button
            key={s.n}
            type="button"
            aria-current={current ? "step" : undefined}
            onClick={() => onSelect?.(s.n)}
            className={clsx(
              "relative flex h-10 items-center gap-3 rounded-control px-3 text-left text-[14px] transition-colors",
              current ? "text-fg" : "text-fg-muted hover:bg-white/[0.04] hover:text-fg",
            )}
          >
            {current && (
              <motion.span
                layoutId="rail-active"
                className="absolute inset-0 rounded-control bg-surface-2"
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
              />
            )}
            <span
              className={clsx(
                "relative inline-flex size-[22px] shrink-0 items-center justify-center rounded-[6px] font-mono text-[12px] tabular",
                s.state === "done" && "bg-white/10 text-fg",
                s.state === "current" && "bg-fg text-on-light",
                s.state === "stale" && "bg-warn/15 text-warn",
                s.state === "todo" && "text-fg-subtle ring-1 ring-line-strong",
              )}
            >
              {s.state === "done" ? <Check className="size-3.5" strokeWidth={2.5} /> : s.n}
            </span>
            <span className="relative flex-1">{s.label}</span>
            {s.state === "stale" && (
              <span className="relative font-mono text-[10px] tracking-wider text-warn uppercase">outdated</span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
