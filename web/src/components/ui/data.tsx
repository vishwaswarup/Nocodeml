"use client";

import { clsx } from "clsx";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { motion } from "motion/react";
import type { ReactNode } from "react";
import { StatusIcon, type Status } from "./badge";

export type Column = { key: string; label: string; dtype?: string; numeric?: boolean };

/** Dense, readable data table: tabular numbers, visible missing values, sortable headers. */
export function DataTable({
  columns, rows, sortBy, descending, onSort, caption,
}: {
  columns: Column[];
  rows: Record<string, unknown>[];
  sortBy?: string;
  descending?: boolean;
  onSort?: (key: string) => void;
  caption?: string;
}) {
  return (
    <div className="overflow-x-auto rounded-card bg-surface ring-1 ring-line">
      <table className="w-full border-collapse text-[13px]">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line">
            {columns.map((c) => {
              const active = sortBy === c.key;
              const Icon = active ? (descending ? ArrowDown : ArrowUp) : ArrowUpDown;
              return (
                <th
                  key={c.key}
                  scope="col"
                  aria-sort={active ? (descending ? "descending" : "ascending") : undefined}
                  className={clsx("px-3 py-2.5 font-normal whitespace-nowrap", c.numeric ? "text-right" : "text-left")}
                >
                  <button
                    type="button"
                    onClick={() => onSort?.(c.key)}
                    className={clsx(
                      "group inline-flex items-center gap-1.5 text-fg-muted hover:text-fg",
                      c.numeric && "flex-row-reverse",
                    )}
                  >
                    <span className="text-fg">{c.label}</span>
                    {c.dtype && <span className="font-mono text-[11px] text-fg-subtle">{c.dtype}</span>}
                    <Icon className={clsx("size-3", active ? "text-fg" : "opacity-0 group-hover:opacity-60")} />
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 last:border-0 hover:bg-white/[0.025]">
              {columns.map((c) => {
                const v = r[c.key];
                const missing = v === null || v === undefined || v === "";
                return (
                  <td key={c.key} className={clsx("px-3 py-2 whitespace-nowrap", c.numeric ? "text-right tabular font-mono text-[12.5px]" : "text-fg")}>
                    {missing ? (
                      <span className="inline-flex items-center rounded-[5px] bg-warn/10 px-1.5 font-mono text-[11px] text-warn">
                        missing
                      </span>
                    ) : (
                      String(v)
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Metric({ label, value, sub, emphasis = false }: {
  label: string; value: ReactNode; sub?: ReactNode; emphasis?: boolean;
}) {
  return (
    <div className="rounded-card bg-surface p-4">
      <p className="text-[13px] text-fg-muted">{label}</p>
      <p className={clsx("mt-1.5 tabular", emphasis ? "text-[32px] leading-none" : "text-[24px] leading-tight")}>{value}</p>
      {sub && <p className="mt-1.5 text-[12px] text-fg-subtle">{sub}</p>}
    </div>
  );
}

/** Horizontal bar for a 0..1 metric, with a baseline marker. */
export function MetricBar({ label, value, baseline }: { label: string; value: number; baseline?: number }) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-[13px]">
        <span className="text-fg-muted">{label}</span>
        <span className="font-mono tabular text-fg">{value.toFixed(3)}</span>
      </div>
      <div className="relative h-1.5 rounded-full bg-surface-3">
        <motion.div
          className="h-full rounded-full bg-fg"
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        />
        {baseline !== undefined && (
          <span
            className="absolute -top-1 h-3.5 w-px bg-ember"
            style={{ left: `${baseline * 100}%` }}
            title={`Baseline ${baseline.toFixed(3)}`}
          />
        )}
      </div>
    </div>
  );
}

export function QualityRow({ status, title, detail }: { status: Status; title: string; detail?: string }) {
  return (
    <div className="flex gap-3 py-3">
      <StatusIcon status={status} />
      <div className="min-w-0">
        <p className="text-[15px] text-fg">{title}</p>
        {detail && <p className="mt-0.5 text-[13px] leading-snug text-fg-muted">{detail}</p>}
      </div>
    </div>
  );
}

/** Thin progress line, as at the top of Palette's form; fills with the signature gradient. */
export function ProgressLine({ value }: { value: number }) {
  return (
    <div className="h-[3px] w-full overflow-hidden rounded-full bg-surface-3" role="progressbar"
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}>
      <motion.div
        className="h-full rounded-full signature-gradient-x"
        animate={{ width: `${value * 100}%` }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      />
    </div>
  );
}
