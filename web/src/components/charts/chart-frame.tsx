"use client";

import { Table2 } from "lucide-react";
import { useState, type ReactNode } from "react";

export type TableView = { columns: string[]; rows: (string | number)[][]; note?: string };

/**
 * Card around a chart: title, subtitle, and a "Table view" twin. Every chart has one, so no value
 * depends on hovering or on seeing colour.
 */
export function ChartFrame({ title, subtitle, table, children }: {
  title: string; subtitle?: string; table: TableView; children: ReactNode;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className="rounded-card bg-surface p-5">
      <figcaption className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="text-[16px] tracking-[-0.02em]">{title}</p>
          {subtitle && <p className="mt-0.5 text-[12.5px] leading-snug text-fg-muted">{subtitle}</p>}
        </div>
        <button type="button" aria-pressed={asTable} onClick={() => setAsTable(!asTable)}
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 text-[12px] transition-colors ${
            asTable ? "bg-fg text-on-light" : "bg-surface-3 text-fg-muted hover:text-fg"}`}>
          <Table2 className="size-3.5" aria-hidden /> Table
        </button>
      </figcaption>
      {asTable ? (
        <div className="max-h-[340px] overflow-auto rounded-control ring-1 ring-line">
          <table className="w-full text-[12.5px]">
            <thead className="sticky top-0 bg-surface-2">
              <tr>{table.columns.map((c) => <th key={c} scope="col" className="px-3 py-2 text-left font-normal text-fg-muted">{c}</th>)}</tr>
            </thead>
            <tbody>
              {table.rows.map((r, i) => (
                <tr key={i} className="border-t border-line/60">
                  {r.map((v, j) => <td key={j} className={`px-3 py-1.5 ${typeof v === "number" ? "font-mono tabular" : ""}`}>{v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          {table.note && <p className="border-t border-line/60 px-3 py-2 text-[12px] text-fg-subtle">{table.note}</p>}
        </div>
      ) : children}
    </figure>
  );
}

export type TipRow = { label: string; value: string };

/** Hover readout. Values lead (strong), labels follow (muted). Text only: never innerHTML. */
export function Tooltip({ x, y, title, rows }: { x: number; y: number; title?: string; rows: TipRow[] }) {
  return (
    <div role="status" className="pointer-events-none absolute z-10 min-w-[120px] rounded-control bg-surface-3 px-3 py-2 shadow-[0_8px_24px_rgb(0_0_0/0.45)] ring-1 ring-line-strong"
      style={{ left: x, top: y, transform: "translate(-50%, calc(-100% - 12px))" }}>
      {title && <p className="mb-1 text-[11.5px] text-fg-muted">{title}</p>}
      {rows.map((r) => (
        <p key={r.label} className="flex items-baseline justify-between gap-4 text-[12.5px]">
          <span className="text-fg-muted">{r.label}</span>
          <span className="font-mono tabular text-fg">{r.value}</span>
        </p>
      ))}
    </div>
  );
}

/** Pointer position relative to an element, in that element's own CSS pixels. */
export function localPoint(e: { clientX: number; clientY: number }, el: Element) {
  const r = el.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
}
