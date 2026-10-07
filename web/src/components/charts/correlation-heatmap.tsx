"use client";

import { useRef, useState } from "react";
import type { CorrelationData } from "@/lib/types";
import { ChartFrame, Tooltip } from "./chart-frame";

const W = 560, L = 104, TOP = 104;

export const strength = (r: number) => {
  const a = Math.abs(r);
  const word = a < 0.1 ? "negligible" : a < 0.3 ? "weak" : a < 0.5 ? "moderate" : a < 0.7 ? "strong" : "very strong";
  return a < 0.1 ? word : `${word} ${r > 0 ? "positive" : "negative"}`;
};
const short = (s: string, n = 16) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

/**
 * Diverging heatmap: blue = move together, orange = move opposite, dark = unrelated. Two hues with a
 * neutral middle, as a correlation needs. Every cell has a tooltip, and a ranked table is the twin.
 */
export function CorrelationHeatmap({ d }: { d: CorrelationData }) {
  const n = d.columns.length;
  const cell = Math.max(14, Math.min(44, Math.floor((W - L - 8) / n)));
  const gridW = cell * n, H = TOP + gridW + 46;
  const ref = useRef<HTMLDivElement>(null);
  const [hov, setHov] = useState<{ i: number; j: number; x: number; y: number } | null>(null);
  const showNums = cell >= 34;
  const fill = (r: number | null, diag: boolean) => (r === null || diag ? { c: "var(--color-surface-3)", o: diag ? 0.5 : 0.25 }
    : { c: r >= 0 ? "var(--color-sky)" : "var(--color-ember)", o: Math.abs(r) < 0.05 ? 0.06 : 0.12 + 0.88 * Math.abs(r) });

  return (
    <ChartFrame title="Correlations" subtitle="How strongly each pair of numeric columns moves together (-1 to +1). A strong link to the target is promising; a near-perfect link between two features means one is redundant."
      table={{ columns: ["Column A", "Column B", "r", "Strength"], rows: d.top_pairs.map((p) => [p.a, p.b, +p.r.toFixed(3), strength(p.r)]),
        note: `The ${d.top_pairs.length} strongest pairs, strongest first.${d.note ? " " + d.note : ""}` }}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block h-auto w-full max-w-[560px]" role="img" onPointerLeave={() => setHov(null)}
          aria-label={`Correlation heatmap of ${n} numeric columns. Open the table view for the strongest pairs.`}>
          {d.columns.map((c, j) => (
            <text key={`c${c}`} transform={`translate(${L + j * cell + cell / 2 + 3} ${TOP - 6}) rotate(-50)`} fontSize={10.5}
              className={c === d.target ? "fill-fg" : "fill-fg-subtle"} fontWeight={c === d.target ? 600 : 400}>{short(c)}{c === d.target ? " (target)" : ""}</text>
          ))}
          {d.columns.map((c, i) => (
            <text key={`r${c}`} x={L - 8} y={TOP + i * cell + cell / 2 + 4} textAnchor="end" fontSize={10.5}
              className={c === d.target ? "fill-fg" : "fill-fg-subtle"} fontWeight={c === d.target ? 600 : 400}>{short(c, 15)}</text>
          ))}
          {d.matrix.map((row, i) => row.map((r, j) => {
            const f = fill(r, i === j);
            const on = hov?.i === i && hov.j === j;
            return (
              <g key={`${i}-${j}`} tabIndex={0} role="img" aria-label={`${d.columns[i]} and ${d.columns[j]}: ${r === null ? "no value" : `r = ${r.toFixed(2)}, ${strength(r)}`}`}
                onPointerMove={(e) => { const p = ref.current!.getBoundingClientRect(); setHov({ i, j, x: e.clientX - p.left, y: e.clientY - p.top }); }}
                onFocus={() => setHov({ i, j, x: ((L + j * cell + cell / 2) / W) * (ref.current?.clientWidth ?? W), y: ((TOP + i * cell) / H) * (ref.current?.clientHeight ?? H) })}
                onBlur={() => setHov(null)}>
                <rect x={L + j * cell + 0.75} y={TOP + i * cell + 0.75} width={cell - 1.5} height={cell - 1.5} rx={3} fill={f.c} fillOpacity={f.o}
                  stroke={on ? "var(--color-fg)" : "none"} strokeWidth={1.5} />
                {showNums && r !== null && i !== j && <text x={L + j * cell + cell / 2} y={TOP + i * cell + cell / 2 + 4} textAnchor="middle" fontSize={11}
                  className={Math.abs(r) > 0.6 ? "fill-on-light" : "fill-fg-muted"}>{r.toFixed(2)}</text>}
              </g>
            );
          }))}
          <defs>
            <linearGradient id="corr-legend" x1="0" x2="1">
              <stop offset="0" stopColor="var(--color-ember)" /><stop offset="0.5" stopColor="var(--color-surface-3)" stopOpacity={0.5} /><stop offset="1" stopColor="var(--color-sky)" />
            </linearGradient>
          </defs>
          <rect x={L} y={TOP + gridW + 18} width={Math.min(200, gridW)} height={8} rx={4} fill="url(#corr-legend)" />
          <text x={L} y={TOP + gridW + 40} fontSize={10.5} className="fill-fg-subtle">-1 opposite</text>
          <text x={L + Math.min(200, gridW) / 2} y={TOP + gridW + 40} fontSize={10.5} textAnchor="middle" className="fill-fg-subtle">0</text>
          <text x={L + Math.min(200, gridW)} y={TOP + gridW + 40} fontSize={10.5} textAnchor="end" className="fill-fg-subtle">+1 together</text>
        </svg>
        {hov && (() => { const r = d.matrix[hov.i][hov.j]; return (
          <Tooltip x={hov.x} y={hov.y} title={hov.i === hov.j ? "A column with itself" : r === null ? "Not enough overlapping rows" : strength(r)}
            rows={[{ label: `${short(d.columns[hov.i], 20)} × ${short(d.columns[hov.j], 20)}`, value: r === null ? "–" : r.toFixed(3) }]} />); })()}
      </div>
      {d.note && <p className="mt-2 text-[12px] text-fg-subtle">{d.note}</p>}
    </ChartFrame>
  );
}
