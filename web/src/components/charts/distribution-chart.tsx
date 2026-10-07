"use client";

import { useRef, useState } from "react";
import { linear, niceTicks, tick } from "@/lib/scales";
import type { DistributionData } from "@/lib/types";
import { ChartFrame, Tooltip } from "./chart-frame";

const W = 520, H = 340, M = { l: 46, r: 14, t: 12, b: 34 };
const BOX_H = 40;                                  // strip under the histogram
const PLOT_B = H - M.b - BOX_H - 8;                // bottom of the histogram
const MAX_BAR = 24, GAP = 1.5;

const n2 = (v: number | null | undefined) => (v === null || v === undefined ? "–" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : String(+v.toFixed(2)));

/** Histogram + box plot on one shared x axis: shape, centre, spread and outliers at a glance. */
export function DistributionChart({ d, title, subtitle }: { d: DistributionData; title?: string; subtitle?: string }) {
  const lo = d.edges[0], hi = d.edges[d.edges.length - 1];
  const maxC = Math.max(...d.counts);
  const yt = niceTicks(0, maxC, 4), yTop = Math.max(maxC, yt[yt.length - 1]);
  const xs = linear(lo, hi, M.l, W - M.r), ys = linear(0, yTop, PLOT_B, M.t);
  const xt = niceTicks(lo, hi, 5);
  const ref = useRef<HTMLDivElement>(null);
  const [hov, setHov] = useState<{ kind: "bin" | "box"; i: number; x: number; y: number } | null>(null);
  const b = d.box;
  const by = PLOT_B + 8 + BOX_H / 2;

  const at = (e: React.PointerEvent) => { const r = ref.current!.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

  return (
    <ChartFrame title={title ?? `Distribution of ${d.column}`} subtitle={subtitle ?? "How the values are spread. The box shows the middle half of the rows; dots are unusually large or small values."}
      table={{ columns: ["From", "To", "Rows"], rows: [...d.counts.map((c, i) => [+d.edges[i].toFixed(3), +d.edges[i + 1].toFixed(3), c] as (string | number)[]),
        ...[["Minimum", b.min], ["25th percentile", b.q1], ["Median", b.median], ["75th percentile", b.q3], ["Maximum", b.max]].map(([k, v]) => [k as string, "", +(v as number).toFixed(3)] as (string | number)[])],
        note: `${d.n.toLocaleString()} values${d.missing ? `, ${d.missing.toLocaleString()} missing` : ""}. Mean ${n2(d.stats.mean)}, spread ${n2(d.stats.std)}, skewness ${n2(d.stats.skewness)}, ${b.outlier_count} unusual values.` }}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" onPointerLeave={() => setHov(null)}
          aria-label={`Histogram and box plot of ${d.column}: ${d.n} values, median ${n2(b.median)}, middle half from ${n2(b.q1)} to ${n2(b.q3)}, ${b.outlier_count} unusual values.`}>
          {yt.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={ys(t)} y2={ys(t)} stroke="var(--color-line)" strokeWidth={1} />
              <text x={M.l - 8} y={ys(t) + 4} textAnchor="end" className="fill-fg-subtle" fontSize={11}>{t}</text>
            </g>
          ))}
          {xt.map((t) => <text key={t} x={xs(t)} y={H - M.b + 18} textAnchor="middle" className="fill-fg-subtle" fontSize={11}>{tick(t)}</text>)}
          <text x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>{d.column}</text>
          {d.counts.map((c, i) => {
            if (!c) return null;
            const x0 = xs(d.edges[i]), x1 = xs(d.edges[i + 1]), band = x1 - x0, bw = Math.min(MAX_BAR, band - GAP), cx = x0 + band / 2;
            const top = ys(c), h = PLOT_B - top, r = Math.min(4, h, bw / 2);
            return (
              <g key={i} tabIndex={0} role="img" aria-label={`${c} rows from ${n2(d.edges[i])} to ${n2(d.edges[i + 1])}`}
                onPointerMove={(e) => setHov({ kind: "bin", i, ...at(e) })}
                onFocus={() => setHov({ kind: "bin", i, x: (cx / W) * (ref.current?.clientWidth ?? W), y: (top / H) * (ref.current?.clientHeight ?? H) + 14 })}
                onBlur={() => setHov(null)}>
                <rect x={x0} y={M.t} width={band} height={PLOT_B - M.t} fill="transparent" />
                <path d={`M${cx - bw / 2} ${PLOT_B} V${top + r} Q${cx - bw / 2} ${top} ${cx - bw / 2 + r} ${top} H${cx + bw / 2 - r} Q${cx + bw / 2} ${top} ${cx + bw / 2} ${top + r} V${PLOT_B} Z`}
                  fill="var(--color-sky)" fillOpacity={hov?.kind === "bin" && hov.i === i ? 1 : 0.8} />
              </g>
            );
          })}
          <g tabIndex={0} role="img" aria-label={`Box plot: median ${n2(b.median)}, middle half ${n2(b.q1)} to ${n2(b.q3)}`}
            onPointerMove={(e) => setHov({ kind: "box", i: 0, ...at(e) })}
            onFocus={() => setHov({ kind: "box", i: 0, x: (xs(b.median) / W) * (ref.current?.clientWidth ?? W), y: (by / H) * (ref.current?.clientHeight ?? H) })}
            onBlur={() => setHov(null)}>
            <rect x={M.l} y={by - BOX_H / 2} width={W - M.l - M.r} height={BOX_H} fill="transparent" />
            <line x1={xs(b.whisker_low)} x2={xs(b.whisker_high)} y1={by} y2={by} stroke="var(--color-fg-subtle)" strokeWidth={1.5} />
            {[b.whisker_low, b.whisker_high].map((v, i) => <line key={i} x1={xs(v)} x2={xs(v)} y1={by - 6} y2={by + 6} stroke="var(--color-fg-subtle)" strokeWidth={1.5} />)}
            <rect x={xs(b.q1)} y={by - 9} width={Math.max(2, xs(b.q3) - xs(b.q1))} height={18} rx={4} fill="var(--color-sky)" fillOpacity={0.3} stroke="var(--color-sky)" strokeWidth={1.5} />
            <line x1={xs(b.median)} x2={xs(b.median)} y1={by - 9} y2={by + 9} stroke="var(--color-fg)" strokeWidth={2} />
            {b.outliers.map((v, i) => <circle key={i} cx={xs(v)} cy={by} r={2.8} fill="var(--color-sky)" fillOpacity={0.7} stroke="var(--color-surface)" strokeWidth={1} />)}
          </g>
        </svg>
        {hov?.kind === "bin" && <Tooltip x={hov.x} y={hov.y} rows={[{ label: "Rows", value: d.counts[hov.i].toLocaleString() },
          { label: "Range", value: `${n2(d.edges[hov.i])} to ${n2(d.edges[hov.i + 1])}` }]} />}
        {hov?.kind === "box" && <Tooltip x={hov.x} y={hov.y} title="Summary" rows={[{ label: "Median", value: n2(b.median) }, { label: "Middle half", value: `${n2(b.q1)} to ${n2(b.q3)}` },
          { label: "Typical range", value: `${n2(b.whisker_low)} to ${n2(b.whisker_high)}` }, { label: "Unusual values", value: String(b.outlier_count) }]} />}
      </div>
      <p className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-fg-muted">
        <span>Median <span className="font-mono text-fg tabular">{n2(b.median)}</span></span>
        <span>Mean <span className="font-mono text-fg tabular">{n2(d.stats.mean)}</span></span>
        <span>Skewness <span className="font-mono text-fg tabular">{n2(d.stats.skewness)}</span></span>
        <span>Unusual values <span className="font-mono text-fg tabular">{b.outlier_count}</span></span>
        {d.missing > 0 && <span className="text-warn">{d.missing.toLocaleString()} missing</span>}
      </p>
    </ChartFrame>
  );
}
