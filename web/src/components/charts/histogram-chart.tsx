"use client";

import { useRef, useState } from "react";
import { linear, niceTicks, tick } from "@/lib/scales";
import { ChartFrame, Tooltip } from "./chart-frame";

const W = 520, H = 310, M = { l: 44, r: 14, t: 26, b: 40 };
const BINS = 20, MAX_BAR = 24, GAP = 2;

/** Distribution of errors (actual - predicted). A centred, narrow spread means unbiased, consistent predictions. */
export function HistogramChart({ values }: { values: number[] }) {
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const counts = Array<number>(BINS).fill(0);
  values.forEach((v) => { counts[Math.min(BINS - 1, Math.floor(((v - lo) / span) * BINS))]++; });
  const edge = (i: number) => lo + (span * i) / BINS;
  const maxC = Math.max(...counts);
  const yt = niceTicks(0, maxC, 4);
  const yTop = Math.max(maxC, yt[yt.length - 1]);   // the scale must always cover the tallest bar
  const x = linear(lo, hi, M.l, W - M.r), y = linear(0, yTop, H - M.b, M.t);
  const band = (W - M.l - M.r) / BINS, bw = Math.min(MAX_BAR, band - GAP);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  const ref = useRef<HTMLDivElement>(null);
  const [hov, setHov] = useState<{ i: number; x: number; y: number } | null>(null);
  const xt = niceTicks(lo, hi, 5);

  return (
    <ChartFrame title="Error distribution" subtitle="How far off each prediction was (actual minus predicted). Centred on 0 and narrow is best."
      table={{ columns: ["From", "To", "Rows"], rows: counts.map((c, i) => [+edge(i).toFixed(2), +edge(i + 1).toFixed(2), c]),
        note: `Average error ${mean.toFixed(2)} (bias), typical spread ${sd.toFixed(2)}.` }}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img"
          aria-label={`Histogram of ${values.length} prediction errors; average ${mean.toFixed(1)}, spread ${sd.toFixed(1)}.`}
          onPointerLeave={() => setHov(null)}>
          {yt.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--color-line)" strokeWidth={1} />
              <text x={M.l - 8} y={y(t) + 4} textAnchor="end" className="fill-fg-subtle" fontSize={11}>{t}</text>
            </g>
          ))}
          {xt.map((t) => <text key={t} x={x(t)} y={H - M.b + 18} textAnchor="middle" className="fill-fg-subtle" fontSize={11}>{tick(t)}</text>)}
          <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>Error</text>
          {lo <= 0 && hi >= 0 && (
            <g>
              <line x1={x(0)} x2={x(0)} y1={M.t} y2={H - M.b} stroke="var(--color-fg-subtle)" strokeWidth={1} />
              <text x={x(0)} y={M.t - 9} textAnchor="middle" className="fill-fg-muted" fontSize={11}>no error</text>
            </g>
          )}
          {counts.map((c, i) => {
            if (!c) return null;
            const cx = M.l + band * i + band / 2, top = y(c), h = H - M.b - top, r = Math.min(4, h);
            return (
              <g key={i} tabIndex={0} role="img" aria-label={`${c} rows with error from ${edge(i).toFixed(1)} to ${edge(i + 1).toFixed(1)}`}
                onPointerMove={(e) => { const p = ref.current!.getBoundingClientRect(); setHov({ i, x: e.clientX - p.left, y: e.clientY - p.top }); }}
                onFocus={() => setHov({ i, x: (cx / W) * (ref.current?.clientWidth ?? W), y: (top / H) * (ref.current?.clientHeight ?? H) + 12 })}
                onBlur={() => setHov(null)}>
                <rect x={cx - band / 2} y={M.t} width={band} height={H - M.b - M.t} fill="transparent" />
                <path d={`M${cx - bw / 2} ${H - M.b} V${top + r} Q${cx - bw / 2} ${top} ${cx - bw / 2 + r} ${top} H${cx + bw / 2 - r} Q${cx + bw / 2} ${top} ${cx + bw / 2} ${top + r} V${H - M.b} Z`}
                  fill="var(--color-sky)" fillOpacity={hov?.i === i ? 1 : 0.8} />
              </g>
            );
          })}
        </svg>
        {hov && <Tooltip x={hov.x} y={hov.y} rows={[{ label: "Rows", value: String(counts[hov.i]) },
          { label: "Error range", value: `${edge(hov.i).toFixed(1)} to ${edge(hov.i + 1).toFixed(1)}` }]} />}
      </div>
      <p className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12.5px] text-fg-muted">
        <span>Average error <span className="font-mono text-fg tabular">{mean.toFixed(2)}</span> {Math.abs(mean) < sd * 0.1 ? "(unbiased)" : mean > 0 ? "(tends to under-predict)" : "(tends to over-predict)"}</span>
        <span>Typical spread <span className="font-mono text-fg tabular">{sd.toFixed(2)}</span></span>
      </p>
    </ChartFrame>
  );
}
