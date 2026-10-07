"use client";

import { useRef, useState } from "react";
import { linear, niceTicks, tick } from "@/lib/scales";
import { ChartFrame, Tooltip, type TableView, type TipRow } from "./chart-frame";

const W = 520, H = 360, M = { l: 56, r: 14, t: 10, b: 42 };
const HIT = 24; // px: the pointer only needs to be near a point, not on it

/** Generic scatter plot with nearest-point hover. Optional diagonal (shared axes) or fitted trend line. */
export function PointsChart({ x, y, xLabel, yLabel, title, subtitle, table, ariaLabel, diagonal = false, trend = false, tip }: {
  x: number[]; y: number[]; xLabel: string; yLabel: string; title: string; subtitle?: string; table: TableView;
  ariaLabel: string; diagonal?: boolean; trend?: boolean; tip: (i: number) => TipRow[];
}) {
  const xmin = Math.min(...x), xmax = Math.max(...x), ymin = Math.min(...y), ymax = Math.max(...y);
  const lo = diagonal ? Math.min(xmin, ymin) : xmin, hi = diagonal ? Math.max(xmax, ymax) : xmax;
  const ylo = diagonal ? lo : ymin, yhi = diagonal ? hi : ymax;
  const px = (hi - lo) * 0.04 || 1, py = (yhi - ylo) * 0.04 || 1;
  const xt = niceTicks(lo - px, hi + px, 5), yt = niceTicks(ylo - py, yhi + py, 5);
  const x0 = Math.min(xt[0], lo - px), x1 = Math.max(xt[xt.length - 1], hi + px);
  const y0 = Math.min(yt[0], ylo - py), y1 = Math.max(yt[yt.length - 1], yhi + py);
  const sx = linear(x0, x1, M.l, W - M.r), sy = linear(y0, y1, H - M.b, M.t);
  const ref = useRef<HTMLDivElement>(null);
  const [hov, setHov] = useState<number | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  let line: [number, number, number, number] | null = null;
  if (trend && x.length > 2) {
    const mx = x.reduce((a, b) => a + b, 0) / x.length, my = y.reduce((a, b) => a + b, 0) / y.length;
    const sxx = x.reduce((a, v) => a + (v - mx) ** 2, 0), sxy = x.reduce((a, v, i) => a + (v - mx) * (y[i] - my), 0);
    if (sxx > 0) { const b = sxy / sxx, a = my - b * mx; line = [x0, a + b * x0, x1, a + b * x1]; }
  }

  const find = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), k = W / r.width;
    const cx = (e.clientX - r.left) * k, cy = (e.clientY - r.top) * k;
    let best = -1, bd = (HIT * k) ** 2;
    for (let i = 0; i < x.length; i++) {
      const dx = sx(x[i]) - cx, dy = sy(y[i]) - cy, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best >= 0 ? best : null;
  };

  return (
    <ChartFrame title={title} subtitle={subtitle} table={table}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full touch-none" role="img" aria-label={ariaLabel}
          onPointerMove={(e) => {
            const i = find(e); setHov(i);
            if (i !== null && ref.current) {
              const r = e.currentTarget.getBoundingClientRect(), o = ref.current.getBoundingClientRect();
              setPos({ x: (sx(x[i]) / W) * r.width + r.left - o.left, y: (sy(y[i]) / H) * r.height + r.top - o.top });
            }
          }}
          onPointerLeave={() => setHov(null)}>
          {yt.map((t) => (
            <g key={`y${t}`}>
              <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} stroke="var(--color-line)" strokeWidth={1} />
              <text x={M.l - 8} y={sy(t) + 4} textAnchor="end" className="fill-fg-subtle" fontSize={11}>{tick(t)}</text>
            </g>
          ))}
          {xt.map((t) => <text key={`x${t}`} x={sx(t)} y={H - M.b + 18} textAnchor="middle" className="fill-fg-subtle" fontSize={11}>{tick(t)}</text>)}
          <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>{xLabel}</text>
          <text transform={`translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>{yLabel}</text>
          {diagonal && <line x1={sx(x0)} y1={sy(y0)} x2={sx(x1)} y2={sy(y1)} stroke="var(--color-fg-subtle)" strokeWidth={1} />}
          {x.map((v, i) => <circle key={i} cx={sx(v)} cy={sy(y[i])} r={2.6} fill="var(--color-sky)" fillOpacity={0.55} />)}
          {line && <line x1={sx(line[0])} y1={sy(line[1])} x2={sx(line[2])} y2={sy(line[3])} stroke="var(--color-ember)" strokeWidth={1.6} strokeLinecap="round" />}
          {hov !== null && <circle cx={sx(x[hov])} cy={sy(y[hov])} r={6} fill="var(--color-sky)" stroke="var(--color-surface)" strokeWidth={2} />}
        </svg>
        {hov !== null && <Tooltip x={pos.x} y={pos.y} rows={tip(hov)} />}
      </div>
      {line && <p className="mt-2 flex items-center gap-2 text-[12px] text-fg-subtle"><span className="inline-block h-0.5 w-5 rounded bg-ember" aria-hidden />fitted trend line</p>}
    </ChartFrame>
  );
}

/** Actual vs predicted (Results). Points on the diagonal are perfect predictions. */
export function ScatterChart({ actual, predicted }: { actual: number[]; predicted: number[] }) {
  const worst = actual.map((a, i) => ({ a, p: predicted[i], e: predicted[i] - a })).sort((m, n) => Math.abs(n.e) - Math.abs(m.e)).slice(0, 25);
  return (
    <PointsChart x={actual} y={predicted} xLabel="Actual" yLabel="Predicted" diagonal title="Actual vs predicted"
      subtitle="Each dot is one test row. The closer to the diagonal, the better the prediction."
      ariaLabel={`Scatter plot of actual against predicted values, ${actual.length} points. Open the table view for the largest errors.`}
      table={{ columns: ["Actual", "Predicted", "Error"], rows: worst.map((w) => [+w.a.toFixed(2), +w.p.toFixed(2), +w.e.toFixed(2)]),
        note: `The ${worst.length} largest errors, out of ${actual.length} plotted rows.` }}
      tip={(i) => [{ label: "Actual", value: actual[i].toFixed(2) }, { label: "Predicted", value: predicted[i].toFixed(2) },
        { label: "Error", value: (predicted[i] - actual[i]).toFixed(2) }]} />
  );
}
