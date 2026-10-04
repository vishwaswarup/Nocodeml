"use client";

import { useRef, useState } from "react";
import { linear, niceTicks, tick } from "@/lib/scales";
import { ChartFrame, Tooltip } from "./chart-frame";

const W = 520, H = 360, M = { l: 56, r: 14, t: 10, b: 42 };
const HIT = 24; // px: pointer only needs to be near a point, not on it

/** Actual vs predicted. Points on the diagonal are perfect predictions. */
export function ScatterChart({ actual, predicted }: { actual: number[]; predicted: number[] }) {
  const all = [...actual, ...predicted];
  const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.04 || 1;
  const ticks = niceTicks(lo - pad, hi + pad, 5);
  const d0 = Math.min(ticks[0], lo - pad), d1 = Math.max(ticks[ticks.length - 1], hi + pad);
  const x = linear(d0, d1, M.l, W - M.r), y = linear(d0, d1, H - M.b, M.t);
  const ref = useRef<HTMLDivElement>(null);
  const [hov, setHov] = useState<number | null>(null);
  const [tip, setTip] = useState({ x: 0, y: 0 });

  const find = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), k = W / r.width;
    const px = (e.clientX - r.left) * k, py = (e.clientY - r.top) * k;
    let best = -1, bd = (HIT * k) ** 2;
    for (let i = 0; i < actual.length; i++) {
      const dx = x(actual[i]) - px, dy = y(predicted[i]) - py, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best >= 0 ? best : null;
  };

  const worst = actual.map((a, i) => ({ a, p: predicted[i], e: predicted[i] - a }))
    .sort((m, n) => Math.abs(n.e) - Math.abs(m.e)).slice(0, 25);
  return (
    <ChartFrame title="Actual vs predicted" subtitle="Each dot is one test row. The closer to the diagonal, the better the prediction."
      table={{ columns: ["Actual", "Predicted", "Error"], rows: worst.map((w) => [+w.a.toFixed(2), +w.p.toFixed(2), +w.e.toFixed(2)]),
        note: `The ${worst.length} largest errors, out of ${actual.length} plotted rows.` }}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full touch-none" role="img"
          aria-label={`Scatter plot of actual against predicted values, ${actual.length} points. Open the table view for the largest errors.`}
          onPointerMove={(e) => {
            const i = find(e); setHov(i);
            if (i !== null && ref.current) { const r = e.currentTarget.getBoundingClientRect(), o = ref.current.getBoundingClientRect();
              setTip({ x: (x(actual[i]) / W) * r.width + r.left - o.left, y: (y(predicted[i]) / H) * r.height + r.top - o.top }); }
          }}
          onPointerLeave={() => setHov(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--color-line)" strokeWidth={1} />
              <text x={M.l - 8} y={y(t) + 4} textAnchor="end" className="fill-fg-subtle" fontSize={11}>{tick(t)}</text>
              <text x={x(t)} y={H - M.b + 18} textAnchor="middle" className="fill-fg-subtle" fontSize={11}>{tick(t)}</text>
            </g>
          ))}
          <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>Actual</text>
          <text transform={`translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>Predicted</text>
          <line x1={x(d0)} y1={y(d0)} x2={x(d1)} y2={y(d1)} stroke="var(--color-fg-subtle)" strokeWidth={1} />
          {actual.map((a, i) => <circle key={i} cx={x(a)} cy={y(predicted[i])} r={2.6} fill="var(--color-sky)" fillOpacity={0.55} />)}
          {hov !== null && <circle cx={x(actual[hov])} cy={y(predicted[hov])} r={6} fill="var(--color-sky)" stroke="var(--color-surface)" strokeWidth={2} />}
        </svg>
        {hov !== null && <Tooltip x={tip.x} y={tip.y} rows={[
          { label: "Actual", value: actual[hov].toFixed(2) }, { label: "Predicted", value: predicted[hov].toFixed(2) },
          { label: "Error", value: (predicted[hov] - actual[hov]).toFixed(2) }]} />}
      </div>
    </ChartFrame>
  );
}
