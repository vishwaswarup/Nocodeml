"use client";

import { useRef, useState } from "react";
import { linear } from "@/lib/scales";
import { ChartFrame, Tooltip } from "./chart-frame";

const W = 520, H = 340, M = { l: 46, r: 14, t: 10, b: 40 };
const TICKS = [0, 0.25, 0.5, 0.75, 1];

export function RocChart({ fpr, tpr, auc }: { fpr: number[]; tpr: number[]; auc: number | null }) {
  const x = linear(0, 1, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState({ x: 0, y: 0 });

  const pts = fpr.map((f, i) => [x(f), y(tpr[i])] as const);
  const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
  const area = `${line} L${x(fpr[fpr.length - 1]).toFixed(1)} ${y(0)} L${x(fpr[0]).toFixed(1)} ${y(0)} Z`;

  const nearest = (clientX: number, svg: SVGSVGElement) => {
    const r = svg.getBoundingClientRect();
    const fx = ((clientX - r.left) / r.width) * W;
    let best = 0, bd = Infinity;
    pts.forEach(([px], i) => { const d = Math.abs(px - fx); if (d < bd) { bd = d; best = i; } });
    return best;
  };
  const place = (i: number) => {
    const el = ref.current; if (!el) return;
    const r = el.getBoundingClientRect(); const svg = el.querySelector("svg")!.getBoundingClientRect();
    setTip({ x: (pts[i][0] / W) * svg.width + (svg.left - r.left), y: (pts[i][1] / H) * svg.height + (svg.top - r.top) });
  };

  return (
    <ChartFrame title="ROC curve" subtitle="How well the model separates the two classes at every decision threshold. Closer to the top-left is better; the diagonal is a coin flip."
      table={{ columns: ["False positive rate", "True positive rate"], rows: fpr.map((f, i) => [+f.toFixed(3), +tpr[i].toFixed(3)]),
        note: auc !== null ? `Area under the curve (ROC-AUC): ${auc.toFixed(3)}` : undefined }}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full touch-none" role="img" tabIndex={0}
          aria-label={`ROC curve${auc !== null ? `, area under the curve ${auc.toFixed(2)}` : ""}. Use left and right arrow keys to read values.`}
          onPointerMove={(e) => { const i = nearest(e.clientX, e.currentTarget); setHover(i); place(i); }}
          onPointerLeave={() => setHover(null)}
          onBlur={() => setHover(null)}
          onKeyDown={(e) => {
            if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
            e.preventDefault();
            const i = Math.min(pts.length - 1, Math.max(0, (hover ?? 0) + (e.key === "ArrowRight" ? 1 : -1)));
            setHover(i); place(i);
          }}>
          {TICKS.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--color-line)" strokeWidth={1} />
              <text x={M.l - 8} y={y(t) + 4} textAnchor="end" className="fill-fg-subtle" fontSize={11}>{t}</text>
              <text x={x(t)} y={H - M.b + 18} textAnchor="middle" className="fill-fg-subtle" fontSize={11}>{t}</text>
            </g>
          ))}
          <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>False positive rate</text>
          <text transform={`translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>True positive rate</text>
          <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--color-fg-subtle)" strokeWidth={1} />
          <text x={x(0.62)} y={y(0.62) + 16} className="fill-fg-subtle" fontSize={11} transform={`rotate(-33 ${x(0.62)} ${y(0.62) + 16})`}>chance</text>
          <path d={area} fill="var(--color-sky)" fillOpacity={0.1} />
          <path d={line} fill="none" stroke="var(--color-sky)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          {auc !== null && <text x={W - M.r - 8} y={H - M.b - 12} textAnchor="end" className="fill-fg" fontSize={13}>AUC {auc.toFixed(3)}</text>}
          {hover !== null && (
            <g>
              <line x1={pts[hover][0]} x2={pts[hover][0]} y1={M.t} y2={H - M.b} stroke="var(--color-line-strong)" strokeWidth={1} />
              <circle cx={pts[hover][0]} cy={pts[hover][1]} r={6} fill="var(--color-sky)" stroke="var(--color-surface)" strokeWidth={2} />
            </g>
          )}
        </svg>
        {hover !== null && <Tooltip x={tip.x} y={tip.y} rows={[{ label: "True positive rate", value: tpr[hover].toFixed(3) }, { label: "False positive rate", value: fpr[hover].toFixed(3) }]} />}
      </div>
    </ChartFrame>
  );
}
