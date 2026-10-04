"use client";

import { useRef, useState } from "react";
import { ChartFrame, localPoint, Tooltip } from "./chart-frame";

/** Heatmap: one hue, darker = more rows. The table twin lists every cell with its share of the actual class. */
export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const n = labels.length;
  const max = Math.max(1, ...matrix.flat());
  const rowTotals = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const cell = Math.min(84, Math.floor(340 / n));
  const L = 92, T = 34, W = L + cell * n + 8, H = T + cell * n + 34;
  const ref = useRef<HTMLDivElement>(null);
  const [hov, setHov] = useState<{ i: number; j: number; x: number; y: number } | null>(null);
  const binary = n === 2;
  const role = (i: number, j: number) => (binary ? (i === 0 ? (j === 0 ? "true negative" : "false positive") : (j === 0 ? "false negative" : "true positive")) : undefined);
  const pct = (i: number, j: number) => (rowTotals[i] ? (100 * matrix[i][j]) / rowTotals[i] : 0);
  const showNums = cell >= 34;

  const rows = matrix.flatMap((r, i) => r.map((v, j) => [`${labels[i]}`, `${labels[j]}`, v, `${pct(i, j).toFixed(1)}%`] as (string | number)[]));
  return (
    <ChartFrame title="Confusion matrix" subtitle={binary
      ? `Rows are what really happened, columns are what the model said. "${labels[1]}" is treated as the positive class.`
      : "Rows are what really happened, columns are what the model said. The diagonal is correct."}
      table={{ columns: ["Actual", "Predicted", "Rows", "% of actual"], rows }}>
      <div ref={ref} className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block h-auto w-full max-w-[460px]" role="img"
          aria-label={`Confusion matrix, ${n} by ${n} classes`} onPointerLeave={() => setHov(null)}>
          <text x={L + (cell * n) / 2} y={14} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>Predicted</text>
          <text transform={`translate(12 ${T + (cell * n) / 2}) rotate(-90)`} textAnchor="middle" className="fill-fg-muted" fontSize={11.5}>Actual</text>
          {labels.map((l, k) => (
            <g key={l}>
              <text x={L + k * cell + cell / 2} y={T - 6} textAnchor="middle" className="fill-fg-subtle" fontSize={11}>{l.length > 9 ? l.slice(0, 8) + "…" : l}</text>
              <text x={L - 8} y={T + k * cell + cell / 2 + 4} textAnchor="end" className="fill-fg-subtle" fontSize={11}>{l.length > 9 ? l.slice(0, 8) + "…" : l}</text>
            </g>
          ))}
          {matrix.map((r, i) => r.map((v, j) => {
            const a = v / max;
            return (
              <g key={`${i}-${j}`} tabIndex={0} role="img" aria-label={`Actual ${labels[i]}, predicted ${labels[j]}: ${v} rows`}
                onPointerMove={(e) => { const s = ref.current!; const p = localPoint(e, s); setHov({ i, j, x: p.x, y: p.y }); }}
                onFocus={() => setHov({ i, j, x: ((L + j * cell + cell / 2) / W) * (ref.current?.clientWidth ?? W), y: ((T + i * cell) / H) * (ref.current?.clientHeight ?? H) })}
                onBlur={() => setHov(null)}>
                <rect x={L + j * cell + 1} y={T + i * cell + 1} width={cell - 2} height={cell - 2} rx={4}
                  fill="var(--color-sky)" fillOpacity={v === 0 ? 0.05 : 0.18 + 0.82 * a}
                  stroke={hov?.i === i && hov.j === j ? "var(--color-fg)" : "none"} strokeWidth={1.5} />
                {showNums && <text x={L + j * cell + cell / 2} y={T + i * cell + cell / 2 + 5} textAnchor="middle" fontSize={14}
                  className={a > 0.55 ? "fill-on-light" : "fill-fg"}>{v}</text>}
              </g>
            );
          }))}
        </svg>
        {hov && <Tooltip x={hov.x} y={hov.y} title={role(hov.i, hov.j)}
          rows={[{ label: `Actual ${labels[hov.i]}, predicted ${labels[hov.j]}`, value: String(matrix[hov.i][hov.j]) }, { label: `Share of actual ${labels[hov.i]}`, value: `${pct(hov.i, hov.j).toFixed(1)}%` }]} />}
      </div>
    </ChartFrame>
  );
}
