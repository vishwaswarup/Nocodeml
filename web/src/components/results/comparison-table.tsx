import type { ModelResult, Task } from "@/lib/types";
import { f3, num, TABLE_METRICS, type Source } from "./metrics";

/**
 * Side-by-side metrics. There is deliberately no overall "best model" score: the right choice depends on
 * the goal. The best value in each column is just marked, so trade-offs stay visible.
 */
export function ComparisonTable({ models, task, source, selected, onSelect }: {
  models: ModelResult[]; task: Task; source: Source; selected: string; onSelect: (key: string) => void;
}) {
  const defs = TABLE_METRICS[task];
  const val = (m: ModelResult, k: string) => num(m.metrics[source]?.[k]);
  const best = Object.fromEntries(defs.map((d) => {
    const vs = models.map((m) => val(m, d.key)).filter((v): v is number => v !== null);
    return [d.key, vs.length ? (d.lowerIsBetter ? Math.min(...vs) : Math.max(...vs)) : null];
  }));
  const baseline = models[0]?.baseline ?? {};
  const baselineLabel = task === "classification" ? "Baseline: always predict the most common class" : "Baseline: always predict the average";

  return (
    <div className="overflow-x-auto rounded-card bg-surface ring-1 ring-line">
      <table className="w-full min-w-[560px] text-[14px]">
        <caption className="sr-only">Model comparison on the {source} data</caption>
        <thead>
          <tr className="border-b border-line text-left text-fg-muted">
            <th scope="col" className="px-4 py-3 font-normal">Model</th>
            {defs.map((d) => (
              <th key={d.key} scope="col" title={d.hint} className="px-4 py-3 text-right font-normal">
                {d.label}{d.lowerIsBetter && <span className="ml-1 text-[11px] text-fg-subtle">lower is better</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const on = m.model_key === selected;
            return (
              <tr key={m.model_key} onClick={() => onSelect(m.model_key)}
                className={`cursor-pointer border-b border-line/60 transition-colors ${on ? "bg-surface-2" : "hover:bg-white/[0.03]"}`}>
                <th scope="row" className="px-4 py-3 text-left font-normal">
                  <button type="button" onClick={() => onSelect(m.model_key)} aria-pressed={on} className="text-left">
                    {m.name}
                  </button>
                </th>
                {defs.map((d) => {
                  const v = val(m, d.key);
                  const isBest = v !== null && best[d.key] === v && models.length > 1;
                  return (
                    <td key={d.key} className={`px-4 py-3 text-right font-mono tabular ${isBest ? "text-fg" : "text-fg-muted"}`}>
                      <span className={isBest ? "inline-flex items-center gap-1.5 font-semibold" : ""}>
                        {isBest && <span className="size-1.5 rounded-full bg-ember" aria-hidden />}
                        {f3(v)}
                        {isBest && <span className="sr-only"> (best in this column)</span>}
                      </span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
          <tr className="text-fg-subtle">
            <th scope="row" className="px-4 py-2.5 text-left text-[13px] font-normal">{baselineLabel}</th>
            {defs.map((d) => <td key={d.key} className="px-4 py-2.5 text-right font-mono text-[13px] tabular">{f3(baseline[d.key])}</td>)}
          </tr>
        </tbody>
      </table>
      <p className="border-t border-line/60 px-4 py-2.5 text-[12px] text-fg-subtle">
        <span className="mr-1.5 inline-block size-1.5 rounded-full bg-ember align-middle" aria-hidden />marks the best value per column. Pick the model that fits your goal; there is no single winner.
      </p>
    </div>
  );
}
