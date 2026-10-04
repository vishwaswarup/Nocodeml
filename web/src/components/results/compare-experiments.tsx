"use client";

import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import type { Comparison, ExperimentSummary, Task } from "@/lib/types";
import { LOWER_IS_BETTER, TABLE_METRICS } from "./metrics";

const SECTION_LABEL: Record<string, string> = {
  dataset: "Dataset", preprocessing: "Preprocessing", feature_engineering: "Feature engineering", split: "Split", models: "Models and settings",
};

/** "What changed, and did it help?": this run against another run of the same project. */
export function CompareExperiments({ projectId, current, all, task }: {
  projectId: string; current: number; all: ExperimentSummary[]; task: Task;
}) {
  const others = all.filter((e) => e.number !== current);
  const [other, setOther] = useState<number | null>(null);
  const [data, setData] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const target = other !== null && others.some((e) => e.number === other) ? other : null;

  useEffect(() => {
    if (target === null) return;
    let alive = true;
    api<Comparison>(`/projects/${projectId}/compare?a=${target}&b=${current}`)
      .then((d) => { if (alive) { setData(d); setError(null); } })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, target, current]);

  if (others.length === 0) return null;
  const defs = TABLE_METRICS[task];
  const changed = data?.config_diff.filter((d) => !d.same) ?? [];

  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-h3">Compare with another run</p>
          <p className="mt-1 text-[13px] text-fg-muted">See exactly what changed between two experiments and how the numbers moved.</p>
        </div>
        <Select label={`Compare experiment #${current} with`} value={target?.toString() ?? ""} onChange={(e) => { setOther(e.target.value ? Number(e.target.value) : null); setData(null); }} className="w-56">
          <option value="">Choose a run…</option>
          {others.map((e) => <option key={e.number} value={e.number}>#{e.number} · v{e.pipeline_version}</option>)}
        </Select>
      </div>
      {error && <p role="alert" className="mt-3 text-[13px] text-fail">{error}</p>}
      {target !== null && !data && !error && <Skeleton className="mt-4 h-24" />}
      {data && target !== null && (
        <div className="mt-5 space-y-5">
          <div>
            <p className="mb-2 text-[13px] text-fg-muted">What changed (#{target} → #{current})</p>
            {changed.length === 0 ? <p className="text-[14px]">Nothing: the settings are identical.</p> : (
              <div className="space-y-1.5">
                {changed.map((d) => (
                  <details key={d.section} className="rounded-control bg-surface-2 px-3.5 py-2.5">
                    <summary className="cursor-pointer text-[14px]">{SECTION_LABEL[d.section] ?? d.section} <span className="text-fg-subtle">differs</span></summary>
                    <div className="mt-2 grid gap-3 sm:grid-cols-2">
                      {[["Before", d.before], ["After", d.after]].map(([t, v]) => (
                        <div key={t as string}><p className="text-[12px] text-fg-subtle">{t as string}</p>
                          <pre className="mt-1 max-h-48 overflow-auto rounded-[8px] bg-bg-deep p-2.5 font-mono text-[11.5px] leading-relaxed">{JSON.stringify(v, null, 1)}</pre></div>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            )}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-[13.5px]">
              <caption className="sr-only">Metric changes between experiments</caption>
              <thead><tr className="border-b border-line text-left text-fg-muted">
                <th scope="col" className="py-2 pr-4 font-normal">Model</th>
                {defs.map((d) => <th key={d.key} scope="col" className="px-3 py-2 text-right font-normal">{d.label}</th>)}
              </tr></thead>
              <tbody>
                {Object.entries(data.metrics).map(([model, ms]) => (
                  <tr key={model} className="border-b border-line/60">
                    <th scope="row" className="py-2.5 pr-4 text-left font-normal">{all.find((e) => e.number === current)?.models.find((m) => m.model_key === model)?.name ?? model}</th>
                    {defs.map((d) => {
                      const v = ms[d.key];
                      if (!v || v.delta === null) return <td key={d.key} className="px-3 py-2.5 text-right text-fg-subtle">–</td>;
                      const better = (LOWER_IS_BETTER.has(d.key) ? -v.delta : v.delta) > 0.0005;
                      const worse = (LOWER_IS_BETTER.has(d.key) ? -v.delta : v.delta) < -0.0005;
                      const Icon = better ? ArrowUp : worse ? ArrowDown : Minus;
                      return (
                        <td key={d.key} className="px-3 py-2.5 text-right font-mono tabular">
                          <span className="block text-fg">{v.b?.toFixed(3)}</span>
                          <span className={`inline-flex items-center gap-0.5 text-[11.5px] ${better ? "text-pass" : worse ? "text-fail" : "text-fg-subtle"}`}>
                            <Icon className="size-3" aria-hidden />
                            {v.delta > 0 ? "+" : ""}{v.delta.toFixed(3)}<span className="sr-only">{better ? " improved" : worse ? " got worse" : " unchanged"}</span>
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}
