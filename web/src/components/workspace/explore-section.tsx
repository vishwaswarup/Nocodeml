"use client";

import { Info } from "lucide-react";
import { useMemo, useState } from "react";
import { BarList } from "@/components/charts/bar-list";
import { CorrelationHeatmap, strength } from "@/components/charts/correlation-heatmap";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { PointsChart } from "@/components/charts/scatter-chart";
import { Segmented } from "@/components/ui/choice";
import { ErrorState, Skeleton } from "@/components/ui/feedback";
import { Select } from "@/components/ui/select";
import { int } from "@/lib/format";
import { useChart } from "@/lib/use-chart";
import type {
  CategoryData, ChartSpec, ClassBalanceData, CorrelationData, DatasetProfile, DistributionData, MissingData, ScatterData,
} from "@/lib/types";

type Tab = "auto" | "distribution" | "categories" | "relationships" | "missing";

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

function Categories({ d, title }: { d: CategoryData; title?: string }) {
  const items = [...d.items.map((i) => ({ label: i.label, value: i.count, detail: pct(i.share) })),
    ...(d.other > 0 ? [{ label: `Other (${d.n_unique - d.items.length} values)`, value: d.other, detail: pct(d.other_share) }] : [])];
  return (
    <BarList title={title ?? `Values of ${d.column}`} items={items} format={int} ariaLabel={`Bar chart of the most common values of ${d.column}`}
      subtitle={`${d.n_unique.toLocaleString()} distinct values in ${d.n.toLocaleString()} rows${d.missing ? `; ${d.missing.toLocaleString()} missing` : ""}.`}
      table={{ columns: ["Value", "Rows", "Share"], rows: items.map((i) => [i.label, i.value, i.detail]) }} />
  );
}

function ClassBalance({ d }: { d: ClassBalanceData }) {
  return (
    <BarList title={`Class balance of ${d.column}`} items={d.items.map((i) => ({ label: i.label, value: i.count, detail: pct(i.share) }))} format={int}
      ariaLabel={`Bar chart of how many rows belong to each class of ${d.column}`}
      subtitle={`${d.n.toLocaleString()} rows with a value.`}
      footnote={d.imbalanced ? `The smallest class is ${pct(d.minority_ratio)} of rows. Accuracy alone would be misleading here; use a stratified split and judge by F1 or ROC-AUC.` : "The classes are reasonably balanced."}
      table={{ columns: ["Class", "Rows", "Share"], rows: d.items.map((i) => [i.label, i.count, pct(i.share)]) }} />
  );
}

function Missing({ d }: { d: MissingData }) {
  if (d.columns.length === 0) {
    return <div className="rounded-card bg-surface p-5 text-[14px] text-fg-muted"><p className="text-[16px] text-fg">Missing values</p><p className="mt-2">No column has any missing values.</p></div>;
  }
  return (
    <BarList title="Missing values per column" subtitle="Share of rows with no value, only for columns that have gaps."
      items={d.columns.map((c) => ({ label: c.column, value: c.pct, detail: `${int(c.missing)} rows` }))} format={(v) => `${v}%`} max={Math.max(100, ...d.columns.map((c) => c.pct))}
      ariaLabel="Bar chart of the percentage of missing values in each column"
      footnote={`${int(d.rows_with_missing)} of ${int(d.n_rows)} rows (${d.rows_with_missing_pct}%) have at least one missing value. Dropping all of them would lose that much data, so filling is usually better.`}
      table={{ columns: ["Column", "Missing rows", "Missing %"], rows: d.columns.map((c) => [c.column, c.missing, `${c.pct}%`]) }} />
  );
}

function Scatter({ d }: { d: ScatterData }) {
  return (
    <PointsChart x={d.x} y={d.y} xLabel={d.x_column} yLabel={d.y_column} trend title={`${d.y_column} vs ${d.x_column}`}
      subtitle={`${d.r === null ? "" : `Correlation r = ${d.r.toFixed(2)} (${strength(d.r)}). `}${d.n_shown < d.n_total ? `Showing a random sample of ${int(d.n_shown)} of ${int(d.n_total)} rows.` : `${int(d.n_total)} rows.`}`}
      ariaLabel={`Scatter plot of ${d.y_column} against ${d.x_column}, ${d.n_shown} points`}
      table={{ columns: [d.x_column, d.y_column], rows: d.x.slice(0, 40).map((v, i) => [+v.toFixed(3), +d.y[i].toFixed(3)]), note: `The first 40 of ${d.n_shown} plotted points.` }}
      tip={(i) => [{ label: d.x_column, value: String(+d.x[i].toFixed(3)) }, { label: d.y_column, value: String(+d.y[i].toFixed(3)) }]} />
  );
}

function ForSpec({ spec }: { spec: ChartSpec }) {
  return (
    <div className={spec.type === "correlation" && spec.data.columns.length > 9 ? "lg:col-span-2" : ""}>
      <p className="mb-2 flex gap-2 text-[12.5px] leading-snug text-fg-muted"><Info className="mt-0.5 size-3.5 shrink-0 text-fg-subtle" aria-hidden /><span><span className="text-fg-subtle">Why this chart: </span>{spec.why}</span></p>
      {spec.type === "class_balance" && <ClassBalance d={spec.data} />}
      {spec.type === "distribution" && <DistributionChart d={spec.data} title={spec.title} />}
      {spec.type === "categories" && <Categories d={spec.data} title={spec.title} />}
      {spec.type === "missing" && <Missing d={spec.data} />}
      {spec.type === "correlation" && <CorrelationHeatmap d={spec.data} />}
      {spec.type === "scatter" && <Scatter d={spec.data} />}
    </div>
  );
}

const q = (o: Record<string, string | number | null | undefined>) =>
  Object.entries(o).filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");

export function ExploreSection({ projectId, datasetId, target, profile }: {
  projectId: string; datasetId: string; target: string | null; profile: DatasetProfile;
}) {
  const [tab, setTab] = useState<Tab>("auto");
  const base = `/projects/${projectId}/datasets/${datasetId}/charts`;
  const numeric = useMemo(() => profile.numerical.filter((c) => !profile.columns[c].identifier_like), [profile]);
  const categorical = useMemo(() => profile.categorical.filter((c) => profile.columns[c].unique >= 2 && profile.columns[c].unique <= 5000), [profile]);
  const [col, setCol] = useState<string | null>(null);
  const [bins, setBins] = useState<"15" | "30" | "50">("30");
  const [sx, setSx] = useState<string | null>(null);
  const [sy, setSy] = useState<string | null>(null);

  const defaultNum = numeric.find((c) => Math.abs(profile.columns[c].skewness ?? 0) > 1) ?? numeric[0] ?? null;
  const numCol = col && numeric.includes(col) ? col : defaultNum;
  const catCol = col && categorical.includes(col) ? col : categorical[0] ?? null;

  const auto = useChart<ChartSpec[]>(tab === "auto" ? `${base}/auto?${q({ target })}` : null);
  const dist = useChart<DistributionData>(tab === "distribution" && numCol ? `${base}/distribution?${q({ column: numCol, bins })}` : null);
  const cats = useChart<CategoryData>(tab === "categories" && catCol ? `${base}/categories?${q({ column: catCol })}` : null);
  const corr = useChart<CorrelationData>(tab === "relationships" && numeric.length >= 2 ? `${base}/correlation?${q({ target })}` : null);
  const miss = useChart<MissingData>(tab === "missing" ? `${base}/missing` : null);
  const topPair = corr.data?.top_pairs.find((p) => numeric.includes(p.a) && numeric.includes(p.b));
  const xCol = sx && numeric.includes(sx) ? sx : topPair?.a ?? numeric[0] ?? null;
  const yCol = sy && numeric.includes(sy) && sy !== xCol ? sy : topPair?.b ?? numeric.find((c) => c !== xCol) ?? null;
  // Wait for the correlations (they pick the default pair) unless the user already chose axes: avoids drawing
  // an arbitrary pair first and then jumping to the strongest one.
  const axesKnown = corr.data !== null || corr.error !== null || (sx !== null && sy !== null);
  const sc = useChart<ScatterData>(tab === "relationships" && axesKnown && xCol && yCol && xCol !== yCol ? `${base}/scatter?${q({ x: xCol, y: yCol })}` : null);

  const dim = (stale: boolean) => `transition-opacity duration-150 ${stale ? "opacity-50" : ""}`;
  const loading = <Skeleton className="h-[380px] rounded-card" />;
  const err = (m: string | null) => (m ? <ErrorState title="Couldn't draw this chart" body={m} /> : null);
  const empty = (t: string) => <div className="rounded-card border border-dashed border-line-strong p-8 text-center text-[14px] text-fg-muted">{t}</div>;

  return (
    <section aria-label="Explore the data">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-h2">Explore</h2>
          <p className="mt-1 text-[14px] text-fg-muted">Charts of your data. Each one has a Table view with the same numbers.</p>
        </div>
        <Segmented label="Chart type" value={tab} onChange={(t) => { setTab(t); setCol(null); }} options={[
          { value: "auto", label: "Auto" }, { value: "distribution", label: "Distributions" }, { value: "categories", label: "Categories" },
          { value: "relationships", label: "Relationships" }, { value: "missing", label: "Missing values" }]} />
      </div>

      {tab === "auto" && (
        err(auto.error) ?? (!auto.data ? <div className="grid gap-4 lg:grid-cols-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-[380px] rounded-card" />)}</div> : (
          <div className={`grid gap-x-4 gap-y-8 lg:grid-cols-2 ${dim(auto.stale)}`}>{auto.data.map((s) => <ForSpec key={s.id} spec={s} />)}</div>))
      )}

      {tab === "distribution" && (numeric.length === 0 ? empty("This dataset has no numeric columns to plot.") : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <Select label="Column" value={numCol ?? ""} onChange={(e) => setCol(e.target.value)} className="w-64">
              {numeric.map((c) => <option key={c} value={c}>{c}</option>)}
            </Select>
            <div><p className="mb-1.5 text-[13px] text-fg-muted">Bars</p><Segmented label="Number of bars" value={bins} onChange={setBins} options={[{ value: "15", label: "15" }, { value: "30", label: "30" }, { value: "50", label: "50" }]} /></div>
          </div>
          {err(dist.error) ?? (dist.data ? <div className={`max-w-[640px] ${dim(dist.stale)}`}><DistributionChart d={dist.data} /></div> : loading)}
        </div>))}

      {tab === "categories" && (categorical.length === 0 ? empty("This dataset has no text columns to plot.") : (
        <div className="space-y-4">
          <Select label="Column" value={catCol ?? ""} onChange={(e) => setCol(e.target.value)} className="w-64">
            {categorical.map((c) => <option key={c} value={c}>{c}</option>)}
          </Select>
          {err(cats.error) ?? (cats.data ? <div className={`max-w-[640px] ${dim(cats.stale)}`}><Categories d={cats.data} /></div> : loading)}
        </div>))}

      {tab === "relationships" && (numeric.length < 2 ? empty("Relationships need at least two numeric columns.") : (
        <div className="space-y-6">
          {err(corr.error) ?? (corr.data ? <div className={`max-w-[760px] ${dim(corr.stale)}`}><CorrelationHeatmap d={corr.data} /></div> : loading)}
          <div className="flex flex-wrap items-end gap-4">
            <Select label="Horizontal axis" value={xCol ?? ""} onChange={(e) => setSx(e.target.value)} className="w-56">{numeric.map((c) => <option key={c} value={c}>{c}</option>)}</Select>
            <Select label="Vertical axis" value={yCol ?? ""} onChange={(e) => setSy(e.target.value)} className="w-56">{numeric.filter((c) => c !== xCol).map((c) => <option key={c} value={c}>{c}</option>)}</Select>
          </div>
          {err(sc.error) ?? (sc.data ? <div className={`max-w-[640px] ${dim(sc.stale)}`}><Scatter d={sc.data} /></div> : xCol && yCol ? loading : null)}
        </div>))}

      {tab === "missing" && (err(miss.error) ?? (miss.data ? <div className={`max-w-[640px] ${dim(miss.stale)}`}><Missing d={miss.data} /></div> : loading))}
    </section>
  );
}
