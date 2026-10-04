"use client";

import { ArrowRight, Download, Play } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { GuidedSteps } from "@/components/guided-steps";
import { Logo } from "@/components/logo";
import { AccordionItem } from "@/components/ui/accordion";
import { CountBadge, StatusPill, Tag } from "@/components/ui/badge";
import { Button, IconButton } from "@/components/ui/button";
import { Card, Eyebrow } from "@/components/ui/card";
import { ChoiceList, Segmented, Switch } from "@/components/ui/choice";
import { DataTable, Metric, MetricBar, QualityRow, type Column } from "@/components/ui/data";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { Field } from "@/components/ui/field";
import { SectionRail, type SectionItem } from "@/components/ui/section-rail";

function Block({ id, title, note, children }: { id: string; title: string; note?: string; children: ReactNode }) {
  return (
    <section id={id} className="border-t border-line py-14">
      <div className="mb-8 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-h2">{title}</h2>
        {note && <p className="max-w-md text-[14px] text-fg-muted">{note}</p>}
      </div>
      {children}
    </section>
  );
}

const swatches: { group: string; items: { name: string; cls: string; hex: string; ring?: boolean }[] }[] = [
  { group: "Canvas & surfaces", items: [
    { name: "bg-deep", cls: "bg-bg-deep", hex: "#0F0F0F", ring: true },
    { name: "bg", cls: "bg-bg", hex: "#141414", ring: true },
    { name: "surface", cls: "bg-surface", hex: "#202020" },
    { name: "surface-2", cls: "bg-surface-2", hex: "#2A2A2A" },
    { name: "surface-3", cls: "bg-surface-3", hex: "#3A3A3A" },
  ] },
  { group: "Text", items: [
    { name: "fg", cls: "bg-fg", hex: "#FFFFFF" },
    { name: "fg-muted", cls: "bg-fg-muted", hex: "#999999" },
    { name: "fg-subtle", cls: "bg-fg-subtle", hex: "#6B6B6B" },
  ] },
  { group: "Brand", items: [
    { name: "ember", cls: "bg-ember", hex: "#FC6435" },
    { name: "amber", cls: "bg-amber", hex: "#FFB638" },
    { name: "sky", cls: "bg-sky", hex: "#80BDFF" },
  ] },
  { group: "Status", items: [
    { name: "pass", cls: "bg-pass", hex: "#4CCF8F" },
    { name: "warn", cls: "bg-warn", hex: "#FFB638" },
    { name: "fail", cls: "bg-fail", hex: "#FF5D5D" },
    { name: "info", cls: "bg-info", hex: "#80BDFF" },
  ] },
];

const sample = [
  { customer_id: 1000, age: 41.5, income: 18394, gender: "M", churn: 1 },
  { customer_id: 1001, age: null, income: 31877, gender: "F", churn: 0 },
  { customer_id: 1002, age: 36.2, income: 22410, gender: "F", churn: 0 },
  { customer_id: 1003, age: 52.9, income: 47120, gender: "M", churn: 1 },
  { customer_id: 1004, age: 29.0, income: 15890, gender: null, churn: 0 },
  { customer_id: 1005, age: 44.4, income: 26305, gender: "M", churn: 0 },
];
const sampleCols: Column[] = [
  { key: "customer_id", label: "customer_id", dtype: "int", numeric: true },
  { key: "age", label: "age", dtype: "float", numeric: true, decimals: 1 },
  { key: "income", label: "income", dtype: "float", numeric: true, decimals: 0 },
  { key: "gender", label: "gender", dtype: "str" },
  { key: "churn", label: "churn", dtype: "int", numeric: true },
];

const rail: SectionItem[] = [
  { n: 0, label: "Dataset", state: "done" },
  { n: 1, label: "Preprocessing", state: "done" },
  { n: 2, label: "Features", state: "current" },
  { n: 3, label: "Split", state: "stale" },
  { n: 4, label: "Models", state: "stale" },
  { n: 5, label: "Regularization", state: "todo" },
  { n: 6, label: "Training", state: "todo" },
  { n: 7, label: "Results", state: "todo" },
];

export function DesignShowcase() {
  const [strategy, setStrategy] = useState<"median" | "mean" | "mode" | "knn">("median");
  const [scaler, setScaler] = useState<"standard" | "minmax" | "robust" | "none">("standard");
  const [strat, setStrat] = useState(true);
  const [loading, setLoading] = useState(false);
  const [sortBy, setSortBy] = useState<string | undefined>("age");
  const [desc, setDesc] = useState(false);
  const [railItems, setRailItems] = useState(rail);

  const rows = useMemo(() => {
    if (!sortBy) return sample;
    return [...sample].sort((a, b) => {
      const x = a[sortBy as keyof typeof a], y = b[sortBy as keyof typeof b];
      if (x === null) return 1;
      if (y === null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * (desc ? -1 : 1);
    });
  }, [sortBy, desc]);

  return (
    <main className="mx-auto max-w-[1120px] px-4 pb-24 sm:px-8">
      <header className="flex flex-col gap-6 py-14 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="mb-8"><Logo /></div>
          <Eyebrow>Design system · v0.1</Eyebrow>
          <h1 className="mt-3 text-h1">The building blocks</h1>
          <p className="mt-3 max-w-xl text-[17px] text-fg-muted">
            Palette&apos;s look (near-black canvas, white pill actions, the ember-to-sky gradient), tuned for a dense,
            serious data tool.
          </p>
        </div>
        <nav className="flex flex-wrap gap-1.5 text-[13px]">
          {["colors", "type", "actions", "guided", "controls", "data", "results", "states"].map((s) => (
            <a key={s} href={`#${s}`} className="rounded-full bg-surface px-3 py-1.5 text-fg-muted hover:text-fg">{s}</a>
          ))}
        </nav>
      </header>

      <Block id="colors" title="Colour" note="Status colours always come with an icon and a word, never colour alone.">
        <div className="grid gap-8 md:grid-cols-2">
          {swatches.map((g) => (
            <div key={g.group}>
              <Eyebrow className="mb-3">{g.group}</Eyebrow>
              <div className="flex flex-wrap gap-3">
                {g.items.map((s) => (
                  <div key={s.name} className="w-[92px]">
                    <div className={`h-14 rounded-control ${s.cls} ${s.ring ? "ring-1 ring-line-strong" : ""}`} />
                    <p className="mt-2 text-[13px]">{s.name}</p>
                    <p className="font-mono text-[11px] text-fg-subtle">{s.hex}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-10 h-16 rounded-card signature-gradient-x" aria-label="Signature gradient" />
      </Block>

      <Block id="type" title="Type" note="Geist for everything, DM Mono for data types, labels and numbers that line up.">
        <div className="space-y-6">
          <p className="text-display">Build pipelines you can trust</p>
          <p className="text-h1">Section heading, 44 px</p>
          <p className="text-h2">Card heading, 28 px</p>
          <p className="text-h3">Sub-heading, 20 px</p>
          <p className="max-w-2xl text-[17px] text-fg-muted">
            Body text at 17 px in muted grey. NoCodeML explains every recommendation, for example: median imputation
            was chosen because <span className="text-fg">age</span> is skewed, so the mean would be pulled by outliers.
          </p>
          <p className="text-[13px] text-fg-subtle">Caption, 13 px: Preprocessing fitted only on training rows.</p>
          <Eyebrow>Mono label · float64 · 0.9142</Eyebrow>
        </div>
      </Block>

      <Block id="actions" title="Actions & badges">
        <div className="flex flex-wrap items-center gap-3">
          <Button>Get started</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="danger">Delete project</Button>
          <Button icon={<Play className="size-4" />} loading={loading} onClick={() => { setLoading(true); setTimeout(() => setLoading(false), 1600); }}>
            {loading ? "Training…" : "Train models"}
          </Button>
          <Button disabled>Disabled</Button>
          <Button size="sm" variant="secondary" icon={<Download className="size-3.5" />}>pipeline.pkl</Button>
          <Button size="lg">Large <ArrowRight className="size-4" /></Button>
          <IconButton label="Up" active={false}><ArrowRight className="size-4 -rotate-90" /></IconButton>
          <IconButton label="Down"><ArrowRight className="size-4 rotate-90" /></IconButton>
        </div>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <CountBadge value="15">quality checks run</CountBadge>
          <Tag>Random Forest</Tag>
          <Tag>v3</Tag>
          <StatusPill status="pass">Leakage-safe</StatusPill>
          <StatusPill status="warn">Class imbalance</StatusPill>
          <StatusPill status="fail">Below baseline</StatusPill>
          <StatusPill status="info">Recommendation</StatusPill>
        </div>
      </Block>

      <Block id="guided" title="Guided steps" note="Palette's one-question-at-a-time form, used for decisions that beginners need walking through.">
        <Card className="px-4 py-12 sm:px-12"><GuidedSteps /></Card>
      </Block>

      <Block id="controls" title="Workspace controls" note="Denser than the guided flow; every recommendation carries its reason.">
        <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
          <Card className="p-3">
            <Eyebrow className="px-3 pt-2 pb-3">Pipeline</Eyebrow>
            <SectionRail items={railItems} onSelect={(n) => setRailItems((items) => items.map((i) => ({
              ...i, state: i.n === n ? "current" : i.state === "current" ? "done" : i.state,
            })))} />
          </Card>
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="p-5">
              <p className="text-[15px]">Missing values · <span className="font-mono text-[13px] text-fg-muted">age</span></p>
              <p className="mt-1 mb-4 text-[13px] text-fg-muted">20 of 400 rows (5.0%) are empty.</p>
              <Segmented label="Imputation" value={strategy} onChange={setStrategy} options={[
                { value: "median", label: "Median" }, { value: "mean", label: "Mean" },
                { value: "mode", label: "Mode" }, { value: "knn", label: "KNN" },
              ]} />
              <div className="mt-5 grid grid-cols-2 gap-3">
                <Field label="Test size" defaultValue="0.20" suffix="ratio" />
                <Field label="Random seed" defaultValue="42" hint="Same seed, same split." />
              </div>
            </Card>
            <Card className="p-5">
              <p className="mb-3 text-[15px]">Scaling</p>
              <ChoiceList label="Scaler" value={scaler} onChange={setScaler} options={[
                { value: "standard", label: "StandardScaler", recommended: true,
                  description: "Logistic Regression and SVM are sensitive to feature scale." },
                { value: "robust", label: "RobustScaler", description: "Better when outliers are present." },
                { value: "none", label: "No scaling", description: "Fine for tree models only." },
              ]} />
              <div className="mt-4 flex items-center justify-between rounded-control bg-surface-2 px-4 py-3">
                <div>
                  <p className="text-[14px]">Stratified split</p>
                  <p className="text-[12px] text-fg-muted">Keeps the 18% minority class in both sets.</p>
                </div>
                <Switch checked={strat} onChange={setStrat} label="Stratified split" />
              </div>
            </Card>
          </div>
        </div>
      </Block>

      <Block id="data" title="Data" note="Missing values are visible, numbers align, headers sort.">
        <DataTable caption="Sample of customers.csv" columns={sampleCols} rows={rows} sortBy={sortBy} descending={desc}
          onSort={(k) => { if (k === sortBy) setDesc(!desc); else { setSortBy(k); setDesc(false); } }} />
      </Block>

      <Block id="results" title="Results" note="Model performance and pipeline quality are kept visibly separate.">
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <Card className="p-5">
            <div className="mb-5 flex items-center justify-between">
              <p className="text-h3">Random Forest</p>
              <Tag>Experiment #2 · v3</Tag>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <Metric label="F1" value="0.381" sub="baseline 0.000" emphasis />
              <Metric label="ROC-AUC" value="0.689" />
              <Metric label="Accuracy" value="0.870" sub="misleading: 83% is the majority class" />
            </div>
            <div className="mt-6 space-y-4">
              <MetricBar label="Recall" value={0.286} baseline={0} />
              <MetricBar label="Precision" value={0.571} baseline={0.17} />
              <MetricBar label="Balanced accuracy" value={0.619} baseline={0.5} />
            </div>
          </Card>
          <Card className="p-5">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-h3">Pipeline health</p>
              <span className="font-mono text-[13px] text-fg-muted tabular">86.7 / 100</span>
            </div>
            <div className="divide-y divide-line">
              <QualityRow status="pass" title="Preprocessing fitted only on training data" />
              <QualityRow status="pass" title="Train/test separation valid" />
              <QualityRow status="warn" title="Class imbalance detected" detail="Minority class is 17.2%. Judge models by F1 / ROC-AUC, not accuracy." />
              <QualityRow status="fail" title="Logistic Regression does not beat the baseline" detail="F1 0.000 vs trivial baseline 0.000." />
            </div>
          </Card>
        </div>
        <div className="mt-6 grid gap-2 lg:w-2/3">
          <AccordionItem title="Why is accuracy marked as misleading?">
            83% of customers did not churn, so a model that always says &ldquo;no churn&rdquo; scores 83% accuracy while
            catching nobody. F1 and ROC-AUC reveal that.
          </AccordionItem>
          <AccordionItem title="What does the pipeline health score measure?">
            Whether the experiment was built correctly (no leakage, proper split, baseline comparison), not how
            accurate the model is.
          </AccordionItem>
        </div>
      </Block>

      <Block id="states" title="Loading, empty, error">
        <div className="grid gap-6 lg:grid-cols-3">
          <Card className="space-y-3 p-5">
            <Skeleton className="h-5 w-1/2" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <div className="grid grid-cols-3 gap-2 pt-2">
              <Skeleton className="h-16" /><Skeleton className="h-16" /><Skeleton className="h-16" />
            </div>
          </Card>
          <EmptyState title="No experiments yet" body="Configure your pipeline, then train to see results here."
            action={<Button size="sm">Go to models</Button>} />
          <ErrorState title="This pipeline can't be trained yet" issues={[
            "'age' has missing values; choose an imputation strategy, drop rows, or drop the column.",
            "Categorical column 'gender' needs an encoding (or drop it).",
          ]} onRetry={() => {}} />
        </div>
      </Block>
    </main>
  );
}
