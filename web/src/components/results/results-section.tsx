"use client";

import { AlertTriangle, Lock } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, Eyebrow } from "@/components/ui/card";
import { Segmented } from "@/components/ui/choice";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import type { ExperimentDetail, ExperimentSummary, PipelineVersion, ProjectState } from "@/lib/types";
import { ArtifactsCard } from "./artifacts-card";
import { CompareExperiments } from "./compare-experiments";
import { ComparisonTable } from "./comparison-table";
import { HealthCard } from "./health-card";
import { SOURCE_LABEL, sourcesOf, type Source } from "./metrics";
import { ModelDetail } from "./model-detail";

const when = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const whenShort = (iso: string) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

export function ResultsSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  if (!state.pipeline) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Nothing to show yet" body="Set up your pipeline and train it to see results here."
          action={<Button size="sm" onClick={() => goTo(0)}>Go to Dataset</Button>} />
      </div>
    );
  }
  return <Loaded projectId={projectId} pipeline={state.pipeline} goTo={goTo} onSaved={onSaved} />;
}

function Loaded({ projectId, pipeline, goTo, onSaved }: {
  projectId: string; pipeline: PipelineVersion; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const [list, setList] = useState<ExperimentSummary[] | null>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [detail, setDetail] = useState<{ n: number; d: ExperimentDetail } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [confirm, setConfirm] = useState(false);
  const [finalizing, setFinalizing] = useState(false);

  useEffect(() => {
    let alive = true;
    api<ExperimentSummary[]>(`/projects/${projectId}/experiments`)
      .then((l) => { if (alive) setList(l); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, pipeline.version, pipeline.status]);

  // default to the newest run that still matches the pipeline, else the newest run
  const newestFirst = useMemo(() => [...(list ?? [])].sort((a, b) => b.number - a.number), [list]);
  const number = picked ?? newestFirst.find((e) => e.current)?.number ?? newestFirst[0]?.number ?? null;

  useEffect(() => {
    if (number === null) return;
    let alive = true;
    api<ExperimentDetail>(`/projects/${projectId}/experiments/${number}`)
      .then((d) => { if (alive) setDetail({ n: number, d }); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, number]);

  const exp = detail && detail.n === number ? detail.d : null;
  const result = exp?.experiment.result;
  const sources = result ? sourcesOf(result.models) : [];
  const activeSource: Source = source && sources.includes(source) ? source : (sources.includes("test") ? "test" : sources[0]);
  const activeModel = result ? (result.models.find((m) => m.model_key === model) ?? result.models[0]) : null;
  const modelNames = useMemo(() => Object.fromEntries((result?.models ?? []).map((m) => [m.model_key, m.name])), [result]);

  const finalize = async () => {
    setFinalizing(true);
    setError(null);
    try {
      const v = await api<PipelineVersion>(`/projects/${projectId}/pipeline/finalize`, { method: "POST" });
      setConfirm(false);
      onSaved(`Pipeline finalized as v${v.version}.0. Editing it later creates a new draft version.`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setFinalizing(false);
    }
  };

  if (error && !list) return <div className="mx-auto max-w-xl pt-6"><ErrorState title="Couldn't load results" body={error} /></div>;
  if (list === null) return <div className="space-y-4"><Skeleton className="h-10 w-64" /><Skeleton className="h-52" /><Skeleton className="h-80" /></div>;
  if (list.length === 0) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="No experiments yet" body="Train your models to see the comparison, charts and pipeline health here."
          action={<Button size="sm" onClick={() => goTo(6)}>Go to training</Button>} />
      </div>
    );
  }

  const fails = result?.quality.checks.filter((c) => c.status === "fail").length ?? 0;
  const canFinalize = !!exp?.current && pipeline.status === "draft" && fails === 0;
  const finalized = pipeline.status === "finalized";

  return (
    <div className="mx-auto max-w-[1180px]">
      <Eyebrow>Section 7 · Results</Eyebrow>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
        <h1 className="text-h1">Results</h1>
        <Select label="Experiment" value={String(number)} onChange={(e) => { setPicked(Number(e.target.value)); setModel(null); setConfirm(false); }} className="w-80 max-w-full">
          {newestFirst.map((e) => (
            <option key={e.number} value={e.number}>#{e.number} · v{e.pipeline_version} · {e.current ? "current" : "outdated"} · {whenShort(e.finished_at)}{e.source === "colab" ? " · Colab" : ""}</option>
          ))}
        </Select>
      </div>

      {exp && !exp.current && (
        <div role="status" className="mt-5 flex flex-wrap items-center justify-between gap-3 rounded-card bg-warn/[0.07] p-4 ring-1 ring-warn/25">
          <p className="flex items-start gap-2.5 text-[14px]">
            <AlertTriangle className="mt-0.5 size-5 shrink-0 text-warn" aria-hidden />
            <span>These results are outdated: the pipeline changed after this run (it ran on v{exp.experiment.pipeline_version}, you are on v{pipeline.version}).</span>
          </p>
          <Button size="sm" variant="secondary" onClick={() => goTo(6)}>Re-run in Training</Button>
        </div>
      )}
      {error && <div className="mt-5"><ErrorState title="Something went wrong" body={error} /></div>}

      {!result || !activeModel ? (
        <div className="mt-8 space-y-4"><Skeleton className="h-44 rounded-card" /><Skeleton className="h-96 rounded-card" /></div>
      ) : (
        <>
          <p className="mt-3 text-[14px] text-fg-muted">
            {result.models.length} model{result.models.length === 1 ? "" : "s"} · {result.task} ·{" "}
            {result.split.mode === "cv" ? `${result.split.n_folds}-fold cross-validation` : `${result.split.n_train.toLocaleString()} train / ${result.split.n_test.toLocaleString()} test rows`}
            {" "}· {when(result.finished_at)}
          </p>

          <section className="mt-8">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-h2">Model comparison</h2>
              {sources.length > 1 && (
                <Segmented label="Evaluated on" value={activeSource} onChange={setSource}
                  options={sources.map((s) => ({ value: s, label: SOURCE_LABEL[s] }))} />
              )}
            </div>
            <ComparisonTable models={result.models} task={result.task} source={activeSource} selected={activeModel.model_key}
              onSelect={(k) => { setModel(k); document.getElementById("model-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }); }} />
            {activeSource === "cv" && <p className="mt-2 text-[12.5px] text-fg-subtle">Cross-validation: every row was tested once, by a model that never saw it. Numbers combine all folds.</p>}
          </section>

          <section id="model-detail" className="mt-12 scroll-mt-20">
            {result.models.length > 1 && (
              <div className="mb-5"><Segmented label="Model" value={activeModel.model_key} onChange={setModel}
                options={result.models.map((m) => ({ value: m.model_key, label: m.name }))} /></div>
            )}
            <ModelDetail model={activeModel} task={result.task} source={activeSource} />
          </section>

          <section className="mt-12 grid grid-cols-[minmax(0,1fr)] items-start gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <HealthCard checks={result.quality.checks} score={result.quality.score} models={modelNames} />
            <div className="min-w-0 space-y-4">
              <ArtifactsCard projectId={projectId} number={exp.experiment.number} />
              <Card className="p-6">
                <p className="text-h3">Finalize this pipeline</p>
                {finalized ? (
                  <p className="mt-2 flex items-center gap-2 text-[14px]"><Lock className="size-4 text-pass" aria-hidden /> Finalized as v{pipeline.version}.0. It can no longer change.</p>
                ) : (
                  <>
                    <p className="mt-1 text-[13px] text-fg-muted">Freezes this exact configuration and its results. You can keep experimenting afterwards: that creates a new version.</p>
                    {!exp.current && <p className="mt-2 text-[13px] text-warn">Only a run that matches the current pipeline can be finalized.</p>}
                    {fails > 0 && <p className="mt-2 text-[13px] text-fail">Fix the failed health checks first.</p>}
                    {confirm ? (
                      <div className="mt-4 flex items-center gap-2">
                        <Button loading={finalizing} onClick={finalize}>Yes, finalize v{pipeline.version}</Button>
                        <Button variant="ghost" onClick={() => setConfirm(false)}>Cancel</Button>
                      </div>
                    ) : (
                      <Button className="mt-4" disabled={!canFinalize} onClick={() => setConfirm(true)}>Finalize pipeline</Button>
                    )}
                  </>
                )}
              </Card>
            </div>
          </section>

          <section className="mt-4">
            <CompareExperiments projectId={projectId} current={exp.experiment.number} all={list} task={result.task} />
          </section>

          <section className="mt-4">
            <Card className="p-6">
              <p className="text-h3">How this was produced</p>
              <p className="mt-1 text-[13px] text-fg-muted">Everything needed to reproduce this run exactly.</p>
              <dl className="mt-4 grid gap-x-8 gap-y-3 text-[13px] sm:grid-cols-2">
                {[
                  ["Experiment", `#${exp.experiment.number} on pipeline v${exp.experiment.pipeline_version}`],
                  ["Trained on", result.source === "colab" ? "Google Colab (every score here was computed by NoCodeML from the predictions it returned)" : "NoCodeML servers"],
                  ["Random seed", String(result.random_state)],
                  ["Dataset fingerprint", result.dataset_fingerprint],
                  ["Configuration hash", result.config_hash],
                  ["Started", when(result.started_at)],
                  ["Finished", when(result.finished_at)],
                  ["Environment", Object.entries(result.environment).map(([k, v]) => `${k} ${v}`).join(" · ")],
                ].map(([k, v]) => (
                  <div key={k} className="min-w-0"><dt className="text-fg-muted">{k}</dt><dd className="mt-0.5 break-words font-mono text-[12.5px]">{v}</dd></div>
                ))}
              </dl>
            </Card>
          </section>
        </>
      )}
    </div>
  );
}
