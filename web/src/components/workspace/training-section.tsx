"use client";

import { ChevronDown, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { StatusIcon } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, Eyebrow } from "@/components/ui/card";
import { Switch } from "@/components/ui/choice";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { asModels, sameModels, trainingParams, useModels } from "@/lib/models";
import { pipelineBody } from "@/lib/pipeline";
import { useModelIssues } from "@/lib/use-model-issues";
import type { Job, ModelConfig, ModelDefaults, ModelInfo, PipelineConfig, PipelineSaved, ProjectState } from "@/lib/types";
import { ParamField } from "./param-field";
import { TunePanel } from "./tune-panel";

type Validation = { valid: boolean; issues: string[] };

export function TrainingSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const cfg = state.pipeline?.config;
  if (!cfg || cfg.models.length === 0) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Choose your models first" body="Training needs at least one model."
          action={<Button size="sm" onClick={() => goTo(4)}>Go to Models</Button>} />
      </div>
    );
  }
  return <Loaded projectId={projectId} cfg={cfg} goTo={goTo} onSaved={onSaved} />;
}

function summary(cfg: PipelineConfig) {
  const pre = cfg.preprocessing as Record<string, unknown[] | string | boolean>;
  const n = (k: string) => (pre[k] as unknown[] | undefined)?.length ?? 0;
  const parts = [
    n("drop_columns") && `${n("drop_columns")} dropped`, n("missing_values") && `${n("missing_values")} filled`,
    n("encoding") && `${n("encoding")} encoded`, n("outliers") && `${n("outliers")} outlier rules`,
    pre.drop_duplicates && "duplicates removed", pre.scaling !== "none" && `${pre.scaling} scaling`,
  ].filter(Boolean);
  const sp = cfg.split as { method: string; test_size: number; n_splits: number; stratify: boolean; time_column: string | null };
  const cv = sp.method.includes("fold") || sp.method === "time_series";
  return {
    pre: parts.length ? parts.join(" · ") : "none",
    split: `${({ train_test: "train / test", train_val_test: "train / validation / test", k_fold: "k-fold", stratified_k_fold: "stratified k-fold", time_series: "time series" } as Record<string, string>)[sp.method] ?? sp.method}${cv ? ` · ${sp.n_splits} folds` : ` · ${Math.round(sp.test_size * 100)}% test`}${sp.stratify ? " · stratified" : ""}${sp.time_column ? ` · by ${sp.time_column}` : ""}`,
  };
}

function Loaded({ projectId, cfg, goTo, onSaved }: {
  projectId: string; cfg: PipelineConfig; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const { models, error: loadError } = useModels(cfg.dataset.task);
  const saved = useMemo(() => asModels(cfg.models), [cfg]);
  const [draft, setDraft] = useState<ModelConfig[]>(saved);
  const [defaults, setDefaults] = useState<ModelDefaults | null>(null);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [saving, setSaving] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const issues = useModelIssues(projectId, cfg, draft);
  const byKey = useMemo(() => Object.fromEntries((models ?? []).map((m) => [m.key, m])), [models]);
  const dirty = !sameModels(draft, saved);
  const hasIssues = Object.values(issues).some((l) => l.length > 0);
  const busy = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    let alive = true;
    api<ModelDefaults>(`/projects/${projectId}/pipeline/model-defaults`).then((d) => { if (alive) setDefaults(d); }).catch(() => {});
    return () => { alive = false; };
  }, [projectId, cfg.version]);

  // is the saved pipeline trainable?
  useEffect(() => {
    let alive = true;
    api<Validation>(`/projects/${projectId}/pipeline/validate`, { method: "POST" })
      .then((v) => { if (alive) setValidation(v); }).catch((e: Error) => { if (alive) setValidation({ valid: false, issues: [e.message] }); });
    return () => { alive = false; };
  }, [projectId, cfg.version]);

  // re-attach to a job that is already running (e.g. after a page reload)
  useEffect(() => {
    let alive = true;
    api<Job | null>(`/projects/${projectId}/training/active`).then((j) => { if (alive && j) setJob(j); }).catch(() => {});
    return () => { alive = false; };
  }, [projectId]);

  // poll until the job finishes
  const jobId = job?.id;
  useEffect(() => {
    if (!busy || !jobId) return;
    let alive = true;
    const t = setInterval(() => {
      setNow(Date.now());
      api<Job>(`/projects/${projectId}/training/${jobId}`).then((j) => {
        if (!alive) return;
        setJob(j);
        if (j.status === "succeeded") onSaved(`Training finished. Experiment #${j.experiment_number} is ready in Results.`);
      }).catch(() => {});
    }, 1000);
    return () => { alive = false; clearInterval(t); };
  }, [busy, jobId, projectId, onSaved]);

  const update = (key: string, fn: (m: ModelConfig) => ModelConfig) => setDraft((d) => d.map((m) => (m.model_key === key ? fn(m) : m)));

  const persist = useCallback(async (): Promise<boolean> => {
    setSaving(true);
    setError(null);
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, {
        method: "PUT", body: JSON.stringify(pipelineBody(cfg, { models: draft as unknown as PipelineConfig["models"] })) });
      setDraft(asModels(res.version.config.models));
      onSaved(res.impact?.changed.length ? res.impact.message : "Settings saved.");
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    } finally {
      setSaving(false);
    }
  }, [projectId, cfg, draft, onSaved]);

  const train = async () => {
    if (dirty && !(await persist())) return;
    setStarting(true);
    setError(null);
    try {
      setJob(await api<Job>(`/projects/${projectId}/training`, { method: "POST" }));
      setNow(Date.now());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const sum = summary(cfg);
  const elapsed = job?.started_at ? Math.max(0, Math.round(((job.finished_at ? new Date(job.finished_at).getTime() : now) - new Date(job.started_at).getTime()) / 1000)) : 0;
  const ready = validation?.valid && !hasIssues;

  return (
    <div className="mx-auto max-w-[1180px]">
      <Eyebrow>Section 6 · Train</Eyebrow>
      <h1 className="mt-2 text-h1">Settings and training</h1>
      <p className="mt-2 max-w-2xl text-[15px] text-fg-muted">
        Fine-tune each model, or leave it on NoCodeML&apos;s recommended starting values. Then train: every model runs on the same split.
      </p>
      {(error || loadError) && <div className="mt-6"><ErrorState title="Something went wrong" body={error ?? loadError ?? ""} /></div>}

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_360px]">
        <div className="min-w-0 space-y-4">
          {!models ? [0, 1].map((i) => <Skeleton key={i} className="h-44 rounded-card" />) : draft.map((m) => {
            const info = byKey[m.model_key];
            return info ? (
              <HyperCard key={m.model_key} info={info} model={m} defaults={defaults?.[m.model_key]} problems={issues[m.model_key] ?? []}
                onChange={(fn) => update(m.model_key, fn)} />
            ) : null;
          })}
          {dirty && (
            <div className="flex items-center gap-3">
              <Button variant="secondary" loading={saving} disabled={hasIssues} onClick={persist}>Save settings</Button>
              <span className="text-[13px] text-fg-subtle">You have unsaved changes.</span>
            </div>
          )}
        </div>

        <aside className="space-y-3 lg:sticky lg:top-[76px] lg:self-start">
          <Card className="p-5">
            <p className="text-h3">Ready to train?</p>
            <dl className="mt-3 divide-y divide-line text-[13px]">
              {[
                ["Predict", `${cfg.dataset.target_column} (${cfg.dataset.task})`],
                ["Preprocessing", sum.pre],
                ["Split", sum.split],
                ["Models", draft.map((m) => byKey[m.model_key]?.name ?? m.model_key).join(", ")],
              ].map(([k, v]) => (
                <div key={k} className="grid grid-cols-[96px_1fr] gap-3 py-2"><dt className="text-fg-muted">{k}</dt><dd>{v}</dd></div>
              ))}
            </dl>
            {!validation ? <Skeleton className="mt-3 h-10" /> : validation.valid ? (
              <div className="mt-3 flex gap-2.5 rounded-control bg-pass/[0.07] px-3 py-2.5 text-[13px] text-fg-muted">
                <StatusIcon status="pass" /><span>Everything checks out. Preprocessing is fitted on training rows only.</span>
              </div>
            ) : (
              <div className="mt-3 rounded-control bg-warn/[0.07] px-3.5 py-3">
                <p className="mb-1.5 text-[13px] text-warn">Fix these before training</p>
                <ul className="space-y-1">{validation.issues.slice(0, 5).map((i) => <li key={i} className="text-[12.5px] leading-snug text-fg-muted">{i}</li>)}</ul>
                <Button variant="ghost" size="sm" className="mt-2" onClick={() => goTo(1)}>Open Preprocessing</Button>
              </div>
            )}
            <Button className="mt-4 w-full" size="lg" icon={<Play className="size-4" />} loading={starting || busy || saving}
              disabled={!ready || !validation || busy} onClick={train}>
              {busy ? "Training…" : dirty ? "Save and train" : `Train ${draft.length} model${draft.length === 1 ? "" : "s"}`}
            </Button>
          </Card>

          {job && (
            <Card className="p-5" aria-live="polite">
              {busy && (
                <>
                  <p className="text-[15px]">{job.status === "queued" ? "Waiting to start…" : `Training ${draft.length} model${draft.length === 1 ? "" : "s"}…`}</p>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-surface-3">
                    <div className="h-full w-1/3 animate-pulse rounded-full signature-gradient-x" />
                  </div>
                  <p className="mt-2 font-mono text-[12px] text-fg-subtle tabular">{elapsed}s elapsed</p>
                </>
              )}
              {job.status === "succeeded" && (
                <>
                  <p className="flex items-center gap-2 text-[15px]"><StatusIcon status="pass" /> Finished in {elapsed}s</p>
                  {job.warnings.map((w) => <p key={w} className="mt-2 text-[12.5px] text-warn">{w}</p>)}
                  <Button className="mt-3 w-full" onClick={() => goTo(7)}>View results</Button>
                </>
              )}
              {job.status === "failed" && (
                <>
                  <p className="flex items-center gap-2 text-[15px]"><StatusIcon status="fail" /> Training didn&apos;t finish</p>
                  <p className="mt-2 text-[13px] text-fg-muted">{job.error}</p>
                  {job.issues.length > 0 && <ul className="mt-2 space-y-1">{job.issues.map((i) => <li key={i} className="text-[12.5px] text-fg-muted">{i}</li>)}</ul>}
                </>
              )}
            </Card>
          )}
        </aside>
      </div>
    </div>
  );
}

function HyperCard({ info, model, defaults, problems, onChange }: {
  info: ModelInfo; model: ModelConfig; defaults?: ModelDefaults[string]; problems: string[];
  onChange: (fn: (m: ModelConfig) => ModelConfig) => void;
}) {
  const [advanced, setAdvanced] = useState(false);
  const params = trainingParams(info);
  const main = params.filter((p) => !p.advanced), adv = params.filter((p) => p.advanced);
  const rec = defaults?.recommended ?? {};
  const touched = params.some((p) => p.name in model.hyperparameters);
  const value = (name: string, dflt: unknown) =>
    name in model.hyperparameters ? model.hyperparameters[name] : model.use_recommended_defaults && name in rec ? rec[name] : dflt;
  const tuned = (name: string) => !!model.search && name in model.search.space;
  const fromRec = (name: string) => !(name in model.hyperparameters) && model.use_recommended_defaults && name in rec;
  const set = (name: string, v: unknown) => onChange((m) => ({ ...m, hyperparameters: { ...m.hyperparameters, [name]: v } }));
  const resetParams = () => onChange((m) => ({
    ...m, hyperparameters: Object.fromEntries(Object.entries(m.hyperparameters).filter(([k]) => !params.some((p) => p.name === k))) }));
  const recChips = params.filter((p) => p.name in rec);

  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-h3">{info.name}</p>
          <p className="mt-1 text-[13px] text-fg-muted">
            {model.use_recommended_defaults
              ? "NoCodeML picked these as a starting point based on your dataset size and this model."
              : "Using the library defaults. Change anything you like."}
          </p>
        </div>
        <label className="flex cursor-pointer items-center gap-2.5 text-[14px]">
          Use recommended values
          <Switch checked={model.use_recommended_defaults} label={`${info.name}: use recommended values`}
            onChange={(v) => onChange((m) => ({ ...m, use_recommended_defaults: v }))} />
        </label>
      </div>

      {model.use_recommended_defaults && recChips.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {recChips.map((p) => (
            <span key={p.name} className="rounded-full bg-surface-2 px-3 py-1 font-mono text-[12px]">
              {p.name} <span className="text-fg">{String(value(p.name, p.default) ?? "no limit")}</span>
            </span>
          ))}
        </div>
      )}
      {params.length === 0 && <p className="mt-4 text-[13.5px] text-fg-muted">This model has no other settings. Its regularization was set in the previous section.</p>}

      {main.length > 0 && (
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          {main.map((p) => <ParamField key={p.name} hp={p} value={value(p.name, p.default)} onChange={(v) => set(p.name, v)}
            disabled={tuned(p.name)} note={tuned(p.name) ? "(chosen by the search)" : fromRec(p.name) ? "(recommended)" : undefined} />)}
        </div>
      )}

      {adv.length > 0 && (
        <div className="mt-4">
          <button type="button" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}
            className="flex items-center gap-1.5 text-[13px] text-fg-muted hover:text-fg">
            Advanced parameters <ChevronDown className={`size-4 transition-transform ${advanced ? "rotate-180" : ""}`} />
          </button>
          {advanced && (
            <div className="mt-3 grid gap-4 sm:grid-cols-2">
              {adv.map((p) => <ParamField key={p.name} hp={p} value={value(p.name, p.default)} onChange={(v) => set(p.name, v)}
                disabled={tuned(p.name)} note={tuned(p.name) ? "(chosen by the search)" : fromRec(p.name) ? "(recommended)" : undefined} />)}
            </div>
          )}
        </div>
      )}

      <TunePanel info={info} model={model} onChange={onChange} />

      {touched && <Button variant="ghost" size="sm" className="mt-3" onClick={resetParams}>Reset to {model.use_recommended_defaults ? "recommended" : "defaults"}</Button>}
      {problems.length > 0 && (
        <ul role="alert" className="mt-4 space-y-1 rounded-control bg-fail/[0.07] px-3.5 py-3">
          {problems.map((p) => <li key={p} className="text-[13px] text-fail">{p}</li>)}
        </ul>
      )}
    </Card>
  );
}
