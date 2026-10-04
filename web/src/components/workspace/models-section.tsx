"use client";

import { AlertTriangle, Check, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { CountBadge } from "@/components/ui/badge";
import { Eyebrow } from "@/components/ui/card";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { asModels, DESCRIPTIONS, emptyModel, sameModels, STARTER, useModels } from "@/lib/models";
import { pipelineBody } from "@/lib/pipeline";
import type { ModelConfig, ModelInfo, PipelineConfig, PipelineSaved, ProjectState } from "@/lib/types";

const MAX_MODELS = 5;
const LEVEL = { low: "text-pass", medium: "text-warn", high: "text-ember" } as const;

export function ModelsSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  if (!state.pipeline) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Choose what to predict first" body="The available models depend on your task. Pick the target in the Dataset section."
          action={<Button size="sm" onClick={() => goTo(0)}>Go to Dataset</Button>} />
      </div>
    );
  }
  return <Loaded projectId={projectId} cfg={state.pipeline.config} goTo={goTo} onSaved={onSaved} />;
}

function Loaded({ projectId, cfg, goTo, onSaved }: {
  projectId: string; cfg: PipelineConfig; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const task = cfg.dataset.task;
  const { models, error: loadError } = useModels(task);
  const saved = useMemo(() => asModels(cfg.models), [cfg]);
  const [draft, setDraft] = useState<ModelConfig[]>(saved);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = !sameModels(draft, saved);
  const picked = new Set(draft.map((m) => m.model_key));
  const full = draft.length >= MAX_MODELS;
  const byKey = useMemo(() => Object.fromEntries((models ?? []).map((m) => [m.key, m])), [models]);
  const scaling = (cfg.preprocessing as { scaling?: string }).scaling ?? "none";
  const needScaling = draft.map((m) => byKey[m.model_key]).filter((m): m is ModelInfo => !!m && m.requires_scaling);

  const toggle = (key: string) => setDraft((d) =>
    d.some((m) => m.model_key === key) ? d.filter((m) => m.model_key !== key) : d.length >= MAX_MODELS ? d : [...d, emptyModel(key)]);
  const starter = () => setDraft((d) => {
    const keep = d.filter((m) => !STARTER[task].includes(m.model_key));
    return [...keep.filter(() => false), ...STARTER[task].map((k) => d.find((m) => m.model_key === k) ?? emptyModel(k))];
  });

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, {
        method: "PUT", body: JSON.stringify(pipelineBody(cfg, { models: draft as unknown as PipelineConfig["models"] })) });
      onSaved(res.impact?.changed.length ? res.impact.message : "Models saved.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1180px]">
      <Eyebrow>Section 4 · Models</Eyebrow>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-h1">Choose your models</h1>
          <p className="mt-2 max-w-xl text-[15px] text-fg-muted">
            Pick up to {MAX_MODELS}. They are all trained on the same split and compared side by side, so you can see which approach suits your data.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <CountBadge value={`${draft.length}/${MAX_MODELS}`}>selected</CountBadge>
          <Button variant="secondary" size="sm" icon={<Sparkles className="size-3.5" />} onClick={starter} disabled={!models}>Starter set</Button>
        </div>
      </div>
      <p className="mt-3 text-[13px] text-fg-subtle">
        Showing {task} models. Starter set: {STARTER[task].map((k) => byKey[k]?.name ?? k).join(", ")}, a linear model, a tree ensemble and a third approach to compare against.
      </p>

      {(error || loadError) && <div className="mt-6"><ErrorState title="Something went wrong" body={error ?? loadError ?? ""} /></div>}

      {needScaling.length > 0 && scaling === "none" && (
        <div role="alert" className="mt-6 flex gap-3 rounded-card bg-warn/[0.07] p-4 ring-1 ring-warn/25">
          <AlertTriangle className="mt-0.5 size-5 shrink-0 text-warn" />
          <div className="text-[14px]">
            <p>{needScaling.map((m) => m.name).join(", ")} {needScaling.length === 1 ? "is" : "are"} sensitive to feature scale, but no scaling is set.</p>
            <p className="mt-0.5 text-fg-muted">Without scaling, features with big numbers can dominate the result.</p>
            <button className="mt-2 text-fg underline underline-offset-4" onClick={() => goTo(1)}>Set scaling in Preprocessing</button>
          </div>
        </div>
      )}

      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {!models ? [0, 1, 2, 3, 4, 5].map((i) => <Skeleton key={i} className="h-[172px] rounded-card" />) : models.map((m) => {
          const on = picked.has(m.key);
          const blocked = !on && full;
          return (
            <button key={m.key} type="button" role="checkbox" aria-checked={on} aria-disabled={blocked} onClick={() => !blocked && toggle(m.key)}
              title={blocked ? `You can pick up to ${MAX_MODELS} models` : undefined}
              className={`flex flex-col rounded-card p-5 text-left ring-1 transition-[background-color,box-shadow,opacity] ${
                on ? "bg-surface-2 ring-fg/70" : "bg-surface ring-line hover:ring-line-strong"} ${blocked ? "cursor-not-allowed opacity-40" : ""}`}>
              <span className="flex items-start justify-between gap-3">
                <span className="text-[18px] tracking-[-0.02em]">{m.name}</span>
                <span className={`mt-0.5 inline-flex size-[22px] shrink-0 items-center justify-center rounded-[6px] ring-1 transition-colors ${on ? "bg-fg text-on-light ring-fg" : "ring-fg-subtle"}`}>
                  {on && <Check className="size-3.5" strokeWidth={3} />}
                </span>
              </span>
              <span className="mt-2 flex-1 text-[13.5px] leading-snug text-fg-muted">{DESCRIPTIONS[m.key]}</span>
              <span className="mt-4 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11.5px] text-fg-subtle">
                <span>speed <span className={LEVEL[m.cost === "low" ? "low" : m.cost === "high" ? "high" : "medium"]}>{m.cost === "low" ? "fast" : m.cost === "high" ? "slow" : "medium"}</span></span>
                <span>explainable <span className="text-fg-muted">{m.interpretability}</span></span>
                {m.requires_scaling && <span className="text-fg-muted">needs scaling</span>}
              </span>
            </button>
          );
        })}
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <Button loading={saving} disabled={!dirty || draft.length === 0} onClick={save}>{dirty ? "Save models" : "Saved"}</Button>
        {draft.length === 0 && <span className="text-[13px] text-fg-subtle">Pick at least one model to continue.</span>}
        {dirty && draft.length > 0 && <span className="text-[13px] text-fg-subtle">You have unsaved changes.</span>}
        {!dirty && draft.length > 0 && (
          <Button variant="ghost" onClick={() => goTo(5)}>Next: regularization →</Button>
        )}
      </div>
    </div>
  );
}
