"use client";

import { Info } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, Eyebrow } from "@/components/ui/card";
import { Segmented } from "@/components/ui/choice";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { Field } from "@/components/ui/field";
import { api } from "@/lib/api";
import { asModels, penaltyHelp, penaltyLabel, sameModels, useModels } from "@/lib/models";
import { pipelineBody } from "@/lib/pipeline";
import { useModelIssues } from "@/lib/use-model-issues";
import type { ModelConfig, ModelDefaults, ModelInfo, PipelineConfig, PipelineSaved, ProjectState } from "@/lib/types";
import { ParamField } from "./param-field";

export function RegularizationSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const cfg = state.pipeline?.config;
  if (!cfg || cfg.models.length === 0) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Choose your models first" body="Regularization options depend on which models you pick."
          action={<Button size="sm" onClick={() => goTo(4)}>Go to Models</Button>} />
      </div>
    );
  }
  return <Loaded projectId={projectId} cfg={cfg} goTo={goTo} onSaved={onSaved} />;
}

function Loaded({ projectId, cfg, goTo, onSaved }: {
  projectId: string; cfg: PipelineConfig; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const { models, error: loadError } = useModels(cfg.dataset.task);
  const saved = useMemo(() => asModels(cfg.models), [cfg]);
  const [draft, setDraft] = useState<ModelConfig[]>(saved);
  const [defaults, setDefaults] = useState<ModelDefaults | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const issues = useModelIssues(projectId, cfg, draft);
  const dirty = !sameModels(draft, saved);
  const byKey = useMemo(() => Object.fromEntries((models ?? []).map((m) => [m.key, m])), [models]);

  useEffect(() => {
    let alive = true;
    api<ModelDefaults>(`/projects/${projectId}/pipeline/model-defaults`)
      .then((d) => { if (alive) setDefaults(d); }).catch(() => { /* fall back to registry defaults */ });
    return () => { alive = false; };
  }, [projectId, cfg.version]);

  const update = (key: string, fn: (m: ModelConfig) => ModelConfig) =>
    setDraft((d) => d.map((m) => (m.model_key === key ? fn(m) : m)));

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, {
        method: "PUT", body: JSON.stringify(pipelineBody(cfg, { models: draft as unknown as PipelineConfig["models"] })) });
      setDraft(asModels(res.version.config.models));
      onSaved(res.impact?.changed.length ? res.impact.message : "Regularization saved.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const hasIssues = Object.values(issues).some((l) => l.length > 0);

  return (
    <div className="mx-auto max-w-[900px]">
      <Eyebrow>Section 5 · Regularization</Eyebrow>
      <h1 className="mt-2 text-h1">Regularization</h1>
      <p className="mt-2 max-w-2xl text-[15px] text-fg-muted">
        Regularization stops a model from memorizing the training data. Which controls exist depends on the model, so you only see options that apply.
        Leave everything alone to use sensible defaults.
      </p>
      {(error || loadError) && <div className="mt-6"><ErrorState title="Something went wrong" body={error ?? loadError ?? ""} /></div>}

      <div className="mt-8 space-y-4">
        {!models ? [0, 1].map((i) => <Skeleton key={i} className="h-48 rounded-card" />) : draft.map((m) => {
          const info = byKey[m.model_key];
          if (!info) return null;
          return (
            <ModelRegCard key={m.model_key} info={info} model={m} defaults={defaults?.[m.model_key]} problems={issues[m.model_key] ?? []}
              onChange={(fn) => update(m.model_key, fn)} />
          );
        })}
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <Button loading={saving} disabled={!dirty || hasIssues} onClick={save}>{dirty ? "Save regularization" : "Saved"}</Button>
        {dirty && <span className="text-[13px] text-fg-subtle">{hasIssues ? "Fix the highlighted settings to save." : "You have unsaved changes."}</span>}
        {!dirty && <Button variant="ghost" onClick={() => goTo(6)}>Next: hyperparameters and training →</Button>}
      </div>
    </div>
  );
}

function ModelRegCard({ info, model, defaults, problems, onChange }: {
  info: ModelInfo; model: ModelConfig; defaults?: ModelDefaults[string]; problems: string[];
  onChange: (fn: (m: ModelConfig) => ModelConfig) => void;
}) {
  const reg = info.regularization;
  const r = model.regularization as { type?: string; strength?: number; l1_ratio?: number };
  const type = r.type ?? (info.key === "logistic_regression" ? "l2" : reg.options[0]);
  const strengthName = info.key === "ridge" || info.key === "lasso" ? "alpha" : "C";
  const edited = Object.keys(model.regularization).length > 0 || reg.complexity_params.some((p) => p in model.hyperparameters);

  const setReg = (patch: Record<string, unknown>) => onChange((m) => {
    const next: Record<string, unknown> = { type, strength: r.strength ?? 1, ...m.regularization, ...patch };
    if (next.type === "none") delete next.strength;
    if (next.type !== "elasticnet") delete next.l1_ratio;
    else next.l1_ratio ??= 0.5;
    return { ...m, regularization: next };
  });
  const effective = (hpName: string, fallback: unknown) =>
    hpName in model.hyperparameters ? model.hyperparameters[hpName]
      : model.use_recommended_defaults && defaults && hpName in defaults.recommended ? defaults.recommended[hpName] : fallback;
  const reset = () => onChange((m) => ({
    ...m, regularization: {}, hyperparameters: Object.fromEntries(Object.entries(m.hyperparameters).filter(([k]) => !reg.complexity_params.includes(k))),
  }));

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-h3">{info.name}</p>
          <p className="mt-1 max-w-xl text-[13.5px] text-fg-muted">{reg.note}</p>
        </div>
        {edited && <Button variant="ghost" size="sm" onClick={reset}>Reset to defaults</Button>}
      </div>

      {reg.kind === "none" && (
        <p className="mt-4 flex gap-2 rounded-control bg-surface-2 px-3.5 py-3 text-[14px] text-fg-muted">
          <Info className="mt-0.5 size-4 shrink-0" />Nothing to configure. If you want regularization on a linear model, add Ridge or Lasso in the Models section.
        </p>
      )}

      {reg.kind === "penalty" && (
        <div className="mt-5 space-y-5">
          {reg.options.length > 1 ? (
            <div>
              <p className="mb-2 text-[13px] text-fg-muted">Penalty</p>
              <Segmented label={`${info.name} penalty`} value={type} onChange={(t) => setReg({ type: t })}
                options={reg.options.map((o) => ({ value: o, label: penaltyLabel[o] ?? o }))} />
            </div>
          ) : (
            <p className="text-[14px]">Penalty: <span className="rounded-full bg-white/10 px-2.5 py-0.5">{penaltyLabel[reg.options[0]] ?? reg.options[0]}</span></p>
          )}
          <p className="text-[13px] text-fg-muted">{penaltyHelp[type]}</p>
          {type !== "none" && (
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label={strengthName === "C" ? "C (inverse strength)" : "alpha (strength)"} type="number" step="any" min={0}
                value={String(r.strength ?? 1)} onChange={(e) => { const n = Number(e.target.value); if (Number.isFinite(n)) setReg({ strength: n }); }}
                hint={strengthName === "C" ? "Smaller C = stronger regularization. Default 1." : "Larger alpha = stronger regularization. Default 1."} />
              {type === "elasticnet" && (
                <Field label="l1_ratio" type="number" step="0.1" min={0} max={1} value={String(r.l1_ratio ?? 0.5)}
                  onChange={(e) => { const n = Number(e.target.value); if (Number.isFinite(n)) setReg({ l1_ratio: n }); }}
                  hint="0 = pure L2, 1 = pure L1." />
              )}
            </div>
          )}
        </div>
      )}

      {reg.kind === "complexity" && (
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          {info.hyperparameters.filter((h) => reg.complexity_params.includes(h.name)).map((h) => (
            <ParamField key={h.name} hp={h} value={effective(h.name, h.default)}
              onChange={(v) => onChange((m) => ({ ...m, hyperparameters: { ...m.hyperparameters, [h.name]: v } }))}
              note={!(h.name in model.hyperparameters) && model.use_recommended_defaults && defaults && h.name in defaults.recommended ? "(recommended for your data)" : undefined} />
          ))}
        </div>
      )}

      {problems.length > 0 && (
        <ul role="alert" className="mt-4 space-y-1 rounded-control bg-fail/[0.07] px-3.5 py-3">
          {problems.map((p) => <li key={p} className="text-[13px] text-fail">{p}</li>)}
        </ul>
      )}
    </Card>
  );
}
