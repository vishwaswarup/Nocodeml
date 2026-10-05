"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { StatusIcon } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, Eyebrow } from "@/components/ui/card";
import { Segmented } from "@/components/ui/choice";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { applyFeatureAction, featFromConfig, sameFeat } from "@/lib/features";
import { pipelineBody } from "@/lib/pipeline";
import type { DatasetProfile, FeatureEngineering, PipelineSaved, Preview, ProjectState, Recommendation } from "@/lib/types";
import { DatePartsEditor, NumericTransforms } from "./feature-editors";
import { RecommendationPanel, type Group } from "./recommendation-panel";

const GROUPS: Group[] = [
  { kinds: ["numeric_transform"], label: "Transform numbers" },
  { kinds: ["date_features"], label: "Dates → numbers" },
];

export function FeaturesSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  if (!state.pipeline) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Choose what to predict first" body="Feature engineering works on your dataset's columns. Pick the target in the Dataset section."
          action={<Button size="sm" onClick={() => goTo(0)}>Go to Dataset</Button>} />
      </div>
    );
  }
  return <Loaded projectId={projectId} pipeline={state.pipeline} onSaved={onSaved} />;
}

function Loaded({ projectId, pipeline, onSaved }: {
  projectId: string; pipeline: NonNullable<ProjectState["pipeline"]>; onSaved: (notice?: string) => void;
}) {
  const cfg = pipeline.config;
  const saved = useMemo(() => featFromConfig(cfg), [cfg]);
  const [draft, setDraft] = useState<FeatureEngineering>(saved);
  const [mode, setMode] = useState<"recommend" | "manual">(() => (saved.numeric_transforms.length + saved.date_features.length > 0 ? "manual" : "recommend"));
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [recs, setRecs] = useState<Recommendation[] | null>(null);
  const [recError, setRecError] = useState<string | null>(null);
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [recKey, setRecKey] = useState(0);
  const dirty = !sameFeat(draft, saved);
  const dropped = useMemo(() => new Set((cfg.preprocessing as { drop_columns?: string[] }).drop_columns ?? []), [cfg]);

  useEffect(() => {
    let alive = true;
    api<DatasetProfile>(`/projects/${projectId}/datasets/${cfg.dataset.dataset_id}/profile?target=${encodeURIComponent(cfg.dataset.target_column)}`)
      .then((p) => { if (alive) setProfile(p); }).catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, cfg.dataset]);

  useEffect(() => {
    if (mode !== "recommend") return;
    let alive = true;
    api<Recommendation[]>(`/projects/${projectId}/pipeline/recommendations/features`)
      .then((r) => { if (alive) { setRecs(r); setRecError(null); } }).catch((e: Error) => { if (alive) setRecError(e.message); });
    return () => { alive = false; };
  }, [projectId, mode, recKey]);

  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => {
      setPreviewing(true);
      api<Preview>(`/projects/${projectId}/pipeline/preview`, { method: "POST", body: JSON.stringify(pipelineBody(cfg, { feature_engineering: draft })) })
        .then((p) => { if (alive) { setPreview(p); setError(null); } }).catch((e: Error) => { if (alive) setError(e.message); })
        .finally(() => { if (alive) setPreviewing(false); });
    }, 350);
    return () => { alive = false; clearTimeout(t); };
  }, [projectId, cfg, draft]);

  const apply = useCallback((picked: Recommendation[]) => {
    setDraft((d) => picked.reduce((acc, r) => applyFeatureAction(acc, r.action), d));
    setApplied((s) => new Set([...s, ...picked.map((r) => r.id)]));
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, { method: "PUT", body: JSON.stringify(pipelineBody(cfg, { feature_engineering: draft })) });
      setDraft(featFromConfig(res.version.config));
      onSaved(res.impact?.changed.length ? res.impact.message : "Features saved.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const usable = (kind: string) => Object.entries(profile?.columns ?? {}).filter(([n, c]) =>
    c.kind === kind && n !== cfg.dataset.target_column && !dropped.has(n) && !c.identifier_like);
  const bad = !!preview && preview.feature_issues.length > 0;

  return (
    <div className="mx-auto max-w-[1180px]">
      <Eyebrow>Section 2 · Features</Eyebrow>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-h1">Feature engineering</h1>
          <p className="mt-2 max-w-xl text-[15px] text-fg-muted">
            Create new columns from existing ones when they help a model see a pattern. This step is optional: skip it if you&apos;re unsure.
          </p>
        </div>
        <Segmented label="Mode" value={mode} onChange={setMode} options={[
          { value: "recommend", label: "NoCodeML recommends" }, { value: "manual", label: "Manual" }]} />
      </div>
      {error && <div className="mt-6"><ErrorState title="Something went wrong" body={error} /></div>}

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_340px]">
        <div className="min-w-0 space-y-4">
          {mode === "recommend" ? (
            <RecommendationPanel recs={recs} error={recError} applied={applied} onApply={apply} groups={GROUPS} noun="suggestions"
              onIgnore={() => setMode("manual")} onRetry={() => setRecKey((k) => k + 1)} />
          ) : !profile ? (
            <><Skeleton className="h-64 rounded-card" /><Skeleton className="h-48 rounded-card" /></>
          ) : (
            <>
              <NumericTransforms columns={usable("numerical")} draft={draft} onChange={setDraft} />
              <DatePartsEditor columns={usable("datetime")} draft={draft} onChange={setDraft} />
            </>
          )}
        </div>
        <aside className="space-y-3 lg:sticky lg:top-[76px] lg:self-start">
          <Card className="p-5">
            <div className="flex items-center justify-between">
              <p className="text-h3">New columns</p>
              {previewing && <span className="text-[12px] text-fg-subtle">updating…</span>}
            </div>
            {!preview ? <Skeleton className="mt-4 h-24" /> : bad ? (
              <div className="mt-3">
                <p className="mb-2 text-[13px] text-warn">Fix these first</p>
                <ul className="space-y-1.5">{preview.feature_issues.map((i) => (
                  <li key={i} className="flex gap-2 text-[13px] leading-snug text-fg-muted"><span className="mt-1.5 size-1 shrink-0 rounded-full bg-warn" aria-hidden />{i}</li>))}</ul>
              </div>
            ) : preview.features.length === 0 ? (
              <p className="mt-3 text-[13.5px] text-fg-muted">No new columns yet. Your data is used as it is.</p>
            ) : (
              <ul className="mt-3 divide-y divide-line">
                {preview.features.map((f) => (
                  <li key={f.name} className="py-2.5">
                    <p className="font-mono text-[12.5px]">{f.name}</p>
                    <p className="mt-0.5 truncate font-mono text-[11.5px] text-fg-subtle tabular">{f.sample.map((v) => (v === null ? "∅" : typeof v === "number" ? +v.toFixed(2) : v)).join(", ")} …</p>
                  </li>
                ))}
              </ul>
            )}
            {preview && !bad && preview.features.length > 0 && (
              <div className="mt-3 flex gap-2.5 rounded-control bg-pass/[0.07] px-3 py-2.5 text-[13px] text-fg-muted">
                <StatusIcon status="pass" /><span>{preview.features.length} new column{preview.features.length === 1 ? "" : "s"}. Row-by-row maths only, so nothing leaks from the test set.</span>
              </div>
            )}
          </Card>
          {mode === "recommend" && applied.size > 0 && <Button variant="secondary" size="sm" className="w-full" onClick={() => setMode("manual")}>Review or change in Manual</Button>}
          <Button className="w-full" loading={saving} disabled={!dirty || bad} onClick={save}>{dirty ? "Save features" : "Saved"}</Button>
          {dirty && <p className="text-center text-[12px] text-fg-subtle">You have unsaved changes.</p>}
        </aside>
      </div>
    </div>
  );
}
