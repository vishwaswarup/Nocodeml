"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Eyebrow } from "@/components/ui/card";
import { Segmented, Switch } from "@/components/ui/choice";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { applyAll, bodyFor, fromConfig, sameDraft, type Draft } from "@/lib/preprocessing";
import type { DatasetProfile, PipelineSaved, Preview, ProjectState, Recommendation, ScalingStrategy } from "@/lib/types";
import { ColumnEditor } from "./column-editor";
import { PreviewCard } from "./preview-card";
import { PREPROCESSING_GROUPS, RecommendationPanel } from "./recommendation-panel";

type Mode = "recommend" | "manual";

const SCALERS: { value: ScalingStrategy; label: string }[] = [
  { value: "standard", label: "Standard" }, { value: "min_max", label: "Min-max" },
  { value: "robust", label: "Robust" }, { value: "none", label: "None" },
];
const SCALER_HELP: Record<ScalingStrategy, string> = {
  standard: "Centres each feature at 0 with unit spread. A good default for linear models, SVM and KNN.",
  min_max: "Squeezes each feature into 0–1. Keeps the shape of the distribution.",
  robust: "Uses the median and IQR, so outliers have little influence.",
  none: "Leave values as they are. Fine for tree models, which don't care about scale.",
};

export function PreprocessingSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const pipeline = state.pipeline;
  if (!pipeline) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Choose what to predict first" body="Preprocessing depends on your target column. Pick it in the Dataset section."
          action={<Button size="sm" onClick={() => goTo(0)}>Go to Dataset</Button>} />
      </div>
    );
  }
  return <Loaded projectId={projectId} pipeline={pipeline} goTo={goTo} onSaved={onSaved} />;
}

function Loaded({ projectId, pipeline, onSaved }: {
  projectId: string; pipeline: NonNullable<ProjectState["pipeline"]>; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  const cfg = pipeline.config;
  const target = cfg.dataset.target_column;
  const saved = useMemo(() => fromConfig(cfg), [cfg]);

  const [draft, setDraft] = useState<Draft>(saved);
  const [mode, setMode] = useState<Mode>(() => (sameDraft(saved, fromConfig({ ...cfg, preprocessing: {}, feature_engineering: {} })) ? "recommend" : "manual"));
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [recs, setRecs] = useState<Recommendation[] | null>(null);
  const [recError, setRecError] = useState<string | null>(null);
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [recKey, setRecKey] = useState(0);

  const dirty = !sameDraft(draft, saved);

  useEffect(() => {
    let alive = true;
    api<DatasetProfile>(`/projects/${projectId}/datasets/${cfg.dataset.dataset_id}/profile?target=${encodeURIComponent(target)}`)
      .then((p) => { if (alive) setProfile(p); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, cfg.dataset.dataset_id, target]);

  useEffect(() => {
    if (mode !== "recommend") return;
    let alive = true;
    api<Recommendation[]>(`/projects/${projectId}/pipeline/recommendations/preprocessing`)
      .then((r) => { if (alive) { setRecs(r); setRecError(null); } })
      .catch((e: Error) => { if (alive) setRecError(e.message); });
    return () => { alive = false; };
  }, [projectId, mode, recKey]);

  // live validation + before/after of the unsaved draft
  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => {
      setPreviewing(true);
      api<Preview>(`/projects/${projectId}/pipeline/preview`, { method: "POST", body: JSON.stringify(bodyFor(cfg, draft)) })
        .then((p) => { if (alive) { setPreview(p); setError(null); } })
        .catch((e: Error) => { if (alive) setError(e.message); })
        .finally(() => { if (alive) setPreviewing(false); });
    }, 350);
    return () => { alive = false; clearTimeout(t); };
  }, [projectId, cfg, draft]);

  const columns = useMemo(() => Object.entries(profile?.columns ?? {}), [profile]);

  const apply = useCallback((picked: Recommendation[]) => {
    setDraft((d) => applyAll(d, picked));
    setApplied((s) => new Set([...s, ...picked.map((r) => r.id)]));
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, { method: "PUT", body: JSON.stringify(bodyFor(cfg, draft)) });
      setDraft(fromConfig(res.version.config));   // adopt exactly what the server stored
      onSaved(res.impact?.changed.length ? res.impact.message : "Preprocessing saved.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const dupes = profile?.n_duplicate_rows ?? 0;

  return (
    <div className="mx-auto max-w-[1180px]">
      <Eyebrow>Section 1 · Prepare</Eyebrow>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-h1">Preprocessing</h1>
          <p className="mt-2 max-w-xl text-[15px] text-fg-muted">
            Decide how each column is cleaned. Everything here is fitted on the training rows only, so the test set stays untouched.
          </p>
        </div>
        <Segmented label="Mode" value={mode} onChange={setMode} options={[
          { value: "recommend", label: "NoCodeML recommends" }, { value: "manual", label: "Manual" },
        ]} />
      </div>

      {error && <div className="mt-6"><ErrorState title="Something went wrong" body={error} /></div>}

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_340px]">
        <div className="min-w-0 space-y-10">
          {mode === "recommend" && (
            <RecommendationPanel recs={recs} error={recError} applied={applied} onApply={apply} groups={PREPROCESSING_GROUPS}
              onIgnore={() => setMode("manual")} onRetry={() => setRecKey((k) => k + 1)} />
          )}

          {mode === "manual" && (
            <>
              <section>
                <h2 className="mb-1 text-h3">Columns</h2>
                <p className="mb-3 text-[14px] text-fg-muted">Rows marked with a warning icon still need a choice before you can train.</p>
                {!profile ? <Skeleton className="h-72 rounded-card" /> : (
                  <ColumnEditor columns={columns} target={target} draft={draft} onChange={setDraft} />
                )}
              </section>

              <section className="grid gap-6 sm:grid-cols-2">
                <div>
                  <h2 className="mb-1 text-h3">Scaling</h2>
                  <p className="mb-3 text-[14px] text-fg-muted">{SCALER_HELP[draft.pre.scaling]}</p>
                  <Segmented label="Scaling" value={draft.pre.scaling} options={SCALERS}
                    onChange={(v) => setDraft((d) => ({ ...d, pre: { ...d.pre, scaling: v } }))} />
                </div>
                <div>
                  <h2 className="mb-1 text-h3">Duplicate rows</h2>
                  <p className="mb-3 text-[14px] text-fg-muted">
                    {dupes ? `${dupes} rows are exact duplicates.` : "No duplicate rows in this dataset."} Duplicates can end up in both training and test data.
                  </p>
                  <div className="flex items-center gap-3 text-[14px]">
                    <Switch checked={draft.pre.drop_duplicates} label="Remove duplicate rows"
                      onChange={(v) => setDraft((d) => ({ ...d, pre: { ...d.pre, drop_duplicates: v } }))} />
                    Remove duplicates
                  </div>
                </div>
              </section>
            </>
          )}
        </div>

        <aside className="space-y-3 lg:sticky lg:top-[76px] lg:self-start">
          <PreviewCard preview={preview} loading={previewing} />
          {mode === "recommend" && applied.size > 0 && (
            <Button variant="secondary" size="sm" className="w-full" onClick={() => setMode("manual")}>Review or change in Manual</Button>
          )}
          <Button className="w-full" loading={saving} disabled={!dirty} onClick={save}>
            {dirty ? "Save preprocessing" : "Saved"}
          </Button>
          {dirty && <p className="text-center text-[12px] text-fg-subtle">You have unsaved changes.</p>}
        </aside>
      </div>
    </div>
  );
}
