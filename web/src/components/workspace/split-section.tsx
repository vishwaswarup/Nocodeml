"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Eyebrow } from "@/components/ui/card";
import { Segmented } from "@/components/ui/choice";
import { EmptyState, ErrorState } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { pipelineBody } from "@/lib/pipeline";
import { applySplitActions, sameSplit, splitFromConfig } from "@/lib/split";
import type { DatasetProfile, PipelineSaved, Preview, ProjectState, Recommendation, SplitConfig } from "@/lib/types";
import { RecommendationPanel } from "./recommendation-panel";
import { SplitControls } from "./split-controls";
import { SplitPreview } from "./split-preview";

type Mode = "recommend" | "manual";

export function SplitSection({ projectId, state, goTo, onSaved }: {
  projectId: string; state: ProjectState; goTo: (n: number) => void; onSaved: (notice?: string) => void;
}) {
  if (!state.pipeline) {
    return (
      <div className="mx-auto max-w-xl pt-6">
        <EmptyState title="Choose what to predict first" body="The split depends on your target column. Pick it in the Dataset section."
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
  const saved = useMemo(() => splitFromConfig(cfg), [cfg]);
  const [draft, setDraft] = useState<SplitConfig>(saved);
  const [mode, setMode] = useState<Mode>("recommend");
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [recs, setRecs] = useState<Recommendation[] | null>(null);
  const [recError, setRecError] = useState<string | null>(null);
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [recKey, setRecKey] = useState(0);
  const dirty = !sameSplit(draft, saved);

  useEffect(() => {
    let alive = true;
    api<DatasetProfile>(`/projects/${projectId}/datasets/${cfg.dataset.dataset_id}/profile?target=${encodeURIComponent(cfg.dataset.target_column)}`)
      .then((p) => { if (alive) setProfile(p); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, cfg.dataset]);

  useEffect(() => {
    if (mode !== "recommend") return;
    let alive = true;
    api<Recommendation[]>(`/projects/${projectId}/pipeline/recommendations/split`)
      .then((r) => { if (alive) { setRecs(r); setRecError(null); } })
      .catch((e: Error) => { if (alive) setRecError(e.message); });
    return () => { alive = false; };
  }, [projectId, mode, recKey]);

  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => {
      setPreviewing(true);
      api<Preview>(`/projects/${projectId}/pipeline/preview`, { method: "POST", body: JSON.stringify(pipelineBody(cfg, { split: draft })) })
        .then((p) => { if (alive) { setPreview(p); setError(null); } })
        .catch((e: Error) => { if (alive) setError(e.message); })
        .finally(() => { if (alive) setPreviewing(false); });
    }, 350);
    return () => { alive = false; clearTimeout(t); };
  }, [projectId, cfg, draft]);

  const apply = useCallback((picked: Recommendation[]) => {
    setDraft((d) => applySplitActions(d, picked));
    setApplied((s) => new Set([...s, ...picked.map((r) => r.id)]));
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, { method: "PUT", body: JSON.stringify(pipelineBody(cfg, { split: draft })) });
      setDraft(splitFromConfig(res.version.config));
      onSaved(res.impact?.changed.length ? res.impact.message : "Split saved.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const dateColumns = profile?.datetime ?? [];
  const invalid = !!preview && !preview.split;

  return (
    <div className="mx-auto max-w-[1180px]">
      <Eyebrow>Section 3 · Split</Eyebrow>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-h1">Dataset split</h1>
          <p className="mt-2 max-w-xl text-[15px] text-fg-muted">
            Decide which rows the models learn from and which rows they are judged on. The judged rows must never be used for learning.
          </p>
        </div>
        <Segmented label="Mode" value={mode} onChange={setMode} options={[
          { value: "recommend", label: "NoCodeML recommends" }, { value: "manual", label: "Manual" }]} />
      </div>

      {error && <div className="mt-6"><ErrorState title="Something went wrong" body={error} /></div>}

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_340px]">
        <div className="min-w-0">
          {mode === "recommend" ? (
            <RecommendationPanel recs={recs} error={recError} applied={applied} onApply={apply} noun="suggestions"
              onIgnore={() => setMode("manual")} onRetry={() => setRecKey((k) => k + 1)} />
          ) : (
            <SplitControls draft={draft} onChange={setDraft} task={cfg.dataset.task} dateColumns={dateColumns} />
          )}
        </div>
        <aside className="space-y-3 lg:sticky lg:top-[76px] lg:self-start">
          <SplitPreview preview={preview} loading={previewing} />
          {mode === "recommend" && applied.size > 0 && (
            <Button variant="secondary" size="sm" className="w-full" onClick={() => setMode("manual")}>Review or change in Manual</Button>
          )}
          <Button className="w-full" loading={saving} disabled={!dirty || invalid} onClick={save}>{dirty ? "Save split" : "Saved"}</Button>
          {dirty && <p className="text-center text-[12px] text-fg-subtle">You have unsaved changes.</p>}
        </aside>
      </div>
    </div>
  );
}
