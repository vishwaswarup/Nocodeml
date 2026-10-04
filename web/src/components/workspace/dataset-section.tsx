"use client";

import { AlertTriangle, Check, FileSpreadsheet, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { StatusIcon, Tag } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, Eyebrow } from "@/components/ui/card";
import { Segmented } from "@/components/ui/choice";
import { ErrorState, Skeleton } from "@/components/ui/feedback";
import { Select } from "@/components/ui/select";
import { api, ApiError } from "@/lib/api";
import { bytes, fmt, int } from "@/lib/format";
import type { DatasetInfo, DatasetProfile, PipelineSaved, ProjectState, Task } from "@/lib/types";
import { DataViewer } from "./data-viewer";
import { Dropzone, MAX_UPLOAD } from "./dropzone";

const KIND_LABEL = { numerical: "num", categorical: "cat", datetime: "date" } as const;

export function DatasetSection({ projectId, state, onChanged }: {
  projectId: string; state: ProjectState; onChanged: () => void;
}) {
  const saved = state.pipeline?.config.dataset ?? null;
  const datasetId = saved?.dataset_id ?? state.datasets[0]?.id ?? null;

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<{ title: string; issues?: string[] } | null>(null);

  const upload = async (file: File, replaceId?: string) => {
    if (!file.name.toLowerCase().endsWith(".csv")) return setUploadError({ title: "Only .csv files are supported for now." });
    if (file.size > MAX_UPLOAD) return setUploadError({ title: `This file is ${bytes(file.size)}; the limit is 100 MB.` });
    setUploadError(null);
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    if (replaceId) form.append("dataset_id", replaceId);
    try {
      await api(`/projects/${projectId}/datasets`, { method: "POST", body: form });
      onChanged();
    } catch (e) {
      setUploadError({ title: (e as Error).message, issues: e instanceof ApiError ? e.issues : undefined });
    } finally {
      setUploading(false);
    }
  };

  if (!datasetId) {
    return (
      <div className="mx-auto max-w-3xl">
        <Eyebrow>Section 0</Eyebrow>
        <h1 className="mt-2 text-h1">Upload a dataset</h1>
        <p className="mt-2 mb-8 text-[16px] text-fg-muted">Start with a CSV. NoCodeML profiles it before you make any decisions.</p>
        <Dropzone onFile={(f) => upload(f)} busy={uploading} />
        {uploadError && <div className="mt-4"><ErrorState title={uploadError.title} issues={uploadError.issues} /></div>}
      </div>
    );
  }

  return (
    <LoadedDataset key={datasetId} projectId={projectId} datasetId={datasetId} state={state} onChanged={onChanged}
      onReplace={(f) => upload(f, datasetId)} uploading={uploading} uploadError={uploadError} />
  );
}

function LoadedDataset({ projectId, datasetId, state, onChanged, onReplace, uploading, uploadError }: {
  projectId: string; datasetId: string; state: ProjectState; onChanged: () => void;
  onReplace: (f: File) => void; uploading: boolean; uploadError: { title: string; issues?: string[] } | null;
}) {
  const saved = state.pipeline?.config.dataset ?? null;
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState<string | null>(saved?.target_column ?? null);
  const [task, setTask] = useState<Task | null>(saved?.task ?? null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "warn" | "err"; text: string } | null>(null);
  const replaceInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    api<DatasetInfo>(`/projects/${projectId}/datasets/${datasetId}`)
      .then((d) => { if (alive) setInfo(d); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, datasetId, uploading]);

  useEffect(() => {
    let alive = true;
    const qs = target ? `?target=${encodeURIComponent(target)}` : "";
    api<DatasetProfile>(`/projects/${projectId}/datasets/${datasetId}/profile${qs}`)
      .then((p) => {
        if (!alive) return;
        setProfile(p);
        if (!target && p.target_candidates[0]) setTarget(p.target_candidates[0].column);
        // follow the inferred task unless the user already saved a choice for this target
        if (p.target && !(saved && saved.target_column === p.target.column)) setTask(p.target.inferred_task);
      })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, datasetId, target, saved, uploading]);

  const latest = info?.versions[info.versions.length - 1];
  const dirty = !saved || saved.target_column !== target || saved.task !== task || saved.version !== latest?.version;

  const saveTarget = async () => {
    if (!target || !task || !latest) return;
    setSaving(true);
    setMessage(null);
    const dataset = { dataset_id: datasetId, version: latest.version, target_column: target, task };
    const c = state.pipeline?.config;
    // Keep every other section as it is; only the dataset reference changes.
    const body = c
      ? { dataset, preprocessing: c.preprocessing, feature_engineering: c.feature_engineering, split: c.split, models: c.models }
      : { dataset };
    try {
      const res = await api<PipelineSaved>(`/projects/${projectId}/pipeline`, { method: "PUT", body: JSON.stringify(body) });
      setMessage(res.impact && res.impact.changed.length
        ? { kind: "warn", text: res.impact.message }
        : { kind: "ok", text: res.created ? "Saved. Your pipeline starts from this dataset." : "Saved." });
      onChanged();
    } catch (e) {
      setMessage({ kind: "err", text: (e as Error).message });
    } finally {
      setSaving(false);
    }
  };

  const columns = useMemo(() => Object.entries(profile?.columns ?? {}), [profile]);
  const warnings = useMemo(() => profile?.warnings ?? [], [profile]);
  const colWarnings = useMemo(() => {
    const m: Record<string, string[]> = {};
    for (const w of warnings) if (w.column) (m[w.column] ??= []).push(w.code);
    return m;
  }, [warnings]);

  if (error) return <ErrorState title="Couldn't load the dataset" body={error} />;

  return (
    <div className="mx-auto max-w-[1180px] space-y-10">
      {/* header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>Section 0 · Explore</Eyebrow>
          <h1 className="mt-2 flex items-center gap-3 text-h1">
            <FileSpreadsheet className="size-8 text-fg-muted" strokeWidth={1.5} />
            {latest?.filename ?? <Skeleton className="h-9 w-56" />}
          </h1>
          {latest && (
            <p className="mt-2 text-[14px] text-fg-muted">
              Version {latest.version} · {bytes(latest.size_bytes)} · uploaded {new Date(latest.created_at).toLocaleString()}
            </p>
          )}
        </div>
        <div>
          <input ref={replaceInput} type="file" accept=".csv,text/csv" className="sr-only" aria-label="Upload new version"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onReplace(f); e.target.value = ""; }} />
          <Button variant="secondary" size="sm" icon={<RefreshCw className="size-3.5" />} loading={uploading}
            onClick={() => replaceInput.current?.click()}>Upload new version</Button>
        </div>
      </div>
      {uploadError && <ErrorState title={uploadError.title} issues={uploadError.issues} />}

      {/* overview */}
      {!profile ? (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">{[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-[92px]" />)}</div>
      ) : (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <Stat label="Rows" value={int(profile.n_rows)} />
          <Stat label="Columns" value={int(profile.n_columns)}
            sub={`${profile.numerical.length} num · ${profile.categorical.length} cat · ${profile.datetime.length} date`} />
          <Stat label="Missing values" value={`${profile.missing_value_pct}%`} tone={profile.missing_value_pct > 0 ? "warn" : undefined} />
          <Stat label="Duplicate rows" value={int(profile.n_duplicate_rows)} tone={profile.n_duplicate_rows > 0 ? "warn" : undefined} />
          <Stat label="Warnings" value={String(warnings.length)} tone={warnings.length ? "warn" : undefined} />
        </div>
      )}

      {/* target + health */}
      <div className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
        <Card className="p-6">
          <p className="text-h3">What do you want to predict?</p>
          <p className="mt-1 mb-5 text-[14px] text-fg-muted">Pick the target column. NoCodeML suggests the likeliest ones first.</p>
          {!profile ? <Skeleton className="h-10" /> : (
            <Select label="Target column" value={target ?? ""} onChange={(e) => { setTarget(e.target.value); setMessage(null); }}>
              <optgroup label="Suggested">
                {profile.target_candidates.slice(0, 3).map((c) => <option key={c.column} value={c.column}>{c.column}</option>)}
              </optgroup>
              <optgroup label="All columns">
                {columns.filter(([n]) => !profile.target_candidates.slice(0, 3).some((c) => c.column === n))
                  .map(([n]) => <option key={n} value={n}>{n}</option>)}
              </optgroup>
            </Select>
          )}
          {profile?.target && task && (
            <div className="mt-5">
              <p className="mb-2 text-[13px] text-fg-muted">Task</p>
              <Segmented label="Task" value={task} onChange={(t) => { setTask(t); setMessage(null); }} options={[
                { value: "classification", label: "Classification" }, { value: "regression", label: "Regression" },
              ]} />
              <p className="mt-2 text-[13px] text-fg-subtle">
                Inferred: <span className="text-fg-muted">{profile.target.inferred_task}</span>
                {profile.target.inferred_task === "classification"
                  ? ` (${int(profile.columns[profile.target.column]?.unique ?? 0)} distinct values)`
                  : " (continuous numbers)"}
              </p>
            </div>
          )}
          {profile?.target?.class_distribution && task === "classification" && (
            <ClassBalance dist={profile.target.class_distribution} />
          )}
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Button onClick={saveTarget} loading={saving} disabled={!dirty || !target || !task || !latest}>
              {saved && !dirty ? <><Check className="size-4" /> Saved</> : "Save target"}
            </Button>
            {message && (
              <p role={message.kind === "err" ? "alert" : "status"}
                className={`text-[13px] ${message.kind === "err" ? "text-fail" : message.kind === "warn" ? "text-warn" : "text-pass"}`}>
                {message.text}
              </p>
            )}
          </div>
        </Card>

        <Card className="p-6">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-h3">Dataset health</p>
            {profile && <span className="text-[13px] text-fg-subtle">{warnings.length ? `${warnings.length} to review` : "All clear"}</span>}
          </div>
          {!profile ? <div className="space-y-3 pt-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-10" />)}</div>
            : warnings.length === 0 ? (
              <div className="flex gap-3 py-3"><StatusIcon status="pass" /><p className="text-[15px]">No issues found in this dataset.</p></div>
            ) : (
              <ul className="max-h-[360px] divide-y divide-line overflow-y-auto pr-1">
                {warnings.map((w, i) => (
                  <li key={i} className="flex gap-3 py-3">
                    <StatusIcon status={w.severity === "warning" ? "warn" : "info"} />
                    <p className="text-[14px] leading-snug text-fg-muted">{w.message}</p>
                  </li>
                ))}
              </ul>
            )}
          <p className="mt-3 text-[12px] text-fg-subtle">These are flags, not changes. You decide what to do in Preprocessing.</p>
        </Card>
      </div>

      {/* columns */}
      <section>
        <h2 className="mb-1 text-h2">Columns</h2>
        <p className="mb-4 text-[14px] text-fg-muted">Type, completeness and a summary of every column.</p>
        {!profile ? <Skeleton className="h-64 rounded-card" /> : (
          <div className="overflow-x-auto rounded-card bg-surface ring-1 ring-line">
            <table className="w-full text-[13px]">
              <caption className="sr-only">Column statistics</caption>
              <thead>
                <tr className="border-b border-line text-left text-fg-muted">
                  <th scope="col" className="px-4 py-2.5 font-normal">Column</th>
                  <th scope="col" className="px-4 py-2.5 font-normal">Type</th>
                  <th scope="col" className="px-4 py-2.5 text-right font-normal">Missing</th>
                  <th scope="col" className="px-4 py-2.5 text-right font-normal">Unique</th>
                  <th scope="col" className="px-4 py-2.5 font-normal">Summary</th>
                  <th scope="col" className="px-4 py-2.5 font-normal">Flags</th>
                </tr>
              </thead>
              <tbody>
                {columns.map(([name, c]) => (
                  <tr key={name} className="border-b border-line/60 last:border-0">
                    <td className="px-4 py-2.5 whitespace-nowrap">
                      {name}
                      {name === target && <span className="ml-2 rounded-full bg-ember/12 px-2 py-0.5 text-[11px] text-ember">target</span>}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-[12px] text-fg-muted">{KIND_LABEL[c.kind]} <span className="text-fg-subtle">{c.dtype.replace("64", "")}</span></td>
                    <td className={`px-4 py-2.5 text-right font-mono text-[12.5px] tabular ${c.missing_pct > 0 ? "text-warn" : "text-fg-subtle"}`}>{c.missing_pct}%</td>
                    <td className="px-4 py-2.5 text-right font-mono text-[12.5px] tabular">{int(c.unique)}</td>
                    <td className="px-4 py-2.5 text-fg-muted"><Summary c={c} /></td>
                    <td className="px-4 py-2.5">
                      <div className="flex flex-wrap gap-1">
                        {(colWarnings[name] ?? []).filter((code) => code !== "missing_values").map((code) => (
                          <span key={code} className="inline-flex items-center gap-1 rounded-full bg-warn/10 px-2 py-0.5 text-[11px] text-warn">
                            <AlertTriangle className="size-3" />{code.replaceAll("_", " ")}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* raw rows */}
      <section>
        <h2 className="mb-1 text-h2">Raw data</h2>
        <p className="mb-4 text-[14px] text-fg-muted">Exactly what you uploaded. Sort by any column, search across all of them.</p>
        <DataViewer projectId={projectId} datasetId={datasetId} key={latest?.version} />
      </section>
    </div>
  );
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "warn" }) {
  return (
    <div className="rounded-card bg-surface p-4">
      <p className="text-[13px] text-fg-muted">{label}</p>
      <p className={`mt-1.5 text-[26px] leading-tight tabular ${tone === "warn" ? "text-warn" : ""}`}>{value}</p>
      {sub && <p className="mt-1 text-[12px] text-fg-subtle">{sub}</p>}
    </div>
  );
}

function Summary({ c }: { c: DatasetProfile["columns"][string] }) {
  if (c.kind === "numerical")
    return <span className="font-mono text-[12px] tabular">mean {fmt(c.mean)} · median {fmt(c.median)} · {fmt(c.min)} – {fmt(c.max)}</span>;
  if (c.kind === "datetime")
    return <span className="font-mono text-[12px]">{String(c.min ?? "–").slice(0, 10)} → {String(c.max ?? "–").slice(0, 10)}</span>;
  return <span>most common <span className="text-fg">{fmt(c.mode)}</span></span>;
}

function ClassBalance({ dist }: { dist: NonNullable<NonNullable<DatasetProfile["target"]>["class_distribution"]> }) {
  const entries = Object.entries(dist.proportions).slice(0, 8);
  return (
    <div className="mt-5">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[13px] text-fg-muted">Class balance</p>
        {dist.imbalanced && <Tag className="bg-warn/12 text-warn">imbalanced</Tag>}
      </div>
      <div className="space-y-2">
        {entries.map(([label, p]) => (
          <div key={label} className="grid grid-cols-[90px_1fr_52px] items-center gap-3 text-[13px]">
            <span className="truncate font-mono text-[12px]">{label}</span>
            <span className="h-1.5 rounded-full bg-surface-3">
              <span className="block h-full rounded-full bg-fg" style={{ width: `${p * 100}%` }} />
            </span>
            <span className="text-right font-mono text-[12px] text-fg-muted tabular">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      {dist.imbalanced && (
        <p className="mt-2 text-[12px] text-fg-subtle">
          The smallest class is {(dist.minority_ratio * 100).toFixed(1)}% of rows, so accuracy alone would be misleading.
          NoCodeML will suggest a stratified split.
        </p>
      )}
    </div>
  );
}
