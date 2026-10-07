"use client";

import { LogOut, Plus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState, type FormEvent } from "react";
import { RequireAuth, useAuth } from "@/components/auth-provider";
import { Logo } from "@/components/logo";
import { StatusPill, Tag } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback";
import { Field } from "@/components/ui/field";
import { api } from "@/lib/api";

type ProjectSummary = { id: string; name: string; updated_at: string; pipeline_version: string | null; status: string };

function ago(iso: string) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

function Projects() {
  const { user, signOut } = useAuth();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  const [reloadKey, setReloadKey] = useState(0);
  const [confirmId, setConfirmId] = useState<string | null>(null);   // project whose delete is awaiting confirmation
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api<ProjectSummary[]>("/projects")
      .then((p) => { if (alive) setProjects(p); })
      .catch((e: Error) => {
        if (!alive) return;
        setError(e instanceof TypeError ? "Can't reach the NoCodeML API. Is it running?" : e.message);
      });
    return () => { alive = false; };
  }, [reloadKey]);

  const load = useCallback(() => {
    setError(null);
    setReloadKey((k) => k + 1);
  }, []);

  const create = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await api("/projects", { method: "POST", body: JSON.stringify({ name }) });
      setName("");
      setCreating(false);
      load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const remove = async (id: string) => {
    setDeletingId(id);
    try {
      await api(`/projects/${id}`, { method: "DELETE" });
      setProjects((list) => list?.filter((p) => p.id !== id) ?? null);
      setConfirmId(null);
    } catch (err) {
      setConfirmId(null);
      setError((err as Error).message);
    } finally {
      setDeletingId(null);
    }
  };

  const avatar = user?.user_metadata?.avatar_url as string | undefined;
  const display = (user?.user_metadata?.full_name as string | undefined) ?? user?.email;

  return (
    <div className="min-h-screen">
      <header className="flex h-16 items-center justify-between border-b border-line px-4 sm:px-8">
        <Link href="/" aria-label="NoCodeML home"><Logo /></Link>
        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-2 text-[14px] text-fg-muted sm:flex">
            {avatar
              // eslint-disable-next-line @next/next/no-img-element
              ? <img src={avatar} alt="" className="size-7 rounded-full" referrerPolicy="no-referrer" />
              : <span className="inline-flex size-7 items-center justify-center rounded-full bg-surface-3 text-[12px] text-fg">{display?.[0]?.toUpperCase()}</span>}
            {display}
          </span>
          <Button variant="ghost" size="sm" icon={<LogOut className="size-3.5" />} onClick={signOut}>Sign out</Button>
        </div>
      </header>

      <main className="mx-auto max-w-[1080px] px-4 py-12 sm:px-8">
        <div className="mb-8 flex items-end justify-between gap-4">
          <div>
            <h1 className="text-h1">Projects</h1>
            <p className="mt-2 text-[15px] text-fg-muted">Each project holds a dataset, its pipeline versions and experiments.</p>
          </div>
          {!creating && <Button icon={<Plus className="size-4" />} onClick={() => setCreating(true)}>New project</Button>}
        </div>

        {creating && (
          <form onSubmit={create} className="mb-6 flex flex-col gap-3 rounded-card bg-surface p-5 sm:flex-row sm:items-end">
            <Field label="Project name" autoFocus required maxLength={120} value={name}
              onChange={(e) => setName(e.target.value)} placeholder="Customer churn" className="flex-1" />
            <div className="flex gap-2">
              <Button type="submit" loading={saving}>Create</Button>
              <Button type="button" variant="ghost" onClick={() => { setCreating(false); setName(""); }}>Cancel</Button>
            </div>
          </form>
        )}

        {error && <div className="mb-6"><ErrorState title="Something went wrong" body={error} onRetry={load} /></div>}

        {projects === null && !error ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => <Skeleton key={i} className="h-[132px] rounded-card" />)}
          </div>
        ) : projects && projects.length === 0 ? (
          <EmptyState title="No projects yet" body="Create a project, upload a CSV, and run your first comparison."
            action={!creating && <Button size="sm" onClick={() => setCreating(true)}>New project</Button>} />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {projects?.map((p) => (
              <div key={p.id} className="group relative">
                {confirmId === p.id ? (
                  <div role="alertdialog" aria-label={`Delete ${p.name}`}
                    className="flex h-full min-h-[132px] flex-col justify-between rounded-card bg-surface p-5 ring-1 ring-fail/40">
                    <p className="text-[14px] leading-snug text-fg-muted">
                      Delete <span className="text-fg">{p.name}</span>? This permanently removes its datasets, pipeline
                      versions, experiments and reports. It can&apos;t be undone.
                    </p>
                    <div className="mt-4 flex items-center gap-2">
                      <Button variant="danger" size="sm" autoFocus loading={deletingId === p.id} onClick={() => remove(p.id)}>Yes, delete</Button>
                      <Button variant="ghost" size="sm" disabled={deletingId === p.id} onClick={() => setConfirmId(null)}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <Link href={`/projects/${p.id}`}
                      className="flex h-full flex-col rounded-card bg-surface p-5 ring-1 ring-transparent transition-[background-color,box-shadow] hover:bg-surface-2 hover:ring-line-strong">
                      <div className="flex items-start justify-between gap-3 pr-8">
                        <p className="text-[18px] leading-snug tracking-[-0.02em]">{p.name}</p>
                        {p.pipeline_version && <Tag>{p.pipeline_version}</Tag>}
                      </div>
                      <div className="mt-auto flex items-center justify-between pt-8 text-[13px] text-fg-subtle">
                        <span>Updated {ago(p.updated_at)}</span>
                        {p.status === "Finalized" ? <StatusPill status="pass">Finalized</StatusPill>
                          : p.status === "Experimenting" ? <StatusPill status="info">Experimenting</StatusPill>
                          : <span className="text-fg-subtle">Empty</span>}
                      </div>
                    </Link>
                    <button type="button" aria-label={`Delete project ${p.name}`} title="Delete project"
                      onClick={() => setConfirmId(p.id)}
                      className="absolute top-3 right-3 inline-flex size-8 items-center justify-center rounded-full text-fg-subtle transition-colors hover:bg-fail/15 hover:text-fail focus-visible:text-fail">
                      <Trash2 className="size-4" aria-hidden />
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default function ProjectsPage() {
  return <RequireAuth><Projects /></RequireAuth>;
}
