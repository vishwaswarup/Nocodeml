"use client";

import { ArrowLeft, ArrowRight, ChevronRight, LogOut } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/components/auth-provider";
import { Logo } from "@/components/logo";
import { Tag } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorState, Skeleton } from "@/components/ui/feedback";
import { SectionRail, type SectionItem } from "@/components/ui/section-rail";
import { api, ApiError } from "@/lib/api";
import type { ProjectState } from "@/lib/types";
import { DatasetSection } from "./dataset-section";
import { PreprocessingSection } from "./preprocessing-section";
import { ModelsSection } from "./models-section";
import { RegularizationSection } from "./regularization-section";
import { ResultsSection } from "../results/results-section";
import { SplitSection } from "./split-section";
import { TrainingSection } from "./training-section";
import { isConfigured } from "@/lib/preprocessing";
import { FeaturesSection } from "./features-section";
import { isFeatConfigured } from "@/lib/features";
import { SECTIONS } from "./sections";

export function Workspace({ projectId }: { projectId: string }) {
  const { signOut } = useAuth();
  const [state, setState] = useState<ProjectState | null>(null);
  const [error, setError] = useState<{ status: number; message: string } | null>(null);
  const [section, setSection] = useState(0);
  const [reloadKey, setReloadKey] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const goTo = useCallback((n: number) => { setSection(n); setNotice(null); }, []);

  useEffect(() => {
    let alive = true;
    api<ProjectState>(`/projects/${projectId}`)
      .then((s) => { if (alive) setState(s); })
      .catch((e: Error) => {
        if (!alive) return;
        setError({ status: e instanceof ApiError ? e.status : 0,
          message: e instanceof TypeError ? "Can't reach the NoCodeML API. Is it running?" : e.message });
      });
    return () => { alive = false; };
  }, [projectId, reloadKey]);

  const reload = useCallback(() => { setError(null); setReloadKey((k) => k + 1); }, []);

  const hasPipeline = !!state?.pipeline;
  const items: SectionItem[] = SECTIONS.map((s) => ({
    n: s.n,
    label: s.label,
    state: s.n === section ? "current"
      : s.n === 0 && hasPipeline ? "done"
      : s.n === 1 && state?.pipeline && isConfigured(state.pipeline.config) ? "done"
      : s.n === 2 && state?.pipeline && isFeatConfigured(state.pipeline.config) ? "done" : "todo",
  }));
  const canContinue = section === 0 ? hasPipeline : section < SECTIONS.length - 1;
  const saved = useCallback((msg?: string) => { setNotice(msg ?? null); reload(); }, [reload]);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 flex h-14 shrink-0 items-center justify-between border-b border-line bg-bg/85 px-4 backdrop-blur-md sm:px-6">
        <div className="flex min-w-0 items-center gap-2 text-[14px]">
          <Link href="/projects" aria-label="All projects"><Logo size={18} /></Link>
          <ChevronRight className="hidden size-4 shrink-0 text-fg-subtle sm:block" aria-hidden />
          <Link href="/projects" className="hidden text-fg-muted hover:text-fg sm:inline">Projects</Link>
          <ChevronRight className="size-4 shrink-0 text-fg-subtle" aria-hidden />
          <span className="truncate text-fg">{state?.project.name ?? "…"}</span>
          {state?.pipeline && <Tag className="ml-1">v{state.pipeline.version}</Tag>}
        </div>
        <div className="flex items-center gap-3">
          {state && <span className="hidden text-[13px] text-fg-subtle sm:inline">Saved</span>}
          <Button variant="ghost" size="sm" icon={<LogOut className="size-3.5" />} onClick={signOut}>Sign out</Button>
        </div>
      </header>

      <div className="flex flex-1">
        <aside className="hidden w-[248px] shrink-0 border-r border-line p-3 md:block">
          <div className="sticky top-[68px]">
            <p className="px-3 pt-2 pb-3 font-mono text-[11px] tracking-[0.08em] text-fg-subtle uppercase">Pipeline</p>
            <SectionRail items={items} onSelect={goTo} />
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          {/* compact section picker on small screens */}
          <div className="flex gap-1 overflow-x-auto border-b border-line px-3 py-2 md:hidden">
            {SECTIONS.map((s) => (
              <button key={s.n} onClick={() => goTo(s.n)} aria-current={s.n === section ? "step" : undefined}
                className={`shrink-0 rounded-full px-3 py-1.5 text-[13px] ${s.n === section ? "bg-fg text-on-light" : "bg-surface text-fg-muted"}`}>
                {s.n} {s.label}
              </button>
            ))}
          </div>

          <main className="flex-1 px-4 py-8 sm:px-8 lg:px-12">
            {notice && (
              <div role="status" className="mx-auto mb-6 flex max-w-[1180px] items-start justify-between gap-4 rounded-card bg-surface px-4 py-3 text-[14px] text-fg-muted ring-1 ring-line">
                <span>{notice}</span>
                <button onClick={() => setNotice(null)} className="shrink-0 text-fg-subtle hover:text-fg" aria-label="Dismiss">Dismiss</button>
              </div>
            )}
            {error ? (
              <div className="mx-auto max-w-xl pt-10">
                <ErrorState title={error.status === 404 ? "Project not found" : "Couldn't load this project"}
                  body={error.status === 404 ? "It may have been deleted, or it belongs to another account." : error.message}
                  onRetry={error.status === 404 ? undefined : reload} />
              </div>
            ) : !state ? (
              <div className="space-y-4">
                <Skeleton className="h-9 w-64" />
                <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}</div>
                <Skeleton className="h-80" />
              </div>
            ) : section === 0 ? (
              <DatasetSection projectId={projectId} state={state} onChanged={reload} />
            ) : section === 1 ? (
              <PreprocessingSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : section === 2 ? (
              <FeaturesSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : section === 3 ? (
              <SplitSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : section === 4 ? (
              <ModelsSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : section === 5 ? (
              <RegularizationSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : section === 6 ? (
              <TrainingSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : section === 7 ? (
              <ResultsSection projectId={projectId} state={state} goTo={goTo} onSaved={saved} />
            ) : null}
          </main>

          <footer className="sticky bottom-0 flex h-16 items-center justify-between border-t border-line bg-bg/85 px-4 backdrop-blur-md sm:px-8">
            <Button variant="ghost" icon={<ArrowLeft className="size-4" />} disabled={section === 0}
              onClick={() => goTo(Math.max(0, section - 1))}>Back</Button>
            <Button disabled={!canContinue} onClick={() => goTo(Math.min(SECTIONS.length - 1, section + 1))}>
              Continue <ArrowRight className="size-4" />
            </Button>
          </footer>
        </div>
      </div>
    </div>
  );
}
