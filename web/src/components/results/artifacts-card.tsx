"use client";

import { Download, ShieldAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import { bytes } from "@/lib/format";
import type { Artifact } from "@/lib/types";
import { KIND_LABEL } from "./metrics";

export function ArtifactsCard({ projectId, number }: { projectId: string; number: number }) {
  const [items, setItems] = useState<Artifact[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api<Artifact[]>(`/projects/${projectId}/experiments/${number}/artifacts`)
      .then((a) => { if (alive) setItems(a); })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, number]);

  const download = async (a: Artifact) => {
    setBusy(a.id);
    setError(null);
    try {
      const { url } = await api<{ url: string }>(`/projects/${projectId}/artifacts/${a.id}/url`);
      window.open(url, "_blank", "noopener");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const shown = (items ?? []).filter((a) => a.storage_path.split("/").pop() !== "SECURITY.txt");
  return (
    <Card className="p-6">
      <p className="text-h3">Download</p>
      <p className="mt-1 text-[13px] text-fg-muted">The full pipeline (preprocessing + model) takes raw rows and returns predictions.</p>
      {error && <p role="alert" className="mt-3 text-[13px] text-fail">{error}</p>}
      <ul className="mt-3 divide-y divide-line">
        {items === null ? <li className="py-3 text-[13px] text-fg-subtle">Loading…</li>
          : shown.length === 0 ? <li className="py-3 text-[13px] text-fg-subtle">No files were stored for this run.</li>
          : shown.map((a) => {
            const name = a.storage_path.split("/").pop()!;
            return (
              <li key={a.id} className="flex items-center justify-between gap-3 py-2.5">
                <span className="min-w-0">
                  <span className="block truncate font-mono text-[13px]">{name}</span>
                  <span className="text-[12px] text-fg-subtle">{KIND_LABEL[a.kind] ?? a.kind} · {bytes(a.size_bytes)}</span>
                </span>
                <Button size="sm" variant="secondary" loading={busy === a.id} icon={<Download className="size-3.5" />} onClick={() => download(a)}
                  aria-label={`Download ${name}`}>Download</Button>
              </li>
            );
          })}
      </ul>
      <p className="mt-3 flex gap-2 rounded-control bg-warn/[0.07] px-3 py-2.5 text-[12.5px] leading-snug text-fg-muted">
        <ShieldAlert className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
        <span>.pkl files are Python pickles: opening one runs code. Only load files you created yourself, and never ones from someone else.</span>
      </p>
    </Card>
  );
}
