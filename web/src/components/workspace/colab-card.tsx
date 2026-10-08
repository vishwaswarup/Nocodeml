"use client";

import { Download, ExternalLink, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { StatusIcon } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/feedback";
import { ApiError, api, apiFile } from "@/lib/api";
import { COLAB_NOTEBOOK } from "@/lib/site";

type Imported = { experiment_number: number };

/** Train on the researcher's own Google Colab: we prepare the data, Colab trains, we score the predictions it returns. */
export function ColabCard({ projectId, ready, disabled, onImported }: {
  projectId: string; ready: boolean; disabled: boolean; onImported: (experimentNumber: number) => void;
}) {
  const [preparing, setPreparing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [error, setError] = useState<{ title: string; issues?: string[] } | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const fail = (e: unknown) => setError({ title: (e as Error).message, issues: e instanceof ApiError ? e.issues : undefined });

  const getBundle = async () => {
    setPreparing(true);
    setError(null);
    try {
      const { blob, filename } = await apiFile(`/projects/${projectId}/colab/bundle`, { method: "POST" });
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement("a"), { href: url, download: filename });
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setDownloaded(true);
    } catch (e) {
      fail(e);
    } finally {
      setPreparing(false);
    }
  };

  const sendResults = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await api<Imported>(`/projects/${projectId}/colab/results`, { method: "POST", body: form });
      onImported(res.experiment_number);
    } catch (e) {
      fail(e);
    } finally {
      setUploading(false);
    }
  };

  const off = !ready || disabled;
  return (
    <Card className="p-5">
      <p className="text-h3">Or train on Google Colab</p>
      <p className="mt-1.5 text-[13px] leading-snug text-fg-muted">
        Free: the training runs in your own Google Colab, not on our servers. We prepare the data here; Colab trains; you send
        back its predictions and we work out every score ourselves.
      </p>
      <ol className="mt-4 space-y-3.5 text-[13.5px]">
        <li className="flex gap-3">
          <span className="mt-0.5 font-mono text-[12px] text-fg-subtle">1</span>
          <div className="min-w-0 flex-1">
            <p>Download your prepared data</p>
            <Button size="sm" variant="secondary" className="mt-2" icon={<Download className="size-3.5" />} loading={preparing}
              disabled={off} onClick={getBundle}>Download data bundle</Button>
            {downloaded && <p className="mt-1.5 flex items-center gap-1.5 text-[12px] text-pass"><StatusIcon status="pass" /> Downloaded</p>}
          </div>
        </li>
        <li className="flex gap-3">
          <span className="mt-0.5 font-mono text-[12px] text-fg-subtle">2</span>
          <div className="min-w-0 flex-1">
            <p>Open the notebook, choose the bundle when asked, and choose <span className="text-fg">Runtime → Run all</span></p>
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
              <a href={COLAB_NOTEBOOK.openUrl} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-fg underline underline-offset-4 hover:no-underline">
                Open in Colab <ExternalLink className="size-3.5" aria-hidden />
              </a>
              <a href={COLAB_NOTEBOOK.downloadPath} download className="text-[12.5px] text-fg-muted underline underline-offset-4 hover:text-fg">
                or download the notebook
              </a>
            </div>
          </div>
        </li>
        <li className="flex gap-3">
          <span className="mt-0.5 font-mono text-[12px] text-fg-subtle">3</span>
          <div className="min-w-0 flex-1">
            <p>Upload the <span className="font-mono text-[12.5px]">nocodeml_results.json</span> it downloads</p>
            <input ref={picker} type="file" accept=".json,application/json" className="sr-only" aria-label="Choose results file"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void sendResults(f); e.target.value = ""; }} />
            <Button size="sm" variant="secondary" className="mt-2" icon={<Upload className="size-3.5" />} loading={uploading}
              disabled={off} onClick={() => picker.current?.click()}>Upload results</Button>
          </div>
        </li>
      </ol>
      {!ready && <p className="mt-4 text-[12.5px] text-fg-subtle">Fix the items above first, then the data can be prepared.</p>}
      {error && <div className="mt-4"><ErrorState title={error.title} issues={error.issues} /></div>}
    </Card>
  );
}
