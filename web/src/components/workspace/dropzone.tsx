"use client";

import { clsx } from "clsx";
import { FileUp } from "lucide-react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";

// The server enforces the real limit (NOCODEML_MAX_UPLOAD_MB); this only lets the page refuse early with a clear message.
const configuredMb = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB);
export const MAX_UPLOAD_MB = Number.isFinite(configuredMb) && configuredMb >= 1 ? Math.min(100, configuredMb) : 100;
export const MAX_UPLOAD = MAX_UPLOAD_MB * 1024 * 1024;
export const DATASET_EXTENSIONS = [".csv", ".xlsx", ".parquet"];
export const DATASET_ACCEPT = DATASET_EXTENSIONS.join(",");

export function Dropzone({ onFile, busy }: { onFile: (f: File) => void; busy: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}
      className={clsx("flex flex-col items-center rounded-card border border-dashed px-6 py-16 text-center transition-colors",
        over ? "border-fg bg-white/[0.03]" : "border-line-strong")}
    >
      <span className="mb-5 inline-flex size-12 items-center justify-center rounded-full bg-surface-2 text-fg">
        <FileUp className="size-5" />
      </span>
      <p className="text-[19px] tracking-[-0.02em]">{busy ? "Uploading and profiling…" : "Drop a dataset here"}</p>
      <p className="mt-1.5 text-[14px] text-fg-muted">CSV, Excel (.xlsx, first sheet) or Parquet. Up to {MAX_UPLOAD_MB} MB. Every column is profiled automatically.</p>
      <input ref={input} type="file" accept={DATASET_ACCEPT} className="sr-only" aria-label="Choose dataset file"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      <Button className="mt-6" loading={busy} onClick={() => input.current?.click()}>Choose file</Button>
    </div>
  );
}
