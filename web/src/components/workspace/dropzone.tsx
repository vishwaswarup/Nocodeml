"use client";

import { clsx } from "clsx";
import { FileUp } from "lucide-react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";

export const MAX_UPLOAD = 100 * 1024 * 1024;

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
      <p className="text-[19px] tracking-[-0.02em]">{busy ? "Uploading and profiling…" : "Drop a CSV file here"}</p>
      <p className="mt-1.5 text-[14px] text-fg-muted">Up to 100 MB. Every column is profiled automatically.</p>
      <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" aria-label="Choose CSV file"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      <Button className="mt-6" loading={busy} onClick={() => input.current?.click()}>Choose file</Button>
    </div>
  );
}
