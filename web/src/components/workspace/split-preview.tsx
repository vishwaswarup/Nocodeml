import { Card } from "@/components/ui/card";
import { StatusIcon } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/feedback";
import { int } from "@/lib/format";
import type { Preview } from "@/lib/types";

/** How the (post-preprocessing) rows are divided, as a proportional bar. */
export function SplitPreview({ preview, loading }: { preview: Preview | null; loading: boolean }) {
  const s = preview?.split;
  const total = s ? s.n_train + s.n_test + s.n_validation : 0;
  const parts = s ? [
    { label: "Train", n: s.n_train, cls: "bg-fg" },
    ...(s.n_validation ? [{ label: "Validation", n: s.n_validation, cls: "bg-sky" }] : []),
    { label: "Test", n: s.n_test, cls: "bg-ember" },
  ] : [];
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <p className="text-h3">Your split</p>
        {loading && <span className="text-[12px] text-fg-subtle">updating…</span>}
      </div>
      {!preview ? <Skeleton className="mt-4 h-24" /> : s ? (
        <>
          <p className="mt-1 text-[13px] text-fg-muted">
            {s.mode === "cv" ? `${s.n_folds} folds. Each fold trains on one part and tests on another:` : `${int(total)} rows after preprocessing:`}
          </p>
          <div className="mt-3 flex h-3 overflow-hidden rounded-full bg-surface-3" role="img"
            aria-label={parts.map((p) => `${p.label} ${p.n} rows`).join(", ")}>
            {parts.map((p) => <span key={p.label} className={p.cls} style={{ width: `${(100 * p.n) / total}%` }} />)}
          </div>
          <ul className="mt-3 space-y-1.5">
            {parts.map((p) => (
              <li key={p.label} className="flex items-center justify-between text-[13px]">
                <span className="flex items-center gap-2 text-fg-muted"><span className={`size-2 rounded-full ${p.cls}`} />{p.label}{s.mode === "cv" ? " (per fold)" : ""}</span>
                <span className="font-mono tabular">{int(p.n)} <span className="text-fg-subtle">· {((100 * p.n) / total).toFixed(0)}%</span></span>
              </li>
            ))}
          </ul>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {s.stratified && <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-[12px]">stratified</span>}
            {s.chronological && <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-[12px]">chronological</span>}
          </div>
          {s.notes.map((n) => <p key={n} className="mt-2 text-[12px] text-warn">{n}</p>)}
          <div className="mt-3 flex gap-2.5 rounded-control bg-pass/[0.07] px-3 py-2.5 text-[13px] text-fg-muted">
            <StatusIcon status="pass" /><span>Valid split. {s.mode === "holdout" ? "The test rows are never seen during training." : "Preprocessing is re-fitted inside every fold."}</span>
          </div>
        </>
      ) : (
        <div className="mt-3">
          <p className="mb-2 text-[13px] text-warn">Fix these first</p>
          <ul className="space-y-1.5">
            {preview.split_issues.map((i) => (
              <li key={i} className="flex gap-2 text-[13px] leading-snug text-fg-muted">
                <span className="mt-1.5 size-1 shrink-0 rounded-full bg-warn" aria-hidden />{i}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
