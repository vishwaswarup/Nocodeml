import { ArrowRight, Check } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/feedback";
import { StatusIcon } from "@/components/ui/badge";
import { int } from "@/lib/format";
import type { Preview } from "@/lib/types";

function Row({ label, a, b, good }: { label: string; a: string; b?: string; good?: boolean }) {
  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-x-3 py-2.5 text-[14px]">
      <span className="text-fg-muted">{label}</span>
      <span className="text-right font-mono tabular text-fg-muted">{a}</span>
      <ArrowRight className="size-3.5 text-fg-subtle" aria-hidden />
      <span className={`min-w-[52px] text-right font-mono tabular ${b === undefined ? "text-fg-subtle" : good ? "text-pass" : "text-fg"}`}>{b ?? "–"}</span>
    </div>
  );
}

/** Before/after of the draft (live), plus what still blocks training. */
export function PreviewCard({ preview, loading }: { preview: Preview | null; loading: boolean }) {
  const b = preview?.before, a = preview?.after;
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <p className="text-h3">Before → after</p>
        {loading && <span className="text-[12px] text-fg-subtle">updating…</span>}
      </div>
      {!preview ? (
        <div className="mt-4 space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-7" />)}</div>
      ) : (
        <>
          <div className="mt-2 divide-y divide-line">
            <Row label="Rows" a={int(b!.rows)} b={a ? int(a.rows) : undefined} good={!!a && a.rows !== b!.rows} />
            <Row label="Columns" a={int(b!.columns)} b={a ? int(a.columns) : undefined} />
            <Row label="Missing values" a={`${b!.missing_pct}%`} b={a ? `${a.missing_pct}%` : undefined} good={!!a && a.missing_cells === 0 && b!.missing_cells > 0} />
            <Row label="Duplicate rows" a={int(b!.duplicate_rows)} b={a ? int(a.duplicate_rows) : undefined} good={!!a && a.duplicate_rows === 0 && b!.duplicate_rows > 0} />
          </div>
          {a ? (
            <div className="mt-3 flex gap-2.5 rounded-control bg-pass/[0.07] px-3 py-2.5 text-[13px] text-fg-muted">
              <StatusIcon status="pass" />
              <span>Ready to continue. Preprocessing will be fitted on training rows only, never on the test set.</span>
            </div>
          ) : (
            <div className="mt-3">
              <p className="mb-2 flex items-center gap-1.5 text-[13px] text-warn">Fix these before training</p>
              <ul className="space-y-1.5">
                {preview.issues.map((i) => (
                  <li key={i} className="flex gap-2 text-[13px] leading-snug text-fg-muted">
                    <span className="mt-1.5 size-1 shrink-0 rounded-full bg-warn" aria-hidden />{i}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {a && preview.steps.filter((s) => s.rows_removed > 0 && s.step !== "drop_missing_target").map((s) => (
            <p key={s.step + s.detail} className="mt-2 flex items-center gap-1.5 text-[12px] text-fg-subtle">
              <Check className="size-3" /> {s.rows_removed} rows removed ({s.step.replaceAll("_", " ")}{s.detail ? `: ${s.detail}` : ""})
            </p>
          ))}
          {a && <p className="mt-3 text-[11.5px] leading-snug text-fg-subtle">{preview.note}</p>}
        </>
      )}
    </Card>
  );
}
