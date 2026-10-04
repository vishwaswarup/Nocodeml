"use client";

import { Check, Sparkles } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState, Skeleton } from "@/components/ui/feedback";
import type { Recommendation } from "@/lib/types";

const GROUPS: { kinds: string[]; label: string }[] = [
  { kinds: ["drop_column", "drop_duplicates"], label: "Clean up" },
  { kinds: ["impute"], label: "Missing values" },
  { kinds: ["encode", "date_features"], label: "Text and dates → numbers" },
  { kinds: ["outliers", "scale"], label: "Ranges" },
];

/**
 * "NoCodeML recommends": every suggestion has a reason; the user ticks what to apply.
 * Nothing is applied until "Apply selected" is pressed.
 */
export function RecommendationPanel({ recs, error, applied, onApply, onIgnore, onRetry }: {
  recs: Recommendation[] | null; error: string | null; applied: Set<string>;
  onApply: (picked: Recommendation[]) => void; onIgnore: () => void; onRetry: () => void;
}) {
  // Everything not yet applied is ticked by default; the user unticks what they don't want.
  const [unticked, setUnticked] = useState<Set<string>>(new Set());
  const isPicked = (id: string) => !unticked.has(id);

  if (error) return <ErrorState title="Couldn't get recommendations" body={error} onRetry={onRetry} />;
  if (!recs) return <div className="space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-[76px] rounded-card" />)}</div>;
  if (recs.length === 0) {
    return <Card className="p-6 text-[15px] text-fg-muted">Nothing to suggest: this dataset looks ready to use as it is.</Card>;
  }

  const pending = recs.filter((r) => !applied.has(r.id));
  const chosen = pending.filter((r) => isPicked(r.id));
  const toggle = (id: string) => setUnticked((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <p className="flex items-center gap-2 text-[15px]">
          <Sparkles className="size-4 text-ember" aria-hidden />
          NoCodeML recommends {recs.length} changes
          {applied.size > 0 && <span className="text-fg-subtle">· {applied.size} applied</span>}
        </p>
        <div className="flex gap-2">
          <Button size="sm" disabled={chosen.length === 0} onClick={() => onApply(chosen)}>
            Apply selected{chosen.length ? ` (${chosen.length})` : ""}
          </Button>
          <Button size="sm" variant="ghost" onClick={onIgnore}>Ignore</Button>
        </div>
      </div>

      <div className="space-y-6">
        {GROUPS.map((g) => {
          const items = recs.filter((r) => g.kinds.includes(r.kind));
          if (!items.length) return null;
          return (
            <section key={g.label} aria-label={g.label}>
              <p className="mb-2 font-mono text-[11px] tracking-[0.08em] text-fg-subtle uppercase">{g.label}</p>
              <ul className="space-y-2">
                {items.map((r) => {
                  const done = applied.has(r.id);
                  return (
                    <li key={r.id}>
                      <label className={`flex cursor-pointer gap-3.5 rounded-card p-4 ring-1 transition-colors ${
                        done ? "bg-pass/[0.05] ring-pass/20" : isPicked(r.id) ? "bg-surface-2 ring-line-strong" : "bg-surface ring-line"}`}>
                        <span className="mt-0.5">
                          {done ? (
                            <span className="inline-flex size-[18px] items-center justify-center rounded-[5px] bg-pass text-on-light" aria-label="Applied"><Check className="size-3.5" strokeWidth={3} /></span>
                          ) : (
                            <>
                              <input type="checkbox" className="peer sr-only" checked={isPicked(r.id)} onChange={() => toggle(r.id)} aria-label={`Apply: ${r.title}`} />
                              <span className="inline-flex size-[18px] items-center justify-center rounded-[5px] ring-1 ring-fg-subtle peer-checked:bg-fg peer-checked:ring-fg peer-focus-visible:outline-2 peer-focus-visible:outline-fg">
                                {isPicked(r.id) && <Check className="size-3.5 text-on-light" strokeWidth={3} />}
                              </span>
                            </>
                          )}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="flex flex-wrap items-baseline justify-between gap-x-3">
                            <span className="text-[15px]">{r.title}</span>
                            <span className="font-mono text-[11px] text-fg-subtle tabular" title="How sure the rule is">
                              {done ? "applied" : `confidence ${r.confidence.toFixed(2)}`}
                            </span>
                          </span>
                          <span className="mt-1 block text-[13.5px] leading-snug text-fg-muted">{r.reason}</span>
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </section>
          );
        })}
      </div>
    </div>
  );
}
