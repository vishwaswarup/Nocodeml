"use client";

import { useState } from "react";
import { QualityRow } from "@/components/ui/data";
import { Card } from "@/components/ui/card";
import type { QualityCheck } from "@/lib/types";

const ORDER = { fail: 0, warn: 1, pass: 2, na: 3 } as const;

/** Pipeline quality = was the experiment built soundly? It is NOT how accurate the models are. */
export function HealthCard({ checks, score, models }: { checks: QualityCheck[]; score: number | null; models: Record<string, string> }) {
  const [showPassed, setShowPassed] = useState(false);
  const sorted = [...checks].filter((c) => c.status !== "na").sort((a, b) => ORDER[a.status] - ORDER[b.status]);
  const count = (s: string) => checks.filter((c) => c.status === s).length;
  const attention = sorted.filter((c) => c.status !== "pass"), passed = sorted.filter((c) => c.status === "pass");
  const label = (c: QualityCheck) => (c.model_key && models[c.model_key] && !c.title.includes(models[c.model_key]) ? `${c.title} (${models[c.model_key]})` : c.title);
  return (
    <Card className="p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="text-h3">Pipeline health</p>
        <p className="font-mono text-[13px] text-fg-muted tabular">
          <span className="text-[22px] text-fg">{score ?? "–"}</span> / 100
        </p>
      </div>
      <p className="mt-1 text-[13px] text-fg-muted">
        Was this experiment built correctly? This score says nothing about accuracy: a 99% model can have poor health if data leaked.
      </p>
      <p className="mt-3 flex gap-4 text-[13px]">
        <span className="text-pass">{count("pass")} passed</span>
        <span className={count("warn") ? "text-warn" : "text-fg-subtle"}>{count("warn")} to review</span>
        <span className={count("fail") ? "text-fail" : "text-fg-subtle"}>{count("fail")} failed</span>
      </p>
      <div className="mt-2 divide-y divide-line">
        {attention.map((c, i) => <QualityRow key={c.id + i} status={c.status as "warn" | "fail"} title={label(c)} detail={c.detail} />)}
        {showPassed && passed.map((c, i) => <QualityRow key={c.id + i} status="pass" title={label(c)} detail={c.detail} />)}
      </div>
      {passed.length > 0 && (
        <button type="button" aria-expanded={showPassed} onClick={() => setShowPassed(!showPassed)} className="mt-3 text-[13px] text-fg-muted underline-offset-4 hover:text-fg hover:underline">
          {showPassed ? "Hide" : "Show"} {passed.length} passed check{passed.length === 1 ? "" : "s"}
        </button>
      )}
    </Card>
  );
}
