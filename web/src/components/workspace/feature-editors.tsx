"use client";

import { clsx } from "clsx";
import { Card } from "@/components/ui/card";
import {
  DATE_PARTS, dateParts, hasTransform, setDateParts, toggleTransform, transformBlocked, TRANSFORMS,
} from "@/lib/features";
import type { ColumnProfile, FeatureEngineering } from "@/lib/types";

function Chip({ on, disabled, title, onClick, children }: {
  on: boolean; disabled?: boolean; title?: string; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button type="button" aria-pressed={on} disabled={disabled} title={title} onClick={onClick}
      className={clsx("h-8 rounded-full px-3.5 text-[13px] transition-colors disabled:cursor-not-allowed disabled:opacity-35",
        on ? "bg-fg text-on-light" : "bg-surface-2 text-fg-muted hover:text-fg")}>
      {children}
    </button>
  );
}

export function NumericTransforms({ columns, draft, onChange }: {
  columns: [string, ColumnProfile][]; draft: FeatureEngineering; onChange: (d: FeatureEngineering) => void;
}) {
  return (
    <Card className="p-6">
      <p className="text-h3">Numeric transforms</p>
      <p className="mt-1 text-[13.5px] text-fg-muted">
        Add a transformed copy of a number column. The original is kept; the new column is named <span className="font-mono text-[12.5px]">column__transform</span>.
      </p>
      {columns.length === 0 ? <p className="mt-4 text-[14px] text-fg-subtle">No numeric columns available to transform.</p> : (
        <ul className="mt-4 divide-y divide-line">
          {columns.map(([name, c]) => {
            const skew = c.skewness ?? 0;
            return (
              <li key={name} className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 py-3">
                <div className="min-w-0">
                  <p className="text-[14.5px]">{name}</p>
                  <p className="font-mono text-[11.5px] text-fg-subtle tabular">
                    skew {(Math.abs(skew) < 0.05 ? 0 : skew).toFixed(1)} · min {typeof c.min === "number" ? c.min.toLocaleString() : "–"}
                    {skew > 1 && <span className="ml-2 font-sans text-ember">right-skewed</span>}
                  </p>
                </div>
                <div className="flex flex-wrap gap-1.5" role="group" aria-label={`Transforms for ${name}`}>
                  {TRANSFORMS.map((t) => {
                    const blocked = transformBlocked(c, t.value);
                    return (
                      <Chip key={t.value} on={hasTransform(draft, name, t.value)} disabled={!!blocked && !hasTransform(draft, name, t.value)}
                        title={blocked ?? t.help} onClick={() => onChange(toggleTransform(draft, name, t.value))}>
                        {t.label}
                      </Chip>
                    );
                  })}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}

export function DatePartsEditor({ columns, draft, onChange }: {
  columns: [string, ColumnProfile][]; draft: FeatureEngineering; onChange: (d: FeatureEngineering) => void;
}) {
  return (
    <Card className="p-6">
      <p className="text-h3">Date parts</p>
      <p className="mt-1 text-[13.5px] text-fg-muted">Models can&apos;t use a raw date, so turn it into numbers. The original date column is replaced.</p>
      {columns.length === 0 ? <p className="mt-4 text-[14px] text-fg-subtle">This dataset has no date columns.</p> : (
        <ul className="mt-4 divide-y divide-line">
          {columns.map(([name, c]) => {
            const parts = dateParts(draft, name);
            return (
              <li key={name} className="py-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-[14.5px]">{name}</p>
                  <p className="font-mono text-[11.5px] text-fg-subtle">{String(c.min ?? "").slice(0, 10)} → {String(c.max ?? "").slice(0, 10)}{c.has_time ? " · has time of day" : ""}</p>
                </div>
                <div className="mt-2.5 flex flex-wrap gap-1.5" role="group" aria-label={`Date parts for ${name}`}>
                  {DATE_PARTS.map((p) => {
                    const noTime = p.value === "hour" && !c.has_time;
                    return (
                      <Chip key={p.value} on={parts.includes(p.value)} disabled={noTime && !parts.includes(p.value)}
                        title={noTime ? "This column has dates only, no time of day." : undefined}
                        onClick={() => onChange(setDateParts(draft, name, parts.includes(p.value) ? parts.filter((x) => x !== p.value) : [...parts, p.value]))}>
                        {p.label}
                      </Chip>
                    );
                  })}
                </div>
                {parts.length === 0 && <p className="mt-2 text-[12.5px] text-warn">Not used yet. Pick parts here, or drop the column in Preprocessing.</p>}
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
