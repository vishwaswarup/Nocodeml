"use client";

import { AlertTriangle } from "lucide-react";
import { CompactSelect } from "@/components/ui/select";
import {
  ENCODING_OPTIONS, getEncoding, getMissing, getOutlier, getRole, MISSING_OPTIONS, OUTLIER_OPTIONS, setEncoding,
  setMissing, setOutlier, setRole, type Draft, type Role,
} from "@/lib/preprocessing";
import type { ColumnProfile, EncodingStrategy, MissingStrategy, OutlierStrategy } from "@/lib/types";

/** One row per column: choose what to do with it. Mirrors the "manual" mode in the spec. */
export function ColumnEditor({ columns, target, draft, onChange }: {
  columns: [string, ColumnProfile][]; target: string; draft: Draft; onChange: (d: Draft) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-card bg-surface ring-1 ring-line">
      <table className="w-full min-w-[640px] table-fixed text-[13px]">
        <caption className="sr-only">Preprocessing choices per column</caption>
        <thead>
          <tr className="border-b border-line text-left text-fg-muted">
            <th scope="col" className="w-[22%] px-4 py-2.5 font-normal">Column</th>
            <th scope="col" className="px-2 py-2.5 font-normal">Use as</th>
            <th scope="col" className="px-2 py-2.5 font-normal">Missing values</th>
            <th scope="col" className="px-2 py-2.5 font-normal">Encoding</th>
            <th scope="col" className="px-2 py-2.5 font-normal">Outliers</th>
          </tr>
        </thead>
        <tbody>
          {columns.map(([name, c]) => {
            const isTarget = name === target;
            const role = getRole(draft, name);
            const dropped = role === "drop";
            const miss = getMissing(draft, name);
            const needsMissing = !dropped && c.missing_pct > 0 && !miss;
            const needsEncoding = !dropped && c.kind === "categorical" && !getEncoding(draft, name);
            const needsDate = !dropped && c.kind === "datetime" && role === "use";
            const attention = needsMissing || needsEncoding || needsDate;
            return (
              <tr key={name} className={`border-b border-line/60 last:border-0 ${dropped ? "opacity-60" : ""}`}>
                <td className="px-4 py-2.5 whitespace-nowrap">
                  <span className="flex items-center gap-2">
                    {attention && <AlertTriangle className="size-3.5 text-warn" aria-label="Needs a choice" />}
                    <span>{name}</span>
                    {isTarget && <span className="rounded-full bg-ember/12 px-2 py-0.5 text-[11px] text-ember">target</span>}
                  </span>
                  <span className="mt-0.5 block font-mono text-[11px] text-fg-subtle">
                    {c.kind === "numerical" ? "num" : c.kind === "categorical" ? "cat" : "date"} · {c.missing_pct}% missing
                    {c.identifier_like ? " · identifier?" : ""}
                  </span>
                </td>
                {isTarget ? (
                  <td colSpan={4} className="px-2 py-2.5 text-fg-subtle">The value the model predicts. It is never changed here.</td>
                ) : (
                  <>
                    <td className="px-2 py-2.5">
                      <CompactSelect label={`${name}: use as`} value={role} onChange={(e) => onChange(setRole(draft, name, e.target.value as Role))}>
                        <option value="use">{c.kind === "datetime" ? "Choose…" : "Feature"}</option>
                        {c.kind === "datetime" && <option value="dateparts">Date parts</option>}
                        <option value="drop">Drop</option>
                      </CompactSelect>
                    </td>
                    <td className="px-2 py-2.5">
                      {c.missing_pct > 0 && !dropped ? (
                        <div className="flex items-center gap-1.5">
                          <CompactSelect label={`${name}: missing values`} value={miss?.strategy ?? ""}
                            onChange={(e) => onChange(setMissing(draft, name, e.target.value as MissingStrategy | ""))}>
                            <option value="">Choose…</option>
                            {MISSING_OPTIONS[c.kind].map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                          </CompactSelect>
                          {miss?.strategy === "constant" && (
                            <input aria-label={`${name}: fill value`} placeholder="value" defaultValue={String(miss.constant_value ?? "")}
                              onChange={(e) => {
                                const raw = e.target.value;
                                onChange(setMissing(draft, name, "constant", raw === "" ? null : c.kind === "numerical" ? Number(raw) : raw));
                              }}
                              className="h-8 w-20 rounded-[8px] bg-surface-2 px-2 text-[13px] ring-1 ring-line outline-none focus:ring-fg/60" />
                          )}
                        </div>
                      ) : <span className="text-fg-subtle">{dropped ? "–" : "none missing"}</span>}
                    </td>
                    <td className="px-2 py-2.5">
                      {c.kind === "categorical" && !dropped ? (
                        <CompactSelect label={`${name}: encoding`} value={getEncoding(draft, name)}
                          onChange={(e) => onChange(setEncoding(draft, name, e.target.value as EncodingStrategy | ""))}>
                          <option value="">Choose…</option>
                          {ENCODING_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </CompactSelect>
                      ) : <span className="text-fg-subtle">–</span>}
                    </td>
                    <td className="px-2 py-2.5">
                      {c.kind === "numerical" && !dropped ? (
                        <CompactSelect label={`${name}: outliers`} value={getOutlier(draft, name) || "keep"}
                          onChange={(e) => onChange(setOutlier(draft, name, e.target.value as OutlierStrategy))}>
                          {OUTLIER_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </CompactSelect>
                      ) : <span className="text-fg-subtle">–</span>}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
