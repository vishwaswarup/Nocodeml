"use client";

import { clsx } from "clsx";
import { useState } from "react";
import { Switch } from "@/components/ui/choice";
import { Field } from "@/components/ui/field";
import { Select } from "@/components/ui/select";
import type { ModelConfig, ModelInfo, SearchConfig, TunableInfo } from "@/lib/types";

/** Starting choices when the search is switched on: the settings that usually matter most, in this order. */
const PRIORITY = ["n_estimators", "max_depth", "learning_rate", "C", "alpha", "n_neighbors", "min_samples_leaf", "kernel"];
const MAX_CANDIDATES = 40;

const combos = (space: SearchConfig["space"]) => Object.values(space).reduce((n, v) => n * Math.max(1, v.length), 1);
const label = (v: unknown) => (v === null ? "None" : String(v));

/** "0.1, 1, 10" -> [0.1, 1, 10]; null if any piece isn't a number. */
function parseNumbers(text: string, integer: boolean): number[] | null {
  const parts = text.split(/[,\s]+/).filter(Boolean);
  const nums = parts.map(Number);
  if (!parts.length || nums.some((n) => !Number.isFinite(n) || (integer && !Number.isInteger(n)))) return null;
  return nums;
}

/** Keeps `n_iter` achievable: a random search can't sample more settings than exist. */
function normalise(s: SearchConfig): SearchConfig {
  const total = combos(s.space);
  return { ...s, n_iter: Math.max(2, Math.min(s.n_iter, total)) };
}

function initial(info: ModelInfo): SearchConfig {
  const usable = info.tunable.filter((t) => t.suggested.length >= 2);
  const ranked = [...usable].sort((a, b) => {
    const ia = PRIORITY.indexOf(a.name), ib = PRIORITY.indexOf(b.name);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  const space = Object.fromEntries(ranked.slice(0, 2).map((t) => [t.name, t.suggested]));
  return normalise({ method: "random", space, n_iter: 8, cv_folds: 3 });
}

export function TunePanel({ info, model, onChange }: {
  info: ModelInfo; model: ModelConfig; onChange: (fn: (m: ModelConfig) => ModelConfig) => void;
}) {
  const search = model.search ?? null;
  const set = (s: SearchConfig | null) => onChange((m) => ({ ...m, search: s ? normalise(s) : null }));
  const total = search ? (search.method === "grid" ? combos(search.space) : search.n_iter) : 0;
  const fits = search ? total * search.cv_folds : 0;

  return (
    <div className="mt-6 rounded-card bg-surface-2/60 p-5 ring-1 ring-line">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1 basis-72">
          <p className="text-[15px]">Find the best settings automatically</p>
          <p className="mt-1 text-[13px] text-fg-muted">
            Tries different values, scores each with cross-validation on the training rows only, then trains the winner.
            Your test data never influences the choice.
          </p>
        </div>
        <Switch checked={!!search} label={`${info.name}: tune automatically`}
          onChange={(on) => set(on ? initial(info) : null)} />
      </div>

      {search && (
        <>
          <div className="mt-5 grid gap-4 sm:grid-cols-3">
            <Select label="Method" value={search.method} onChange={(e) => set({ ...search, method: e.target.value as SearchConfig["method"] })}
              hint={search.method === "grid" ? "Tries every combination." : "Tries a random sample of combinations. Faster."}>
              <option value="random">Random search</option>
              <option value="grid">Grid search</option>
            </Select>
            {search.method === "random" && (
              <Field label="Settings to try" type="number" min={2} max={MAX_CANDIDATES} step={1} value={search.n_iter}
                suffix={`max ${Math.min(MAX_CANDIDATES, combos(search.space))}`}
                onChange={(e) => { const n = Math.round(Number(e.target.value)); if (Number.isFinite(n)) set({ ...search, n_iter: n }); }} />
            )}
            <Select label="Cross-validation folds" value={String(search.cv_folds)} onChange={(e) => set({ ...search, cv_folds: Number(e.target.value) })}
              hint="More folds: steadier scores, slower.">
              {[2, 3, 4, 5].map((n) => <option key={n} value={n}>{n} folds</option>)}
            </Select>
          </div>

          <fieldset className="mt-5">
            <legend className="mb-2 text-[13px] text-fg-muted">Settings to tune and the values to try</legend>
            <div className="space-y-2.5">
              {info.tunable.filter((t) => t.suggested.length > 0 || t.name in search.space).map((t) => (
                <ParamRow key={t.name} t={t} values={search.space[t.name] ?? null}
                  onChange={(values) => {
                    const space = { ...search.space };
                    if (values === null) delete space[t.name]; else space[t.name] = values;
                    set({ ...search, space });
                  }} />
              ))}
            </div>
          </fieldset>

          <p className={clsx("mt-4 text-[13px]", Object.keys(search.space).length === 0 || total > MAX_CANDIDATES ? "text-warn" : "text-fg-muted")} aria-live="polite">
            {Object.keys(search.space).length === 0
              ? "Pick at least one setting to tune."
              : `${search.method === "grid" ? `Will try all ${total} combinations` : `Will try ${total} of the ${combos(search.space)} possible combinations`}, ${search.cv_folds} folds each: ${fits} extra model fits per training run.`}
            {fits > 60 && " This can take a while."}
          </p>
          {Object.keys(model.hyperparameters).some((k) => k in search.space) && (
            <p className="mt-1 text-[12.5px] text-fg-subtle">Values you set by hand for a tuned setting are ignored; the search decides.</p>
          )}
        </>
      )}
    </div>
  );
}

function ParamRow({ t, values, onChange }: { t: TunableInfo; values: unknown[] | null; onChange: (v: unknown[] | null) => void }) {
  const on = values !== null;
  const options = t.kind === "bool" ? [true, false] : t.kind === "choice" ? t.choices.filter((c) => c !== null) : null;
  const range = t.min !== null || t.max !== null ? `${t.min ?? "…"} to ${t.max ?? "…"}` : "";
  return (
    <div className="grid gap-2 sm:grid-cols-[200px_1fr] sm:items-start">
      <label className="flex cursor-pointer items-center gap-2.5 pt-2 text-[14px]">
        <input type="checkbox" className="size-4 accent-white" checked={on} aria-label={`Tune ${t.name}`}
          onChange={(e) => onChange(e.target.checked ? (t.suggested.length ? t.suggested : []) : null)} />
        <span className="font-mono text-[13px]">{t.name}</span>
      </label>
      {on && (options ? (
        <div className="flex flex-wrap gap-2" role="group" aria-label={`Values to try for ${t.name}`}>
          {options.map((o) => {
            const picked = values.some((v) => v === o);
            return (
              <button key={label(o)} type="button" aria-pressed={picked}
                onClick={() => onChange(picked ? values.filter((v) => v !== o) : [...values, o])}
                className={clsx("rounded-full px-3.5 py-1.5 text-[13px] transition-colors",
                  picked ? "bg-fg text-on-light" : "bg-surface-3 text-fg-muted hover:text-fg")}>
                {label(o)}
              </button>
            );
          })}
        </div>
      ) : (
        <NumberList t={t} values={values as number[]} range={range} onChange={onChange} />
      ))}
    </div>
  );
}

/** Comma-separated numbers. Keeps what you type until it parses, so a half-typed "0." never gets reverted. */
function NumberList({ t, values, range, onChange }: { t: TunableInfo; values: number[]; range: string; onChange: (v: unknown[]) => void }) {
  const integer = t.kind === "int" || t.kind === "optional_int";
  const [text, setText] = useState(values.join(", "));
  const parsed = parseNumbers(text, integer);
  const bad = parsed === null && text.trim() !== "";
  return (
    <Field label="" aria-label={`Values to try for ${t.name}`} value={text} suffix={range} placeholder="e.g. 0.1, 1, 10"
      error={bad ? (integer ? "Whole numbers separated by commas." : "Numbers separated by commas.") : undefined}
      className="[&>label]:sr-only"
      onChange={(e) => {
        setText(e.target.value);
        const p = parseNumbers(e.target.value, integer);
        if (p) onChange(p);
      }} />
  );
}
