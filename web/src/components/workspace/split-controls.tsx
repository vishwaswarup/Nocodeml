"use client";

import { Switch } from "@/components/ui/choice";
import { ChoiceList } from "@/components/ui/choice";
import { Field } from "@/components/ui/field";
import { Select } from "@/components/ui/select";
import { isCV, isHoldout, METHODS, withMethod } from "@/lib/split";
import type { SplitConfig, SplitMethod, Task } from "@/lib/types";

export function SplitControls({ draft, onChange, task, dateColumns }: {
  draft: SplitConfig; onChange: (s: SplitConfig) => void; task: Task; dateColumns: string[];
}) {
  const set = (patch: Partial<SplitConfig>) => onChange({ ...draft, ...patch });
  const pct = (v: number) => String(Math.round(v * 100));
  const parsePct = (raw: string, fallback: number) => {
    const n = Number(raw);
    return Number.isFinite(n) && raw !== "" ? n / 100 : fallback;
  };
  const methods = METHODS.filter((m) => !m.classificationOnly || task === "classification");
  const canStratify = task === "classification" && isHoldout(draft.method) && !draft.time_column;

  return (
    <div className="space-y-8">
      <section>
        <h2 className="mb-1 text-h3">Method</h2>
        <p className="mb-3 text-[14px] text-fg-muted">How the rows are divided between training and testing.</p>
        <ChoiceList<SplitMethod> label="Split method" value={draft.method}
          onChange={(m) => onChange(withMethod(draft, m))}
          options={methods.map((m) => ({ value: m.value, label: m.label, description: m.description }))} />
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        {isHoldout(draft.method) && (
          <Field label="Test set" type="number" min={5} max={50} step={5} suffix="% of rows" value={pct(draft.test_size)}
            onChange={(e) => set({ test_size: parsePct(e.target.value, draft.test_size) })}
            hint="Rows held back to measure the model honestly." />
        )}
        {draft.method === "train_val_test" && (
          <Field label="Validation set" type="number" min={5} max={40} step={5} suffix="% of rows" value={pct(draft.validation_size)}
            onChange={(e) => set({ validation_size: parsePct(e.target.value, draft.validation_size) })}
            hint="Used to compare settings before the final test." />
        )}
        {isCV(draft.method) && (
          <Field label="Number of folds" type="number" min={2} max={20} step={1} value={String(draft.n_splits)}
            onChange={(e) => set({ n_splits: Math.max(2, Math.round(Number(e.target.value) || draft.n_splits)) })}
            hint="More folds is slower but uses more data for each training run." />
        )}
        <Field label="Random seed" type="number" value={String(draft.random_state)}
          onChange={(e) => set({ random_state: Math.round(Number(e.target.value) || 0) })}
          hint="Same seed, same split: makes results reproducible." />
      </section>

      {canStratify && (
        <section className="flex items-center justify-between gap-4 rounded-card bg-surface p-4">
          <div>
            <p className="text-[15px]">Stratify by the target</p>
            <p className="text-[13px] text-fg-muted">Keeps class proportions the same in the training and test sets. Recommended when one class is rare.</p>
          </div>
          <Switch checked={draft.stratify} onChange={(v) => set({ stratify: v })} label="Stratify" />
        </section>
      )}

      {(draft.method === "time_series" || isHoldout(draft.method)) && (
        <section>
          <Select label={draft.method === "time_series" ? "Date column (required)" : "Order rows by a date column (optional)"}
            value={draft.time_column ?? ""}
            onChange={(e) => onChange({ ...draft, time_column: e.target.value || null, stratify: e.target.value ? false : draft.stratify })}
            hint={draft.method === "time_series"
              ? "Each fold trains on earlier dates and tests on later ones."
              : "If set, the test set is the most recent rows instead of a random sample. Stratifying is turned off."}>
            <option value="">{draft.method === "time_series" ? "Choose a column…" : "No: split randomly"}</option>
            {dateColumns.map((c) => <option key={c} value={c}>{c}</option>)}
          </Select>
          {dateColumns.length === 0 && <p className="mt-2 text-[12px] text-fg-subtle">This dataset has no date columns.</p>}
        </section>
      )}
    </div>
  );
}
