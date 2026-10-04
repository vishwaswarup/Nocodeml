import { Eyebrow } from "@/components/ui/card";

const plans: Record<string, string[]> = {
  preprocessing: ["Missing values per column (drop, mean, median, mode, constant, KNN)", "Encoding, scaling, outliers, duplicates", "NoCodeML recommendations with reasons: apply, modify or ignore"],
  features: ["Log, square root, absolute value, square", "Date parts: year, month, day, weekday, quarter, weekend"],
  split: ["Train/test, train/validation/test, k-fold, stratified, time series", "Recommendations, e.g. stratify when classes are imbalanced"],
  models: ["Up to five models for your task", "Each with cost, interpretability and whether it needs scaling"],
  regularization: ["Model-aware: L1/L2/Elastic Net where it applies, tree complexity limits otherwise"],
  training: ["Recommended or manual hyperparameters", "Train in the background with live status"],
  results: ["Side-by-side comparison, confusion matrix, ROC, residuals", "Pipeline health checks, finalize and download"],
};

export function ComingNext({ section }: { section: { n: number; key: string; label: string } }) {
  return (
    <div className="mx-auto max-w-2xl pt-6">
      <Eyebrow>Section {section.n}</Eyebrow>
      <h1 className="mt-2 text-h1">{section.label}</h1>
      <div className="mt-8 rounded-card border border-dashed border-line-strong p-6">
        <p className="text-[15px] text-fg">This section is next in the build.</p>
        <ul className="mt-3 space-y-1.5">
          {(plans[section.key] ?? []).map((p) => (
            <li key={p} className="flex gap-2 text-[14px] text-fg-muted">
              <span className="mt-2 size-1 shrink-0 rounded-full bg-fg-subtle" aria-hidden />{p}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
