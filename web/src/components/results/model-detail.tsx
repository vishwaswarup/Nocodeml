import { ConfusionMatrix } from "@/components/charts/confusion-matrix";
import { HistogramChart } from "@/components/charts/histogram-chart";
import { RocChart } from "@/components/charts/roc-chart";
import { ScatterChart } from "@/components/charts/scatter-chart";
import { Metric } from "@/components/ui/data";
import type { ModelResult, Task } from "@/lib/types";
import { f3, num, type Source } from "./metrics";

function Hyper({ params }: { params: Record<string, unknown> }) {
  const entries = Object.entries(params).filter(([k]) => k !== "random_state");
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([k, v]) => (
        <span key={k} className="rounded-full bg-surface-2 px-2.5 py-1 font-mono text-[11.5px] text-fg-muted">
          {k} <span className="text-fg">{v === null ? "none" : String(v)}</span>
        </span>
      ))}
    </div>
  );
}

export function ModelDetail({ model, task, source }: { model: ModelResult; task: Task; source: Source }) {
  const m = model.metrics[source];
  const pm = model.primary_metric;
  const train = num(model.metrics.train?.[pm]);
  const held = num(m?.[pm]);
  const gap = train !== null && held !== null ? train - held : null;
  const base = model.baseline;
  const metricName = pm === "r2" ? "R²" : pm === "f1" ? "F1" : pm.toUpperCase();

  if (!m) return <p className="text-[14px] text-fg-muted">No results for this data source.</p>;
  const cm = m.confusion_matrix, roc = m.roc_curve, avp = m.actual_vs_predicted, res = m.residuals;
  const folds = model.fold_primary_scores;
  const mean = folds.length ? folds.reduce((a, b) => a + b, 0) / folds.length : null;
  const sd = mean !== null ? Math.sqrt(folds.reduce((a, b) => a + (b - mean) ** 2, 0) / folds.length) : null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <h3 className="text-h2">{model.name}</h3>
        <Hyper params={model.hyperparameters} />
      </div>

      {task === "classification" ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="F1" value={f3(m.f1)} sub={`baseline ${f3(base.f1)}`} emphasis />
          <Metric label="ROC-AUC" value={f3(m.roc_auc)} sub="0.5 = coin flip" />
          <Metric label="Accuracy" value={f3(m.accuracy)} sub={`baseline ${f3(base.accuracy)}`} />
          <Metric label="Recall" value={f3(m.recall)} sub={`precision ${f3(m.precision)}`} />
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="R²" value={f3(m.r2)} sub={`baseline ${f3(base.r2)}`} emphasis />
          <Metric label="RMSE" value={f3(m.rmse)} sub="typical error size" />
          <Metric label="MAE" value={f3(m.mae)} sub="average absolute error" />
          <Metric label="MAPE" value={num(m.mape) === null ? "–" : `${((m.mape as number) * 100).toFixed(1)}%`} sub="average % error" />
        </div>
      )}

      <div className="flex flex-wrap gap-x-8 gap-y-2 text-[13px] text-fg-muted">
        {gap !== null && source !== "cv" && (
          <p>
            {metricName} on training rows <span className="font-mono text-fg tabular">{f3(train)}</span> vs held-out{" "}
            <span className="font-mono text-fg tabular">{f3(held)}</span>
            {gap > 0.15 ? <span className="ml-2 text-warn">large gap: the model may be memorizing the training data</span> : <span className="ml-2 text-fg-subtle">gap {gap.toFixed(3)}</span>}
          </p>
        )}
        {source === "cv" && mean !== null && sd !== null && (
          <p>{metricName} per fold: <span className="font-mono text-fg tabular">{folds.map((f) => f.toFixed(2)).join(", ")}</span> (mean {mean.toFixed(3)} ± {sd.toFixed(3)}). Out-of-fold predictions.</p>
        )}
        <p>Trained in <span className="font-mono text-fg tabular">{model.fit_seconds}s</span>{model.n_outlier_rows_removed > 0 && ` · ${model.n_outlier_rows_removed} outlier rows removed from training`}</p>
      </div>

      {task === "classification" ? (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-2">
          {cm && <ConfusionMatrix labels={cm.labels} matrix={cm.matrix} />}
          {roc ? <RocChart fpr={roc.fpr} tpr={roc.tpr} auc={num(m.roc_auc)} />
            : <div className="rounded-card bg-surface p-5 text-[14px] text-fg-muted">
                <p className="text-[16px] text-fg">ROC curve</p>
                <p className="mt-2">The ROC curve is shown for two-class problems. This target has {cm?.labels.length ?? "several"} classes, so use the confusion matrix and the numbers above.</p>
              </div>}
        </div>
      ) : (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-2">
          {avp && <ScatterChart actual={avp.actual} predicted={avp.predicted} />}
          {res && <HistogramChart values={res} />}
        </div>
      )}
    </div>
  );
}
