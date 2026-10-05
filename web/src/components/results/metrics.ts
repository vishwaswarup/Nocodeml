import type { ModelResult, Task } from "@/lib/types";

export type MetricDef = { key: string; label: string; lowerIsBetter?: boolean; hint: string };

export const TABLE_METRICS: Record<Task, MetricDef[]> = {
  classification: [
    { key: "accuracy", label: "Accuracy", hint: "Share of rows predicted correctly. Misleading when one class dominates." },
    { key: "precision", label: "Precision", hint: "Of the rows predicted positive, how many really were." },
    { key: "recall", label: "Recall", hint: "Of the rows that really were positive, how many were found." },
    { key: "f1", label: "F1", hint: "Balance of precision and recall." },
    { key: "roc_auc", label: "ROC-AUC", hint: "How well the model ranks positives above negatives. 0.5 is a coin flip." },
  ],
  regression: [
    { key: "r2", label: "R²", hint: "Share of the variation explained. 0 is no better than predicting the average." },
    { key: "rmse", label: "RMSE", lowerIsBetter: true, hint: "Typical error size, punishing big misses. Same units as the target." },
    { key: "mae", label: "MAE", lowerIsBetter: true, hint: "Average absolute error, in the same units as the target." },
  ],
};

export const LOWER_IS_BETTER = new Set(["rmse", "mae", "mse", "mape"]);

export type Source = "test" | "validation" | "cv";
export const SOURCE_LABEL: Record<Source, string> = { test: "Test set", validation: "Validation set", cv: "Cross-validation" };

export const sourcesOf = (models: ModelResult[]): Source[] =>
  (["test", "validation", "cv"] as Source[]).filter((s) => models.some((m) => s in m.metrics));

export const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
export const f3 = (v: unknown) => (num(v) === null ? "–" : (v as number).toFixed(3));

export const KIND_LABEL: Record<string, string> = {
  pipeline: "Full pipeline", model: "Model only", configuration: "Configuration", metrics: "Metrics & record", report: "PDF report", other: "Notes",
};
