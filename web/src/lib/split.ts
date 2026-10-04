import { sameJson } from "./canon";
import type { PipelineConfig, Recommendation, SplitConfig, SplitMethod } from "./types";

const DEFAULT: SplitConfig = {
  method: "train_test", test_size: 0.2, validation_size: 0, n_splits: 5, stratify: false, time_column: null, random_state: 42,
};

export const splitFromConfig = (c: PipelineConfig): SplitConfig => ({ ...DEFAULT, ...(c.split as Partial<SplitConfig>) });
export const sameSplit = (a: SplitConfig, b: SplitConfig) => sameJson(a, b);
export const isCV = (m: SplitMethod) => m === "k_fold" || m === "stratified_k_fold" || m === "time_series";
export const isHoldout = (m: SplitMethod) => !isCV(m);

/** Switching method resets the fields that don't apply, so the draft never carries stale settings. */
export function withMethod(s: SplitConfig, method: SplitMethod): SplitConfig {
  return {
    ...s, method,
    validation_size: method === "train_val_test" ? (s.validation_size || 0.15) : 0,
    stratify: isHoldout(method) && !s.time_column ? s.stratify : false,
    time_column: method === "time_series" || isHoldout(method) ? s.time_column : null,
  };
}

/** Mirror of engine apply_split_actions: patches merge in order. */
export function applySplitActions(s: SplitConfig, recs: Recommendation[]): SplitConfig {
  return recs.reduce((acc, r) => ({ ...acc, ...((r.action.patch as Partial<SplitConfig>) ?? {}) }), s);
}

export const METHODS: { value: SplitMethod; label: string; description: string; classificationOnly?: boolean }[] = [
  { value: "train_test", label: "Train / test", description: "Train on most rows, test on a held-back part. Fastest and simplest." },
  { value: "train_val_test", label: "Train / validation / test", description: "Adds a validation set for tuning, so the test set stays untouched until the end." },
  { value: "k_fold", label: "K-fold cross-validation", description: "Train and test several times, so every row is tested once. More reliable on small data." },
  { value: "stratified_k_fold", label: "Stratified k-fold", description: "K-fold that keeps class proportions in every fold.", classificationOnly: true },
  { value: "time_series", label: "Time-series cross-validation", description: "Each fold trains on earlier rows and tests on later ones. Needs a date column." },
];
