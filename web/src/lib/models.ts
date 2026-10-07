"use client";

import { useEffect, useState } from "react";
import { api } from "./api";
import { sameJson } from "./canon";
import type { HyperParamInfo, ModelConfig, ModelInfo, Task } from "./types";

const cache = new Map<Task, Promise<ModelInfo[]>>();

/** Model registry metadata for a task (fetched once per task and cached). */
export function useModels(task: Task) {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let p = cache.get(task);
    if (!p) { p = api<ModelInfo[]>(`/models?task=${task}`); cache.set(task, p); }
    p.then((m) => { if (alive) setModels(m); })
      .catch((e: Error) => { cache.delete(task); if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [task]);
  return { models, error };
}

export const emptyModel = (key: string): ModelConfig =>
  ({ model_key: key, regularization: {}, hyperparameters: {}, use_recommended_defaults: true });

export const asModels = (raw: unknown[]): ModelConfig[] =>
  (raw as Partial<ModelConfig>[]).map((m) => ({ ...emptyModel(m.model_key as string), ...m }));

export const sameModels = (a: ModelConfig[], b: ModelConfig[]) => sameJson(a, b);

export const DESCRIPTIONS: Record<string, string> = {
  logistic_regression: "Draws a straight boundary between classes. Fast and easy to explain.",
  knn: "Predicts from the most similar rows. No training step, but slow on large data.",
  decision_tree: "A flowchart of yes/no questions. Easy to read, but can memorize the data.",
  random_forest: "Many decision trees voting together. A strong default that rarely needs tuning.",
  svm: "Finds the widest margin between classes. Strong on small, clean data; slow on large.",
  linear_regression: "Fits a straight line (or plane) through the data. The simplest baseline.",
  ridge: "Linear regression that shrinks its coefficients (L2). Steadier when features are correlated.",
  lasso: "Linear regression that can zero out features (L1). Doubles as feature selection.",
  gradient_boosting: "Trees built one after another, each fixing the last one's mistakes. Often the most accurate.",
  xgboost: "A fast, heavily tuned form of gradient boosting with built-in L1/L2 penalties. Often a top performer on tables.",
};

/** A small, diverse starting set: a linear model, a tree ensemble and one more. */
export const STARTER: Record<Task, string[]> = {
  classification: ["logistic_regression", "random_forest", "knn"],
  regression: ["ridge", "random_forest", "gradient_boosting"],
};

/** Hyperparameters controlled in the Regularization section, hence hidden from the Training section. */
export function regulationParams(m: ModelInfo): Set<string> {
  const r = m.regularization;
  if (r.kind === "complexity") return new Set(r.complexity_params);
  if (r.kind === "penalty") return new Set(["alpha"]); // ridge/lasso: alpha IS the regularization strength
  return new Set();
}

export const trainingParams = (m: ModelInfo): HyperParamInfo[] =>
  m.hyperparameters.filter((h) => !regulationParams(m).has(h.name));

export const penaltyLabel: Record<string, string> = { none: "None", l1: "L1 (lasso)", l2: "L2 (ridge)", elasticnet: "Elastic net" };
export const penaltyHelp: Record<string, string> = {
  none: "No penalty: the model can fit the training data as closely as it likes. Risky with many features.",
  l1: "Pushes the least useful coefficients to exactly zero, so some features drop out.",
  l2: "Shrinks all coefficients towards zero without removing any. A safe, common default.",
  elasticnet: "A mix of L1 and L2: shrinks coefficients and drops some features.",
};
