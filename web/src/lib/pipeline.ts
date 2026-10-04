import type { PipelineConfig } from "./types";

type Sections = Pick<PipelineConfig, "preprocessing" | "feature_engineering" | "split" | "models">;

/** Body for PUT /pipeline and POST /pipeline/preview: the saved config with some sections replaced. */
export function pipelineBody(c: PipelineConfig, override: Partial<Sections> = {}) {
  return {
    dataset: c.dataset,
    preprocessing: override.preprocessing ?? c.preprocessing,
    feature_engineering: override.feature_engineering ?? c.feature_engineering,
    split: override.split ?? c.split,
    models: override.models ?? c.models,
  };
}
