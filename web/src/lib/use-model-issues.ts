"use client";

import { useEffect, useState } from "react";
import { api } from "./api";
import { pipelineBody } from "./pipeline";
import type { ModelConfig, PipelineConfig } from "./types";

/** Debounced server-side check of the draft models' settings (no dataset needed, so it's instant). */
export function useModelIssues(projectId: string, cfg: PipelineConfig, models: ModelConfig[]) {
  const [issues, setIssues] = useState<Record<string, string[]>>({});
  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => {
      api<{ model_issues: Record<string, string[]> }>(`/projects/${projectId}/pipeline/validate-models`, {
        method: "POST", body: JSON.stringify(pipelineBody(cfg, { models: models as unknown as PipelineConfig["models"] })),
      }).then((r) => { if (alive) setIssues(r.model_issues); }).catch(() => { /* shown on save instead */ });
    }, 300);
    return () => { alive = false; clearTimeout(t); };
  }, [projectId, cfg, models]);
  return issues;
}
