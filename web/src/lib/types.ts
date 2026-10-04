/** Shapes returned by the NoCodeML API (see engine/nocodeml_engine/api). */

export type Task = "classification" | "regression";

export type DatasetRef = { dataset_id: string; version: number; target_column: string; task: Task };

export type PipelineConfig = {
  pipeline_id: string;
  version: number;
  dataset: DatasetRef;
  preprocessing: Record<string, unknown>;
  feature_engineering: Record<string, unknown>;
  split: Record<string, unknown>;
  models: Record<string, unknown>[];
};

export type PipelineVersion = {
  pipeline_id: string;
  version: number;
  status: "draft" | "finalized";
  config: PipelineConfig;
  config_hash: string;
  created_at: string;
  finalized_experiment_id: string | null;
};

export type ProjectState = {
  project: { id: string; name: string; description: string; task: Task | null; updated_at: string };
  datasets: { id: string; name: string; created_at: string }[];
  pipeline: PipelineVersion | null;
  experiments: unknown[];
};

export type DatasetInfo = {
  dataset_id: string;
  versions: { version: number; filename: string; n_rows: number; n_columns: number; size_bytes: number; created_at: string }[];
};

export type ColumnKind = "numerical" | "categorical" | "datetime";

export type ColumnProfile = {
  kind: ColumnKind;
  dtype: string;
  count: number;
  missing_pct: number;
  unique: number;
  identifier_like?: boolean;
  mean?: number; median?: number; min?: number | string; max?: number | string; std?: number;
  skewness?: number; outlier_count?: number;
  mode?: string | number | null; cardinality?: number; frequencies?: Record<string, number>;
  range_days?: number; inferred_frequency?: string | null;
};

export type ProfileWarning = { code: string; severity: string; column: string | null; message: string };

export type DatasetProfile = {
  n_rows: number;
  n_columns: number;
  columns: Record<string, ColumnProfile>;
  numerical: string[];
  categorical: string[];
  datetime: string[];
  n_duplicate_rows: number;
  missing_value_pct: number;
  warnings: ProfileWarning[];
  target_candidates: { column: string; task: Task; score: number }[];
  target: null | {
    column: string;
    inferred_task: Task;
    class_distribution?: { counts: Record<string, number>; proportions: Record<string, number>; minority_ratio: number; imbalanced: boolean };
  };
};

export type RowsPage = {
  columns: { name: string; dtype: string }[];
  rows: Record<string, unknown>[];
  total: number;
  page: number;
  page_size: number;
};

export type ChangeImpact = { changed: string[]; affected_sections: string[]; reruns_training: boolean; stale_experiment_ids: string[]; message: string };

export type PipelineSaved = { version: PipelineVersion; created: boolean; impact: ChangeImpact | null };
