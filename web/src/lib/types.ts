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
  range_days?: number; inferred_frequency?: string | null; has_time?: boolean;
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

/* ---- Section 1: preprocessing ---- */

export type MissingStrategy = "drop_rows" | "drop_column" | "mean" | "median" | "mode" | "constant" | "knn";
export type EncodingStrategy = "one_hot" | "ordinal" | "label" | "frequency" | "target";
export type OutlierStrategy = "iqr" | "z_score" | "winsorize" | "isolation_forest" | "keep";
export type ScalingStrategy = "standard" | "min_max" | "robust" | "none";

export type MissingRule = { column: string; strategy: MissingStrategy; constant_value?: string | number | null; n_neighbors?: number };
export type EncodingRule = { column: string; strategy: EncodingStrategy };
export type OutlierRule = { column: string; strategy: OutlierStrategy; threshold: number };

export type Preprocessing = {
  drop_columns: string[];
  missing_values: MissingRule[];
  encoding: EncodingRule[];
  scaling: ScalingStrategy;
  scale_columns: string[];
  outliers: OutlierRule[];
  drop_duplicates: boolean;
};

export type DateRule = { column: string; extract: string[] };
export type NumericRule = { column: string; transform: string };
export type FeatureEngineering = { numeric_transforms: NumericRule[]; date_features: DateRule[] };

export type Recommendation = {
  id: string;
  kind: string;
  column: string | null;
  title: string;
  reason: string;
  confidence: number;
  action: Record<string, unknown> & { type: string };
  optional?: boolean; // shown, but not ticked by default
};

export type DataShape = { rows: number; columns: number; missing_cells: number; missing_pct: number; duplicate_rows: number };
export type SplitMethod = "train_test" | "train_val_test" | "k_fold" | "stratified_k_fold" | "time_series";
export type SplitConfig = {
  method: SplitMethod; test_size: number; validation_size: number; n_splits: number;
  stratify: boolean; time_column: string | null; random_state: number;
};
export type SplitSummary = {
  mode: "holdout" | "cv"; method: SplitMethod; stratified: boolean; chronological: boolean; n_folds: number;
  n_train: number; n_test: number; n_validation: number; notes: string[];
};

export type Preview = {
  before: DataShape;
  after: DataShape | null;
  issues: string[];
  steps: { step: string; rows_removed: number; detail: string }[];
  split: SplitSummary | null;
  split_issues: string[];
  model_issues: Record<string, string[]>;
  features: NewFeature[];
  feature_issues: string[];
  note?: string;
};

/* ---- Sections 4-6: models ---- */

export type HyperParamInfo = {
  name: string; kind: "int" | "float" | "choice" | "bool" | "optional_int"; default: unknown;
  choices: unknown[]; min: number | null; max: number | null; advanced: boolean; description: string;
};
export type RegInfo = { kind: "penalty" | "complexity" | "none"; options: string[]; complexity_params: string[]; note: string };
export type ModelInfo = {
  key: string; name: string; task: Task; requires_scaling: boolean; cost: "low" | "medium" | "high";
  interpretability: "low" | "medium" | "high"; hyperparameters: HyperParamInfo[]; regularization: RegInfo;
};
export type ModelConfig = {
  model_key: string; regularization: Record<string, unknown>; hyperparameters: Record<string, unknown>;
  use_recommended_defaults: boolean;
};
export type ModelDefaults = Record<string, { recommended: Record<string, unknown>; defaults: Record<string, unknown>; n_rows: number; n_features: number }>;

export type Job = {
  id: string; project_id: string; status: "queued" | "running" | "succeeded" | "failed";
  created_at: string; started_at: string | null; finished_at: string | null;
  experiment_number: number | null; error: string | null; issues: string[]; warnings: string[];
};

/* ---- Section 7: results ---- */

export type Metrics = Record<string, unknown> & {
  confusion_matrix?: { labels: string[]; matrix: number[][] };
  roc_curve?: { fpr: number[]; tpr: number[] };
  actual_vs_predicted?: { actual: number[]; predicted: number[] };
  residuals?: number[];
};
export type ModelResult = {
  model_key: string; name: string; hyperparameters: Record<string, unknown>;
  metrics: Record<string, Metrics>; fold_primary_scores: number[]; primary_metric: string;
  baseline: Record<string, number | null>; n_rows_fitted: number; n_outlier_rows_removed: number; fit_seconds: number;
};
export type QualityCheck = { id: string; status: "pass" | "warn" | "fail" | "na"; title: string; detail: string; model_key: string | null };
export type ExperimentResult = {
  experiment_id: string; pipeline_id: string; pipeline_version: number; dataset_fingerprint: string; config_hash: string;
  task: Task; random_state: number; started_at: string; finished_at: string; environment: Record<string, string>;
  split: Record<string, unknown> & { mode: "holdout" | "cv"; method: string; n_folds: number; n_train: number; n_test: number };
  preparation_log: { step: string; rows_removed: number; detail: string }[];
  models: ModelResult[]; quality: { checks: QualityCheck[]; score: number | null };
};
export type ExperimentSummary = {
  number: number; experiment_id: string; pipeline_version: number; parent_number: number | null; current: boolean;
  finished_at: string; quality_score: number | null; models: { model_key: string; name: string; metric: string; value: number | null }[];
};
export type ExperimentDetail = {
  experiment: { number: number; pipeline_version: number; parent_number: number | null; result: ExperimentResult };
  current: boolean;
};
export type Artifact = { id: string; kind: string; bucket: string; storage_path: string; size_bytes: number };
export type Comparison = {
  a: number; b: number;
  config_diff: { section: string; same: boolean; before: unknown; after: unknown }[];
  metrics: Record<string, Record<string, { a: number | null; b: number | null; delta: number | null }>>;
};

/* ---- Section 2: features ---- */

export type NewFeature = { name: string; source: string; op: string; sample: (number | string | null)[] };

/* ---- Section 0: charts ---- */

export type BoxStats = { min: number; q1: number; median: number; q3: number; max: number; whisker_low: number; whisker_high: number; outliers: number[]; outlier_count: number };
export type DistributionData = {
  column: string; n: number; missing: number; edges: number[]; counts: number[]; box: BoxStats;
  stats: { mean: number | null; std: number | null; skewness: number | null };
};
export type CategoryData = { column: string; n: number; missing: number; n_unique: number; items: { label: string; count: number; share: number }[]; other: number; other_share: number };
export type ClassBalanceData = { column: string; n: number; imbalanced: boolean; minority_ratio: number; items: { label: string; count: number; share: number }[] };
export type MissingData = { n_rows: number; columns: { column: string; missing: number; pct: number }[]; rows_with_missing: number; rows_with_missing_pct: number };
export type ScatterData = { x_column: string; y_column: string; n_total: number; n_shown: number; r: number | null; x: number[]; y: number[] };
export type CorrelationData = { columns: string[]; matrix: (number | null)[][]; target: string | null; note: string | null; top_pairs: { a: string; b: string; r: number }[] };
export type ChartSpec =
  | { id: string; type: "class_balance"; title: string; why: string; data: ClassBalanceData }
  | { id: string; type: "distribution"; title: string; why: string; data: DistributionData }
  | { id: string; type: "categories"; title: string; why: string; data: CategoryData }
  | { id: string; type: "missing"; title: string; why: string; data: MissingData }
  | { id: string; type: "correlation"; title: string; why: string; data: CorrelationData }
  | { id: string; type: "scatter"; title: string; why: string; data: ScatterData };
