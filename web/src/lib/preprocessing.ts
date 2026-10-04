import type {
  ColumnKind, EncodingStrategy, FeatureEngineering, MissingStrategy, OutlierStrategy, PipelineConfig,
  Preprocessing, Recommendation,
} from "./types";

/** Everything the Preprocessing section edits. */
export type Draft = { pre: Preprocessing; fe: FeatureEngineering };

export type Role = "use" | "drop" | "dateparts";

const EMPTY_PRE: Preprocessing = {
  drop_columns: [], missing_values: [], encoding: [], scaling: "none", scale_columns: [], outliers: [], drop_duplicates: false,
};
const EMPTY_FE: FeatureEngineering = { numeric_transforms: [], date_features: [] };

export const DATE_PARTS = ["year", "month", "day_of_week"];

export function fromConfig(c: PipelineConfig): Draft {
  return {
    pre: { ...EMPTY_PRE, ...structuredClone(c.preprocessing) } as Preprocessing,
    fe: { ...EMPTY_FE, ...structuredClone(c.feature_engineering) } as FeatureEngineering,
  };
}

/**
 * Order-independent JSON with null/undefined fields and the default n_neighbors dropped, so a draft
 * built in the browser compares equal to the same choices as echoed back by the server (which adds
 * defaults such as constant_value: null).
 */
function canon(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(canon);
  if (v && typeof v === "object") {
    return Object.fromEntries(
      Object.entries(v as Record<string, unknown>)
        .filter(([k, x]) => x !== null && x !== undefined && !(k === "n_neighbors" && x === 5))
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, x]) => [k, canon(x)]),
    );
  }
  return v;
}

export const sameDraft = (a: Draft, b: Draft) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));

export function isConfigured(c: PipelineConfig): boolean {
  const d = fromConfig(c);
  return d.pre.drop_columns.length + d.pre.missing_values.length + d.pre.encoding.length + d.pre.outliers.length > 0
    || d.pre.drop_duplicates || d.pre.scaling !== "none" || d.fe.date_features.length > 0;
}

const without = <T extends { column: string }>(l: T[], col: string) => l.filter((r) => r.column !== col);

export function getRole(d: Draft, col: string): Role {
  if (d.pre.drop_columns.includes(col)) return "drop";
  if (d.fe.date_features.some((r) => r.column === col)) return "dateparts";
  return "use";
}

export function setRole(d: Draft, col: string, role: Role): Draft {
  const n = structuredClone(d);
  n.pre.drop_columns = n.pre.drop_columns.filter((c) => c !== col);
  n.fe.date_features = without(n.fe.date_features, col);
  if (role === "drop") {
    n.pre.drop_columns.push(col);
    n.pre.missing_values = without(n.pre.missing_values, col);
    n.pre.encoding = without(n.pre.encoding, col);
    n.pre.outliers = without(n.pre.outliers, col);
  } else if (role === "dateparts") {
    n.fe.date_features.push({ column: col, extract: [...DATE_PARTS] });
  }
  return n;
}

export const getMissing = (d: Draft, col: string) => d.pre.missing_values.find((r) => r.column === col);
export const getEncoding = (d: Draft, col: string) => d.pre.encoding.find((r) => r.column === col)?.strategy ?? "";
export const getOutlier = (d: Draft, col: string) => d.pre.outliers.find((r) => r.column === col)?.strategy ?? "";

export function setMissing(d: Draft, col: string, strategy: MissingStrategy | "", constant?: string | number | null): Draft {
  const n = structuredClone(d);
  n.pre.missing_values = without(n.pre.missing_values, col);
  if (strategy) {
    n.pre.missing_values.push({
      column: col, strategy, n_neighbors: 5, ...(strategy === "constant" ? { constant_value: constant ?? null } : {}),
    });
  }
  return n;
}

export function setEncoding(d: Draft, col: string, strategy: EncodingStrategy | ""): Draft {
  const n = structuredClone(d);
  n.pre.encoding = without(n.pre.encoding, col);
  if (strategy) n.pre.encoding.push({ column: col, strategy });
  return n;
}

export function setOutlier(d: Draft, col: string, strategy: OutlierStrategy | "", threshold?: number): Draft {
  const n = structuredClone(d);
  n.pre.outliers = without(n.pre.outliers, col);
  if (strategy && strategy !== "keep") {
    n.pre.outliers.push({ column: col, strategy, threshold: threshold ?? (strategy === "z_score" ? 3 : 1.5) });
  }
  return n;
}

/** Mirror of engine/nocodeml_engine/recommendations.apply_actions. */
export function applyAction(d: Draft, a: Recommendation["action"]): Draft {
  const col = a.column as string;
  switch (a.type) {
    case "drop_column": return setRole(d, col, "drop");
    case "drop_duplicates": return { ...d, pre: { ...d.pre, drop_duplicates: true } };
    case "impute": return setMissing(d, col, a.strategy as MissingStrategy);
    case "encode": return setEncoding(d, col, a.strategy as EncodingStrategy);
    case "outliers": return setOutlier(d, col, a.strategy as OutlierStrategy, (a.threshold as number) ?? 1.5);
    case "scale": return { ...d, pre: { ...d.pre, scaling: a.strategy as Preprocessing["scaling"] } };
    case "date_features": {
      const n = setRole(d, col, "dateparts");
      n.fe.date_features = n.fe.date_features.map((r) => (r.column === col ? { column: col, extract: a.extract as string[] } : r));
      return n;
    }
    default: throw new Error(`Unknown recommendation action: ${a.type}`);
  }
}

export function applyAll(d: Draft, recs: Recommendation[]): Draft {
  return recs.reduce((acc, r) => applyAction(acc, r.action), d);
}

/** The body for PUT /pipeline and POST /pipeline/preview: saved config with this draft swapped in. */
export function bodyFor(c: PipelineConfig, d: Draft) {
  return {
    dataset: c.dataset, preprocessing: d.pre, feature_engineering: d.fe, split: c.split, models: c.models,
  };
}

export const MISSING_OPTIONS: Record<ColumnKind, { value: MissingStrategy; label: string }[]> = {
  numerical: [
    { value: "median", label: "Fill with median" }, { value: "mean", label: "Fill with mean" },
    { value: "constant", label: "Fill with a value" }, { value: "knn", label: "KNN (nearest rows)" },
    { value: "drop_rows", label: "Drop those rows" },
  ],
  categorical: [
    { value: "mode", label: "Most common value" }, { value: "constant", label: "Fill with a value" },
    { value: "drop_rows", label: "Drop those rows" },
  ],
  datetime: [{ value: "drop_rows", label: "Drop those rows" }],
};

export const ENCODING_OPTIONS: { value: EncodingStrategy; label: string }[] = [
  { value: "one_hot", label: "One-hot" }, { value: "ordinal", label: "Ordinal" }, { value: "label", label: "Label" },
  { value: "frequency", label: "Frequency" }, { value: "target", label: "Target" },
];

export const OUTLIER_OPTIONS: { value: OutlierStrategy; label: string }[] = [
  { value: "keep", label: "Keep" }, { value: "winsorize", label: "Clip to range" },
  { value: "iqr", label: "Remove rows (IQR)" }, { value: "z_score", label: "Remove rows (z-score)" },
  { value: "isolation_forest", label: "Isolation Forest" },
];
