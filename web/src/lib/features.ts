import { sameJson } from "./canon";
import type { ColumnProfile, FeatureEngineering, PipelineConfig, Recommendation } from "./types";

const EMPTY: FeatureEngineering = { numeric_transforms: [], date_features: [] };

export const featFromConfig = (c: PipelineConfig): FeatureEngineering =>
  ({ ...EMPTY, ...structuredClone(c.feature_engineering) }) as FeatureEngineering;
export const sameFeat = (a: FeatureEngineering, b: FeatureEngineering) => sameJson(a, b);
export const isFeatConfigured = (c: PipelineConfig) => {
  const f = featFromConfig(c);
  return f.numeric_transforms.length + f.date_features.length > 0;
};

export const TRANSFORMS = [
  { value: "log", label: "Log", help: "log(1 + x). Compresses a long tail of large values." },
  { value: "sqrt", label: "Square root", help: "Gentler than log. Needs values of 0 or more." },
  { value: "abs", label: "Absolute value", help: "Drops the sign: -5 and 5 become 5." },
  { value: "square", label: "Square", help: "x². Lets a linear model see a curve." },
] as const;

export const DATE_PARTS = [
  { value: "year", label: "Year" }, { value: "month", label: "Month" }, { value: "day", label: "Day of month" },
  { value: "day_of_week", label: "Weekday" }, { value: "quarter", label: "Quarter" }, { value: "is_weekend", label: "Weekend?" },
  { value: "hour", label: "Hour" },
] as const;

/** Why a transform can't be used on this column, or null if it can. */
export function transformBlocked(stats: ColumnProfile, t: string): string | null {
  const lo = typeof stats.min === "number" ? stats.min : null;
  if (lo === null) return null;
  if (t === "log" && lo <= -1) return `Log needs values above -1; this column goes down to ${lo.toLocaleString()}.`;
  if (t === "sqrt" && lo < 0) return `Square root needs values of 0 or more; this column goes down to ${lo.toLocaleString()}.`;
  return null;
}

export const hasTransform = (d: FeatureEngineering, col: string, t: string) =>
  d.numeric_transforms.some((r) => r.column === col && r.transform === t);

export function toggleTransform(d: FeatureEngineering, col: string, t: string): FeatureEngineering {
  const n = structuredClone(d);
  n.numeric_transforms = hasTransform(d, col, t)
    ? n.numeric_transforms.filter((r) => !(r.column === col && r.transform === t))
    : [...n.numeric_transforms, { column: col, transform: t }];
  return n;
}

export const dateParts = (d: FeatureEngineering, col: string) => d.date_features.find((r) => r.column === col)?.extract ?? [];

export function setDateParts(d: FeatureEngineering, col: string, parts: string[]): FeatureEngineering {
  const n = structuredClone(d);
  n.date_features = n.date_features.filter((r) => r.column !== col);
  if (parts.length) n.date_features.push({ column: col, extract: DATE_PARTS.map((p) => p.value).filter((v) => parts.includes(v)) });
  return n;
}

/** Mirror of the engine's apply_actions for the feature-engineering part. */
export function applyFeatureAction(d: FeatureEngineering, a: Recommendation["action"]): FeatureEngineering {
  if (a.type === "numeric_transform") return hasTransform(d, a.column as string, a.transform as string) ? d : toggleTransform(d, a.column as string, a.transform as string);
  if (a.type === "date_features") return setDateParts(d, a.column as string, a.extract as string[]);
  throw new Error(`Unknown recommendation action: ${a.type}`);
}
