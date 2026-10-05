"""Deterministic preprocessing recommendations (Section 1, "NoCodeML recommendation" mode).

Rules over the dataset profile only: no raw data, no LLM. Every recommendation carries a plain-
language reason and a structured `action` that `apply_actions` can apply to a PipelineConfig.
Nothing is ever applied silently: the caller decides which recommendations to accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nocodeml_engine.config import (
    DateFeature, DateFeatureRule, EncodingRule, EncodingStrategy, MissingValueRule,
    MissingValueStrategy, NumericFeatureRule, NumericTransform, OutlierRule, OutlierStrategy, PipelineConfig, ScalingStrategy,
)
from nocodeml_engine.profiling.profiler import DatasetProfile

HIGH_MISSING_PCT = 40.0
ONE_HOT_MAX_UNIQUE = 15
SKEW_LIMIT = 1.0
OUTLIER_SHARE = 0.05


@dataclass
class Recommendation:
    id: str
    kind: str  # drop_column | impute | encode | scale | outliers | drop_duplicates | date_features
    column: str | None
    title: str
    reason: str
    confidence: float
    action: dict[str, Any]
    optional: bool = False  # shown, but not ticked by default: a judgement call the user should make


def recommend_preprocessing(profile: DatasetProfile, target: str,
                            models_need_scaling: bool | None = None) -> list[Recommendation]:
    """`models_need_scaling`: True/False once models are chosen, None while still unknown."""
    recs: list[Recommendation] = []
    dropped: set[str] = set()
    n_rows = max(profile.n_rows, 1)

    def add(kind, column, title, reason, confidence, action):
        recs.append(Recommendation(f"{kind}:{column or '*'}", kind, column, title, reason,
                                   round(confidence, 2), action))

    cols = {c: s for c, s in profile.columns.items() if c != target}

    # 1. Columns that should not be features at all
    for c, s in cols.items():
        if s.get("identifier_like"):
            dropped.add(c)
            add("drop_column", c, f"Drop '{c}'",
                f"'{c}' is unique for every row, so it identifies rows rather than describing them. "
                "A model could memorize it instead of learning patterns.", 0.95,
                {"type": "drop_column", "column": c})
        elif s["unique"] <= 1 and s["count"] > 0:
            dropped.add(c)
            add("drop_column", c, f"Drop '{c}'", f"'{c}' has a single value, so it carries no information.",
                0.99, {"type": "drop_column", "column": c})
        elif s["missing_pct"] > HIGH_MISSING_PCT:
            dropped.add(c)
            add("drop_column", c, f"Drop '{c}'",
                f"'{c}' is {s['missing_pct']:.0f}% empty; filling it in would mostly invent values.", 0.7,
                {"type": "drop_column", "column": c})

    if profile.n_duplicate_rows:
        pct = 100 * profile.n_duplicate_rows / n_rows
        add("drop_duplicates", None, "Remove duplicate rows",
            f"{profile.n_duplicate_rows} rows ({pct:.1f}%) are exact duplicates. Identical rows can land in both "
            "the training and test sets and inflate the score.", 0.85, {"type": "drop_duplicates"})

    # 2. Per-column handling
    outlier_cols = 0
    for c, s in cols.items():
        if c in dropped:
            continue
        kind = s["kind"]
        miss = s["missing_pct"]
        if kind == "datetime":
            from nocodeml_engine.recommendations.features import date_parts_for
            parts = date_parts_for(s)
            names = ", ".join(x.replace("_", " ") for x in parts)
            add("date_features", c, f"Extract {names} from '{c}'",
                "Models can't use raw dates. The date parts capture trend, seasonality and weekday patterns instead.",
                0.8, {"type": "date_features", "column": c, "extract": parts})
            continue
        if kind == "numerical":
            out_share = (s.get("outlier_count") or 0) / max(s["count"], 1)
            skew = s.get("skewness") or 0.0
            if miss > 0:
                skewed = abs(skew) > SKEW_LIMIT or out_share > OUTLIER_SHARE
                strategy = "median" if skewed else "mean"
                why = (f"'{c}' is {miss:.1f}% missing. " + (
                    f"Its distribution is skewed (skewness {skew:.1f}) or has outliers, so the median is a more "
                    "typical value than the mean." if skewed else
                    f"Its distribution is roughly symmetric (skewness {skew:.1f}), so the mean is a fair fill-in."))
                add("impute", c, f"Fill missing '{c}' with the {strategy}", why, 0.9 if skewed else 0.85,
                    {"type": "impute", "column": c, "strategy": strategy})
            if out_share > OUTLIER_SHARE:
                outlier_cols += 1
                add("outliers", c, f"Clip outliers in '{c}'",
                    f"{s['outlier_count']} values ({100 * out_share:.0f}%) lie beyond the usual range. Clipping them to "
                    "the range limits keeps every row but stops extreme values dominating.", 0.6,
                    {"type": "outliers", "column": c, "strategy": "winsorize", "threshold": 1.5})
        else:  # categorical
            if miss > 0:
                add("impute", c, f"Fill missing '{c}' with its most common value",
                    f"'{c}' is {miss:.1f}% missing. Categories have no average, so the most common one is used.",
                    0.8, {"type": "impute", "column": c, "strategy": "mode"})
            card = s.get("cardinality") or s["unique"]
            if card <= ONE_HOT_MAX_UNIQUE:
                add("encode", c, f"One-hot encode '{c}'",
                    f"'{c}' has {card} distinct values with no natural order, so each gets its own 0/1 column.",
                    0.9, {"type": "encode", "column": c, "strategy": "one_hot"})
            else:
                add("encode", c, f"Frequency-encode '{c}'",
                    f"'{c}' has {card} distinct values; one-hot would add {card} columns. Replacing each value by how "
                    "common it is keeps one column.", 0.75, {"type": "encode", "column": c, "strategy": "frequency"})

    # 3. Scaling
    if models_need_scaling is not False:
        robust = outlier_cols >= 2
        strategy = "robust" if robust else "standard"
        if models_need_scaling:
            why = ("Your selected models are sensitive to feature scale, so features should share a common range."
                   + (" Several columns have outliers, so the robust scaler (median and IQR) is safer." if robust else ""))
            conf = 0.9
        else:
            why = ("Logistic regression, KNN, SVM, ridge and lasso are sensitive to feature scale; tree models are not. "
                   "Scaling is a safe default until you choose models."
                   + (" Several columns have outliers, so the robust scaler (median and IQR) is safer." if robust else ""))
            conf = 0.7
        add("scale", None, f"Scale numeric features ({'robust' if robust else 'standard'})", why, conf,
            {"type": "scale", "strategy": strategy})
    return recs


def apply_actions(config: PipelineConfig, actions: list[dict[str, Any]]) -> PipelineConfig:
    """Return a new config with the given recommendation actions applied (reference implementation;
    the web UI mirrors this logic on its draft)."""
    pre = config.preprocessing.model_copy(deep=True)
    fe = config.feature_engineering.model_copy(deep=True)

    def upsert(lst, col, rule):
        lst[:] = [r for r in lst if r.column != col] + [rule]

    for a in actions:
        t = a["type"]
        if t == "drop_column":
            if a["column"] not in pre.drop_columns:
                pre.drop_columns.append(a["column"])
        elif t == "drop_duplicates":
            pre.drop_duplicates = True
        elif t == "impute":
            upsert(pre.missing_values, a["column"],
                   MissingValueRule(column=a["column"], strategy=MissingValueStrategy(a["strategy"])))
        elif t == "encode":
            upsert(pre.encoding, a["column"], EncodingRule(column=a["column"], strategy=EncodingStrategy(a["strategy"])))
        elif t == "outliers":
            upsert(pre.outliers, a["column"], OutlierRule(
                column=a["column"], strategy=OutlierStrategy(a["strategy"]), threshold=a.get("threshold", 1.5)))
        elif t == "scale":
            pre.scaling = ScalingStrategy(a["strategy"])
        elif t == "numeric_transform":
            fe.numeric_transforms = [r for r in fe.numeric_transforms if not (r.column == a["column"] and r.transform.value == a["transform"])] + [
                NumericFeatureRule(column=a["column"], transform=NumericTransform(a["transform"]))]
        elif t == "date_features":
            fe.date_features = [r for r in fe.date_features if r.column != a["column"]] + [
                DateFeatureRule(column=a["column"], extract=[DateFeature(x) for x in a["extract"]])]
        else:
            raise ValueError(f"Unknown recommendation action '{t}'")
    return config.model_copy(update={"preprocessing": pre, "feature_engineering": fe})
