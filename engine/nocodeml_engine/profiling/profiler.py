"""Deterministic dataset profiler (Section 0).

Everything here is computed with pandas/numpy/scipy. The resulting
`DatasetProfile` is what a future recommendation engine consumes - never the
raw data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nocodeml_engine.config import TaskType

ID_NAME_HINTS = ("id", "uuid", "guid", "index", "key")
HIGH_CARDINALITY_ABS = 50
HIGH_CARDINALITY_RATIO = 0.5
NEAR_CONSTANT_RATIO = 0.99
IMBALANCE_MINORITY_RATIO = 0.2
CLASSIFICATION_MAX_UNIQUE = 20


@dataclass
class Warning_:
    code: str
    severity: str  # "info" | "warning"
    column: str | None
    message: str


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: dict[str, dict[str, Any]]
    numerical: list[str]
    categorical: list[str]
    datetime: list[str]
    n_duplicate_rows: int
    missing_value_pct: float
    warnings: list[Warning_] = field(default_factory=list)
    target_candidates: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(v: Any) -> Any:
    """Make numpy/pandas scalars JSON-serializable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def numeric_stats(s: pd.Series) -> dict[str, Any]:
    x = s.dropna().astype(float)
    n = len(s)
    out: dict[str, Any] = {"count": int(len(x)), "missing_pct": round(100 * (n - len(x)) / n, 2) if n else 0.0,
                           "unique": int(s.nunique())}
    if x.empty:
        return out
    q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mode = x.mode()
    out.update(
        mean=x.mean(), median=q2, mode=mode.iloc[0] if not mode.empty else None,
        min=x.min(), max=x.max(), range=x.max() - x.min(),
        variance=x.var(ddof=1) if len(x) > 1 else 0.0,
        std=x.std(ddof=1) if len(x) > 1 else 0.0,
        q1=q1, q2=q2, q3=q3, iqr=iqr,
        skewness=stats.skew(x, bias=False) if len(x) > 2 and x.nunique() > 1 else 0.0,
        kurtosis=stats.kurtosis(x, bias=False) if len(x) > 3 and x.nunique() > 1 else 0.0,
        outlier_count=int(((x < lo) | (x > hi)).sum()),
    )
    return {k: _clean(v) for k, v in out.items()}


def categorical_stats(s: pd.Series, top: int = 20) -> dict[str, Any]:
    x = s.dropna()
    n = len(s)
    vc = x.value_counts()
    return {
        "count": int(len(x)),
        "missing_pct": round(100 * (n - len(x)) / n, 2) if n else 0.0,
        "unique": int(s.nunique()),
        "cardinality": int(s.nunique()),
        "mode": _clean(vc.index[0]) if not vc.empty else None,
        "frequencies": {str(k): int(v) for k, v in vc.head(top).items()},
    }


def datetime_stats(s: pd.Series) -> dict[str, Any]:
    x = s.dropna()
    n = len(s)
    out: dict[str, Any] = {"count": int(len(x)), "missing_pct": round(100 * (n - len(x)) / n, 2) if n else 0.0,
                           "unique": int(s.nunique())}
    if not x.empty:
        out.update(min=x.min(), max=x.max(), range_days=(x.max() - x.min()).days)
        freq = None
        if len(x) >= 3:
            try:
                freq = pd.infer_freq(pd.DatetimeIndex(x.sort_values().unique()))
            except (ValueError, TypeError):
                freq = None
        out["inferred_frequency"] = freq
    return {k: _clean(v) for k, v in out.items()}


def _is_identifier_like(name: str, s: pd.Series, kind: str) -> bool:
    n = s.notna().sum()
    if n == 0 or s.nunique() != n:  # identifiers are unique per row
        return False
    if kind == "categorical":
        return n > 1
    if kind == "numerical":
        lowered = name.lower().replace(" ", "_")
        hinted = any(lowered == h or lowered.endswith("_" + h) or lowered.startswith(h + "_")
                     for h in ID_NAME_HINTS)
        if hinted:
            return True
        # monotonically increasing integers (row numbers)
        if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing and n > 10:
            return True
    return False


def infer_task(target: pd.Series) -> TaskType:
    if not pd.api.types.is_numeric_dtype(target):
        return TaskType.CLASSIFICATION
    if pd.api.types.is_float_dtype(target) and (target.dropna() % 1 != 0).any():
        return TaskType.REGRESSION
    return (TaskType.CLASSIFICATION if target.nunique() <= CLASSIFICATION_MAX_UNIQUE
            else TaskType.REGRESSION)


def class_distribution(target: pd.Series) -> dict[str, Any]:
    vc = target.dropna().value_counts()
    total = int(vc.sum())
    return {
        "counts": {str(k): int(v) for k, v in vc.items()},
        "proportions": {str(k): round(float(v) / total, 4) for k, v in vc.items()},
        "minority_ratio": round(float(vc.min()) / total, 4) if total else None,
        "imbalanced": bool(total and vc.min() / total < IMBALANCE_MINORITY_RATIO
                           and len(vc) >= 2),
    }


def profile_dataset(df: pd.DataFrame, target: str | None = None) -> DatasetProfile:
    numerical = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                 and not pd.api.types.is_bool_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [c for c in df.columns if c not in numerical and c not in datetime_cols]

    total_cells = df.shape[0] * df.shape[1]
    warnings: list[Warning_] = []
    columns: dict[str, dict[str, Any]] = {}

    for c in df.columns:
        if c in numerical:
            kind, st = "numerical", numeric_stats(df[c])
        elif c in datetime_cols:
            kind, st = "datetime", datetime_stats(df[c])
        else:
            kind, st = "categorical", categorical_stats(df[c])
        st["kind"] = kind
        st["dtype"] = str(df[c].dtype)
        columns[str(c)] = st

        n_unique = df[c].nunique()
        if len(df) and n_unique <= 1:
            warnings.append(Warning_("constant_column", "warning", c,
                                     f"'{c}' has a single value and carries no information."))
        elif kind != "datetime" and len(df) > 20:
            top = df[c].value_counts(normalize=True, dropna=True)
            if not top.empty and top.iloc[0] >= NEAR_CONSTANT_RATIO:
                warnings.append(Warning_("near_constant_column", "warning", c,
                                         f"'{c}' is {top.iloc[0]:.1%} one value."))
        if kind != "datetime" and _is_identifier_like(str(c), df[c], kind):
            st["identifier_like"] = True
            warnings.append(Warning_("possible_identifier", "warning", c,
                                     f"'{c}' looks like an identifier (unique per row); it "
                                     "should usually be excluded from features."))
        if kind == "categorical" and not st.get("identifier_like") and (
                n_unique > HIGH_CARDINALITY_ABS
                or (len(df) and n_unique / len(df) > HIGH_CARDINALITY_RATIO and n_unique > 10)):
            warnings.append(Warning_("high_cardinality", "warning", c,
                                     f"'{c}' has {n_unique} distinct values; one-hot encoding "
                                     "would create many columns."))
        if st["missing_pct"] > 0:
            warnings.append(Warning_("missing_values", "warning", c,
                                     f"'{c}' has {st['missing_pct']}% missing values."))

    n_dups = int(df.duplicated().sum())
    if n_dups:
        warnings.append(Warning_("duplicate_rows", "warning", None, f"{n_dups} duplicate rows."))
    if df.columns.duplicated().any():
        warnings.append(Warning_("duplicate_column_names", "warning", None,
                                 "Duplicate column names found."))

    profile = DatasetProfile(
        n_rows=len(df), n_columns=df.shape[1], columns=columns,
        numerical=[str(c) for c in numerical], categorical=[str(c) for c in categorical],
        datetime=[str(c) for c in datetime_cols], n_duplicate_rows=n_dups,
        missing_value_pct=round(100 * int(df.isna().sum().sum()) / total_cells, 2) if total_cells else 0.0,
        warnings=warnings,
    )

    # Target candidates: non-identifier, non-constant, non-datetime columns,
    # ranked with a bias towards low-cardinality labels and last columns.
    for i, c in enumerate(df.columns):
        st = columns[str(c)]
        if st["kind"] == "datetime" or st.get("identifier_like") or df[c].nunique() <= 1:
            continue
        task = infer_task(df[c])
        score = (2 if task is TaskType.CLASSIFICATION and df[c].nunique() <= 10 else 0) + (
            1 if i == len(df.columns) - 1 else 0)
        profile.target_candidates.append({"column": str(c), "task": task.value, "score": score})
    profile.target_candidates.sort(key=lambda t: -t["score"])

    if target is not None:
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found.")
        task = infer_task(df[target])
        info: dict[str, Any] = {"column": target, "inferred_task": task.value}
        if task is TaskType.CLASSIFICATION:
            info["class_distribution"] = dist = class_distribution(df[target])
            if dist["imbalanced"]:
                profile.warnings.append(Warning_(
                    "class_imbalance", "warning", target,
                    f"Class imbalance: minority class is {dist['minority_ratio']:.1%} of rows. "
                    "Accuracy can be misleading; use stratified splitting and F1/ROC-AUC."))
        if df[target].isna().any():
            profile.warnings.append(Warning_("missing_target", "warning", target,
                                             "Target has missing values; those rows will be dropped."))
        profile.target = info
    return profile
