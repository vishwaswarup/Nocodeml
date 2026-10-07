"""Chart data for the dataset explorer (Section 0).

Everything returned is small, aggregated and JSON-safe: histogram bins, box-plot statistics,
value counts, a correlation matrix, a seeded scatter sample. Raw rows never leave this module in
bulk. Deterministic: sampling uses a fixed seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nocodeml_engine.config import TaskType
from nocodeml_engine.profiling.profiler import DatasetProfile, class_distribution, infer_task

MAX_CORR_COLUMNS = 20
SCATTER_POINTS = 800
SEED = 0


class ChartError(ValueError):
    pass


def _f(v: Any) -> float | None:
    v = float(v)
    return round(v, 6) if np.isfinite(v) else None


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ChartError(f"Column '{name}' not found.")
    return df[name]


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    s = _col(df, name)
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise ChartError(f"'{name}' is not a numeric column.")
    return s


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> dict[str, Any]:
    """Histogram bins + box-plot statistics for one numeric column."""
    if not 3 <= bins <= 100:
        raise ChartError("bins must be between 3 and 100.")
    s = _numeric(df, column)
    x = s.dropna().astype(float)
    if x.empty:
        raise ChartError(f"'{column}' has no values to plot.")
    x = x[np.isfinite(x)]
    if x.empty:
        raise ChartError(f"'{column}' has no finite values to plot.")
    counts, edges = np.histogram(x, bins=bins)
    q1, med, q3 = (float(v) for v in x.quantile([0.25, 0.5, 0.75]))
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = x[(x >= lo_f) & (x <= hi_f)]
    out = x[(x < lo_f) | (x > hi_f)].sort_values()
    if len(out) > 60:  # keep the most extreme on both sides, evenly
        idx = np.unique(np.linspace(0, len(out) - 1, 60).astype(int))
        out = out.iloc[idx]
    return {
        "column": column, "n": int(len(x)), "missing": int(s.isna().sum()),
        "edges": [_f(e) for e in edges], "counts": [int(c) for c in counts],
        "box": {"min": _f(x.min()), "q1": _f(q1), "median": _f(med), "q3": _f(q3), "max": _f(x.max()),
                "whisker_low": _f(inside.min()), "whisker_high": _f(inside.max()),
                "outliers": [_f(v) for v in out], "outlier_count": int(((x < lo_f) | (x > hi_f)).sum())},
        "stats": {"mean": _f(x.mean()), "std": _f(x.std(ddof=1)) if len(x) > 1 else 0.0,
                  "skewness": _f(x.skew()) if len(x) > 2 and x.nunique() > 1 else 0.0},
    }


def categories(df: pd.DataFrame, column: str, top: int = 12) -> dict[str, Any]:
    """Most common values of a column (everything else folded into 'Other')."""
    if not 3 <= top <= 30:
        raise ChartError("top must be between 3 and 30.")
    s = _col(df, column)
    vc = s.value_counts(dropna=True)
    total = int(vc.sum())
    shown = vc.head(top)
    other = int(vc.iloc[top:].sum()) if len(vc) > top else 0
    items = [{"label": str(k), "count": int(v), "share": round(int(v) / total, 4) if total else 0.0} for k, v in shown.items()]
    return {"column": column, "n": total, "missing": int(s.isna().sum()), "n_unique": int(s.nunique()),
            "items": items, "other": other, "other_share": round(other / total, 4) if total else 0.0}


def class_balance(df: pd.DataFrame, target: str) -> dict[str, Any]:
    s = _col(df, target).dropna()
    if s.empty:
        raise ChartError(f"'{target}' has no values to plot.")
    d = class_distribution(s)
    return {"column": target, "n": int(len(s)), "imbalanced": d["imbalanced"], "minority_ratio": d["minority_ratio"],
            "items": [{"label": k, "count": v, "share": d["proportions"][k]} for k, v in d["counts"].items()][:30]}


def missing_overview(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    miss = df.isna().sum()
    cols = [{"column": str(c), "missing": int(m), "pct": round(100 * int(m) / n, 2) if n else 0.0}
            for c, m in miss.items() if m > 0]
    cols.sort(key=lambda r: (-r["missing"], r["column"]))
    any_missing = int(df.isna().any(axis=1).sum())
    return {"n_rows": n, "columns": cols, "rows_with_missing": any_missing,
            "rows_with_missing_pct": round(100 * any_missing / n, 2) if n else 0.0}


def scatter(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    if x == y:
        raise ChartError("Choose two different columns.")
    pair = pd.DataFrame({"x": _numeric(df, x), "y": _numeric(df, y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        raise ChartError("Not enough rows with both values to plot.")
    r = pair["x"].corr(pair["y"])
    shown = pair.sample(SCATTER_POINTS, random_state=SEED) if len(pair) > SCATTER_POINTS else pair
    return {"x_column": x, "y_column": y, "n_total": int(len(pair)), "n_shown": int(len(shown)),
            "r": _f(r) if np.isfinite(r) else None,
            "x": [_f(v) for v in shown["x"]], "y": [_f(v) for v in shown["y"]]}


def correlation(df: pd.DataFrame, target: str | None = None, columns: list[str] | None = None,
                exclude: set[str] | None = None) -> dict[str, Any]:
    """Pearson correlation between numeric columns, plus the target when it can be coded as a number."""
    exclude = exclude or set()
    if columns is not None:
        for c in columns:
            _numeric(df, c)
        cols = list(columns)
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                and c != target and c not in exclude and df[c].nunique() > 1]
    if len(cols) > MAX_CORR_COLUMNS:
        cols = sorted(cols, key=lambda c: (-int(df[c].notna().sum()), str(c)))[:MAX_CORR_COLUMNS]
    frame = df[cols].astype(float).copy()
    note = None
    if target and target in df.columns and target not in frame.columns:
        t = df[target]
        if pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t) and infer_task(t) is TaskType.REGRESSION:
            frame[target] = t.astype(float)
        elif t.nunique() == 2:  # two classes: code as 0/1 so "correlation with the target" is meaningful
            order = sorted(t.dropna().unique().tolist(), key=str)
            frame[target] = t.map({order[0]: 0.0, order[1]: 1.0})
            note = f"'{target}' coded as {order[0]} = 0, {order[1]} = 1."
    if frame.shape[1] < 2:
        raise ChartError("Need at least two numeric columns to compute correlations.")
    m = frame.corr(min_periods=5)
    names = [str(c) for c in m.columns]
    matrix = [[_f(m.iloc[i, j]) if np.isfinite(m.iloc[i, j]) else None for j in range(len(names))] for i in range(len(names))]
    pairs = [{"a": names[i], "b": names[j], "r": matrix[i][j]} for i in range(len(names)) for j in range(i + 1, len(names))
             if matrix[i][j] is not None]
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"columns": names, "matrix": matrix, "target": target if target in names else None, "note": note,
            "top_pairs": pairs[:30]}


# ---------------------------------------------------------------------------
# Auto visualize
# ---------------------------------------------------------------------------


def auto_charts(df: pd.DataFrame, profile: DatasetProfile, target: str | None = None) -> list[dict[str, Any]]:
    """A sensible default set of charts, each with the reason it was chosen. Rule-based, not AI."""
    specs: list[dict[str, Any]] = []
    cols = profile.columns
    skip = {c for c, s in cols.items() if s.get("identifier_like")} | ({target} if target else set())

    def add(type_: str, title: str, why: str, data: dict[str, Any], **params: Any) -> None:
        specs.append({"id": f"{type_}:{params.get('column') or params.get('x') or '*'}", "type": type_, "title": title,
                      "why": why, "params": params, "data": data})

    if target and target in df.columns:
        task = infer_task(df[target])
        if task is TaskType.CLASSIFICATION:
            add("class_balance", f"Class balance of '{target}'",
                "Shows how many rows belong to each class. A rare class means accuracy can mislead and the split should be stratified.",
                class_balance(df, target), column=target)
        else:
            add("distribution", f"Distribution of '{target}'", "The value you are predicting: its spread and any extreme values.",
                histogram(df, target), column=target)
    miss = missing_overview(df)
    if miss["columns"]:
        add("missing", "Missing values",
            f"{miss['rows_with_missing_pct']}% of rows have at least one gap. Decide in Preprocessing how to handle each column.", miss)
    numeric = [c for c, s in cols.items() if s["kind"] == "numerical" and c not in skip and s["unique"] > 1]
    corr: dict[str, Any] | None = None
    if len(numeric) >= 3:
        try:
            corr = correlation(df, target=target, exclude=skip - ({target} if target else set()))
            add("correlation", "Correlations",
                "Which numeric columns move together. Very strong pairs carry duplicate information; strong links to the target are promising features.", corr)
        except ChartError:
            pass
    skewed = sorted((c for c in numeric if abs(cols[c].get("skewness") or 0) > 1), key=lambda c: -abs(cols[c].get("skewness") or 0))
    for c in (skewed or numeric)[:3 if skewed else 2]:
        sk = cols[c].get("skewness") or 0
        add("distribution", f"Distribution of '{c}'",
            (f"Strongly skewed (skewness {sk:.1f}): a few very large or small values. Consider a log transform in Features."
             if abs(sk) > 1 else "How the values are spread, and whether there are outliers."), histogram(df, c), column=c)
    cats = [c for c, s in cols.items() if s["kind"] == "categorical" and c not in skip and 2 <= s["unique"] <= 30]
    for c in cats[:2]:
        add("categories", f"Values of '{c}'", f"{cols[c]['unique']} distinct values. Shows which are common and which are rare.", categories(df, c), column=c)
    if corr:
        best = next((p for p in corr["top_pairs"] if p["a"] in numeric and p["b"] in numeric), None)
        if best and abs(best["r"]) >= 0.3:
            add("scatter", f"'{best['a']}' vs '{best['b']}'", f"The most strongly related pair of features (r = {best['r']:.2f}).",
                scatter(df, best["a"], best["b"]), x=best["a"], y=best["b"])
    return specs
