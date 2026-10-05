"""Deterministic feature-engineering recommendations (Section 2). Rules over the dataset profile only."""

from __future__ import annotations

from nocodeml_engine.profiling.profiler import DatasetProfile
from nocodeml_engine.recommendations.preprocessing import Recommendation

SKEW_LIMIT = 1.0
MIN_UNIQUE_FOR_TRANSFORM = 10


def date_parts_for(stats: dict) -> list[str]:
    """Which date parts are worth extracting, given a datetime column's stats."""
    span = stats.get("range_days") or 0
    parts = []
    if span >= 365:
        parts.append("year")
    if span >= 60:
        parts.append("month")
    parts.append("day_of_week")
    if stats.get("has_time"):
        parts.append("hour")
    return parts


def recommend_features(profile: DatasetProfile, target: str, drop_columns: list[str] | None = None) -> list[Recommendation]:
    dropped = set(drop_columns or [])
    recs: list[Recommendation] = []
    for c, s in profile.columns.items():
        if c == target or c in dropped or s.get("identifier_like"):
            continue
        if s["kind"] == "numerical":
            skew, lo = s.get("skewness") or 0.0, s.get("min")
            if skew > SKEW_LIMIT and lo is not None and lo > -1 and s["unique"] > MIN_UNIQUE_FOR_TRANSFORM:
                recs.append(Recommendation(
                    f"numeric_transform:{c}", "numeric_transform", c, f"Add a log version of '{c}'",
                    f"'{c}' is strongly right-skewed (skewness {skew:.1f}): a few very large values dominate. A log transform "
                    "compresses them so the model sees a more even spread. It helps linear models, SVM and KNN most; tree models "
                    f"don't need it. The original column is kept, and the new one is called '{c}__log'.", 0.7,
                    {"type": "numeric_transform", "column": c, "transform": "log"}))
        elif s["kind"] == "datetime":
            parts = date_parts_for(s)
            recs.append(Recommendation(
                f"date_features:{c}", "date_features", c, f"Extract {', '.join(p.replace('_', ' ') for p in parts)} from '{c}'",
                f"Models can't use a raw date. The parts capture what the date tells you: "
                + ("trend over the years, " if "year" in parts else "") + ("seasonality, " if "month" in parts else "")
                + "weekday patterns" + (", and time of day" if "hour" in parts else "") + ". The original column is replaced.",
                0.8, {"type": "date_features", "column": c, "extract": parts}))
    return recs
