"""Before/after preview of a preprocessing draft (Section 16 of the spec).

PREVIEW ONLY: to count output columns it fits the preprocessing on the *whole* dataset. Real
training always fits on training rows only (see training.runner), so these numbers describe the
shape of the data, never model quality.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nocodeml_engine.config import FeatureEngineeringConfig, PipelineConfig, TaskType
from nocodeml_engine.feature_engineering import FeatureEngineer
from nocodeml_engine.feature_engineering import FeatureEngineer, FeatureEngineeringError
from nocodeml_engine.models import MAX_MODELS, ModelConfigError, build_estimator
from nocodeml_engine.preprocessing.engine import build_preprocessor, prepare_frame
from nocodeml_engine.splitting import make_split_plan, validate_split
from nocodeml_engine.training.runner import check_config


def _shape(df: pd.DataFrame) -> dict[str, Any]:
    cells = max(df.shape[0] * df.shape[1], 1)
    miss = int(df.isna().sum().sum())
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "missing_cells": miss,
            "missing_pct": round(100 * miss / cells, 2), "duplicate_rows": int(df.duplicated().sum())}


def preview_split(df: pd.DataFrame, config: PipelineConfig) -> tuple[dict[str, Any] | None, list[str]]:
    """How the rows would be divided, judged on the split settings alone.

    Independent of whether preprocessing is finished: the split only needs the row count after
    row-level preparation, so split problems are never confused with preprocessing problems.
    """
    target = config.dataset.target_column
    if target not in df.columns:
        return None, [f"Target column '{target}' not found in dataset."]
    prepared, _ = prepare_frame(df, config.preprocessing, target)
    if len(prepared) == 0:
        return None, ["No rows left after preprocessing."]
    tc = config.split.time_column
    if tc:
        if tc not in prepared.columns:
            return None, [f"time_column '{tc}' not found."]
        prepared = prepared.sort_values(tc, kind="stable").reset_index(drop=True)
    y = prepared[target]
    issues = validate_split(config.split, y, config.dataset.task)
    if issues:
        return None, issues
    plan = make_split_plan(prepared.drop(columns=[target]), y, config.split, config.dataset.task)
    return {**plan.summary(), "notes": plan.notes}, []


def _safe(v: Any) -> Any:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, float):
        return round(v, 4)
    return v.item() if hasattr(v, "item") else v


def preview_features(df: pd.DataFrame, config: PipelineConfig) -> tuple[list[dict[str, Any]], list[str]]:
    """Which columns the feature-engineering rules would add, with sample values.

    Every rule is checked on its own so all problems are reported together, not one at a time.
    Independent of preprocessing validity (it only needs the row-prepared data).
    """
    target = config.dataset.target_column
    if target not in df.columns:
        return [], [f"Target column '{target}' not found in dataset."]
    prepared, _ = prepare_frame(df, config.preprocessing, target)
    X = prepared.drop(columns=[target])
    fe = config.feature_engineering
    issues: list[str] = []
    for rule in fe.numeric_transforms:
        try:
            FeatureEngineer(FeatureEngineeringConfig(numeric_transforms=[rule])).fit(X)
        except FeatureEngineeringError as e:
            issues.append(str(e))
    for rule in fe.date_features:
        try:
            FeatureEngineer(FeatureEngineeringConfig(date_features=[rule])).fit(X)
        except FeatureEngineeringError as e:
            issues.append(str(e))
    if issues:
        return [], issues
    sample = FeatureEngineer(fe).fit(X).transform(X.head(500))
    origin = {f"{r.column}__{r.transform.value}": (r.column, r.transform.value) for r in fe.numeric_transforms}
    origin.update({f"{r.column}__{p.value}": (r.column, p.value) for r in fe.date_features for p in r.extract})
    new = [c for c in sample.columns if c not in X.columns]
    return [{"name": c, "source": origin.get(c, (c, ""))[0], "op": origin.get(c, (c, ""))[1],
             "sample": [_safe(v) for v in sample[c].head(5)]} for c in new], []


def preview_models(config: PipelineConfig) -> dict[str, list[str]]:
    """Problems with each selected model's settings (key -> messages). Instantiates the estimator,
    so incompatible combinations are caught here instead of at training time."""
    out: dict[str, list[str]] = {}
    if len(config.models) > MAX_MODELS:
        out["_all"] = [f"At most {MAX_MODELS} models can be trained at once."]
    for m in config.models:
        try:
            est = build_estimator(m, config.dataset.task, 1000, 10, config.split.random_state)
            # sklearn only rejects bad values (e.g. C=-1) at fit time unless asked to check now.
            # _validate_params is sklearn's own pre-fit check; tests pin that it still exists.
            # (XGBoost isn't a scikit-learn class and has no such hook; its values are range-checked by the registry.)
            if hasattr(type(est), "_parameter_constraints"):
                est._validate_params()
        except ModelConfigError as e:
            out.setdefault(m.model_key, []).append(str(e))
        except Exception as e:  # noqa: BLE001 - sklearn rejecting a parameter combination
            out.setdefault(m.model_key, []).append(f"These settings can't be used together: {e}")
    return out


def preview_preprocessing(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    before = _shape(df)
    split, split_issues = preview_split(df, config)
    model_issues = preview_models(config)
    features, feature_issues = preview_features(df, config)
    issues = check_config(df, config, require_models=False)
    if issues:
        return {"before": before, "after": None, "issues": issues, "steps": [],
                "split": split, "split_issues": split_issues, "model_issues": model_issues,
                "features": features, "feature_issues": feature_issues}
    target = config.dataset.target_column
    prepared, log = prepare_frame(df, config.preprocessing, target)
    X, y = prepared.drop(columns=[target]), prepared[target]
    Xf = FeatureEngineer(config.feature_engineering).fit(X).transform(X)
    y_num = pd.Series(pd.factorize(y)[0], index=y.index) if config.dataset.task is TaskType.CLASSIFICATION \
        else y.astype(float)
    out = build_preprocessor(Xf, config.preprocessing, config.dataset.task).fit_transform(Xf, y_num)
    after = _shape(out)
    after["rows"] = int(len(prepared))
    after["duplicate_rows"] = int(prepared.duplicated().sum())
    return {"before": before, "after": after, "issues": [], "steps": log.steps, "split": split,
            "split_issues": split_issues, "model_issues": model_issues, "features": features, "feature_issues": feature_issues,
            "note": "Preview fits on the full dataset to count columns. Training fits on training rows only."}
