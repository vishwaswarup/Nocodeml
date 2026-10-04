"""Before/after preview of a preprocessing draft (Section 16 of the spec).

PREVIEW ONLY: to count output columns it fits the preprocessing on the *whole* dataset. Real
training always fits on training rows only (see training.runner), so these numbers describe the
shape of the data, never model quality.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nocodeml_engine.config import PipelineConfig, TaskType
from nocodeml_engine.feature_engineering import FeatureEngineer
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
    issues = check_config(df, config, require_models=False)
    if issues:
        return {"before": before, "after": None, "issues": issues, "steps": [],
                "split": split, "split_issues": split_issues, "model_issues": model_issues}
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
            "split_issues": split_issues, "model_issues": model_issues,
            "note": "Preview fits on the full dataset to count columns. Training fits on training rows only."}
