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
from nocodeml_engine.preprocessing.engine import build_preprocessor, prepare_frame
from nocodeml_engine.splitting import make_split_plan
from nocodeml_engine.training.runner import check_config


def _shape(df: pd.DataFrame) -> dict[str, Any]:
    cells = max(df.shape[0] * df.shape[1], 1)
    miss = int(df.isna().sum().sum())
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "missing_cells": miss,
            "missing_pct": round(100 * miss / cells, 2), "duplicate_rows": int(df.duplicated().sum())}


def preview_preprocessing(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    before = _shape(df)
    issues = check_config(df, config, require_models=False)
    if issues:
        return {"before": before, "after": None, "issues": issues, "steps": [], "split": None}
    target = config.dataset.target_column
    prepared, log = prepare_frame(df, config.preprocessing, target)
    X, y = prepared.drop(columns=[target]), prepared[target]
    Xf = FeatureEngineer(config.feature_engineering).fit(X).transform(X)
    y_num = pd.Series(pd.factorize(y)[0], index=y.index) if config.dataset.task is TaskType.CLASSIFICATION \
        else y.astype(float)
    out = build_preprocessor(Xf, config.preprocessing, config.dataset.task).fit_transform(Xf, y_num)
    plan_info = None
    ordered = prepared.sort_values(config.split.time_column, kind="stable").reset_index(drop=True) \
        if config.split.time_column and config.split.time_column in prepared.columns else prepared
    plan = make_split_plan(ordered.drop(columns=[target]), ordered[target], config.split, config.dataset.task)
    plan_info = {**plan.summary(), "notes": plan.notes}
    after = _shape(out)
    after["rows"] = int(len(prepared))
    after["duplicate_rows"] = int(prepared.duplicated().sum())
    return {"before": before, "after": after, "issues": [], "steps": log.steps, "split": plan_info,
            "note": "Preview fits on the full dataset to count columns. Training fits on training rows only."}
