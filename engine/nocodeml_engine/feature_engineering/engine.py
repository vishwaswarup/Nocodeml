"""Feature engineering (Section 2).

`FeatureEngineer` is a stateless row-wise transformer, so applying it before
the split cannot leak information. It runs as the first step of the exported
pipeline so raw input is accepted at prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nocodeml_engine.config import DateFeature, FeatureEngineeringConfig, NumericTransform


class FeatureEngineeringError(ValueError):
    pass


def numeric_feature_name(col: str, t: NumericTransform) -> str:
    return f"{col}__{t.value}"


def date_feature_name(col: str, f: DateFeature) -> str:
    return f"{col}__{f.value}"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeatureEngineeringConfig | None = None):
        self.config = config

    def _cfg(self) -> FeatureEngineeringConfig:
        return self.config or FeatureEngineeringConfig()

    def fit(self, X: pd.DataFrame, y=None):
        cfg = self._cfg()
        for rule in cfg.numeric_transforms:
            if rule.column not in X.columns:
                raise FeatureEngineeringError(f"Column '{rule.column}' not found.")
            if not pd.api.types.is_numeric_dtype(X[rule.column]):
                raise FeatureEngineeringError(
                    f"{rule.transform.value} needs a numeric column; '{rule.column}' is not.")
            lo = X[rule.column].min()
            if rule.transform is NumericTransform.LOG and lo <= -1:
                raise FeatureEngineeringError(
                    f"log (log1p) needs values > -1; '{rule.column}' has min {lo}.")
            if rule.transform is NumericTransform.SQRT and lo < 0:
                raise FeatureEngineeringError(
                    f"sqrt needs non-negative values; '{rule.column}' has min {lo}.")
        for rule in cfg.date_features:
            if rule.column not in X.columns:
                raise FeatureEngineeringError(f"Column '{rule.column}' not found.")
            if not rule.extract:
                raise FeatureEngineeringError(f"Choose at least one date part to extract from '{rule.column}'.")
            col = X[rule.column]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                # pandas would "parse" 1.0, 2.0 as nanoseconds since 1970; numbers are never dates here
                raise FeatureEngineeringError(f"'{rule.column}' does not look like a date column, so date parts can't be extracted.")
            if not pd.api.types.is_datetime64_any_dtype(col):
                vals = col.dropna()
                parsed = pd.to_datetime(vals, errors="coerce", format="mixed") if len(vals) else vals
                if len(vals) == 0 or parsed.notna().mean() < 0.95:
                    raise FeatureEngineeringError(f"'{rule.column}' does not look like a date column, so date parts can't be extracted.")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self._cfg()
        X = X.copy()
        for c in X.columns:  # booleans behave as 0/1 numerics downstream
            if pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(int)
        for rule in cfg.numeric_transforms:
            x = X[rule.column].astype(float)
            fn = {
                NumericTransform.LOG: np.log1p,
                NumericTransform.SQRT: np.sqrt,
                NumericTransform.ABS: np.abs,
                NumericTransform.SQUARE: np.square,
            }[rule.transform]
            X[numeric_feature_name(rule.column, rule.transform)] = fn(x)
        for rule in cfg.date_features:
            d = pd.to_datetime(X[rule.column], errors="coerce", format="mixed")
            for f in rule.extract:
                val = {
                    DateFeature.YEAR: d.dt.year,
                    DateFeature.MONTH: d.dt.month,
                    DateFeature.DAY: d.dt.day,
                    DateFeature.DAY_OF_WEEK: d.dt.dayofweek,
                    DateFeature.QUARTER: d.dt.quarter,
                    DateFeature.IS_WEEKEND: (d.dt.dayofweek >= 5).astype(float).where(d.notna()),
                    DateFeature.HOUR: d.dt.hour,
                }[f]
                X[date_feature_name(rule.column, f)] = val.astype(float)
            X = X.drop(columns=[rule.column])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.transform(pd.DataFrame(columns=self.feature_names_in_)).columns)
