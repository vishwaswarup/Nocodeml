"""Preprocessing engine (Section 1).

Leakage rule: nothing that learns statistics from data is applied before the
train/test split. Row-level operations with no learned state (duplicate
removal, dropping rows with a missing value) run in `prepare_frame` before the
split. Everything fitted (imputation, winsorization, encoding, scaling,
outlier bounds) lives in `build_preprocessor` / `OutlierRowFilter` and is
fitted on TRAIN rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from nocodeml_engine.config import (
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    PreprocessingConfig,
    ScalingStrategy,
    TaskType,
)


class PreprocessingError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


# ---------------------------------------------------------------------------
# Row-level preparation (stateless, runs before the split)
# ---------------------------------------------------------------------------


@dataclass
class PreparationLog:
    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, before: int, after: int, detail: str = "") -> None:
        self.steps.append({"step": step, "rows_before": before, "rows_after": after,
                           "rows_removed": before - after, "detail": detail})


def prepare_frame(df: pd.DataFrame, cfg: PreprocessingConfig, target: str
                  ) -> tuple[pd.DataFrame, PreparationLog]:
    log = PreparationLog()
    n = len(df)
    df = df.dropna(subset=[target])
    log.add("drop_missing_target", n, len(df), "rows without a target cannot be used")

    if cfg.drop_duplicates:
        n = len(df)
        df = df.drop_duplicates()
        log.add("drop_duplicates", n, len(df))

    for rule in cfg.missing_values:
        if rule.strategy is MissingValueStrategy.DROP_ROWS and rule.column in df.columns:
            n = len(df)
            df = df.dropna(subset=[rule.column])
            log.add("drop_rows_missing", n, len(df), rule.column)
    return df.reset_index(drop=True), log


# ---------------------------------------------------------------------------
# Fitted components
# ---------------------------------------------------------------------------


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its relative frequency in the training data."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return pd.DataFrame({c: X[c].map(self.maps_[c]).fillna(0.0).astype(float)
                             for c in self.feature_names_in_}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values to per-column IQR fences learned on the training data."""

    def __init__(self, k: dict[str, float] | None = None):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.bounds_ = {}
        for c in X.columns:
            q1, q3 = X[c].quantile([0.25, 0.75])
            k = (self.k or {}).get(c, 1.5)
            self.bounds_[c] = (q1 - k * (q3 - q1), q3 + k * (q3 - q1))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class OutlierRowFilter:
    """Removes outlier rows from TRAINING data only; bounds are learned on train.

    Test rows are never filtered: dropping evaluation rows would flatter the
    metrics, and at prediction time every input row needs an answer.
    """

    def __init__(self, cfg: PreprocessingConfig, random_state: int = 42):
        self.rules = [r for r in cfg.outliers
                      if r.strategy in (OutlierStrategy.IQR, OutlierStrategy.Z_SCORE,
                                        OutlierStrategy.ISOLATION_FOREST)]
        self.random_state = random_state
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.iforest_: IsolationForest | None = None
        self.iforest_cols_: list[str] = []

    @property
    def active(self) -> bool:
        return bool(self.rules)

    def fit_filter(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, int]:
        if not self.active:
            return X, y, 0
        keep = pd.Series(True, index=X.index)
        for r in self.rules:
            if r.strategy is OutlierStrategy.ISOLATION_FOREST:
                self.iforest_cols_.append(r.column)
                continue
            x = X[r.column]
            if r.strategy is OutlierStrategy.IQR:
                q1, q3 = x.quantile([0.25, 0.75])
                lo, hi = q1 - r.threshold * (q3 - q1), q3 + r.threshold * (q3 - q1)
            else:
                mu, sd = x.mean(), x.std(ddof=0)
                lo, hi = mu - r.threshold * sd, mu + r.threshold * sd
            self.bounds_[r.column] = (lo, hi)
            keep &= ~((x < lo) | (x > hi))  # NaN compares False -> kept
        if self.iforest_cols_:
            sub = X[self.iforest_cols_]
            complete = sub.notna().all(axis=1)
            if complete.sum() > 10:
                self.iforest_ = IsolationForest(random_state=self.random_state).fit(sub[complete])
                pred = pd.Series(self.iforest_.predict(sub[complete]), index=sub.index[complete])
                keep.loc[pred.index[pred == -1]] = False
        removed = int((~keep).sum())
        return X[keep], y[keep], removed


# ---------------------------------------------------------------------------
# Validation + construction
# ---------------------------------------------------------------------------

_NUMERIC_IMPUTE = {MissingValueStrategy.MEAN, MissingValueStrategy.MEDIAN, MissingValueStrategy.KNN}


def _kinds(X: pd.DataFrame) -> dict[str, str]:
    out = {}
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            out[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(X[c]):
            out[c] = "numerical"
        else:
            out[c] = "categorical"
    return out


def dropped_columns(cfg: PreprocessingConfig) -> set[str]:
    return set(cfg.drop_columns) | {r.column for r in cfg.missing_values
                                    if r.strategy is MissingValueStrategy.DROP_COLUMN}


def validate_preprocessing(X: pd.DataFrame, cfg: PreprocessingConfig) -> list[str]:
    """Return human-readable problems that would make the pipeline invalid.

    `X` is the feature frame after feature engineering (excluding the target).
    """
    issues: list[str] = []
    kinds = _kinds(X)
    dropped = dropped_columns(cfg)

    def known(col: str, what: str) -> bool:
        if col not in X.columns:
            issues.append(f"{what}: column '{col}' does not exist.")
            return False
        return True

    for c in cfg.drop_columns:
        known(c, "drop_columns")
    seen: set[tuple[str, str]] = set()
    for r in cfg.missing_values:
        if (r.column, "mv") in seen:
            issues.append(f"missing_values: multiple rules for '{r.column}'.")
        seen.add((r.column, "mv"))
        if not known(r.column, "missing_values"):
            continue
        k = kinds[r.column]
        if r.strategy in _NUMERIC_IMPUTE and k != "numerical":
            issues.append(f"{r.strategy.value} imputation needs a numeric column; '{r.column}' is {k}.")
        if r.strategy is MissingValueStrategy.CONSTANT and r.constant_value is None:
            issues.append(f"constant imputation for '{r.column}' needs a constant_value.")
        if r.strategy is MissingValueStrategy.CONSTANT and k == "numerical" \
                and not isinstance(r.constant_value, (int, float)):
            issues.append(f"constant for numeric column '{r.column}' must be a number.")
    enc = {}
    for r in cfg.encoding:
        if not known(r.column, "encoding"):
            continue
        if kinds[r.column] != "categorical":
            issues.append(f"Encoding applies to categorical columns; '{r.column}' is {kinds[r.column]}.")
        if r.column in enc:
            issues.append(f"encoding: multiple rules for '{r.column}'.")
        enc[r.column] = r.strategy
    for r in cfg.outliers:
        if known(r.column, "outliers") and r.strategy is not OutlierStrategy.KEEP \
                and kinds[r.column] != "numerical":
            issues.append(f"Outlier handling needs a numeric column; '{r.column}' is {kinds[r.column]}.")
        if r.strategy is OutlierStrategy.Z_SCORE and r.threshold < 1:
            issues.append(f"z-score threshold for '{r.column}' should be at least 1 (typically 3).")
    for c in cfg.scale_columns:
        if known(c, "scale_columns") and kinds[c] != "numerical":
            issues.append(f"Scaling needs a numeric column; '{c}' is {kinds[c]}.")

    imputed = {r.column for r in cfg.missing_values
               if r.strategy not in (MissingValueStrategy.DROP_ROWS, MissingValueStrategy.DROP_COLUMN)}
    handled_rows = {r.column for r in cfg.missing_values if r.strategy is MissingValueStrategy.DROP_ROWS}
    for c in X.columns:
        if c in dropped:
            continue
        if kinds[c] == "datetime":
            issues.append(f"'{c}' is a datetime column; extract date features or drop it.")
        elif kinds[c] == "categorical" and c not in enc:
            issues.append(f"Categorical column '{c}' needs an encoding (or drop it).")
        if X[c].isna().any() and c not in imputed and c not in handled_rows:
            issues.append(f"'{c}' has missing values; choose an imputation strategy, drop rows, or drop the column.")
    return issues


_SCALERS = {ScalingStrategy.STANDARD: StandardScaler, ScalingStrategy.MIN_MAX: MinMaxScaler,
            ScalingStrategy.ROBUST: RobustScaler}


def build_preprocessor(X: pd.DataFrame, cfg: PreprocessingConfig, task: TaskType,
                       ) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline for feature frame `X`.

    Raises PreprocessingError if the config is not executable on this data.
    """
    issues = validate_preprocessing(X, cfg)
    if issues:
        raise PreprocessingError(issues)

    kinds = _kinds(X)
    dropped = dropped_columns(cfg)
    used = [c for c in X.columns if c not in dropped]
    steps: list[tuple[str, Any]] = []

    def stage(transformers):
        ct = ColumnTransformer(transformers, remainder="passthrough",
                               verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
        return ct

    # Hand each stage only surviving columns so `remainder` drops nothing we want.
    drop_stage = ColumnTransformer([("keep", "passthrough", used)], remainder="drop",
                                   verbose_feature_names_out=False)
    drop_stage.set_output(transform="pandas")
    steps.append(("select", drop_stage))

    # 1. imputation
    groups: dict[tuple, list[str]] = {}
    for r in cfg.missing_values:
        if r.column in dropped or r.strategy in (MissingValueStrategy.DROP_ROWS,):
            continue
        if r.strategy is MissingValueStrategy.KNN:
            groups.setdefault(("knn", r.n_neighbors), []).append(r.column)
        elif r.strategy is MissingValueStrategy.CONSTANT:
            groups.setdefault(("constant", r.constant_value), []).append(r.column)
        else:
            groups.setdefault((r.strategy.value, None), []).append(r.column)
    ts = []
    for i, ((strat, extra), cols) in enumerate(groups.items()):
        if strat == "knn":
            imp = KNNImputer(n_neighbors=extra)
        elif strat == "constant":
            imp = SimpleImputer(strategy="constant", fill_value=extra)
        else:
            imp = SimpleImputer(strategy="most_frequent" if strat == "mode" else strat)
        ts.append((f"impute_{i}_{strat}", imp, cols))
    if ts:
        steps.append(("impute", stage(ts)))

    # 2. winsorization (numeric, learned on train)
    wins = {r.column: r.threshold for r in cfg.outliers
            if r.strategy is OutlierStrategy.WINSORIZE and r.column in used}
    if wins:
        steps.append(("winsorize", stage([("winsorize", Winsorizer(k=wins), list(wins))])))

    # 3. encoding + scaling
    enc_groups: dict[EncodingStrategy, list[str]] = {}
    for r in cfg.encoding:
        if r.column in used:
            enc_groups.setdefault(r.strategy, []).append(r.column)
    ts = []
    for strat, cols in enc_groups.items():
        if strat is EncodingStrategy.ONE_HOT:
            e = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif strat in (EncodingStrategy.ORDINAL, EncodingStrategy.LABEL):
            e = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        elif strat is EncodingStrategy.FREQUENCY:
            e = FrequencyEncoder()
        else:
            e = TargetEncoder(target_type="continuous" if task is TaskType.REGRESSION else "auto",
                              random_state=0)
        ts.append((f"encode_{strat.value}", e, cols))
    encoded = {c for cols in enc_groups.values() for c in cols}
    if cfg.scaling is not ScalingStrategy.NONE:
        scale_cols = cfg.scale_columns or [c for c in used if kinds[c] == "numerical"
                                           and c not in encoded]
        scale_cols = [c for c in scale_cols if c in used]
        if scale_cols:
            ts.append(("scale", _SCALERS[cfg.scaling](), scale_cols))
    if ts:
        steps.append(("encode_scale", stage(ts)))

    return Pipeline(steps)
