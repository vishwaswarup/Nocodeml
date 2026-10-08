"""The exportable artifact: raw input -> FE -> preprocessing -> model -> prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from nocodeml_engine.config import ModelConfig, PipelineConfig, TaskType
from nocodeml_engine.feature_engineering import FeatureEngineer
from nocodeml_engine.models import build_estimator, get_spec
from nocodeml_engine.preprocessing import OutlierRowFilter, build_preprocessor


class FittedPipeline:
    """A fully fitted pipeline that accepts raw rows (target column excluded)."""

    def __init__(self, pipeline: Pipeline, task: TaskType, target: str, model_key: str,
                 input_columns: list[str], label_encoder: LabelEncoder | None,
                 n_rows_fitted: int, n_outlier_rows_removed: int, search: dict[str, Any] | None = None):
        self.pipeline = pipeline
        self.task = task
        self.target = target
        self.model_key = model_key
        self.input_columns = input_columns
        self.label_encoder = label_encoder
        self.n_rows_fitted = n_rows_fitted
        self.n_outlier_rows_removed = n_outlier_rows_removed
        self.search = search     # summary of the hyperparameter search that chose this model's settings, if any

    @property
    def model(self):
        return self.pipeline.named_steps["model"]

    @property
    def classes(self) -> list:
        return list(self.label_encoder.classes_) if self.label_encoder is not None else []

    def _check(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing columns: {missing}")
        return X[self.input_columns]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.pipeline.predict(self._check(X))
        return self.label_encoder.inverse_transform(pred.astype(int)) \
            if self.label_encoder is not None else pred

    def predict_encoded(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._check(X))

    def scores(self, X: pd.DataFrame) -> np.ndarray | None:
        """Positive-class score (binary) or class probabilities (multiclass).

        Returns None when the model offers no usable score (multiclass SVM).
        """
        if self.task is not TaskType.CLASSIFICATION:
            return None
        X = self._check(X)
        k = len(self.classes)
        if hasattr(self.pipeline, "predict_proba"):
            try:
                p = self.pipeline.predict_proba(X)
            except AttributeError:
                p = None
            if p is not None:
                full = np.zeros((len(p), k))
                full[:, np.asarray(self.model.classes_, dtype=int)] = p
                return full[:, 1] if k == 2 else full
        if k == 2 and hasattr(self.pipeline, "decision_function"):
            return np.asarray(self.pipeline.decision_function(X))
        return None


def front_parts(X: pd.DataFrame, y: pd.Series, config: PipelineConfig, classes: list | None = None):
    """Everything before the model, learned from TRAINING rows only: the label encoder, the rows kept after outlier
    removal (`X_raw`, `y_f`), and the unfitted preprocessor. Shared by `fit_pipeline` and the Colab bundle so both
    prepare data in exactly the same way."""
    task = config.dataset.task
    rs = config.split.random_state
    le = None
    if task is TaskType.CLASSIFICATION:
        le = LabelEncoder().fit(classes if classes is not None else y)
        y_enc = pd.Series(le.transform(y), index=y.index)
    else:
        y_enc = y.astype(float)
    fe = FeatureEngineer(config.feature_engineering).fit(X)
    X_fe = fe.transform(X)
    X_fe_f, y_f, removed = OutlierRowFilter(config.preprocessing, rs).fit_filter(X_fe, y_enc)
    X_raw = X.loc[X_fe_f.index]
    preprocessor = build_preprocessor(X_fe_f, config.preprocessing, task)
    return le, preprocessor, X_raw, y_f, removed, X_fe_f.shape[1]


def fit_pipeline(X: pd.DataFrame, y: pd.Series, config: PipelineConfig, model_cfg: ModelConfig,
                 classes: list | None = None, deadline: float | None = None, cancel=None) -> FittedPipeline:
    """Fit FE + preprocessing + model on (X, y). Call with TRAIN rows only."""
    task = config.dataset.task
    rs = config.split.random_state
    le, preprocessor, X_raw, y_f, removed, n_features = front_parts(X, y, config, classes)
    est = build_estimator(model_cfg, task, len(X_raw), n_features, rs)
    pipe = Pipeline([("fe", FeatureEngineer(config.feature_engineering)),
                     ("preprocess", preprocessor), ("model", est)])
    summary = None
    if model_cfg.search is not None:
        from nocodeml_engine.training.search import run_search
        pipe, summary = run_search(pipe, X_raw, y_f, model_cfg, config,
                                   len(le.classes_) if le is not None else None, deadline, cancel)
    else:
        pipe.fit(X_raw, y_f)
    return FittedPipeline(pipe, task, config.dataset.target_column, model_cfg.model_key,
                          list(X.columns), le, len(X_raw), removed, summary)


def resolve_params(model_key: str, params: dict[str, Any]) -> dict[str, Any]:
    """The user-meaningful subset of an estimator's constructor settings, JSON-safe."""
    spec = get_spec(model_key)
    keep = {hp.name for hp in spec.hyperparameters} | {"C", "l1_ratio", "alpha", "solver", "max_depth", "random_state"}
    out = {}
    for k, v in params.items():
        if k in keep:
            out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
            if isinstance(v, float) and not np.isfinite(v):
                out[k] = "inf"
    return out


def resolved_params(fp: FittedPipeline) -> dict[str, Any]:
    return resolve_params(fp.model_key, fp.model.get_params())
