"""The data bundle for a Colab run: per-fold CSV files (preprocessing learned from the training rows only) plus a manifest
describing the models to train. Built entirely by the server, from the same code that trains on the server."""

from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from nocodeml_engine.config import PipelineConfig, SplitMethod, TaskType
from nocodeml_engine.feature_engineering import FeatureEngineer
from nocodeml_engine.models import build_estimator, get_spec
from nocodeml_engine.training.fitted import front_parts
from nocodeml_engine.training.runner import RunSetup, config_hash, dataset_fingerprint, prepare_run
from nocodeml_engine.training.search import scoring_name

FORMAT = 1
SPLITS = ("train", "validation", "test")


def estimator_spec(model_key: str, task: TaskType) -> dict[str, Any]:
    """Which plain library class to build in Colab. Our own thin wrappers map back to the library class they extend."""
    cls = get_spec(model_key).estimators[task]
    wrapped = cls.__module__.startswith("nocodeml_engine")
    base = cls.__mro__[1] if wrapped else cls
    parts = base.__module__.split(".")
    module = ".".join(parts[:2]) if parts[0] == "sklearn" else parts[0]   # the public import path, e.g. sklearn.ensemble
    return {"module": module, "class": base.__name__, "any_labels": wrapped}


def _json_value(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def estimator_params(model_cfg, task: TaskType, n_rows: int, n_features: int, seed: int) -> dict[str, Any]:
    """The exact constructor settings the server would use for this model on a training set of this size."""
    est = build_estimator(model_cfg, task, n_rows, n_features, seed)
    # NaN settings (XGBoost's `missing`) are left out so the library applies its own default; JSON has no NaN.
    return {k: _json_value(v) for k, v in est.get_params(deep=False).items()
            if (v is None or isinstance(v, (bool, int, float, str, np.generic)))
            and not (isinstance(v, (float, np.floating)) and math.isnan(v))}


class FoldFront:
    """One fold's preprocessing, learned on its training rows only."""

    def __init__(self, setup: RunSetup, config: PipelineConfig, fold):
        X, y = setup.X, setup.y
        self.fold = fold
        self.Xtr, self.ytr = X.iloc[fold.train], y.iloc[fold.train]
        self.le, pre, self.X_raw, self.y_f, self.removed, self.n_features = front_parts(
            self.Xtr, self.ytr, config, setup.classes)
        self.pipeline = Pipeline([("fe", FeatureEngineer(config.feature_engineering)), ("preprocess", pre)])

    def fit(self) -> "FoldFront":
        self.pipeline.fit(self.X_raw, self.y_f)
        return self

    @property
    def n_fit(self) -> int:
        return len(self.X_raw)


def _search_spec(model_cfg, task: TaskType, front: FoldFront, config: PipelineConfig) -> dict[str, Any] | None:
    sr = model_cfg.search
    if sr is None:
        return None
    if config.split.method is SplitMethod.TIME_SERIES:
        kind = "timeseries"
    elif task is TaskType.CLASSIFICATION and front.y_f.value_counts().min() >= sr.cv_folds:
        kind = "stratified"
    else:
        kind = "kfold"
    n_cls = len(front.le.classes_) if front.le is not None else None
    return {"method": sr.method, "space": {k: list(v) for k, v in sr.space.items()}, "n_iter": sr.n_iter,
            "cv_folds": sr.cv_folds, "cv": kind, "scoring": scoring_name(task, n_cls)}


def _frame(front: FoldFront, X: pd.DataFrame, y: pd.Series, row_ids: np.ndarray, setup: RunSetup) -> pd.DataFrame:
    feats = front.pipeline.transform(X)
    feats.columns = [str(c) for c in feats.columns]
    target = front.le.transform(y) if front.le is not None else y.astype(float).to_numpy()
    out = feats.reset_index(drop=True)
    out.insert(0, "__row_id", row_ids)
    out["__target"] = target
    return out


def build_bundle(df: pd.DataFrame, config: PipelineConfig) -> tuple[bytes, dict[str, Any]]:
    """Returns (zip bytes, manifest). Raises the same configuration errors as training would."""
    setup = prepare_run(df, config)
    task, seed = config.dataset.task, config.split.random_state
    files: dict[str, bytes] = {}
    folds_meta: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None
    per_model_params: dict[str, list[dict]] = {m.model_key: [] for m in config.models}
    searches: dict[str, dict | None] = {}

    for i, fold in enumerate(setup.plan.folds):
        front = FoldFront(setup, config, fold).fit()
        parts = {"train": (front.Xtr, front.ytr, fold.train), "test": (setup.X.iloc[fold.test], setup.y.iloc[fold.test], fold.test)}
        if fold.validation is not None:
            parts["validation"] = (setup.X.iloc[fold.validation], setup.y.iloc[fold.validation], fold.validation)
        meta: dict[str, Any] = {"index": i, "n_train": len(fold.train), "n_fitted": front.n_fit}
        for split in SPLITS:
            if split not in parts:
                meta[split] = None
                continue
            X, y, ids = parts[split]
            fr = _frame(front, X, y, ids, setup)
            if split == "train":
                fr.insert(1, "__fit", X.index.isin(front.X_raw.index).astype(int))
            cols = [c for c in fr.columns if not c.startswith("__")]
            feature_columns = feature_columns or cols
            if cols != feature_columns:
                raise RuntimeError("Preprocessing produced different columns in different splits.")
            name = f"fold_{i}_{split}.csv"
            files[name] = fr.to_csv(index=False).encode()
            meta[split] = name
        folds_meta.append(meta)
        for m in config.models:
            per_model_params[m.model_key].append(estimator_params(m, task, front.n_fit, front.n_features, seed))
            if i == 0:
                searches[m.model_key] = _search_spec(m, task, front, config)

    manifest = {
        "format": FORMAT, "created": datetime.now(timezone.utc).isoformat(), "task": task.value,
        "target": config.dataset.target_column, "classes": [str(c) for c in setup.classes] if setup.classes else None,
        "n_classes": len(setup.classes) if setup.classes else None, "random_state": seed,
        "config_hash": config_hash(config), "dataset_fingerprint": dataset_fingerprint(df),
        "split": {**{k: v for k, v in setup.plan.summary().items() if k in ("mode", "method", "n_folds")}},
        "feature_columns": feature_columns, "folds": folds_meta,
        "models": [{"key": m.model_key, "name": get_spec(m.model_key).name, "estimator": estimator_spec(m.model_key, task),
                    "params_by_fold": per_model_params[m.model_key], "search": searches[m.model_key]} for m in config.models],
    }
    files["manifest.json"] = json.dumps(manifest, indent=1).encode()
    files["README.txt"] = (
        "NoCodeML data bundle for Google Colab.\n\n1. Open the NoCodeML notebook in Colab.\n2. Run it and upload this zip when asked.\n"
        "3. Download the nocodeml_results.json it produces and upload that file in NoCodeML.\n\n"
        "The CSV files are already preprocessed. Every rule (imputation, encoding, scaling, ...) was learned from the training "
        "rows only. __row_id identifies the row, __fit marks training rows used for fitting, __target is the answer.\n").encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
    return buf.getvalue(), manifest
