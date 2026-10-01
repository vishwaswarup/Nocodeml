"""Training engine: executes a PipelineConfig end to end and records the experiment."""

from __future__ import annotations

import hashlib
import platform
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn

from nocodeml_engine.config import PipelineConfig, TaskType
from nocodeml_engine.evaluation import evaluate, primary_metric
from nocodeml_engine.feature_engineering import FeatureEngineer
from nocodeml_engine.models import MAX_MODELS, ModelConfigError, get_spec, validate_model_config
from nocodeml_engine.preprocessing import PreprocessingError, prepare_frame, validate_preprocessing
from nocodeml_engine.preprocessing.engine import dropped_columns
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.quality import run_quality_checks
from nocodeml_engine.results import ExperimentResult, ModelResult
from nocodeml_engine.splitting import SplitPlan, make_split_plan
from nocodeml_engine.training.fitted import FittedPipeline, fit_pipeline, resolved_params


class ConfigurationError(ValueError):
    pass


@dataclass
class ExperimentRun:
    result: ExperimentResult
    pipelines: dict[str, FittedPipeline]  # model_key -> pipeline fitted for export


def dataset_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def config_hash(config: PipelineConfig) -> str:
    """Hash of the configuration *content*; the version number is deliberately excluded."""
    return hashlib.sha256(config.model_dump_json(exclude={"version"}).encode()).hexdigest()[:16]


def environment_info() -> dict[str, str]:
    import joblib
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__}


def _validate(df: pd.DataFrame, config: PipelineConfig) -> None:
    target = config.dataset.target_column
    if target not in df.columns:
        raise ConfigurationError(f"Target column '{target}' not found in dataset.")
    if not config.models:
        raise ConfigurationError("Select at least one model.")
    if len(config.models) > MAX_MODELS:
        raise ConfigurationError(f"At most {MAX_MODELS} models can be trained at once.")
    keys = [m.model_key for m in config.models]
    if len(set(keys)) != len(keys):
        raise ConfigurationError("Each model can only be selected once.")
    for m in config.models:
        try:
            validate_model_config(m, config.dataset.task)
        except ModelConfigError as e:
            raise ConfigurationError(str(e)) from e
    if target in config.preprocessing.drop_columns:
        raise ConfigurationError("The target column cannot be dropped.")


def _eval(fp: FittedPipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    task = fp.task
    pred = fp.predict_encoded(X)
    if task is TaskType.CLASSIFICATION:
        y_true = fp.label_encoder.transform(y)
        return evaluate(task, y_true, pred, fp.scores(X), fp.classes)
    return evaluate(task, y.astype(float).to_numpy(), pred)


def _baseline_pred(task: TaskType, y_train: pd.Series, n: int, classes: list | None):
    if task is TaskType.CLASSIFICATION:
        majority = y_train.value_counts().idxmax()
        return np.full(n, classes.index(majority))
    return np.full(n, float(y_train.astype(float).mean()))


def _baseline_metrics(task, y_true_raw: pd.Series, preds: np.ndarray, classes):
    if task is TaskType.CLASSIFICATION:
        yt = np.array([classes.index(v) for v in y_true_raw])
        m = evaluate(task, yt, preds, None, classes)
        return {"accuracy": m["accuracy"], "f1": m["f1"]}
    m = evaluate(task, y_true_raw.astype(float).to_numpy(), preds)
    return {"r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"]}


def _train_model(model_cfg, config, X, y, plan: SplitPlan, classes):
    task = config.dataset.task
    pm = primary_metric(task)
    spec = get_spec(model_cfg.model_key)
    t0 = time.perf_counter()
    fold_rows: list[tuple[int, int]] = []
    metrics: dict[str, dict] = {}
    fold_scores: list[float] = []
    baseline: dict = {}
    outliers_removed = 0

    if plan.mode == "holdout":
        f = plan.folds[0]
        Xtr, ytr = X.iloc[f.train], y.iloc[f.train]
        fp = fit_pipeline(Xtr, ytr, config, model_cfg, classes)
        fold_rows.append((fp.n_rows_fitted + fp.n_outlier_rows_removed, len(f.train)))
        metrics["train"] = _eval(fp, Xtr, ytr)
        if f.validation is not None:
            metrics["validation"] = _eval(fp, X.iloc[f.validation], y.iloc[f.validation])
        metrics["test"] = _eval(fp, X.iloc[f.test], y.iloc[f.test])
        yte = y.iloc[f.test]
        baseline = _baseline_metrics(task, yte, _baseline_pred(task, ytr, len(yte), classes), classes)
        outliers_removed = fp.n_outlier_rows_removed
        final = fp
    else:
        oof_idx, oof_true, oof_pred, oof_score, base_pred, train_primary = [], [], [], [], [], []
        for f in plan.folds:
            Xtr, ytr = X.iloc[f.train], y.iloc[f.train]
            Xte, yte = X.iloc[f.test], y.iloc[f.test]
            fp = fit_pipeline(Xtr, ytr, config, model_cfg, classes)
            fold_rows.append((fp.n_rows_fitted + fp.n_outlier_rows_removed, len(f.train)))
            m = _eval(fp, Xte, yte)
            fold_scores.append(m[pm])
            train_primary.append(_eval(fp, Xtr, ytr)[pm])
            outliers_removed += fp.n_outlier_rows_removed
            oof_idx.append(f.test)
            oof_pred.append(fp.predict_encoded(Xte))
            sc = fp.scores(Xte)
            oof_score.append(sc)
            base_pred.append(_baseline_pred(task, ytr, len(yte), classes))
        ix = np.concatenate(oof_idx)
        y_oof = y.iloc[ix]
        pred = np.concatenate(oof_pred)
        score = None if any(s is None for s in oof_score) else np.concatenate(oof_score)
        if task is TaskType.CLASSIFICATION:
            yt = np.array([classes.index(v) for v in y_oof])
            metrics["cv"] = evaluate(task, yt, pred, score, classes)
        else:
            metrics["cv"] = evaluate(task, y_oof.astype(float).to_numpy(), pred)
        metrics["cv"][f"{pm}_fold_mean"] = float(np.mean(fold_scores))
        metrics["cv"][f"{pm}_fold_std"] = float(np.std(fold_scores))
        metrics["train"] = {pm: float(np.mean(train_primary))}
        baseline = _baseline_metrics(task, y_oof, np.concatenate(base_pred), classes)
        final = fit_pipeline(X, y, config, model_cfg, classes)  # exported model sees all data

    result = ModelResult(
        model_key=model_cfg.model_key, name=spec.name, hyperparameters=resolved_params(final),
        metrics=metrics, fold_primary_scores=[float(s) for s in fold_scores], primary_metric=pm,
        baseline=baseline, n_rows_fitted=final.n_rows_fitted,
        n_outlier_rows_removed=outliers_removed, fit_seconds=round(time.perf_counter() - t0, 3))
    return result, final, fold_rows


def run_experiment(df: pd.DataFrame, config: PipelineConfig,
                   experiment_id: str | None = None) -> ExperimentRun:
    started = datetime.now(timezone.utc)
    _validate(df, config)
    target = config.dataset.target_column
    task = config.dataset.task

    profile = profile_dataset(df, target)
    prepared, plog = prepare_frame(df, config.preprocessing, target)
    if config.split.time_column:
        if config.split.time_column not in prepared.columns:
            raise ConfigurationError(f"time_column '{config.split.time_column}' not found.")
        prepared = prepared.sort_values(config.split.time_column, kind="stable").reset_index(drop=True)
    y = prepared[target]
    X = prepared.drop(columns=[target])
    if len(X) == 0:
        raise ConfigurationError("No rows left after preprocessing.")
    if task is TaskType.CLASSIFICATION:
        if y.nunique() < 2:
            raise ConfigurationError("Classification needs at least two classes in the target.")
        classes = sorted(y.unique().tolist(), key=str) if y.dtype == object else sorted(y.unique().tolist())
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ConfigurationError("Regression needs a numeric target.")
        classes = None

    issues = validate_preprocessing(
        FeatureEngineer(config.feature_engineering).fit(X).transform(X), config.preprocessing)
    if issues:
        raise PreprocessingError(issues)

    plan = make_split_plan(X, y, config.split, task)

    results, pipelines, fitted_rows = [], {}, {}
    for m in config.models:
        r, fp, rows = _train_model(m, config, X, y, plan, classes)
        results.append(r)
        pipelines[m.model_key] = fp
        fitted_rows[m.model_key] = rows

    quality = run_quality_checks(
        config=config, profile=profile, plan=plan, X=X, y=y, model_results=results,
        fitted_rows=fitted_rows, dropped_features=dropped_columns(config.preprocessing))

    result = ExperimentResult(
        experiment_id=experiment_id or uuid.uuid4().hex[:12],
        pipeline_id=config.pipeline_id, pipeline_version=config.version,
        dataset_id=config.dataset.dataset_id, dataset_version=config.dataset.version,
        dataset_fingerprint=dataset_fingerprint(df), config_hash=config_hash(config),
        task=task.value, random_state=config.split.random_state,
        started_at=started.isoformat(), finished_at=datetime.now(timezone.utc).isoformat(),
        environment=environment_info(), split={**plan.summary(), "notes": plan.notes},
        preparation_log=plog.steps, models=results, quality=quality)
    return ExperimentRun(result=result, pipelines=pipelines)
