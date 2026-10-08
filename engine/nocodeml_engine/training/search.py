"""Hyperparameter search (grid or random) for one model, leakage-safe.

The search wraps the *whole* pipeline (feature engineering -> preprocessing -> model), so inside every inner
cross-validation fold the imputers, encoders and scalers are re-fitted on that fold's training part only. It is
given the training rows of the outer split and nothing else: validation/test rows never influence which settings
win. The winner is re-fitted on all of the training rows, exactly like a model trained without a search.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from nocodeml_engine.config import ModelConfig, PipelineConfig, SplitMethod, TaskType

TOP_CANDIDATES = 20   # how many tried settings are kept in the stored result


def scoring_name(task: TaskType, n_classes: int | None) -> str:
    """The sklearn scorer matching the experiment's primary metric (F1 for classification, R2 for regression)."""
    if task is TaskType.REGRESSION:
        return "r2"
    return "f1" if n_classes == 2 else "f1_weighted"


def _inner_cv(config: PipelineConfig, task: TaskType, y: pd.Series, folds: int):
    rs = config.split.random_state
    if config.split.method is SplitMethod.TIME_SERIES:
        return TimeSeriesSplit(n_splits=folds)          # rows are already in time order: never train on the future
    if task is TaskType.CLASSIFICATION and y.value_counts().min() >= folds:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs)
    return KFold(n_splits=folds, shuffle=True, random_state=rs)


class _DeadlineScorer:
    """Wraps a scorer so a search stops (raises) once the training time limit has passed."""

    def __init__(self, scorer, deadline: float | None):
        self.scorer, self.deadline = scorer, deadline

    def __call__(self, estimator, X, y):
        if self.deadline is not None and time.monotonic() > self.deadline:
            from nocodeml_engine.training.runner import TrainingTimeout
            raise TrainingTimeout("Training took longer than the time limit and was stopped. "
                                  "Try fewer models, fewer folds or a smaller search.")
        return self.scorer(estimator, X, y)


def run_search(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, model_cfg: ModelConfig, config: PipelineConfig,
               n_classes: int | None, deadline: float | None) -> tuple[Pipeline, dict[str, Any]]:
    """Search `model_cfg.search` on (X, y) and return (best pipeline refitted on all of X, summary)."""
    from sklearn.metrics import get_scorer

    sr = model_cfg.search
    assert sr is not None
    task = config.dataset.task
    scoring = scoring_name(task, n_classes)
    grid = {f"model__{k}": list(v) for k, v in sr.space.items()}
    cv = _inner_cv(config, task, y, sr.cv_folds)
    common: dict[str, Any] = dict(estimator=pipe, scoring=_DeadlineScorer(get_scorer(scoring), deadline), cv=cv,
                                  refit=True, n_jobs=1, error_score="raise", return_train_score=False)
    if sr.method == "grid":
        search = GridSearchCV(param_grid=grid, **common)
    else:
        search = RandomizedSearchCV(param_distributions=grid, n_iter=sr.n_iter,
                                    random_state=config.split.random_state, **common)
    t0 = time.perf_counter()
    search.fit(X, y)
    res = search.cv_results_
    order = np.argsort(res["rank_test_score"], kind="stable")
    candidates = []
    for i in order[:TOP_CANDIDATES]:
        candidates.append({
            "params": {k.removeprefix("model__"): _plain(v) for k, v in res["params"][i].items()},
            "mean_score": float(res["mean_test_score"][i]), "std_score": float(res["std_test_score"][i]),
            "rank": int(res["rank_test_score"][i])})
    best = {k.removeprefix("model__"): _plain(v) for k, v in search.best_params_.items()}
    edge = [name for name, vals in sr.space.items()           # best value sits at an end of a numeric range: widen it
            if len(vals) >= 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
            and best[name] in (min(vals), max(vals))]
    summary = {
        "method": sr.method, "cv_folds": sr.cv_folds, "scoring": scoring,
        "n_candidates": len(res["params"]), "searched": {k: [_plain(v) for v in vs] for k, vs in sr.space.items()},
        "best_params": best, "edge_params": edge,
        "best_score": float(search.best_score_), "candidates": candidates,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    return search.best_estimator_, summary


def _plain(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v
