"""Task-aware evaluation (Section 7). Returns plain, JSON-safe data (no images)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as m

from nocodeml_engine.config import TaskType

MAX_PLOT_POINTS = 2000


def primary_metric(task: TaskType) -> str:
    return "f1" if task is TaskType.CLASSIFICATION else "r2"


def _f(x: Any) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def classification_metrics(y_true, y_pred, y_score, classes: list) -> dict[str, Any]:
    """y_score: (n,) positive-class score for binary, (n, k) probabilities for multiclass, or None."""
    binary = len(classes) == 2
    avg = "binary" if binary else "weighted"
    out: dict[str, Any] = {
        "average": avg,
        "accuracy": _f(m.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _f(m.balanced_accuracy_score(y_true, y_pred)),
        "precision": _f(m.precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": _f(m.recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": _f(m.f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "roc_auc": None,
    }
    labels = list(range(len(classes)))
    out["confusion_matrix"] = {
        "labels": [str(c) for c in classes],
        "matrix": m.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if binary:
        cm = np.array(out["confusion_matrix"]["matrix"])
        tn, fp = cm[0]
        out["specificity"] = _f(tn / (tn + fp)) if (tn + fp) else None
    if y_score is not None and len(np.unique(y_true)) == len(classes):
        try:
            if binary:
                out["roc_auc"] = _f(m.roc_auc_score(y_true, y_score))
                fpr, tpr, _ = m.roc_curve(y_true, y_score)
                step = max(1, len(fpr) // 200)
                out["roc_curve"] = {"fpr": fpr[::step].round(5).tolist(),
                                    "tpr": tpr[::step].round(5).tolist()}
            else:
                out["roc_auc"] = _f(m.roc_auc_score(y_true, y_score, multi_class="ovr",
                                                    average="weighted", labels=labels))
        except ValueError:
            pass
    return out


def regression_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = m.mean_squared_error(y_true, y_pred)
    out: dict[str, Any] = {
        "mae": _f(m.mean_absolute_error(y_true, y_pred)),
        "mse": _f(mse),
        "rmse": _f(np.sqrt(mse)),
        "r2": _f(m.r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }
    nonzero = y_true != 0
    out["mape"] = _f(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) \
        if nonzero.any() else None
    rng = np.random.default_rng(0)
    ix = np.arange(len(y_true))
    if len(ix) > MAX_PLOT_POINTS:
        ix = np.sort(rng.choice(ix, MAX_PLOT_POINTS, replace=False))
    out["actual_vs_predicted"] = {"actual": y_true[ix].round(6).tolist(),
                                  "predicted": y_pred[ix].round(6).tolist()}
    out["residuals"] = (y_true[ix] - y_pred[ix]).round(6).tolist()
    return out


def evaluate(task: TaskType, y_true, y_pred, y_score=None, classes: list | None = None
             ) -> dict[str, Any]:
    if task is TaskType.CLASSIFICATION:
        return classification_metrics(y_true, y_pred, y_score, classes or [])
    return regression_metrics(y_true, y_pred)
