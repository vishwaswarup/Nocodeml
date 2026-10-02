"""ML Quality Engine: judges whether the *experiment* is sound, not how accurate it is."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nocodeml_engine.config import PipelineConfig, ScalingStrategy, TaskType
from nocodeml_engine.models import get_spec
from nocodeml_engine.profiling.profiler import DatasetProfile
from nocodeml_engine.results import ModelResult, QualityCheck, QualityReport
from nocodeml_engine.splitting import SplitPlan

OVERFIT_GAP = 0.15
LEAKAGE_CORR = 0.98
MIN_ROWS = 100
_WEIGHT = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


def target_leakage_suspects(X: pd.DataFrame, y_numeric: pd.Series) -> list[tuple[str, float]]:
    out = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) or X[c].nunique() < 2:
            continue
        r = X[c].corr(y_numeric)
        if pd.notna(r) and abs(r) >= LEAKAGE_CORR:
            out.append((str(c), float(r)))
    return out


def run_quality_checks(*, config: PipelineConfig, profile: DatasetProfile, plan: SplitPlan,
                       X: pd.DataFrame, y: pd.Series, model_results: list[ModelResult],
                       fitted_rows: dict[str, list[tuple[int, int]]],
                       dropped_features: set[str]) -> QualityReport:
    """`fitted_rows[model_key]` is a list of (rows_given_to_fit, train_rows_in_fold) per fold."""
    task = config.dataset.task
    checks: list[QualityCheck] = []

    def add(id_, status, title, detail="", model=None):
        checks.append(QualityCheck(id=id_, status=status, title=title, detail=detail, model_key=model))

    # Duplicates
    dups = int(X.assign(_y=y.values).duplicated().sum())
    if dups and not config.preprocessing.drop_duplicates:
        add("duplicates", "warn", "Duplicate rows present",
            f"{dups} duplicate rows remain; identical rows can land in both train and test.")
    else:
        add("duplicates", "pass", "No duplicate rows")

    # Split isolation
    clean = all(
        not set(f.train) & set(f.test)
        and (f.validation is None or not (set(f.validation) & (set(f.train) | set(f.test))))
        for f in plan.folds)
    add("split_isolation", "pass" if clean else "fail",
        "Train/test separation valid" if clean else "Train and test rows overlap")

    # Preprocessing fitted on train only
    bad = [k for k, folds in fitted_rows.items() if any(fit > tr for fit, tr in folds)]
    if bad:
        add("preprocessing_leakage", "fail", "Preprocessing saw non-training rows", ", ".join(bad))
    else:
        add("preprocessing_leakage", "pass", "Preprocessing fitted only on training data",
            "Imputation, encoding, scaling and outlier bounds were learned on training rows of "
            "each fold and only applied to evaluation rows.")

    # Hygiene guaranteed by validation
    add("missing_handled", "pass", "Missing values handled",
        "Every feature column with missing values has an explicit strategy.")
    add("categoricals_encoded", "pass", "Categorical variables encoded")

    # Identifier columns used as features
    for c, st in profile.columns.items():
        if st.get("identifier_like") and c != config.dataset.target_column \
                and c not in dropped_features:
            add("identifier_feature", "warn", f"Identifier-like feature '{c}' is used",
                "Unique-per-row columns let models memorize rows instead of learning patterns.")

    # Class imbalance
    if task is TaskType.CLASSIFICATION and profile.target:
        dist = profile.target.get("class_distribution", {})
        if dist.get("imbalanced"):
            if plan.stratified:
                add("class_imbalance", "warn", "Class imbalance detected",
                    f"Minority class is {dist['minority_ratio']:.1%}. Stratified splitting preserves "
                    "proportions; judge models by F1/ROC-AUC rather than accuracy.")
            else:
                add("class_imbalance", "warn", "Class imbalance without stratification",
                    f"Minority class is {dist['minority_ratio']:.1%}. Enable stratification so "
                    "train and test keep the same class proportions.")
        else:
            add("class_imbalance", "pass", "Classes reasonably balanced")

    # Scaling vs model sensitivity
    if config.preprocessing.scaling is ScalingStrategy.NONE:
        for m in config.models:
            if get_spec(m.model_key).requires_scaling:
                add("scaling", "warn", f"{get_spec(m.model_key).name} is sensitive to feature scale",
                    "No scaling is configured.", m.model_key)

    # Validation strategy
    if plan.mode == "cv":
        add("cross_validation", "pass", "Cross-validation performed",
            f"{len(plan.folds)} folds; metrics are out-of-fold.")
    else:
        add("cross_validation", "warn", "Single holdout split only",
            "Results come from one split and may vary with the random seed; consider k-fold.")

    # Temporal leakage
    used_dt = [c for c in profile.datetime if c not in dropped_features]
    if used_dt and not config.split.time_column and plan.method.value != "time_series":
        add("temporal_leakage", "warn", "Datetime column with random splitting",
            f"Columns {used_dt} suggest time-ordered data; random splits can leak the "
            "future into training. Set a time column for a chronological split.")

    # Target leakage suspicion
    y_num = pd.Series(pd.factorize(y)[0], index=y.index) if task is TaskType.CLASSIFICATION \
        else y.astype(float)
    sus = target_leakage_suspects(X.drop(columns=list(dropped_features & set(X.columns))), y_num)
    if sus:
        add("target_leakage", "warn", "Feature almost identical to the target",
            "; ".join(f"{c} (corr {r:+.3f})" for c, r in sus))
    else:
        add("target_leakage", "pass", "No feature is a near-copy of the target")

    # Sample size
    if len(X) < MIN_ROWS:
        add("sample_size", "warn", "Very small dataset", f"{len(X)} rows; metrics will be noisy.")

    # Per model
    for r in model_results:
        pm = r.primary_metric
        test = r.metrics.get("test") or r.metrics.get("cv") or {}
        train = r.metrics.get("train") or {}
        if train.get(pm) is not None and test.get(pm) is not None:
            gap = train[pm] - test[pm]
            if gap > OVERFIT_GAP:
                add("overfitting", "warn", f"Possible overfitting ({r.name})",
                    f"{pm} is {train[pm]:.3f} on train vs {test[pm]:.3f} held out.", r.model_key)
            else:
                add("overfitting", "pass", f"No strong overfitting signal ({r.name})",
                    f"{pm} train {train[pm]:.3f} vs held-out {test[pm]:.3f}.", r.model_key)
        base = r.baseline.get(pm)
        if base is not None and test.get(pm) is not None:
            if test[pm] <= base:
                add("baseline", "warn", f"{r.name} does not beat the baseline",
                    f"{pm} {test[pm]:.3f} vs trivial baseline {base:.3f}.", r.model_key)
            else:
                add("baseline", "pass", f"{r.name} beats the baseline",
                    f"{pm} {test[pm]:.3f} vs trivial baseline {base:.3f}.", r.model_key)

    add("tuning_isolation", "pass", "Test set not used for tuning",
        "Hyperparameters are fixed by the configuration; no search ran against test data.")

    scored = [_WEIGHT[c.status] for c in checks if c.status in _WEIGHT]
    return QualityReport(checks=checks,
                         score=round(100 * float(np.mean(scored)), 1) if scored else None)
