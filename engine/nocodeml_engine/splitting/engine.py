"""Split engine (Section 3). Produces index plans; never touches feature values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from nocodeml_engine.config import SplitConfig, SplitMethod, TaskType


class SplitError(ValueError):
    pass


@dataclass
class Fold:
    train: np.ndarray
    test: np.ndarray
    validation: np.ndarray | None = None


@dataclass
class SplitPlan:
    mode: str  # "holdout" | "cv"
    folds: list[Fold]
    method: SplitMethod
    stratified: bool
    chronological: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        f = self.folds[0]
        return {"mode": self.mode, "method": self.method.value, "stratified": self.stratified,
                "chronological": self.chronological, "n_folds": len(self.folds),
                "n_train": int(len(f.train)), "n_test": int(len(f.test)),
                "n_validation": int(len(f.validation)) if f.validation is not None else 0}


def validate_split(cfg: SplitConfig, y: pd.Series, task: TaskType) -> list[str]:
    issues: list[str] = []
    n = len(y)
    if not 0 < cfg.test_size < 1:
        issues.append("test_size must be between 0 and 1.")
    if cfg.method is SplitMethod.TRAIN_VAL_TEST:
        if not 0 < cfg.validation_size < 1 or cfg.validation_size + cfg.test_size >= 1:
            issues.append("validation_size must be > 0 and validation_size + test_size < 1.")
    if cfg.method in (SplitMethod.K_FOLD, SplitMethod.STRATIFIED_K_FOLD, SplitMethod.TIME_SERIES):
        if cfg.n_splits < 2 or cfg.n_splits > n // 2:
            issues.append(f"n_splits must be between 2 and {n // 2} for {n} rows.")
    wants_strat = cfg.stratify or cfg.method is SplitMethod.STRATIFIED_K_FOLD
    if wants_strat:
        if task is not TaskType.CLASSIFICATION:
            issues.append("Stratification only applies to classification targets.")
        else:
            smallest = int(y.value_counts().min())
            need = cfg.n_splits if cfg.method is SplitMethod.STRATIFIED_K_FOLD else 2
            if smallest < need:
                issues.append(f"Smallest class has {smallest} rows; stratification needs at least {need}.")
    if cfg.method is SplitMethod.TIME_SERIES and not cfg.time_column:
        issues.append("time_series splitting requires split.time_column.")
    return issues


def make_split_plan(X: pd.DataFrame, y: pd.Series, cfg: SplitConfig, task: TaskType) -> SplitPlan:
    """Build a split plan over positional indices of X/y (already ordered)."""
    issues = validate_split(cfg, y, task)
    if issues:
        raise SplitError("; ".join(issues))
    n = len(y)
    idx = np.arange(n)
    rs = cfg.random_state
    chrono = bool(cfg.time_column)
    notes: list[str] = []
    strat = cfg.stratify or cfg.method is SplitMethod.STRATIFIED_K_FOLD
    if cfg.stratify and chrono and cfg.method in (SplitMethod.TRAIN_TEST, SplitMethod.TRAIN_VAL_TEST):
        notes.append("stratify ignored: chronological splits keep row order.")
        strat = False

    if cfg.method in (SplitMethod.TRAIN_TEST, SplitMethod.TRAIN_VAL_TEST):
        def cut(ix: np.ndarray, frac: float, stratify_on):
            if chrono:
                k = int(round(len(ix) * (1 - frac)))
                return ix[:k], ix[k:]
            a, b = train_test_split(ix, test_size=frac, random_state=rs, shuffle=True,
                                    stratify=stratify_on)
            return a, b

        rest, test = cut(idx, cfg.test_size, y.iloc[idx] if strat else None)
        validation = None
        if cfg.method is SplitMethod.TRAIN_VAL_TEST:
            frac = cfg.validation_size / (1 - cfg.test_size)
            train, validation = cut(rest, frac, y.iloc[rest] if strat else None)
        else:
            train = rest
        return SplitPlan("holdout", [Fold(train, test, validation)], cfg.method, strat, chrono, notes)

    if cfg.method is SplitMethod.TIME_SERIES:
        splitter = TimeSeriesSplit(n_splits=cfg.n_splits)
        chrono = True
        folds_iter: Iterator = splitter.split(idx)
    elif cfg.method is SplitMethod.STRATIFIED_K_FOLD:
        folds_iter = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=rs).split(idx, y)
    else:
        folds_iter = KFold(cfg.n_splits, shuffle=True, random_state=rs).split(idx)
    return SplitPlan("cv", [Fold(a, b) for a, b in folds_iter], cfg.method,
                     cfg.method is SplitMethod.STRATIFIED_K_FOLD, chrono, notes)
