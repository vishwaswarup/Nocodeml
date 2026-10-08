"""Turn a Colab results file into a normal experiment. The file carries predictions only; every metric, curve, baseline and
quality check is computed here, by the same code that scores a model trained on the server."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from nocodeml_engine.colab.bundle import FoldFront, estimator_params
from nocodeml_engine.config import PipelineConfig, TaskType
from nocodeml_engine.models import get_spec
from nocodeml_engine.preprocessing.engine import dropped_columns
from nocodeml_engine.quality import run_quality_checks
from nocodeml_engine.quality.checks import _WEIGHT
from nocodeml_engine.results import ExperimentResult, ModelResult, QualityCheck
from nocodeml_engine.training.fitted import resolve_params
from nocodeml_engine.training.runner import (
    FoldPreds, assemble_metrics, config_hash, dataset_fingerprint, environment_info, prepare_run,
)
from nocodeml_engine.training.search import TOP_CANDIDATES


class ResultsError(ValueError):
    """The uploaded results file can't be used. The message says why, in words a researcher can act on."""

    def __init__(self, message: str, issues: list[str] | None = None):
        super().__init__(message)
        self.issues = issues or []


def _fail(msg: str):
    raise ResultsError(msg)


def _array(v: Any, n: int, what: str, integer: bool = False, width: int | None = None) -> np.ndarray:
    """A list of numbers of exactly the expected length, finite, of the expected shape."""
    if not isinstance(v, list) or len(v) != n:
        _fail(f"{what}: expected {n} values, got {len(v) if isinstance(v, list) else 'something that is not a list'}.")
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        _fail(f"{what}: contains values that are not numbers.")
    if not np.isfinite(a).all():
        _fail(f"{what}: contains NaN or infinite values.")
    if width is None and a.ndim != 1 or width is not None and a.shape != (n, width):
        _fail(f"{what}: has the wrong shape {a.shape}.")
    if integer and not np.array_equal(a, np.round(a)):
        _fail(f"{what}: class predictions must be whole numbers.")
    return a.astype(int) if integer else a


def _search_summary(model_cfg, raw: Any, what: str) -> dict[str, Any]:
    """The tuning summary the notebook reports. It can't be recomputed here, so it is bounds-checked and labelled."""
    sr = model_cfg.search
    if sr is None:
        _fail(f"{what}: reported a hyperparameter search that was not configured.")
    if not isinstance(raw, dict) or not isinstance(raw.get("best_params"), dict) or not isinstance(raw.get("candidates"), list):
        _fail(f"{what}: the search summary is malformed.")
    best = raw["best_params"]
    for k, v in best.items():
        if k not in sr.space or v not in sr.space[k]:
            _fail(f"{what}: the chosen setting {k}={v!r} is not one of the values that were meant to be searched.")
    cands = []
    for c in raw["candidates"][:TOP_CANDIDATES]:
        try:
            cands.append({"params": {k: c["params"][k] for k in sr.space}, "mean_score": float(c["mean_score"]),
                          "std_score": float(c["std_score"]), "rank": int(c["rank"])})
        except (KeyError, TypeError, ValueError):
            _fail(f"{what}: the search summary is malformed.")
    if not all(math.isfinite(c["mean_score"]) and math.isfinite(c["std_score"]) for c in cands):
        _fail(f"{what}: the search summary contains invalid numbers.")
    best_score = float(raw.get("best_score", float("nan")))
    if not math.isfinite(best_score):
        _fail(f"{what}: the search summary contains invalid numbers.")
    edge = [n for n, vals in sr.space.items()
            if len(vals) >= 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
            and best.get(n) in (min(vals), max(vals))]
    from nocodeml_engine.training.search import scoring_name
    return {"method": sr.method, "cv_folds": sr.cv_folds, "scoring": raw.get("scoring"), "n_candidates": int(raw.get("n_candidates", len(cands))),
            "seconds": float(raw.get("seconds", 0.0)), "searched": {k: list(v) for k, v in sr.space.items()}, "best_params": best,
            "edge_params": edge, "best_score": best_score, "candidates": cands, "reported_by": "colab",
            "prep_scope": "Tuned on data that was already preprocessed using all training rows."}


def import_results(df: pd.DataFrame, config: PipelineConfig, results: Any) -> ExperimentResult:
    if not isinstance(results, dict) or results.get("format") != 1:
        _fail("This isn't a NoCodeML results file (or it is from a different version).")
    if results.get("config_hash") != config_hash(config):
        _fail("These results were made for a different version of your pipeline. Your settings changed since you "
              "downloaded the bundle: download a new bundle and run it again.")
    if results.get("dataset_fingerprint") != dataset_fingerprint(df):
        _fail("These results were made for a different dataset than the one in this project.")
    started = datetime.now(timezone.utc)
    setup = prepare_run(df, config)
    task, plan, y, X, classes = config.dataset.task, setup.plan, setup.y, setup.X, setup.classes
    cls_task = task is TaskType.CLASSIFICATION
    k = len(classes) if classes else None
    seed = config.split.random_state

    models = results.get("models")
    expected = [m.model_key for m in config.models]
    if not isinstance(models, dict) or sorted(models) != sorted(expected):
        _fail(f"The results must contain exactly these models: {', '.join(expected)}.")
    row_ids = results.get("row_ids")
    if not isinstance(row_ids, dict):
        _fail("The results file is missing its row list.")

    fronts = [FoldFront(setup, config, f) for f in plan.folds]   # preprocessing is re-learned here, to count fitted rows
    for i, f in enumerate(plan.folds):
        want = {"train": f.train, "test": f.test, **({"validation": f.validation} if f.validation is not None else {})}
        got = row_ids.get(str(i))
        if not isinstance(got, dict) or sorted(got) != sorted(want):
            _fail(f"Fold {i}: the results don't cover the expected splits.")
        for split, ids in want.items():
            if got[split] != [int(v) for v in ids]:
                _fail(f"Fold {i}: the rows in the '{split}' predictions don't match the rows NoCodeML sent. "
                      "Use the bundle you downloaded for this pipeline, unchanged.")

    model_results: list[ModelResult] = []
    fitted_rows: dict[str, list[tuple[int, int]]] = {}
    cv_mode = plan.mode == "cv"
    for m in config.models:
        spec = get_spec(m.model_key)
        raw = models[m.model_key]
        what = spec.name
        if not isinstance(raw, dict) or not isinstance(raw.get("folds"), list) or len(raw["folds"]) != len(plan.folds):
            _fail(f"{what}: expected results for {len(plan.folds)} fold(s).")
        fold_preds: list[FoldPreds] = []
        search_summary = None
        for i, (f, rf) in enumerate(zip(plan.folds, raw["folds"])):
            if rf.get("index") != i or not isinstance(rf.get("splits"), dict):
                _fail(f"{what}: fold {i} is malformed.")
            sp: FoldPreds = {}
            for split, ids in (("train", f.train), ("validation", f.validation), ("test", f.test)):
                if ids is None:
                    continue
                d = rf["splits"].get(split)
                if not isinstance(d, dict):
                    _fail(f"{what}: fold {i} has no '{split}' predictions.")
                label = f"{what}, fold {i}, {split}"
                pred = _array(d.get("pred"), len(ids), f"{label} predictions", integer=cls_task)
                if cls_task and ((pred < 0) | (pred >= k)).any():
                    _fail(f"{label}: a predicted class is outside the {k} known classes.")
                score = None
                if cls_task and d.get("score") is not None:
                    score = _array(d["score"], len(ids), f"{label} scores", width=None if k == 2 else k)
                sp[split] = (np.asarray(ids), pred, score)
            fold_preds.append(sp)
            if rf.get("search") is not None and i == 0:
                search_summary = _search_summary(m, rf["search"], what)
        try:
            seconds = float(raw.get("seconds", 0.0))
        except (TypeError, ValueError):
            _fail(f"{what}: the training time is not a number.")
        if not 0 <= seconds < 1e7:
            _fail(f"{what}: the training time is not valid.")

        le = fronts[0].le
        metrics, fold_scores, baseline = assemble_metrics(task, classes, le, plan, y, fold_preds)
        fitted_rows[m.model_key] = [(len(fr.Xtr), len(fr.Xtr)) for fr in fronts]
        params = estimator_params(m, task, fronts[0].n_fit, fronts[0].n_features, seed)
        if search_summary:
            params.update(search_summary["best_params"])
        model_results.append(ModelResult(
            model_key=m.model_key, name=spec.name, hyperparameters=resolve_params(m.model_key, params), metrics=metrics,
            fold_primary_scores=[float(s) for s in fold_scores], primary_metric="f1" if cls_task else "r2", baseline=baseline,
            n_rows_fitted=fronts[0].n_fit, n_outlier_rows_removed=int(sum(fr.removed for fr in fronts)),
            fit_seconds=round(seconds, 3), search=search_summary))

    quality = run_quality_checks(config=config, profile=setup.profile, plan=plan, X=X, y=y, model_results=model_results,
                                 fitted_rows=fitted_rows, dropped_features=dropped_columns(config.preprocessing))
    for mr in model_results:
        if mr.search:
            quality.checks.append(QualityCheck(
                id="tuning_preprocessing_scope", status="warn", model_key=mr.model_key,
                title=f"{mr.name}: tuned on pre-processed data",
                detail="On Colab the settings were tuned on training data that was preprocessed once, using all training rows, "
                       "rather than re-learning the preprocessing inside each tuning fold as the server does. The final scores "
                       "still come from rows the search never saw, but tuning scores may be slightly optimistic."))
    scored = [_WEIGHT[c.status] for c in quality.checks if c.status in _WEIGHT]
    quality.score = round(100 * float(np.mean(scored)), 1) if scored else None

    env = {**environment_info(), "trained_on": "Google Colab"}
    for k_, v in (results.get("environment") or {}).items():
        if isinstance(k_, str) and isinstance(v, str) and len(k_) < 30 and len(v) < 100:
            env[f"colab_{k_}"] = v
    return ExperimentResult(
        experiment_id=uuid.uuid4().hex[:12], pipeline_id=config.pipeline_id, pipeline_version=config.version,
        dataset_id=config.dataset.dataset_id, dataset_version=config.dataset.version,
        dataset_fingerprint=dataset_fingerprint(df), config_hash=config_hash(config), task=task.value, random_state=seed,
        started_at=started.isoformat(), finished_at=datetime.now(timezone.utc).isoformat(), environment=env,
        split={**plan.summary(), "notes": plan.notes}, preparation_log=setup.plog.steps, models=model_results, quality=quality,
        source="colab")
