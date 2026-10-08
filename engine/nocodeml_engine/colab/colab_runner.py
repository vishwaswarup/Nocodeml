# NoCodeML Colab runner.
#
# Plain pandas + scikit-learn (+ XGBoost when you chose it). It does NOT need the NoCodeML engine. It reads the bundle
# NoCodeML prepared for you (already preprocessed: every rule was learned from the training rows only), trains the models
# you chose with the settings you chose, and writes the model's PREDICTIONS to a results file. NoCodeML then calculates
# all scores, curves and checks itself from those predictions, so no score is ever typed in or trusted from here.
import importlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd


def _estimator(spec, params):
    """Build the estimator named in the bundle from its (module, class) and parameters."""
    cls = getattr(importlib.import_module(spec["module"]), spec["class"])
    if spec.get("any_labels"):
        cls = _any_label_classifier(cls)
    return cls(**params)


def _any_label_classifier(base):
    """XGBoost insists the labels it sees are exactly 0..k-1. A training fold can lack a rare class, so labels are
    re-indexed internally and mapped back, as scikit-learn's own models do."""
    class AnyLabels(base):
        def fit(self, X, y, **kw):
            self._seen = np.unique(np.asarray(y))
            self._fitting = True
            try:
                return super().fit(X, np.searchsorted(self._seen, np.asarray(y)), **kw)
            finally:
                self._fitting = False

        @property
        def classes_(self):
            return np.arange(len(self._seen)) if getattr(self, "_fitting", False) else self._seen

        def predict(self, X, **kw):
            return self._seen[np.asarray(super().predict(X, **kw)).astype(int)]
    return AnyLabels


def _inner_cv(kind, folds, seed):
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
    if kind == "timeseries":
        return TimeSeriesSplit(n_splits=folds)
    if kind == "stratified":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _scores(est, X, n_classes):
    """Positive-class score (two classes) or class probabilities; None when the model offers neither."""
    if n_classes is None:
        return None
    if hasattr(est, "predict_proba"):
        try:
            p = est.predict_proba(X)
        except AttributeError:
            p = None
        if p is not None:
            full = np.zeros((len(p), n_classes))
            full[:, np.asarray(est.classes_, dtype=int)] = p
            return full[:, 1] if n_classes == 2 else full
    if n_classes == 2 and hasattr(est, "decision_function"):
        return np.asarray(est.decision_function(X))
    return None


def _fit(spec, params, X, y, search, seed):
    est = _estimator(spec, params)
    if not search:
        est.fit(X, y)
        return est, None
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    kw = dict(scoring=search["scoring"], cv=_inner_cv(search["cv"], search["cv_folds"], seed), refit=True, n_jobs=1,
              error_score="raise")
    if search["method"] == "grid":
        s = GridSearchCV(est, search["space"], **kw)
    else:
        s = RandomizedSearchCV(est, search["space"], n_iter=search["n_iter"], random_state=seed, **kw)
    t = time.perf_counter()
    s.fit(X, y)
    res = s.cv_results_
    order = np.argsort(res["rank_test_score"], kind="stable")[:20]
    summary = {"best_params": {k: _plain(v) for k, v in s.best_params_.items()}, "best_score": float(s.best_score_),
               "n_candidates": len(res["params"]), "seconds": round(time.perf_counter() - t, 3),
               "candidates": [{"params": {k: _plain(v) for k, v in res["params"][i].items()},
                               "mean_score": float(res["mean_test_score"][i]), "std_score": float(res["std_test_score"][i]),
                               "rank": int(res["rank_test_score"][i])} for i in order]}
    return s.best_estimator_, summary


def _plain(v):
    return v.item() if isinstance(v, np.generic) else v


def run(bundle_dir, log=print):
    """Train every model on every fold in the bundle and return the results (predictions only)."""
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        man = json.load(f)
    task, n_classes, feats, seed = man["task"], man["n_classes"], man["feature_columns"], man["random_state"]
    keep_scores = n_classes is not None
    out = {"format": man["format"], "config_hash": man["config_hash"], "dataset_fingerprint": man["dataset_fingerprint"],
           "environment": {"python": platform.python_version(), "platform": platform.platform()[:80]},
           "row_ids": {}, "models": {}}
    for lib in ("sklearn", "xgboost", "pandas", "numpy"):
        try:
            out["environment"][lib] = importlib.import_module(lib).__version__
        except Exception:  # noqa: BLE001 - a library you didn't need may simply be absent
            pass

    frames = []
    for fold in man["folds"]:
        fr = {}
        for split in ("train", "validation", "test"):
            if fold.get(split):
                fr[split] = pd.read_csv(os.path.join(bundle_dir, fold[split]), float_precision="round_trip")
        frames.append(fr)
        out["row_ids"][str(fold["index"])] = {s: d["__row_id"].tolist() for s, d in fr.items()}

    cv_mode = man["split"]["mode"] == "cv"
    for m in man["models"]:
        t_model = time.perf_counter()
        folds_out = []
        for i, fr in enumerate(frames):
            train = fr["train"]
            fit_rows = train[train["__fit"] == 1]
            t = time.perf_counter()
            est, search = _fit(m["estimator"], m["params_by_fold"][i], fit_rows[feats], fit_rows["__target"],
                               m.get("search"), seed)
            log(f"{m['name']}: fold {i + 1}/{len(frames)} fitted on {len(fit_rows):,} rows in {time.perf_counter() - t:.1f}s")
            splits = {}
            for split, d in fr.items():
                X = d[feats]
                pred = est.predict(X)
                # In cross-validation only the held-out rows need scores; training rows only need predictions.
                want_scores = keep_scores and not (cv_mode and split == "train")
                sc = _scores(est, X, n_classes) if want_scores else None
                splits[split] = {"pred": (pred.astype(int) if n_classes is not None else pred.astype(float)).tolist(),
                                 "score": None if sc is None else np.asarray(sc, dtype=float).tolist()}
            folds_out.append({"index": man["folds"][i]["index"], "splits": splits, "search": search})
        out["models"][m["key"]] = {"seconds": round(time.perf_counter() - t_model, 3), "folds": folds_out}
        log(f"{m['name']}: done")
    return out


def save(results, path="nocodeml_results.json"):
    with open(path, "w") as f:
        json.dump(results, f, allow_nan=False)
    return path
