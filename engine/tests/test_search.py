import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.artifacts import export_artifacts, load_pipeline
from nocodeml_engine.config import (
    EncodingRule, EncodingStrategy, ModelConfig, PreprocessingConfig, SearchConfig, SplitConfig, SplitMethod, TaskType,
)
from nocodeml_engine.models import ModelConfigError, validate_model_config
from nocodeml_engine.training import ConfigurationError, TrainingTimeout, run_experiment
from nocodeml_engine.training.runner import config_hash

from .conftest import make_config


def rf(**search):
    return ModelConfig(model_key="random_forest", search=SearchConfig(**search))


def with_models(cfg, *models):
    return cfg.model_copy(update={"models": list(models)})


def test_random_search_picks_from_the_space_and_records_every_candidate(churn_df, churn_config):
    cfg = with_models(churn_config, rf(method="random", n_iter=4, cv_folds=3,
                                       space={"max_depth": [2, 4, 8], "n_estimators": [10, 20]}))
    res = run_experiment(churn_df, cfg).result.models[0]
    s = res.search
    assert s["method"] == "random" and s["scoring"] == "f1" and s["n_candidates"] == 4
    assert s["best_params"]["max_depth"] in (2, 4, 8) and s["best_params"]["n_estimators"] in (10, 20)
    assert res.hyperparameters["max_depth"] == s["best_params"]["max_depth"]          # the model really uses the winner
    scores = [c["mean_score"] for c in s["candidates"]]
    assert scores == sorted(scores, reverse=True) and s["candidates"][0]["rank"] == 1
    assert s["best_score"] == pytest.approx(scores[0])
    assert 0 <= res.metrics["test"]["accuracy"] <= 1


def test_grid_search_tries_every_combination(churn_df, churn_config):
    cfg = with_models(churn_config, ModelConfig(model_key="logistic_regression",
                      search=SearchConfig(method="grid", space={"C": [0.01, 0.1, 1.0, 10.0]}, cv_folds=3)))
    s = run_experiment(churn_df, cfg).result.models[0].search
    assert s["n_candidates"] == 4 and sorted(c["params"]["C"] for c in s["candidates"]) == [0.01, 0.1, 1.0, 10.0]


def test_search_is_reproducible(churn_df, churn_config):
    cfg = with_models(churn_config, rf(method="random", n_iter=4, space={"max_depth": [2, 3, 4, 6], "min_samples_leaf": [1, 5, 10]}))
    a = run_experiment(churn_df, cfg).result.models[0]
    b = run_experiment(churn_df, cfg).result.models[0]
    assert a.search["best_params"] == b.search["best_params"]
    assert [c["mean_score"] for c in a.search["candidates"]] == [c["mean_score"] for c in b.search["candidates"]]
    assert a.metrics["test"]["f1"] == b.metrics["test"]["f1"]


def test_tuning_uses_only_training_rows(churn_df, churn_config, monkeypatch):
    """The rows handed to the search are the training rows of the split; none of the test rows may be among them."""
    import nocodeml_engine.training.search as S
    seen = {}
    real = S.run_search

    def spy(pipe, X, y, *a, **k):
        seen["index"] = set(X.index)
        return real(pipe, X, y, *a, **k)

    monkeypatch.setattr(S, "run_search", spy)
    cfg = with_models(churn_config, rf(method="grid", space={"max_depth": [2, 4]}))
    run = run_experiment(churn_df, cfg)
    n_test = run.result.split["n_test"]
    assert len(seen["index"]) < len(churn_df) - n_test + 1            # training rows only
    assert len(seen["index"]) + n_test <= len(churn_df)


def test_tuning_works_with_cross_validation_and_regression(housing_df):
    cfg = make_config(TaskType.REGRESSION, "price", [ModelConfig(model_key="ridge", search=SearchConfig(
        method="grid", space={"alpha": [0.1, 1.0, 10.0]}, cv_folds=2))],
        preprocessing=PreprocessingConfig(encoding=[EncodingRule(column="city", strategy=EncodingStrategy.ONE_HOT)]),
        split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=3))
    m = run_experiment(housing_df, cfg).result.models[0]
    assert m.search["scoring"] == "r2" and m.metrics["cv"]["r2"] > 0.5


def test_exported_pipeline_uses_the_tuned_model_and_scores_raw_rows(churn_df, churn_config, tmp_path):
    cfg = with_models(churn_config, rf(method="grid", space={"max_depth": [2, 6]}))
    run = run_experiment(churn_df, cfg)
    paths = export_artifacts(run, cfg, tmp_path)
    fp = load_pipeline(paths["pipeline_random_forest"])
    assert fp.model.get_params()["max_depth"] == run.result.models[0].search["best_params"]["max_depth"]
    assert len(fp.predict(churn_df.drop(columns=["churn"]).head(4))) == 4


def test_config_hash_is_unchanged_for_configs_without_a_search_and_changes_with_one(churn_config):
    plain = config_hash(churn_config)
    dumped = churn_config.model_dump(mode="json")
    assert all("search" in m for m in dumped["models"])                       # the field exists in the model...
    import json, hashlib
    legacy = {k: v for k, v in dumped.items() if k != "version"}
    for m in legacy["models"]:
        m.pop("search")
    legacy_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    assert plain == legacy_hash                                               # ...but old configs still hash the same
    tuned = with_models(churn_config, rf(method="grid", space={"max_depth": [2, 4]}))
    assert config_hash(tuned) != plain


def test_time_limit_stops_a_search(churn_df, churn_config):
    import time
    cfg = with_models(churn_config, rf(method="grid", space={"max_depth": [2, 4, 6]}))
    with pytest.raises(TrainingTimeout):
        run_experiment(churn_df, cfg, deadline=time.monotonic() + 0.0001)


@pytest.mark.parametrize("model,search,match", [
    ("random_forest", dict(space={}), "at least one"),
    ("random_forest", dict(space={"bogus": [1, 2]}), "can't be tuned"),
    ("random_forest", dict(space={"max_depth": [0, 2]}), "within"),
    ("random_forest", dict(space={"max_depth": [2, 2]}), "same value twice"),
    ("random_forest", dict(space={"max_depth": list(range(1, 12))}), "between 1 and 10"),
    ("random_forest", dict(method="grid", space={"max_depth": list(range(1, 11)), "n_estimators": [10, 20, 30, 40, 50]}), "limit is 40"),
    ("random_forest", dict(method="random", n_iter=31, space={"max_depth": list(range(1, 11)), "n_estimators": [10, 20, 30, 40, 50]},
                           cv_folds=5), "limit is 150"),
    ("random_forest", dict(method="random", n_iter=10, space={"max_depth": [2, 3]}), "only 2 combinations"),
    ("random_forest", dict(space={"max_depth": [2, 3]}, cv_folds=1), "between 2 and 5"),
    ("random_forest", dict(method="random", n_iter=1, space={"max_depth": [2, 3]}), "n_iter"),
])
def test_search_validation_messages(model, search, match):
    cfg = ModelConfig(model_key=model, search=SearchConfig(**{"method": "grid", **search}) if "n_iter" not in search and "method" not in search
                      else SearchConfig(**search))
    with pytest.raises(ModelConfigError, match=match):
        validate_model_config(cfg, TaskType.CLASSIFICATION)


def test_c_cannot_be_tuned_when_the_penalty_is_none_and_run_rejects_bad_search(churn_df, churn_config):
    bad = ModelConfig(model_key="logistic_regression", regularization={"type": "none"},
                      search=SearchConfig(method="grid", space={"C": [1, 2]}))
    with pytest.raises(ModelConfigError, match="penalty is 'none'"):
        validate_model_config(bad, TaskType.CLASSIFICATION)
    with pytest.raises(ConfigurationError):
        run_experiment(churn_df, with_models(churn_config, bad))


def test_models_endpoint_lists_tunable_parameters_with_valid_suggestions():
    import uuid
    from fastapi.testclient import TestClient
    from nocodeml_engine.api.app import create_app
    from nocodeml_engine.models import MODEL_REGISTRY
    from nocodeml_engine.models.registry import tunable_params
    from .fake_supabase import FakeStore, FakeSupabase
    store, uid = FakeStore(), str(uuid.uuid4())
    app = create_app(client_factory=lambda t: (FakeSupabase(store, uid), uid), client_builder=lambda t: FakeSupabase(store, uid))
    with TestClient(app) as c:
        out = c.get("/models").json()
    assert {"C"} <= {t["name"] for m in out if m["key"] == "logistic_regression" for t in m["tunable"]}
    assert all(t["name"] != "max_iter" for m in out for t in m["tunable"])
    # every suggestion we offer must pass the registry's own validation, or the UI would hand users a broken default
    for key, spec in MODEL_REGISTRY.items():
        for task in spec.tasks:
            for name, hp in tunable_params(spec).items():
                if name == "max_iter":
                    continue
                vals = [t for m in out if m["key"] == key and m["task"] == task.value for t in m["tunable"] if t["name"] == name][0]["suggested"]
                if vals:
                    validate_model_config(ModelConfig(model_key=key, search=SearchConfig(
                        method="grid", space={name: vals}, cv_folds=2)), task)


def test_edge_of_range_is_flagged(churn_df, churn_config):
    cfg = with_models(churn_config, ModelConfig(model_key="logistic_regression",
                      search=SearchConfig(method="grid", space={"C": [1e-4, 1e-3, 1e-2]})))     # tiny C everywhere: best is the top end
    s = run_experiment(churn_df, cfg).result.models[0].search
    assert s["edge_params"] in (["C"], [])      # flagged exactly when the winner is an end value
    assert (s["best_params"]["C"] in (1e-4, 1e-2)) == (s["edge_params"] == ["C"])
