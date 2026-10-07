import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.artifacts import export_artifacts, load_pipeline
from nocodeml_engine.config import (
    ModelConfig, PreprocessingConfig, ScalingStrategy, SplitConfig, SplitMethod, TaskType,
)
from nocodeml_engine.models import ModelConfigError, models_for_task, validate_model_config
from nocodeml_engine.models.registry import XGBOOST_AVAILABLE
from nocodeml_engine.training import run_experiment

from .conftest import make_config

needs_xgb = pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost (and its OpenMP runtime) not installed")


def with_models(cfg, *models):
    return cfg.model_copy(update={"models": [ModelConfig(model_key=k, **kw) for k, kw in models]})


def test_gradient_boosting_now_supports_classification(churn_df, churn_config):
    assert "gradient_boosting" in [m.key for m in models_for_task(TaskType.CLASSIFICATION)]
    res = run_experiment(churn_df, with_models(churn_config, ("gradient_boosting", {}))).result
    t = res.models[0].metrics["test"]
    assert 0 <= t["accuracy"] <= 1 and t["roc_auc"] is not None


@needs_xgb
def test_xgboost_classification_end_to_end_and_export(churn_df, churn_config, tmp_path):
    cfg = with_models(churn_config, ("xgboost", {}), ("logistic_regression", {}))
    run = run_experiment(churn_df, cfg)
    t = run.result.models[0].metrics["test"]
    assert t["roc_auc"] is not None and t["confusion_matrix"]["labels"] == ["0", "1"]
    paths = export_artifacts(run, cfg, tmp_path)
    fp = load_pipeline(paths["pipeline_xgboost"])
    preds = fp.predict(churn_df.drop(columns=["churn"]).head(5))            # raw rows, NaN and strings included
    assert set(preds) <= {0, 1} and len(preds) == 5


@needs_xgb
def test_xgboost_is_reproducible(churn_df, churn_config):
    cfg = with_models(churn_config, ("xgboost", {}))
    a = run_experiment(churn_df, cfg).result.models[0].metrics["test"]
    b = run_experiment(churn_df, cfg).result.models[0].metrics["test"]
    assert a["f1"] == b["f1"] and a["roc_auc"] == b["roc_auc"]


@needs_xgb
def test_xgboost_regression_and_cv(housing_df):
    from nocodeml_engine.config import EncodingRule, EncodingStrategy
    cfg = make_config(TaskType.REGRESSION, "price", [ModelConfig(model_key="xgboost", hyperparameters={"n_estimators": 50})],
                      preprocessing=PreprocessingConfig(encoding=[EncodingRule(column="city", strategy=EncodingStrategy.ONE_HOT)]),
                      split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=3))
    m = run_experiment(housing_df, cfg).result.models[0]
    assert m.metrics["cv"]["r2"] > 0.5


@needs_xgb
def test_a_class_missing_from_a_training_fold_does_not_crash():
    """XGBoost normally demands labels 0..k-1; a rare class absent from one fold's training rows used to break that."""
    rng = np.random.default_rng(0)
    n = 90
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = np.where(df.a > 0.3, 1, 0)
    df.loc[:2, "y"] = 2                                   # three rows of a rare third class: some folds won't train on it
    cfg = make_config(TaskType.CLASSIFICATION, "y", [ModelConfig(model_key="xgboost", hyperparameters={"n_estimators": 20})],
                      preprocessing=PreprocessingConfig(scaling=ScalingStrategy.NONE),
                      split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=3, shuffle=False))
    res = run_experiment(df, cfg).result
    assert res.models[0].metrics["cv"]["accuracy"] > 0.5


@needs_xgb
def test_xgboost_parameter_validation_and_regularization_rules(churn_config):
    validate_model_config(ModelConfig(model_key="xgboost", hyperparameters={"reg_alpha": 1.0, "reg_lambda": 2.0, "max_depth": 3}),
                          TaskType.CLASSIFICATION)
    for bad in ({"max_depth": 0}, {"learning_rate": 5}, {"subsample": 0}, {"bogus": 1}, {"n_estimators": 10.5}):
        with pytest.raises(ModelConfigError):
            validate_model_config(ModelConfig(model_key="xgboost", hyperparameters=bad), TaskType.CLASSIFICATION)
    with pytest.raises(ModelConfigError, match="not an L1/L2 penalty"):          # regularized through hyperparameters
        validate_model_config(ModelConfig(model_key="xgboost", regularization={"type": "l2"}), TaskType.CLASSIFICATION)


@needs_xgb
def test_model_preview_accepts_xgboost_and_still_catches_bad_sklearn_values(churn_config):
    from nocodeml_engine.preprocessing.preview import preview_models
    ok = with_models(churn_config, ("xgboost", {"hyperparameters": {"reg_lambda": 5.0, "learning_rate": 0.2}}),
                     ("gradient_boosting", {}))
    assert preview_models(ok) == {}
    bad = with_models(churn_config, ("xgboost", {"hyperparameters": {"reg_lambda": -3.0}}),
                      ("logistic_regression", {"regularization": {"type": "l2", "strength": -1}}))
    out = preview_models(bad)
    assert set(out) == {"xgboost", "logistic_regression"}
