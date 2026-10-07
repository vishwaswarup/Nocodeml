import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.models.registry import XGBOOST_AVAILABLE
from nocodeml_engine.artifacts import export_artifacts, load_pipeline
from nocodeml_engine.config import (
    DateFeature, DateFeatureRule, EncodingRule, EncodingStrategy, FeatureEngineeringConfig,
    MissingValueRule, MissingValueStrategy, ModelConfig, NumericFeatureRule, NumericTransform,
    OutlierRule, OutlierStrategy, PreprocessingConfig, ScalingStrategy, SplitConfig, SplitMethod,
    TaskType,
)
from nocodeml_engine.models import ModelConfigError, models_for_task, validate_model_config
from nocodeml_engine.preprocessing import PreprocessingError
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.splitting import SplitError, make_split_plan
from nocodeml_engine.training import ConfigurationError, run_experiment

from .conftest import make_config


# ---- profiler -------------------------------------------------------------

def test_profiler_flags_identifier_missing_and_imbalance(churn_df):
    df = churn_df.copy()
    df["churn"] = (np.arange(len(df)) % 10 == 0).astype(int)  # 10% positives
    p = profile_dataset(df, "churn")
    codes = {(w.code, w.column) for w in p.warnings}
    assert ("possible_identifier", "customer_id") in codes
    assert ("missing_values", "age") in codes
    assert ("class_imbalance", "churn") in codes
    assert p.target["inferred_task"] == "classification"
    assert p.columns["age"]["outlier_count"] >= 0 and "skewness" in p.columns["age"]


# ---- end to end -----------------------------------------------------------

def test_classification_end_to_end(churn_df, churn_config, tmp_path):
    run = run_experiment(churn_df, churn_config)
    res = run.result
    assert [m.model_key for m in res.models] == ["logistic_regression", "random_forest"]
    for m in res.models:
        t = m.metrics["test"]
        assert 0 <= t["accuracy"] <= 1 and t["roc_auc"] is not None
        assert "roc_curve" in t and t["confusion_matrix"]["labels"] == ["0", "1"]
    statuses = {c.id: c.status for c in res.quality.checks}
    assert statuses["preprocessing_leakage"] == "pass"
    assert statuses["split_isolation"] == "pass"
    assert "identifier_feature" not in statuses  # customer_id was dropped
    assert res.quality.score is not None

    paths = export_artifacts(run, churn_config, tmp_path)
    fp = load_pipeline(paths["pipeline_random_forest"])
    raw = churn_df.drop(columns=["churn"]).head(5)  # raw rows incl. NaN / strings
    preds = fp.predict(raw)
    assert set(preds) <= {0, 1} and len(preds) == 5
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "SECURITY.txt").exists()


def test_reproducible(churn_df, churn_config):
    a = run_experiment(churn_df, churn_config).result
    b = run_experiment(churn_df, churn_config).result
    assert a.config_hash == b.config_hash and a.dataset_fingerprint == b.dataset_fingerprint
    assert a.models[1].metrics["test"]["f1"] == b.models[1].metrics["test"]["f1"]


def test_preprocessing_fitted_on_train_only(churn_df, churn_config):
    run = run_experiment(churn_df, churn_config)
    fp = run.pipelines["logistic_regression"]
    scaler = fp.pipeline.named_steps["preprocess"].named_steps["encode_scale"] \
        .named_transformers_["scale"]
    # Recreate the same split and verify scaler mean == TRAIN mean, not full-data mean.
    prepared = churn_df.copy()
    from nocodeml_engine.splitting import make_split_plan
    X, y = prepared.drop(columns=["churn"]), prepared["churn"]
    plan = make_split_plan(X, y, churn_config.split, TaskType.CLASSIFICATION)
    tr = X.iloc[plan.folds[0].train]
    names = list(scaler.feature_names_in_)
    age_train_imputed = tr["age"].fillna(tr["age"].median())
    assert scaler.mean_[names.index("age")] == pytest.approx(age_train_imputed.mean())
    full = X["age"].fillna(X["age"].median()).mean()
    assert scaler.mean_[names.index("age")] != pytest.approx(full)


def test_regression_and_cv(housing_df):
    cfg = make_config(
        TaskType.REGRESSION, "price",
        [ModelConfig(model_key="ridge", regularization={"type": "l2", "strength": 1.0}),
         ModelConfig(model_key="lasso", hyperparameters={"alpha": 0.5}),
         ModelConfig(model_key="gradient_boosting")],
        preprocessing=PreprocessingConfig(
            encoding=[EncodingRule(column="city", strategy=EncodingStrategy.ONE_HOT)],
            scaling=ScalingStrategy.STANDARD),
        split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=4))
    res = run_experiment(housing_df, cfg).result
    for m in res.models:
        cv = m.metrics["cv"]
        assert cv["r2"] > 0.9 and len(m.fold_primary_scores) == 4
        assert "actual_vs_predicted" in cv and len(cv["residuals"]) == len(housing_df)
        assert m.baseline["r2"] < 0.1
    assert {c.id for c in res.quality.checks if c.status == "pass"} >= {"cross_validation"}


def test_train_val_test_and_encodings_and_outliers(churn_df):
    df = churn_df.copy()
    df["city"] = np.random.default_rng(2).choice(list("abcde"), len(df))
    df["when"] = pd.date_range("2020-01-01", periods=len(df), freq="D").astype(str)
    df = pd.read_csv(__import__("io").StringIO(df.to_csv(index=False)))  # as loaded from CSV
    from nocodeml_engine.dataset.loader import infer_datetimes
    df = infer_datetimes(df)
    assert pd.api.types.is_datetime64_any_dtype(df["when"])
    cfg = make_config(
        TaskType.CLASSIFICATION, "churn",
        [ModelConfig(model_key="knn"), ModelConfig(model_key="decision_tree"),
         ModelConfig(model_key="svm")],
        preprocessing=PreprocessingConfig(
            drop_columns=["customer_id"], drop_duplicates=True,
            missing_values=[MissingValueRule(column="age", strategy=MissingValueStrategy.KNN)],
            encoding=[EncodingRule(column="gender", strategy=EncodingStrategy.ONE_HOT),
                      EncodingRule(column="city", strategy=EncodingStrategy.FREQUENCY)],
            outliers=[OutlierRule(column="income", strategy=OutlierStrategy.IQR),
                      OutlierRule(column="age", strategy=OutlierStrategy.WINSORIZE)],
            scaling=ScalingStrategy.ROBUST),
        feature_engineering=FeatureEngineeringConfig(
            numeric_transforms=[NumericFeatureRule(column="income", transform=NumericTransform.LOG)],
            date_features=[DateFeatureRule(column="when",
                                           extract=[DateFeature.MONTH, DateFeature.IS_WEEKEND])]),
        split=SplitConfig(method=SplitMethod.TRAIN_VAL_TEST, validation_size=0.15, stratify=True))
    run = run_experiment(df, cfg)
    m = run.result.models[0]
    assert "validation" in m.metrics and "test" in m.metrics
    assert m.n_outlier_rows_removed > 0
    assert run.pipelines["knn"].predict(df.drop(columns=["churn"]).head(3)).shape == (3,)


def test_target_encoding(housing_df):
    cfg = make_config(TaskType.REGRESSION, "price", [ModelConfig(model_key="random_forest")],
                      preprocessing=PreprocessingConfig(
                          encoding=[EncodingRule(column="city", strategy=EncodingStrategy.TARGET)]))
    assert run_experiment(housing_df, cfg).result.models[0].metrics["test"]["r2"] > 0.8


def test_time_series_split_and_temporal_warning():
    n = 200
    df = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=n), "x": np.arange(n, dtype=float),
                       "y": np.arange(n) * 2.0 + np.random.default_rng(0).normal(0, 1, n)})
    cfg = make_config(TaskType.REGRESSION, "y", [ModelConfig(model_key="linear_regression")],
                      preprocessing=PreprocessingConfig(drop_columns=["t"]),
                      split=SplitConfig(method=SplitMethod.TIME_SERIES, n_splits=3, time_column="t"))
    res = run_experiment(df, cfg).result
    assert res.split["chronological"] and res.models[0].metrics["cv"]["r2"] > 0.9
    # Random split on time data should warn
    cfg2 = cfg.model_copy(update={
        "split": SplitConfig(), "preprocessing": PreprocessingConfig(),
        "feature_engineering": FeatureEngineeringConfig(date_features=[
            DateFeatureRule(column="t", extract=[DateFeature.YEAR])])})
    ids = {c.id: c.status for c in run_experiment(df, cfg2).result.quality.checks}
    assert ids["temporal_leakage"] == "warn"
    # ...but not when the datetime column is dropped entirely
    assert "temporal_leakage" not in {c.id for c in run_experiment(df, cfg.model_copy(
        update={"split": SplitConfig()})).result.quality.checks}


# ---- validation -----------------------------------------------------------

def test_unresolved_missing_and_unencoded_rejected(churn_df, churn_config):
    bad = churn_config.model_copy(update={"preprocessing": PreprocessingConfig(drop_columns=["customer_id"])})
    with pytest.raises(PreprocessingError) as e:
        run_experiment(churn_df, bad)
    assert any("'age' has missing" in i for i in e.value.issues)
    assert any("'gender'" in i and "encoding" in i for i in e.value.issues)


def test_config_validation(churn_df, churn_config):
    many = churn_config.model_copy(update={"models": [ModelConfig(model_key=k) for k in
                                                     ["logistic_regression", "knn", "decision_tree",
                                                      "random_forest", "svm", "ridge"]]})
    with pytest.raises(ConfigurationError, match="At most 5"):
        run_experiment(churn_df, many)
    with pytest.raises(ConfigurationError, match="does not support classification"):
        run_experiment(churn_df, churn_config.model_copy(update={"models": [ModelConfig(model_key="ridge")]}))


def test_model_aware_regularization():
    C = TaskType.CLASSIFICATION
    validate_model_config(ModelConfig(model_key="logistic_regression",
                                      regularization={"type": "elasticnet", "strength": 0.5,
                                                      "l1_ratio": 0.3}), C)
    with pytest.raises(ModelConfigError, match="not an L1/L2"):
        validate_model_config(ModelConfig(model_key="random_forest", regularization={"type": "l2"}), C)
    with pytest.raises(ModelConfigError, match="does not support regularization"):
        validate_model_config(ModelConfig(model_key="linear_regression", regularization={"type": "l2"}),
                              TaskType.REGRESSION)
    with pytest.raises(ModelConfigError, match="max_depth"):
        validate_model_config(ModelConfig(model_key="random_forest", hyperparameters={"max_depth": 0}), C)
    with pytest.raises(ModelConfigError, match="no hyperparameter"):
        validate_model_config(ModelConfig(model_key="knn", hyperparameters={"bogus": 1}), C)
    assert {m.key for m in models_for_task(TaskType.REGRESSION)} == {
        "linear_regression", "ridge", "lasso", "decision_tree", "random_forest", "gradient_boosting",
        *(["xgboost"] if XGBOOST_AVAILABLE else [])}


def test_logistic_regularization_variants_train(churn_df, churn_config):
    for reg in ({"type": "none"}, {"type": "l1", "strength": 0.5}, {"type": "l2", "strength": 2.0},
                {"type": "elasticnet", "strength": 1.0, "l1_ratio": 0.4}):
        cfg = churn_config.model_copy(update={"models": [
            ModelConfig(model_key="logistic_regression", regularization=reg)]})
        assert run_experiment(churn_df, cfg).result.models[0].metrics["test"]["accuracy"] > 0.4


def test_split_validation():
    y = pd.Series([0] * 98 + [1] * 1 + [2] * 1)
    X = pd.DataFrame({"a": range(100)})
    with pytest.raises(SplitError, match="Smallest class"):
        make_split_plan(X, y, SplitConfig(stratify=True), TaskType.CLASSIFICATION)
    with pytest.raises(SplitError, match="classification"):
        make_split_plan(X, y.astype(float), SplitConfig(stratify=True), TaskType.REGRESSION)
    with pytest.raises(SplitError, match="validation_size"):
        make_split_plan(X, y, SplitConfig(method=SplitMethod.TRAIN_VAL_TEST), TaskType.CLASSIFICATION)


def test_leakage_and_overfit_checks_fire():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "leak": x * 2 + 0.0, "noise": rng.normal(size=n), "y": x * 3})
    cfg = make_config(TaskType.REGRESSION, "y", [ModelConfig(model_key="linear_regression")])
    ids = {c.id: c for c in run_experiment(df, cfg).result.quality.checks}
    assert ids["target_leakage"].status == "warn"
