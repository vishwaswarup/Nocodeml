import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.config import (
    DateFeature, DateFeatureRule, EncodingRule, EncodingStrategy, FeatureEngineeringConfig, MissingValueRule,
    MissingValueStrategy, ModelConfig, NumericFeatureRule, NumericTransform, PreprocessingConfig, ScalingStrategy, TaskType,
)
from nocodeml_engine.feature_engineering import FeatureEngineer, FeatureEngineeringError
from nocodeml_engine.preprocessing.preview import preview_features, preview_preprocessing
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.recommendations import apply_actions, recommend_features, recommend_preprocessing
from nocodeml_engine.training import check_config, run_experiment

from .conftest import make_config


@pytest.fixture
def feat_df():
    rng = np.random.default_rng(8)
    n = 600
    return pd.DataFrame({
        "income": rng.lognormal(10, 1.1, n),                       # strongly right-skewed, positive
        "balance": rng.normal(0, 500, n),                          # symmetric, has negatives
        "age": rng.normal(40, 10, n),
        "opened": pd.date_range("2022-01-01 00:00", periods=n, freq="7h"),   # >1 year, with times of day
        "day_only": pd.date_range("2024-01-01", periods=n, freq="D"),         # dates without times
        "short": pd.date_range("2024-06-01", periods=n, freq="1min"),         # a single day span
        "y": rng.integers(0, 2, n),
    })


def test_hour_extraction_and_validation():
    df = pd.DataFrame({"t": pd.to_datetime(["2024-03-01 07:30", "2024-03-02 23:05", None]), "x": [1.0, 2.0, 3.0]})
    out = FeatureEngineer(FeatureEngineeringConfig(date_features=[DateFeatureRule(column="t", extract=[DateFeature.HOUR])])).fit(df).transform(df)
    assert out["t__hour"].tolist()[:2] == [7.0, 23.0] and np.isnan(out["t__hour"].iloc[2]) and "t" not in out.columns
    bad = FeatureEngineeringConfig(date_features=[DateFeatureRule(column="x", extract=[DateFeature.YEAR])])
    with pytest.raises(FeatureEngineeringError, match="does not look like a date"):
        FeatureEngineer(bad).fit(df)
    empty = FeatureEngineeringConfig(date_features=[DateFeatureRule(column="t", extract=[])])
    with pytest.raises(FeatureEngineeringError, match="at least one date part"):
        FeatureEngineer(empty).fit(df)
    # a text column that really holds dates is accepted
    txt = pd.DataFrame({"d": ["2024-01-01", "2024-02-01", "2024-03-01"]})
    ok = FeatureEngineeringConfig(date_features=[DateFeatureRule(column="d", extract=[DateFeature.MONTH])])
    assert FeatureEngineer(ok).fit(txt).transform(txt)["d__month"].tolist() == [1.0, 2.0, 3.0]


def test_profile_knows_whether_a_date_has_a_time(feat_df):
    cols = profile_dataset(feat_df, "y").columns
    assert cols["opened"]["has_time"] is True and cols["day_only"]["has_time"] is False


def test_feature_recommendations_and_reasons(feat_df):
    recs = {r.id: r for r in recommend_features(profile_dataset(feat_df, "y"), "y")}
    assert recs["numeric_transform:income"].action == {"type": "numeric_transform", "column": "income", "transform": "log"}
    assert "skewness" in recs["numeric_transform:income"].reason and "income__log" in recs["numeric_transform:income"].reason
    assert "numeric_transform:balance" not in recs and "numeric_transform:age" not in recs     # symmetric / negative -> no log
    assert recs["date_features:opened"].action["extract"] == ["year", "month", "day_of_week", "hour"]
    assert recs["date_features:day_only"].action["extract"] == ["year", "month", "day_of_week"]  # no time of day
    assert recs["date_features:short"].action["extract"] == ["day_of_week", "hour"]               # one-day span: no year/month
    assert all(r.reason and r.confidence > 0 for r in recs.values())
    dropped = {r.id for r in recommend_features(profile_dataset(feat_df, "y"), "y", drop_columns=["income", "opened"])}
    assert "numeric_transform:income" not in dropped and "date_features:opened" not in dropped
    assert not any(r.column == "y" for r in recs.values())                                      # never the target


def test_applying_feature_recommendations_trains(feat_df):
    cfg = make_config(TaskType.CLASSIFICATION, "y", [ModelConfig(model_key="logistic_regression"), ModelConfig(model_key="random_forest")])
    prof = profile_dataset(feat_df, "y")
    acts = [r.action for r in recommend_features(prof, "y")]
    # the date columns left unused by recommendations are dropped (Section 1's job); then everything must train
    cfg = cfg.model_copy(update={"preprocessing": PreprocessingConfig(drop_columns=["day_only", "short"], scaling=ScalingStrategy.STANDARD)})
    cfg = apply_actions(cfg, [a for a in acts if a["column"] not in ("day_only", "short")])
    assert [r.transform.value for r in cfg.feature_engineering.numeric_transforms] == ["log"]
    assert check_config(feat_df, cfg) == []
    res = run_experiment(feat_df, cfg).result
    assert {c.id: c.status for c in res.quality.checks}["preprocessing_leakage"] == "pass"
    assert apply_actions(cfg, acts) == apply_actions(apply_actions(cfg, acts), acts)             # idempotent


def test_preview_lists_new_columns_with_samples_and_every_problem(feat_df):
    cfg = make_config(TaskType.CLASSIFICATION, "y", [], feature_engineering=FeatureEngineeringConfig(
        numeric_transforms=[NumericFeatureRule(column="income", transform=NumericTransform.LOG)],
        date_features=[DateFeatureRule(column="opened", extract=[DateFeature.YEAR, DateFeature.HOUR])]))
    new, issues = preview_features(feat_df, cfg)
    assert issues == [] and [f["name"] for f in new] == ["income__log", "opened__year", "opened__hour"]
    log = new[0]
    assert log["source"] == "income" and log["op"] == "log" and len(log["sample"]) == 5
    assert log["sample"][0] == pytest.approx(float(np.log1p(feat_df["income"].iloc[0])), abs=1e-3)
    assert new[2]["sample"][:2] == [0.0, 7.0]                                                     # 7-hourly timestamps

    # all problems are reported together, independently of preprocessing being ready
    bad = make_config(TaskType.CLASSIFICATION, "y", [], feature_engineering=FeatureEngineeringConfig(
        numeric_transforms=[NumericFeatureRule(column="balance", transform=NumericTransform.LOG),
                            NumericFeatureRule(column="balance", transform=NumericTransform.SQRT)],
        date_features=[DateFeatureRule(column="age", extract=[DateFeature.MONTH])]))
    new2, issues2 = preview_features(feat_df, bad)
    assert new2 == [] and len(issues2) == 3
    assert any("log" in i for i in issues2) and any("sqrt" in i for i in issues2) and any("does not look like a date" in i for i in issues2)
    out = preview_preprocessing(feat_df, bad)
    assert out["feature_issues"] == issues2 and out["features"] == []


def test_section_one_recommendation_uses_the_same_date_parts(feat_df):
    recs = {r.id: r for r in recommend_preprocessing(profile_dataset(feat_df, "y"), "y")}
    assert recs["date_features:opened"].action["extract"] == ["year", "month", "day_of_week", "hour"]
