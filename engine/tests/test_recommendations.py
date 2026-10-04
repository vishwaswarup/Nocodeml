import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.config import ModelConfig, PreprocessingConfig
from nocodeml_engine.preprocessing.preview import preview_preprocessing
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.recommendations import apply_actions, recommend_preprocessing
from nocodeml_engine.training import check_config, run_experiment

from .conftest import make_config
from nocodeml_engine.config import TaskType


@pytest.fixture
def messy():
    rng = np.random.default_rng(5)
    n = 500
    df = pd.DataFrame({
        "row_id": np.arange(n),
        "const": ["x"] * n,
        "age": rng.normal(40, 12, n),
        "income": rng.lognormal(10, 1.2, n),            # skewed, with outliers
        "gender": rng.choice(["M", "F"], n),
        "city": [f"city{i}" for i in rng.integers(0, 60, n)],   # high cardinality
        "mostly_empty": np.where(rng.random(n) < 0.7, np.nan, 1.0),
        "joined": pd.date_range("2021-01-01", periods=n, freq="D"),
        "churn": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 40, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "gender"] = np.nan
    return df


@pytest.fixture
def messy_dups(messy):
    """Same data plus 15 exact duplicate rows (which also makes row_id non-unique)."""
    return pd.concat([messy, messy.iloc[:15]], ignore_index=True)


def by_id(recs):
    return {r.id: r for r in recs}


def test_rules_and_reasons(messy):
    recs = by_id(recommend_preprocessing(profile_dataset(messy, "churn"), "churn"))
    assert recs["drop_column:row_id"].action == {"type": "drop_column", "column": "row_id"}
    assert "unique for every row" in recs["drop_column:row_id"].reason
    assert "drop_column:const" in recs and "drop_column:mostly_empty" in recs
    assert recs["impute:income"].action["strategy"] == "median" and "skew" in recs["impute:income"].reason
    assert recs["impute:age"].action["strategy"] == "mean"          # symmetric -> mean
    assert recs["impute:gender"].action["strategy"] == "mode"
    assert recs["encode:gender"].action["strategy"] == "one_hot"
    assert recs["encode:city"].action["strategy"] == "frequency" and "60" in recs["encode:city"].reason
    assert recs["date_features:joined"].action["extract"] == ["year", "month", "day_of_week"]
    assert "drop_duplicates:*" not in recs                           # this fixture has none
    assert recs["outliers:income"].action["strategy"] == "winsorize"
    assert recs["scale:*"].confidence == 0.7                        # models unknown yet
    assert not any(r.column == "churn" for r in recs.values())      # target is never touched
    assert not any(k.endswith(":mostly_empty") and not k.startswith("drop") for k in recs)
    for r in recs.values():
        assert r.reason and 0 < r.confidence <= 1


def test_duplicates_are_recommended_for_removal(messy_dups):
    recs = by_id(recommend_preprocessing(profile_dataset(messy_dups, "churn"), "churn"))
    assert "15 rows" in recs["drop_duplicates:*"].reason
    # repeated rows repeat their IDs too: an id-named, 97%-unique column is still an identifier
    assert "drop_column:row_id" in recs


def test_identifier_detection_boundaries():
    n = 200
    df = pd.DataFrame({
        "customer_id": np.arange(n),                               # exactly unique
        "user_id": np.r_[np.arange(n - 4), [0, 1, 2, 3]],           # 98% unique, id-named -> identifier
        "region_id": np.arange(n) % 5,                             # id-named but only 5 values -> a category, keep
        "salary": np.random.default_rng(0).normal(50, 9, n),        # unique-ish floats but not id-named
        "y": np.arange(n) % 2,
    })
    p = profile_dataset(df, "y")
    flagged = {c for c, s in p.columns.items() if s.get("identifier_like")}
    assert flagged == {"customer_id", "user_id"}


def test_scaling_follows_model_choice(messy):
    prof = profile_dataset(messy, "churn")
    assert "scale:*" not in by_id(recommend_preprocessing(prof, "churn", models_need_scaling=False))
    s = by_id(recommend_preprocessing(prof, "churn", models_need_scaling=True))["scale:*"]
    assert s.confidence == 0.9 and "sensitive to feature scale" in s.reason


def test_applying_every_recommendation_gives_a_valid_trainable_config(messy):
    cfg = make_config(TaskType.CLASSIFICATION, "churn",
                      [ModelConfig(model_key="logistic_regression"), ModelConfig(model_key="random_forest")])
    assert check_config(messy, cfg, require_models=False)  # raw data is not trainable as-is
    recs = recommend_preprocessing(profile_dataset(messy, "churn"), "churn", models_need_scaling=True)
    fixed = apply_actions(cfg, [r.action for r in recs])
    assert check_config(messy, fixed) == []
    res = run_experiment(messy, fixed).result
    assert {c.id: c.status for c in res.quality.checks}["preprocessing_leakage"] == "pass"
    assert len(res.models) == 2


def test_apply_actions_is_pure_and_idempotent(messy):
    cfg = make_config(TaskType.CLASSIFICATION, "churn", [])
    acts = [r.action for r in recommend_preprocessing(profile_dataset(messy, "churn"), "churn")]
    once = apply_actions(cfg, acts)
    assert cfg.preprocessing == PreprocessingConfig()                # original untouched
    assert apply_actions(once, acts) == once                         # applying twice changes nothing
    with pytest.raises(ValueError, match="Unknown"):
        apply_actions(cfg, [{"type": "format_disk"}])


def test_preview_before_after(messy_dups):
    messy = messy_dups
    cfg = make_config(TaskType.CLASSIFICATION, "churn", [])
    raw = preview_preprocessing(messy, cfg)
    assert raw["after"] is None and any("missing values" in i for i in raw["issues"])
    recs = recommend_preprocessing(profile_dataset(messy, "churn"), "churn")
    out = preview_preprocessing(messy, apply_actions(cfg, [r.action for r in recs]))
    assert out["issues"] == [], out["issues"]
    b, a = out["before"], out["after"]
    assert b["rows"] == 515 and a["rows"] == 500                      # duplicates removed
    assert a["duplicate_rows"] == 0 and b["duplicate_rows"] == 15
    assert a["missing_cells"] == 0 and b["missing_cells"] > 0
    # age, income, gender_F, gender_M, city (frequency), joined year/month/weekday  (row_id dropped)
    assert a["columns"] == 8
    assert any(s["step"] == "drop_duplicates" and s["rows_removed"] == 15 for s in out["steps"])
    assert "training fits on training rows only" in out["note"].lower()


# ---- Section 3: split recommendations ------------------------------------------

from nocodeml_engine.config import SplitConfig  # noqa: E402
from nocodeml_engine.recommendations import apply_split_actions, recommend_split  # noqa: E402
from nocodeml_engine.splitting import validate_split  # noqa: E402


def _split_df(n, minority=0.5, with_date=False):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": (rng.random(n) < minority).astype(int)})
    if with_date:
        df["when"] = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


def _recs(df, task=TaskType.CLASSIFICATION):
    return by_id(recommend_split(profile_dataset(df, "y"), task, len(df)))


def test_split_large_balanced_is_plain_train_test():
    r = _recs(_split_df(2000))
    assert set(r) == {"split_method"} and r["split_method"].action["patch"]["method"] == "train_test"
    assert "2,000 rows" in r["split_method"].reason


def test_split_imbalanced_is_stratified_with_reason():
    r = _recs(_split_df(2000, minority=0.1))
    assert r["stratify"].action["patch"] == {"stratify": True} and "%" in r["stratify"].reason
    assert not r["stratify"].optional


def test_split_small_data_gets_cross_validation_and_stratify_is_not_duplicated():
    r = _recs(_split_df(300, minority=0.1))
    assert r["split_method"].action["patch"]["method"] == "stratified_k_fold"
    assert "300 rows" in r["split_method"].reason and "stratify" not in r      # k-fold already stratifies
    reg = _recs(_split_df(300), TaskType.REGRESSION)
    assert reg["split_method"].action["patch"]["method"] == "k_fold" and "stratify" not in reg


def test_split_time_data_is_an_optional_suggestion():
    r = _recs(_split_df(2000, with_date=True))
    assert r["time_split"].optional and r["time_split"].column == "when"
    assert r["time_split"].action["patch"]["time_column"] == "when"
    assert not r["split_method"].optional


def test_applying_split_recommendations_gives_valid_splits():
    for n, minority, date in [(2000, 0.5, False), (2000, 0.1, False), (300, 0.1, False), (2000, 0.1, True)]:
        df = _split_df(n, minority, date)
        recs = recommend_split(profile_dataset(df, "y"), TaskType.CLASSIFICATION, n)
        cfg = make_config(TaskType.CLASSIFICATION, "y", [])
        fixed = apply_split_actions(cfg, [r.action for r in recs if not r.optional])
        assert validate_split(fixed.split, df["y"], TaskType.CLASSIFICATION) == [], (n, minority, date)
    # accepting the optional chronological suggestion also yields a valid split
    recs = recommend_split(profile_dataset(_split_df(2000, with_date=True), "y"), TaskType.CLASSIFICATION, 2000)
    chrono = apply_split_actions(make_config(TaskType.CLASSIFICATION, "y", []), [r.action for r in recs])
    assert chrono.split.time_column == "when" and chrono.split.method.value == "train_test"
    with pytest.raises(ValueError, match="Unknown"):
        apply_split_actions(make_config(TaskType.CLASSIFICATION, "y", []), [{"type": "nope"}])


def test_preview_reports_split_sizes(messy_dups):
    cfg = make_config(TaskType.CLASSIFICATION, "churn", [])
    recs = recommend_preprocessing(profile_dataset(messy_dups, "churn"), "churn")
    cfg = apply_actions(cfg, [r.action for r in recs]).model_copy(
        update={"split": SplitConfig(test_size=0.25, stratify=True)})
    out = preview_preprocessing(messy_dups, cfg)
    assert out["split"]["n_test"] == 125 and out["split"]["n_train"] == 375 and out["split"]["stratified"]
    bad = cfg.model_copy(update={"split": SplitConfig(test_size=1.5)})
    out = preview_preprocessing(messy_dups, bad)
    assert out["split"] is None and any("test_size" in i for i in out["split_issues"])


def test_split_preview_is_independent_of_preprocessing_problems(messy_dups):
    """Raw data (unhandled NaNs, raw dates, unencoded text) must not block the split preview."""
    cfg = make_config(TaskType.CLASSIFICATION, "churn", []).model_copy(update={"split": SplitConfig(test_size=0.2)})
    out = preview_preprocessing(messy_dups, cfg)
    assert out["issues"] and out["after"] is None                      # preprocessing is not ready
    assert out["split"]["n_test"] == 103 and out["split"]["n_train"] == 412 and out["split_issues"] == []
    # duplicates removal changes the row count the split sees
    dedup = cfg.model_copy(update={"preprocessing": PreprocessingConfig(drop_duplicates=True)})
    assert preview_preprocessing(messy_dups, dedup)["split"]["n_test"] == 100
    # a real split problem is reported on its own
    tiny = cfg.model_copy(update={"split": SplitConfig(method="stratified_k_fold", n_splits=3, stratify=True)})
    df = messy_dups.assign(churn=[0] * (len(messy_dups) - 1) + [1])    # a class with a single row
    assert any("Smallest class" in i for i in preview_preprocessing(df, tiny)["split_issues"])
