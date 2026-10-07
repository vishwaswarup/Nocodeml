import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.visualization import (
    ChartError, auto_charts, categories, class_balance, correlation, histogram, missing_overview, scatter,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=n)
    d = pd.DataFrame({
        "customer_id": np.arange(n),
        "skewed": rng.lognormal(1, 1.2, n),
        "x": x,
        "x2": 2 * x + rng.normal(0, 0.3, n),          # strongly related to x
        "noise": rng.normal(size=n),
        "plan": rng.choice(["basic", "pro", "team"], n, p=[0.6, 0.3, 0.1]),
        "city": [f"c{i}" for i in rng.integers(0, 60, n)],
        "churn": (x + rng.normal(0, 1, n) > 1.9).astype(int),
    })
    d.loc[rng.choice(n, 80, replace=False), "noise"] = np.nan
    d.loc[rng.choice(n, 40, replace=False), "plan"] = np.nan
    return d


def test_histogram_and_box_are_consistent_with_the_data(df):
    h = histogram(df, "skewed", bins=20)
    assert len(h["counts"]) == 20 and len(h["edges"]) == 21 and sum(h["counts"]) == h["n"] == 800
    b = h["box"]
    assert b["min"] <= b["whisker_low"] <= b["q1"] <= b["median"] <= b["q3"] <= b["whisker_high"] <= b["max"]
    assert b["median"] == pytest.approx(df["skewed"].median(), rel=1e-4) and b["outlier_count"] > 0
    assert all(o < b["whisker_low"] or o > b["whisker_high"] for o in b["outliers"]) and len(b["outliers"]) <= 60
    assert h["stats"]["skewness"] > 1 and h["missing"] == 0
    assert histogram(df, "noise")["missing"] == 80 and histogram(df, "noise")["n"] == 720


def test_chart_input_errors_are_clear(df):
    with pytest.raises(ChartError, match="not a numeric column"):
        histogram(df, "plan")
    with pytest.raises(ChartError, match="not found"):
        histogram(df, "nope")
    with pytest.raises(ChartError, match="bins"):
        histogram(df, "x", bins=1)
    with pytest.raises(ChartError, match="at least two numeric"):
        correlation(df[["x", "plan"]])
    with pytest.raises(ChartError, match="different columns"):
        scatter(df, "x", "x")
    with pytest.raises(ChartError, match="no values"):
        histogram(pd.DataFrame({"a": [np.nan, np.nan, np.nan]}), "a")


def test_categories_fold_the_tail_into_other(df):
    c = categories(df, "city", top=10)
    assert len(c["items"]) == 10 and c["n_unique"] > 10
    assert sum(i["count"] for i in c["items"]) + c["other"] == 800 and c["other"] > 0
    assert categories(df, "plan")["missing"] == 40 and categories(df, "plan")["n"] == 760
    assert [i["label"] for i in categories(df, "plan")["items"]] == ["basic", "pro", "team"]


def test_correlation_is_symmetric_bounded_and_codes_a_binary_target(df):
    c = correlation(df, target="churn", exclude={"customer_id"})
    m, cols = np.array([[np.nan if v is None else v for v in r] for r in c["matrix"]]), c["columns"]
    assert "customer_id" not in cols and "churn" in cols and c["target"] == "churn" and "coded as" in c["note"]
    assert np.allclose(m, m.T, equal_nan=True) and np.nanmax(np.abs(m)) <= 1.0 + 1e-9 and np.allclose(np.diag(m), 1.0)
    top = c["top_pairs"][0]
    assert {top["a"], top["b"]} == {"x", "x2"} and top["r"] > 0.95
    assert df["x"].corr(df["x2"]) == pytest.approx(top["r"], abs=1e-5)
    # a 3-class target is not coded (an ordinal code would be meaningless)
    d3 = df.assign(churn=np.digitize(df["x"], [-0.5, 0.5]))
    assert correlation(d3, target="churn", exclude={"customer_id"})["target"] is None


def test_correlation_caps_the_number_of_columns():
    wide = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 40)), columns=[f"f{i}" for i in range(40)])
    assert len(correlation(wide)["columns"]) == 20


def test_missing_overview(df):
    m = missing_overview(df)
    assert [c["column"] for c in m["columns"]] == ["noise", "plan"] and m["columns"][0]["missing"] == 80
    assert m["rows_with_missing"] == int(df.isna().any(axis=1).sum()) and m["rows_with_missing_pct"] > 10
    assert missing_overview(pd.DataFrame({"a": [1, 2]}))["columns"] == []


def test_scatter_is_sampled_deterministically_and_reports_r(df):
    s = scatter(df, "x", "x2")
    assert s["n_total"] == 800 and s["n_shown"] == 800 and s["r"] > 0.95
    big = pd.concat([df] * 5, ignore_index=True)
    a, b = scatter(big, "x", "noise"), scatter(big, "x", "noise")
    assert a["n_shown"] == 800 < a["n_total"] and a == b
    assert len(a["x"]) == len(a["y"]) == 800


def test_class_balance_flags_imbalance(df):
    c = class_balance(df, "churn")
    assert c["n"] == 800 and c["imbalanced"] is True and 0 < c["minority_ratio"] < 0.2
    assert sum(i["count"] for i in c["items"]) == 800


def test_auto_picks_relevant_charts_with_reasons(df):
    specs = {s["id"]: s for s in auto_charts(df, profile_dataset(df, "churn"), "churn")}
    types = [s["type"] for s in specs.values()]
    assert types[0] == "class_balance" and "missing" in types and "correlation" in types
    assert "distribution:skewed" in specs and "log transform" in specs["distribution:skewed"]["why"]   # the skewed column, with advice
    assert "distribution:customer_id" not in specs and not any(s["params"].get("column") == "customer_id" for s in specs.values())
    assert "categories:plan" in specs and "categories:city" not in specs          # 60 distinct values: too many to chart as bars
    sc = next(s for s in specs.values() if s["type"] == "scatter")
    assert {sc["params"]["x"], sc["params"]["y"]} == {"x", "x2"} and "most strongly related" in sc["why"]
    assert all(s["why"] and s["title"] and s["data"] for s in specs.values())
    # regression target -> its distribution instead of class balance
    reg = auto_charts(df, profile_dataset(df, "skewed"), "skewed")
    assert reg[0]["type"] == "distribution" and reg[0]["params"]["column"] == "skewed"
    # no target, nothing missing, no numeric columns: still no crash
    assert auto_charts(pd.DataFrame({"a": list("abab") * 5}), profile_dataset(pd.DataFrame({"a": list("abab") * 5})))[0]["type"] == "categories"
