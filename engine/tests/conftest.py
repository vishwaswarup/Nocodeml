import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.config import (
    DatasetRef, EncodingRule, EncodingStrategy, MissingValueRule, MissingValueStrategy,
    ModelConfig, PipelineConfig, PreprocessingConfig, ScalingStrategy, SplitConfig, SplitMethod,
    TaskType,
)


@pytest.fixture
def churn_df():
    rng = np.random.default_rng(0)
    n = 400
    age = rng.normal(40, 12, n)
    income = rng.lognormal(10, 0.5, n)
    gender = rng.choice(["M", "F"], n)
    logit = 0.04 * (age - 40) - 0.00002 * (income - 25000) + (gender == "M") * 0.3
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit - 1.5))).astype(int)
    df = pd.DataFrame({"customer_id": np.arange(1000, 1000 + n), "age": age, "income": income,
                       "gender": gender, "churn": churn})
    df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    return df


@pytest.fixture
def housing_df():
    rng = np.random.default_rng(1)
    n = 300
    sqft = rng.uniform(500, 3000, n)
    rooms = rng.integers(1, 7, n)
    city = rng.choice(["a", "b", "c"], n)
    price = 100 * sqft + 5000 * rooms + (city == "a") * 20000 + rng.normal(0, 8000, n)
    return pd.DataFrame({"sqft": sqft, "rooms": rooms, "city": city, "price": price})


def make_config(task, target, models, **kw):
    return PipelineConfig(pipeline_id="p1", dataset=DatasetRef(
        dataset_id="d1", target_column=target, task=task), models=models, **kw)


@pytest.fixture
def churn_config():
    return make_config(
        TaskType.CLASSIFICATION, "churn",
        [ModelConfig(model_key="logistic_regression"), ModelConfig(model_key="random_forest")],
        preprocessing=PreprocessingConfig(
            drop_columns=["customer_id"],
            missing_values=[MissingValueRule(column="age", strategy=MissingValueStrategy.MEDIAN)],
            encoding=[EncodingRule(column="gender", strategy=EncodingStrategy.ONE_HOT)],
            scaling=ScalingStrategy.STANDARD),
        split=SplitConfig(method=SplitMethod.TRAIN_TEST, stratify=True))
