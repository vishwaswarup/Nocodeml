import io
import re

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from nocodeml_engine.config import (
    EncodingRule, EncodingStrategy, MissingValueRule, MissingValueStrategy, ModelConfig, PreprocessingConfig,
    ScalingStrategy, SplitConfig, SplitMethod, TaskType,
)
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.reports import ReportContext, build_report
from nocodeml_engine.training import run_experiment

from .conftest import make_config


def ctx_for(df, cfg, *, name="Churn study", current=True, number=3):
    run = run_experiment(df, cfg)
    return ReportContext(project_name=name, experiment_number=number, result=run.result, config=cfg, pipeline_label="v2",
                         dataset_filename="data.csv", dataset_version=1,
                         profile=profile_dataset(df, cfg.dataset.target_column).to_dict(), current=current), run


def read(pdf: bytes):
    r = PdfReader(io.BytesIO(pdf))
    return r, "\n".join(p.extract_text() for p in r.pages)


@pytest.fixture
def cls_ctx(churn_df, churn_config):
    cfg = churn_config.model_copy(update={"split": SplitConfig(method=SplitMethod.TRAIN_VAL_TEST, test_size=0.2,
                                                                validation_size=0.15, stratify=True)})
    return ctx_for(churn_df, cfg)[0]


def test_report_has_every_section_in_order_with_real_numbers(cls_ctx):
    pdf = build_report(cls_ctx)
    assert pdf.startswith(b"%PDF")
    reader, text = read(pdf)
    titles = ["1. Dataset overview", "2. Dataset statistics", "3. Preprocessing", "4. Feature engineering", "5. Dataset split",
              "6. Models", "7. Regularization and hyperparameters", "8. Training information", "9. Evaluation metrics",
              "10. Visualizations", "11. Model comparison", "12. Pipeline health", "13. Final pipeline configuration"]
    pos = [text.index(t) for t in titles]
    assert pos == sorted(pos), "sections out of order"
    r = cls_ctx.result
    for m in r.models:
        assert m.name in text
        t = m.metrics["test"]
        for key in ("accuracy", "f1", "roc_auc"):
            assert f"{t[key]:.3f}" in text, (m.name, key)
    assert f"{r.models[0].baseline['f1']:.3f}" in text and "Baseline" in text
    assert "Evaluated on the validation set" in text and "Evaluated on the test set" in text
    assert str(r.config_hash) in text and r.dataset_fingerprint in text and "scikit-learn" in text
    assert f"{r.quality.score}" in text and "PASS" in text
    assert "deliberately no overall" in text                      # no "best model" claim
    assert len(reader.pages) >= 5
    n_images = sum(len(p.images) for p in reader.pages)
    assert n_images >= 2 * len(r.models) + 2, n_images            # confusion+ROC per model, comparison charts
    assert '"model_key": "logistic_regression"' in text           # the config appendix


def test_report_is_deterministic(cls_ctx):
    assert build_report(cls_ctx) == build_report(cls_ctx)


def test_untrusted_text_is_escaped_not_interpreted(churn_df, churn_config):
    evil = "<b>bold</b> & <i>x</i> <font color='red'>"
    df = churn_df.rename(columns={"gender": evil})
    cfg = churn_config.model_copy(update={"preprocessing": PreprocessingConfig(
        drop_columns=["customer_id"], scaling=ScalingStrategy.STANDARD,
        missing_values=[MissingValueRule(column="age", strategy=MissingValueStrategy.MEDIAN)],
        encoding=[EncodingRule(column=evil, strategy=EncodingStrategy.ONE_HOT)])})
    c, _ = ctx_for(df, cfg, name=evil)
    _, text = read(build_report(c))
    assert "<b>bold</b>" in text and "<i>x</i>" in text           # shown literally, never rendered
    assert "&amp;" not in text and "&lt;" not in text             # and not double-escaped


def test_cross_validation_regression_report(housing_df):
    cfg = make_config(TaskType.REGRESSION, "price", [ModelConfig(model_key="ridge"), ModelConfig(model_key="random_forest")],
                      preprocessing=PreprocessingConfig(scaling=ScalingStrategy.STANDARD,
                                                        encoding=[EncodingRule(column="city", strategy=EncodingStrategy.ONE_HOT)]),
                      split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=3))
    c, run = ctx_for(housing_df, cfg)
    reader, text = read(build_report(c))
    assert "cross-validation" in text and "per cross-validation fold" in text
    assert "Evaluated on the test set" not in text
    assert "R²" in text and "RMSE" in text and "MAPE" in text
    assert "ROC" not in text.replace("ROC-AUC", "")                # no classification artefacts
    m = run.result.models[0]
    assert f"{m.metrics['cv']['r2']:.3f}" in text
    assert sum(len(p.images) for p in reader.pages) >= 2 * len(run.result.models) + 2


def test_multiclass_report_skips_roc_and_says_why():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    y = np.digitize(x + rng.normal(0, 0.4, n), [-0.5, 0.5])
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": y})
    cfg = make_config(TaskType.CLASSIFICATION, "y", [ModelConfig(model_key="random_forest")],
                      split=SplitConfig(method=SplitMethod.TRAIN_TEST, stratify=True))
    c, _ = ctx_for(df, cfg)
    _, text = read(build_report(c))
    assert "ROC curve is shown for two-class problems only" in text


def test_outdated_run_is_labelled(cls_ctx):
    cls_ctx.current = False
    _, text = read(build_report(cls_ctx))
    assert "pipeline was changed after this run" in text


def test_many_columns_are_truncated_with_a_note(churn_df, churn_config):
    wide = pd.concat([churn_df, pd.DataFrame(np.random.default_rng(0).normal(size=(len(churn_df), 40)),
                                             columns=[f"feat_{i}" for i in range(40)])], axis=1)
    cfg = churn_config.model_copy(update={"preprocessing": PreprocessingConfig(
        drop_columns=["customer_id"] + [f"feat_{i}" for i in range(40)], scaling=ScalingStrategy.STANDARD,
        missing_values=[MissingValueRule(column="age", strategy=MissingValueStrategy.MEDIAN)],
        encoding=[EncodingRule(column="gender", strategy=EncodingStrategy.ONE_HOT)])})
    c, _ = ctx_for(wide, cfg)
    _, text = read(build_report(c))
    assert re.search(r"Showing 30 of \d+ numerical columns", text)
