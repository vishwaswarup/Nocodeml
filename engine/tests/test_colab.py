import copy
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.colab.bundle import build_bundle
from nocodeml_engine.colab.importer import ResultsError, import_results
from nocodeml_engine.config import (
    EncodingRule, EncodingStrategy, ModelConfig, OutlierRule, OutlierStrategy, PreprocessingConfig, ScalingStrategy,
    SearchConfig, SplitConfig, SplitMethod, TaskType,
)
from nocodeml_engine.models.registry import XGBOOST_AVAILABLE
from nocodeml_engine.training import run_experiment

from .conftest import make_config

RUNNER = Path(__file__).parents[1] / "nocodeml_engine" / "colab" / "colab_runner.py"


def load_runner():
    """The runner exactly as a Colab user gets it: loaded from its own file, with nothing from the engine."""
    spec = importlib.util.spec_from_file_location("colab_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def colab_run(df, cfg, tmp_path):
    """bundle -> unzip -> run the notebook code -> results file (through JSON, as in real life) -> imported experiment."""
    data, manifest = build_bundle(df, cfg)
    d = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(d)
    runner = load_runner()
    results = runner.run(str(d), log=lambda m: None)
    path = runner.save(results, str(tmp_path / "nocodeml_results.json"))
    results = json.loads(Path(path).read_text())
    return import_results(df, cfg, results), results, manifest, d


def assert_same_metrics(server, colab, splits=("train", "test")):
    for ms, mc in zip(server.models, colab.models):
        assert ms.model_key == mc.model_key
        for split in splits:
            if split not in ms.metrics:
                continue
            for key, v in ms.metrics[split].items():
                got = mc.metrics[split][key]
                if isinstance(v, float):
                    assert got == pytest.approx(v, rel=1e-9, abs=1e-12), (ms.model_key, split, key)
                else:
                    assert got == v, (ms.model_key, split, key)
        assert mc.baseline == pytest.approx(ms.baseline) if all(isinstance(x, float) for x in ms.baseline.values()) else mc.baseline == ms.baseline
        assert mc.hyperparameters == ms.hyperparameters
        assert mc.fold_primary_scores == pytest.approx(ms.fold_primary_scores)


def test_runner_is_self_contained_plain_scikit_learn():
    src = RUNNER.read_text()
    assert "nocodeml_engine" not in src.replace("NoCodeML", "")
    assert "import nocodeml" not in src and "from nocodeml" not in src


def test_holdout_classification_matches_server_training_exactly(churn_df, churn_config, tmp_path):
    server = run_experiment(churn_df, churn_config).result
    colab, results, manifest, _ = colab_run(churn_df, churn_config, tmp_path)
    assert colab.source == "colab" and server.source == "server"
    assert colab.environment["trained_on"] == "Google Colab"
    assert_same_metrics(server, colab, ("train", "test"))
    assert {c.id: c.status for c in colab.quality.checks} == {c.id: c.status for c in server.quality.checks}
    assert colab.quality.score == server.quality.score
    assert colab.split["n_train"] == server.split["n_train"] and colab.config_hash == server.config_hash
    assert set(results["models"]) == {"logistic_regression", "random_forest"}
    assert not any(k in json.dumps(results) for k in ("roc_auc", "accuracy", "f1\""))        # predictions only, no metrics


def test_train_validation_test_split_matches(churn_df, churn_config, tmp_path):
    cfg = churn_config.model_copy(update={"split": SplitConfig(method=SplitMethod.TRAIN_VAL_TEST, test_size=0.2,
                                                                validation_size=0.15, stratify=True)})
    server = run_experiment(churn_df, cfg).result
    colab, _, manifest, d = colab_run(churn_df, cfg, tmp_path)
    assert_same_metrics(server, colab, ("train", "validation", "test"))
    assert (d / "fold_0_validation.csv").exists()


def test_cross_validation_regression_matches(housing_df, tmp_path):
    cfg = make_config(TaskType.REGRESSION, "price", [ModelConfig(model_key="ridge"), ModelConfig(model_key="random_forest")],
                      preprocessing=PreprocessingConfig(scaling=ScalingStrategy.STANDARD,
                                                        encoding=[EncodingRule(column="city", strategy=EncodingStrategy.ONE_HOT)]),
                      split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=3))
    server = run_experiment(housing_df, cfg).result
    colab, _, manifest, d = colab_run(housing_df, cfg, tmp_path)
    assert manifest["split"]["mode"] == "cv" and len(manifest["folds"]) == 3
    assert_same_metrics(server, colab, ("cv",))
    for ms, mc in zip(server.models, colab.models):
        assert mc.metrics["train"]["r2"] == pytest.approx(ms.metrics["train"]["r2"])


def test_outlier_rows_are_excluded_from_fitting_only(churn_df, churn_config, tmp_path):
    cfg = churn_config.model_copy(update={"preprocessing": churn_config.preprocessing.model_copy(update={
        "outliers": [OutlierRule(column="income", strategy=OutlierStrategy.IQR, threshold=1.5)]})})
    server = run_experiment(churn_df, cfg).result
    colab, _, manifest, d = colab_run(churn_df, cfg, tmp_path)
    assert server.models[0].n_outlier_rows_removed > 0
    assert colab.models[0].n_outlier_rows_removed == server.models[0].n_outlier_rows_removed
    train = pd.read_csv(d / "fold_0_train.csv")
    assert (train["__fit"] == 0).sum() == server.models[0].n_outlier_rows_removed
    assert not (pd.read_csv(d / "fold_0_test.csv").columns == "__fit").any()          # test rows are never filtered
    assert_same_metrics(server, colab, ("train", "test"))


def test_bundle_preprocessing_is_learned_from_training_rows_only(churn_df, churn_config, tmp_path):
    _, _, manifest, d = colab_run(churn_df, churn_config, tmp_path)
    train, test = pd.read_csv(d / "fold_0_train.csv"), pd.read_csv(d / "fold_0_test.csv")
    assert abs(train["income"][train["__fit"] == 1].mean()) < 1e-9        # standard scaling: centred on the TRAINING rows
    assert abs(test["income"].mean()) > 1e-3                              # the test rows were only transformed, not re-centred
    assert train["age"].notna().all() and test["age"].notna().all()       # imputed
    ids = set(train["__row_id"]) | set(test["__row_id"])
    assert len(ids) == len(train) + len(test)                             # no row is in both files
    assert "churn" not in manifest["feature_columns"] and "customer_id" not in manifest["feature_columns"]


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
def test_xgboost_with_a_search_runs_in_the_notebook_and_is_labelled(churn_df, churn_config, tmp_path):
    cfg = churn_config.model_copy(update={"models": [
        ModelConfig(model_key="xgboost", search=SearchConfig(method="grid", space={"max_depth": [2, 4], "n_estimators": [20, 40]})),
        ModelConfig(model_key="knn", search=SearchConfig(method="random", n_iter=3, space={"n_neighbors": [3, 5, 9, 15]}))]})
    colab, results, manifest, _ = colab_run(churn_df, cfg, tmp_path)
    x = colab.models[0]
    assert x.search["n_candidates"] == 4 and x.search["reported_by"] == "colab"
    assert x.hyperparameters["max_depth"] == x.search["best_params"]["max_depth"]
    ids = {c.id for c in colab.quality.checks}
    assert "tuning_preprocessing_scope" in ids
    assert 0 <= x.metrics["test"]["f1"] <= 1 and x.metrics["test"]["roc_auc"] is not None
    assert colab.quality.score < 100                                       # the extra warning costs something


def test_multiclass_and_scoreless_models(tmp_path):
    rng = np.random.default_rng(3)
    n = 240
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = np.select([df.a > 0.5, df.a < -0.5], ["hi", "lo"], "mid")
    cfg = make_config(TaskType.CLASSIFICATION, "y", [ModelConfig(model_key="svm"), ModelConfig(model_key="random_forest")],
                      preprocessing=PreprocessingConfig(scaling=ScalingStrategy.STANDARD),
                      split=SplitConfig(method=SplitMethod.TRAIN_TEST, stratify=True))
    server = run_experiment(df, cfg).result
    colab, results, _, _ = colab_run(df, cfg, tmp_path)
    assert_same_metrics(server, colab, ("train", "test"))
    assert results["models"]["svm"]["folds"][0]["splits"]["test"]["score"] is None      # multiclass SVM offers no score


# ---- the server never trusts the file ------------------------------------------------------------------------------

@pytest.fixture
def good(churn_df, churn_config, tmp_path):
    _, results, _, _ = colab_run(churn_df, churn_config, tmp_path)
    return churn_df, churn_config, results


def bad(good, mutate, match):
    df, cfg, results = good
    r = copy.deepcopy(results)
    mutate(r)
    with pytest.raises(ResultsError, match=match):
        import_results(df, cfg, r)


def test_tampered_results_are_rejected_with_a_clear_message(good):
    first = lambda r: r["models"]["logistic_regression"]["folds"][0]["splits"]  # noqa: E731
    bad(good, lambda r: r.update(format=2), "isn't a NoCodeML results file")
    bad(good, lambda r: r.update(config_hash="deadbeef"), "different version of your pipeline")
    bad(good, lambda r: r.update(dataset_fingerprint="x"), "different dataset")
    bad(good, lambda r: r["models"].pop("random_forest"), "exactly these models")
    bad(good, lambda r: r["models"].update(extra=r["models"]["random_forest"]), "exactly these models")
    bad(good, lambda r: first(r)["test"]["pred"].pop(), "expected \\d+ values")
    bad(good, lambda r: first(r)["test"]["pred"].__setitem__(0, 7), "outside the 2 known classes")
    bad(good, lambda r: first(r)["test"]["pred"].__setitem__(0, 0.5), "whole numbers")
    bad(good, lambda r: first(r)["test"]["pred"].__setitem__(0, "x"), "not numbers")
    bad(good, lambda r: first(r)["test"]["score"].__setitem__(0, float("inf")), "NaN or infinite")
    bad(good, lambda r: first(r)["test"].update(score=[[0.1, 0.9]] * len(first(r)["test"]["pred"])), "wrong shape")
    bad(good, lambda r: first(r).pop("test"), "no 'test' predictions")
    bad(good, lambda r: r["row_ids"]["0"]["test"].reverse(), "rows in the 'test' predictions don't match")
    bad(good, lambda r: r["row_ids"]["0"].pop("test"), "expected splits")
    bad(good, lambda r: r["models"]["random_forest"].update(seconds="slow"), "not a number")
    bad(good, lambda r: r["models"]["random_forest"].update(seconds=-5), "not valid")
    bad(good, lambda r: r["models"]["random_forest"]["folds"][0]["splits"]["train"].update(pred=None), "expected \\d+ values")
    bad(good, lambda r: r["models"]["random_forest"]["folds"][0].update(search={"best_params": {}, "candidates": []}),
        "not configured")
    bad(good, lambda r: r["models"]["random_forest"].update(folds=[]), "expected results for 1 fold")


def test_a_fabricated_perfect_score_is_impossible_because_only_predictions_are_accepted(good):
    df, cfg, results = good
    r = copy.deepcopy(results)
    r["models"]["random_forest"]["metrics"] = {"test": {"f1": 1.0}}               # extra claims are simply ignored
    imported = import_results(df, cfg, r)
    assert imported.models[1].metrics["test"]["f1"] < 1.0
    assert "metrics" not in imported.models[1].model_dump()["search"] if imported.models[1].search else True


def test_search_summary_must_stay_inside_the_searched_values(churn_df, churn_config, tmp_path):
    cfg = churn_config.model_copy(update={"models": [ModelConfig(model_key="random_forest", search=SearchConfig(
        method="grid", space={"max_depth": [2, 4]}))]})
    _, results, _, _ = colab_run(churn_df, cfg, tmp_path)
    r = copy.deepcopy(results)
    r["models"]["random_forest"]["folds"][0]["search"]["best_params"]["max_depth"] = 99
    with pytest.raises(ResultsError, match="not one of the values"):
        import_results(churn_df, cfg, r)
    r = copy.deepcopy(results)
    r["models"]["random_forest"]["folds"][0]["search"]["candidates"][0]["mean_score"] = "nan?"
    with pytest.raises(ResultsError, match="malformed|invalid"):
        import_results(churn_df, cfg, r)


# ---- over HTTP -------------------------------------------------------------------------------------------------

def _client():
    import uuid
    from fastapi.testclient import TestClient
    from nocodeml_engine.api.app import create_app
    from .fake_supabase import FakeStore, FakeSupabase
    store = FakeStore()
    users = {"tok-a": str(uuid.uuid4()), "tok-b": str(uuid.uuid4())}

    def factory(t):
        if t not in users:
            raise PermissionError("bad")
        return FakeSupabase(store, users[t]), users[t]

    return TestClient(create_app(client_factory=factory, client_builder=lambda t: FakeSupabase(store, users[t])))


def test_full_colab_flow_over_http(churn_df, churn_config, tmp_path):
    from .test_api import A, B, body_for, setup_project
    with _client() as c:
        pid, ds = setup_project(c, churn_df)
        assert c.post(f"/projects/{pid}/colab/bundle", headers=A).status_code == 404            # no pipeline saved yet
        c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
        r = c.post(f"/projects/{pid}/colab/bundle", headers=A)
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert "nocodeml_bundle_v1.zip" in r.headers["content-disposition"]
        assert c.post(f"/projects/{pid}/colab/bundle", headers=B).status_code == 404            # someone else's project
        d = tmp_path / "b"
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(d)
        runner = load_runner()
        res = runner.run(str(d), log=lambda m: None)
        up = c.post(f"/projects/{pid}/colab/results", headers=A,
                    files={"file": ("nocodeml_results.json", io.BytesIO(json.dumps(res).encode()), "application/json")})
        assert up.status_code == 201, up.text
        n = up.json()["experiment_number"]
        exp = c.get(f"/projects/{pid}/experiments/{n}", headers=A).json()
        assert exp["experiment"]["result"]["source"] == "colab" and exp["current"] is True
        assert c.get(f"/projects/{pid}/experiments", headers=A).json()[0]["models"][0]["value"] is not None
        # the same file is refused for another user, and bad uploads give clear errors
        assert c.post(f"/projects/{pid}/colab/results", headers=B,
                      files={"file": ("r.json", io.BytesIO(b"{}"), "application/json")}).status_code == 404
        bad_json = c.post(f"/projects/{pid}/colab/results", headers=A, files={"file": ("r.json", io.BytesIO(b"not json"), "application/json")})
        assert bad_json.status_code == 422 and "isn't valid JSON" in bad_json.json()["detail"]
        wrong = c.post(f"/projects/{pid}/colab/results", headers=A, files={"file": ("r.json", io.BytesIO(b"{}"), "application/json")})
        assert wrong.status_code == 422 and "isn't a NoCodeML results file" in wrong.json()["detail"]
        # changing the pipeline after downloading the bundle makes the results stale
        body = body_for(churn_config, ds["dataset_id"])
        body["split"]["test_size"] = 0.3
        assert c.put(f"/projects/{pid}/pipeline", json=body, headers=A).status_code == 200
        stale = c.post(f"/projects/{pid}/colab/results", headers=A,
                       files={"file": ("r.json", io.BytesIO(json.dumps(res).encode()), "application/json")})
        assert stale.status_code == 422 and "different version of your pipeline" in stale.json()["detail"]


def test_bundle_refuses_an_untrainable_pipeline_and_oversized_results(churn_df, churn_config, monkeypatch):
    from .test_api import A, body_for, setup_project
    with _client() as c:
        pid, ds = setup_project(c, churn_df)
        body = body_for(churn_config, ds["dataset_id"])
        body["preprocessing"]["missing_values"] = []                    # 'age' has gaps and no rule: can't train
        c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
        r = c.post(f"/projects/{pid}/colab/bundle", headers=A)
        assert r.status_code == 422 and r.json()["issues"]
    monkeypatch.setenv("NOCODEML_MAX_RESULTS_MB", "1")
    with _client() as c:
        pid, ds = setup_project(c, churn_df)
        big = io.BytesIO(b" " * (2 * 2**20))
        r = c.post(f"/projects/{pid}/colab/results", headers=A, files={"file": ("r.json", big, "application/json")})
        assert r.status_code == 413


# ---- the notebook ----------------------------------------------------------------------------------------------

def test_notebook_is_in_sync_with_the_runner_and_is_valid(tmp_path):
    from nocodeml_engine.colab.notebook import build_notebook, render
    committed = Path(__file__).parents[2] / "web" / "public" / "nocodeml-colab.ipynb"
    assert committed.read_text() == render(), "run: python -m nocodeml_engine.colab.notebook ../web/public/nocodeml-colab.ipynb"
    nb = json.loads(committed.read_text())
    assert nb["nbformat"] == 4 and [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code", "code"]
    assert len({c["id"] for c in nb["cells"]}) == 4
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "cell", "exec")
    runner_cell = "".join(nb["cells"][2]["source"])
    assert RUNNER.read_text().rstrip("\n") in runner_cell                      # the notebook contains the tested code verbatim


def test_the_notebook_cells_work_end_to_end_outside_colab(churn_df, churn_config, tmp_path, monkeypatch):
    """Execute the notebook's own cells (the upload prompt falls back to a path) and import what they produce."""
    from nocodeml_engine.colab.notebook import build_notebook
    data, _ = build_bundle(churn_df, churn_config)
    zip_path = tmp_path / "bundle.zip"
    zip_path.write_bytes(data)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": str(zip_path))
    ns: dict = {}
    for cell in build_notebook()["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)
    produced = json.loads((tmp_path / "nocodeml_results.json").read_text())
    assert import_results(churn_df, churn_config, produced).source == "colab"
