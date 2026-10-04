import io
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from nocodeml_engine.api.app import create_app

from .fake_supabase import FakeStore, FakeSupabase


@pytest.fixture
def env():
    store = FakeStore()
    users = {"tok-a": str(uuid.uuid4()), "tok-b": str(uuid.uuid4())}

    def factory(token):
        if token not in users:
            raise PermissionError("bad token")
        return FakeSupabase(store, users[token]), users[token]

    app = create_app(client_factory=factory, cors_origins=["http://localhost:3000"])
    with TestClient(app) as c:
        yield c, store, users


H = lambda t: {"Authorization": f"Bearer {t}"}  # noqa: E731
A, B = H("tok-a"), H("tok-b")


def csv_file(df, name="customers.csv"):
    return {"file": (name, io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}


def setup_project(c, churn_df, headers=A):
    pid = c.post("/projects", json={"name": "Churn", "task": "classification"}, headers=headers).json()["id"]
    ds = c.post(f"/projects/{pid}/datasets", files=csv_file(churn_df), data={"target": "churn"},
                headers=headers)
    assert ds.status_code == 201, ds.text
    return pid, ds.json()


def body_for(churn_config, dataset_id):
    b = churn_config.model_dump(mode="json")
    b.pop("pipeline_id"); b.pop("version")
    b["dataset"]["dataset_id"] = dataset_id
    return b


def train_and_wait(c, pid, headers=A, timeout=60):
    r = c.post(f"/projects/{pid}/training", headers=headers)
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/projects/{pid}/training/{jid}", headers=headers).json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("training job did not finish")


# ---- auth ---------------------------------------------------------------

def test_health_and_models_are_public(env):
    c, *_ = env
    assert c.get("/health").json() == {"status": "ok"}
    models = c.get("/models", params={"task": "regression"}).json()
    assert {m["key"] for m in models} == {"linear_regression", "ridge", "lasso", "decision_tree",
                                          "random_forest", "gradient_boosting"}
    rf = next(m for m in models if m["key"] == "random_forest")
    assert rf["regularization"]["kind"] == "complexity" and rf["hyperparameters"]
    assert c.get("/models", params={"task": "bogus"}).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, H("nope")])
def test_requires_valid_token(env, headers):
    c, *_ = env
    assert c.get("/projects", headers=headers).status_code == 401
    assert c.post("/projects", json={"name": "x"}, headers=headers).status_code == 401


def test_cors_only_allows_configured_origin(env):
    c, *_ = env
    ok = c.options("/projects", headers={"Origin": "http://localhost:3000",
                                         "Access-Control-Request-Method": "GET"})
    bad = c.options("/projects", headers={"Origin": "https://evil.example",
                                          "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in bad.headers


# ---- the whole workflow -------------------------------------------------

def test_full_workflow(env, churn_df, churn_config):
    c, store, _ = env
    pid, ds = setup_project(c, churn_df)
    assert c.get("/projects", headers=A).json()[0]["status"] == "Empty"

    # Section 0: dataset versions, viewer + profile
    info = c.get(f"/projects/{pid}/datasets/{ds['dataset_id']}", headers=A).json()
    assert [v["version"] for v in info["versions"]] == [1] and info["versions"][0]["n_rows"] == 400
    assert "profile" not in info["versions"][0]
    assert c.get(f"/projects/{pid}/datasets/{uuid.uuid4()}", headers=A).status_code == 404

    page = c.get(f"/projects/{pid}/datasets/{ds['dataset_id']}/rows",
                 params={"page": 2, "page_size": 25, "sort_by": "age", "descending": True},
                 headers=A).json()
    assert page["total"] == 400 and len(page["rows"]) == 25 and page["page"] == 2
    assert {x["name"] for x in page["columns"]} == {"customer_id", "age", "income", "gender", "churn"}
    found = c.get(f"/projects/{pid}/datasets/{ds['dataset_id']}/rows", params={"search": "f"},
                  headers=A).json()
    assert 0 < found["total"] < 400
    prof = c.get(f"/projects/{pid}/datasets/{ds['dataset_id']}/profile", params={"target": "churn"},
                 headers=A).json()
    assert any(w["code"] == "possible_identifier" for w in prof["warnings"])

    # Section 1-5: save pipeline, validate, then train
    body = body_for(churn_config, ds["dataset_id"])
    saved = c.put(f"/projects/{pid}/pipeline", json=body, headers=A).json()
    assert saved["created"] and saved["version"]["version"] == 1 and saved["version"]["pipeline_id"] == pid
    assert c.post(f"/projects/{pid}/pipeline/validate", headers=A).json() == {"valid": True, "issues": []}

    job = train_and_wait(c, pid)
    assert job["status"] == "succeeded" and job["experiment_number"] == 1, job
    assert c.get("/projects", headers=A).json()[0]["status"] == "Experimenting"

    exps = c.get(f"/projects/{pid}/experiments", headers=A).json()
    assert exps[0]["current"] and exps[0]["quality_score"] is not None
    assert [m["model_key"] for m in exps[0]["models"]] == ["logistic_regression", "random_forest"]
    detail = c.get(f"/projects/{pid}/experiments/1", headers=A).json()
    assert detail["current"] and detail["experiment"]["result"]["quality"]["checks"]

    # artifacts + signed link
    arts = c.get(f"/projects/{pid}/experiments/1/artifacts", headers=A).json()
    assert {"pipeline", "model", "configuration", "metrics"} <= {a["kind"] for a in arts}
    url = c.get(f"/projects/{pid}/artifacts/{arts[0]['id']}/url", headers=A).json()
    assert url["url"].startswith("https://") and url["expires_in"] == 300

    # Modify pipeline -> impact reported, old result stale
    body["preprocessing"]["scaling"] = "min_max"
    edited = c.put(f"/projects/{pid}/pipeline", json=body, headers=A).json()
    assert edited["version"]["version"] == 2 and not edited["created"]
    assert edited["impact"]["changed"] == ["preprocessing"] and "no longer apply" in edited["impact"]["message"]
    assert c.get(f"/projects/{pid}/experiments", headers=A).json()[0]["current"] is False
    assert c.post(f"/projects/{pid}/pipeline/finalize", headers=A).status_code == 409  # no current run

    job2 = train_and_wait(c, pid)
    assert job2["experiment_number"] == 2
    cmp = c.get(f"/projects/{pid}/compare", params={"a": 1, "b": 2}, headers=A).json()
    assert "f1" in cmp["metrics"]["random_forest"]
    assert [d["same"] for d in cmp["config_diff"] if d["section"] == "preprocessing"] == [False]

    fin = c.post(f"/projects/{pid}/pipeline/finalize", headers=A).json()
    assert fin["status"] == "finalized"
    assert c.get("/projects", headers=A).json()[0]["pipeline_version"] == "v2.0"
    assert len(c.get(f"/projects/{pid}/pipeline/versions", headers=A).json()) == 2

    # delete cleans up rows and stored files
    assert c.delete(f"/projects/{pid}", headers=A).status_code == 204
    assert c.get("/projects", headers=A).json() == [] and store.objects == {}


# ---- isolation ----------------------------------------------------------

def test_other_user_gets_404_everywhere(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
    train_and_wait(c, pid)
    d = ds["dataset_id"]
    probes = [("get", f"/projects/{pid}"), ("delete", f"/projects/{pid}"),
              ("get", f"/projects/{pid}/pipeline"), ("get", f"/projects/{pid}/experiments"),
              ("get", f"/projects/{pid}/experiments/1"), ("post", f"/projects/{pid}/training"),
              ("get", f"/projects/{pid}/datasets/{d}/rows"), ("get", f"/projects/{pid}/datasets/{d}"), ("get", f"/projects/{pid}/datasets/{d}/profile"),
              ("post", f"/projects/{pid}/pipeline/finalize"), ("get", f"/projects/{pid}/compare?a=1&b=1")]
    for method, path in probes:
        r = getattr(c, method)(path, headers=B)
        assert r.status_code == 404, (method, path, r.status_code)
        assert "Churn" not in r.text
    assert c.post(f"/projects/{pid}/datasets", files=csv_file(churn_df), headers=B).status_code == 404
    assert c.get("/projects", headers=B).json() == []
    # the owner can still see everything
    assert c.get(f"/projects/{pid}", headers=A).status_code == 200


def test_path_ids_must_be_uuids(env):
    c, *_ = env
    assert c.get("/projects/not-a-uuid", headers=A).status_code == 422


# ---- validation and failure modes --------------------------------------

def test_upload_rejections(env, churn_df, monkeypatch):
    c, store, _ = env
    pid = c.post("/projects", json={"name": "p"}, headers=A).json()["id"]
    r = c.post(f"/projects/{pid}/datasets", files={"file": ("a.xlsx", io.BytesIO(b"x"), "application/x")}, headers=A)
    assert r.status_code == 400 and "Only .csv" in r.json()["detail"]
    r = c.post(f"/projects/{pid}/datasets", files={"file": ("a.csv", io.BytesIO(b"a,b\n"), "text/csv")}, headers=A)
    assert r.status_code == 400
    import nocodeml_engine.api.app as appmod
    import nocodeml_engine.persistence.projects as pm
    monkeypatch.setattr(appmod, "MAX_DATASET_BYTES", 10)
    monkeypatch.setattr(pm, "MAX_DATASET_BYTES", 10)
    r = c.post(f"/projects/{pid}/datasets", files=csv_file(churn_df), headers=A)
    assert r.status_code == 400 and "limit" in r.json()["detail"]
    assert store.objects == {}


def test_invalid_pipeline_is_reported_not_trained(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    body = body_for(churn_config, ds["dataset_id"])
    body["preprocessing"]["missing_values"] = []  # 'age' has NaNs with no strategy
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    rep = c.post(f"/projects/{pid}/pipeline/validate", headers=A).json()
    assert not rep["valid"] and any("'age' has missing" in i for i in rep["issues"])
    job = train_and_wait(c, pid)
    assert job["status"] == "failed" and any("'age'" in i for i in job["issues"])
    assert c.get(f"/projects/{pid}/experiments", headers=A).json() == []


def test_pipeline_rejects_foreign_dataset_and_bad_models(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    body = body_for(churn_config, str(uuid.uuid4()))
    assert c.put(f"/projects/{pid}/pipeline", json=body, headers=A).status_code == 422
    body = body_for(churn_config, ds["dataset_id"])
    body["models"] = [{"model_key": "random_forest"}] * 6
    assert c.put(f"/projects/{pid}/pipeline", json=body, headers=A).status_code == 422
    body["models"] = [{"model_key": "ridge"}]  # regression model on a classification task
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    job = train_and_wait(c, pid)
    assert job["status"] == "failed" and "does not support classification" in job["error"]
    body["models"] = []
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    r = c.post(f"/projects/{pid}/training", headers=A)
    assert r.status_code == 422


def test_only_one_training_job_per_project(env, churn_df, churn_config, monkeypatch):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
    import nocodeml_engine.api.app as appmod
    gate = threading.Event()
    real = appmod.run_experiment
    monkeypatch.setattr(appmod, "run_experiment", lambda *a, **k: (gate.wait(10), real(*a, **k))[1])
    first = c.post(f"/projects/{pid}/training", headers=A)
    assert first.status_code == 202
    assert c.post(f"/projects/{pid}/training", headers=A).status_code == 409
    gate.set()
    jid = first.json()["id"]
    for _ in range(300):
        if c.get(f"/projects/{pid}/training/{jid}", headers=A).json()["status"] == "succeeded":
            break
        time.sleep(0.1)
    else:
        raise AssertionError("job never finished")
    assert c.post(f"/projects/{pid}/training", headers=A).status_code == 202  # free again


def test_job_ids_are_private(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
    jid = c.post(f"/projects/{pid}/training", headers=A).json()["id"]
    assert c.get(f"/projects/{pid}/training/{jid}", headers=B).status_code == 404
    for _ in range(300):  # let the job finish so it can't leak into other tests
        if c.get(f"/projects/{pid}/training/{jid}", headers=A).json()["status"] != "queued" and \
                c.get(f"/projects/{pid}/training/{jid}", headers=A).json()["status"] != "running":
            break
        time.sleep(0.1)
    pid_b = c.post("/projects", json={"name": "mine"}, headers=B).json()["id"]
    assert c.get(f"/projects/{pid_b}/training/{jid}", headers=B).status_code == 404  # wrong project too


def test_unexpected_errors_do_not_leak_details(env, churn_df, monkeypatch):
    c, *_ = env
    import nocodeml_engine.api.app as appmod
    monkeypatch.setattr(appmod.ProjectService, "list_projects",
                        lambda self: (_ for _ in ()).throw(RuntimeError("secret-db-password-xyz")))
    c2 = TestClient(c.app, raise_server_exceptions=False)
    r = c2.get("/projects", headers=A)
    assert r.status_code == 500 and "secret" not in r.text and r.json()["detail"] == "Internal server error."


def test_experiment_number_race_is_retried(env, churn_df, churn_config):
    """If another run grabs the same number first, the next free one is used."""
    c, store, _ = env
    pid, ds = setup_project(c, churn_df)
    c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
    from nocodeml_engine.persistence.repository import SupabasePipelineRepository as Repo
    orig_save = Repo.save_experiment
    calls = {"n": 0}

    def flaky(self, e):
        if e.pipeline_id != pid:
            return orig_save(self, e)
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("duplicate key (simulated concurrent insert)")
        return orig_save(self, e)
    Repo.save_experiment = flaky
    try:
        job = train_and_wait(c, pid)
    finally:
        Repo.save_experiment = orig_save
    assert job["status"] == "succeeded" and job["experiment_number"] == 1 and calls["n"] == 2


# ---- Section 1: recommendations + preview ------------------------------------

def test_preprocessing_recommendations_and_preview(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    assert c.get(f"/projects/{pid}/pipeline/recommendations/preprocessing", headers=A).status_code == 409

    body = body_for(churn_config, ds["dataset_id"])
    body["preprocessing"] = {}
    body["models"] = []
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    recs = c.get(f"/projects/{pid}/pipeline/recommendations/preprocessing", headers=A).json()
    ids = {r["id"] for r in recs}
    assert {"drop_column:customer_id", "impute:age", "encode:gender", "scale:*"} <= ids
    assert all(r["reason"] and r["action"]["type"] for r in recs)

    # the unsaved raw draft is reported as not trainable, with reasons
    bad = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert bad["after"] is None and any("'age' has missing" in i for i in bad["issues"])

    # applying the recommendations (as the UI does) gives a valid preview
    body["preprocessing"] = {
        "drop_columns": ["customer_id"],
        "missing_values": [{"column": "age", "strategy": "median"}],
        "encoding": [{"column": "gender", "strategy": "one_hot"}], "scaling": "standard"}
    good = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert good["issues"] == [] and good["after"]["missing_cells"] == 0
    assert good["before"]["missing_cells"] > 0 and good["after"]["columns"] == 4  # age, income, gender_F/M
    # previewing saves nothing
    assert c.get(f"/projects/{pid}/pipeline", headers=A).json()["version"] == 1
    # foreign users get 404
    for m, path in (("get", f"/projects/{pid}/pipeline/recommendations/preprocessing"),):
        assert getattr(c, m)(path, headers=B).status_code == 404
    assert c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=B).status_code == 404


def test_split_recommendations_model_defaults_and_split_preview(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    for path in ("pipeline/recommendations/split", "pipeline/model-defaults"):
        assert c.get(f"/projects/{pid}/{path}", headers=A).status_code == 409
    body = body_for(churn_config, ds["dataset_id"])
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)

    recs = c.get(f"/projects/{pid}/pipeline/recommendations/split", headers=A).json()
    ids = {r["id"] for r in recs}
    assert "split_method" in ids and all(r["reason"] and r["action"]["type"] == "split" for r in recs)
    d = c.get(f"/projects/{pid}/pipeline/model-defaults", headers=A).json()
    assert set(d) == {"logistic_regression", "random_forest"}
    assert d["random_forest"]["recommended"]["n_estimators"] == 100 and d["random_forest"]["defaults"]["bootstrap"] is True

    pv = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert pv["issues"] == [] and pv["split"]["n_train"] + pv["split"]["n_test"] == 400
    body["split"] = {"method": "train_test", "test_size": 5}
    bad = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert bad["issues"] and bad["split"] is None and any("test_size" in i for i in bad["split_issues"])
    # raw preprocessing (not ready) still previews the split on its own
    body["split"] = {"method": "train_test", "test_size": 0.25}
    body["preprocessing"] = {}
    raw = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert raw["after"] is None and raw["split"]["n_test"] == 100 and raw["split_issues"] == []
    for path in ("pipeline/recommendations/split", "pipeline/model-defaults"):
        assert c.get(f"/projects/{pid}/{path}", headers=B).status_code == 404


def test_model_issues_in_preview_and_active_job(env, churn_df, churn_config, monkeypatch):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    body = body_for(churn_config, ds["dataset_id"])
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    assert c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()["model_issues"] == {}

    bad = dict(body, models=[
        {"model_key": "random_forest", "regularization": {"type": "l2"}},               # L1/L2 on a tree
        {"model_key": "knn", "hyperparameters": {"n_neighbors": 0}},                    # out of range
        {"model_key": "logistic_regression", "regularization": {"type": "l1", "strength": -1}},  # sklearn rejects
        {"model_key": "ridge"}, ])                                                       # wrong task
    mi = c.post(f"/projects/{pid}/pipeline/preview", json=bad, headers=A).json()["model_issues"]
    assert "not an L1/L2" in mi["random_forest"][0] and "within" in mi["knn"][0]
    assert "C" in mi["logistic_regression"][0] and "does not support classification" in mi["ridge"][0]

    vm = c.post(f"/projects/{pid}/pipeline/validate-models", json=bad, headers=A).json()["model_issues"]
    assert vm == mi                                                   # same answer, without loading data
    assert c.post(f"/projects/{pid}/pipeline/validate-models", json=body, headers=A).json() == {"model_issues": {}}
    assert c.post(f"/projects/{pid}/pipeline/validate-models", json=bad, headers=B).status_code == 404

    six = dict(body, models=[{"model_key": "random_forest"}] * 5 + [{"model_key": "knn"}])
    c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=B)  # (foreign: 404, covered elsewhere)
    assert c.post(f"/projects/{pid}/pipeline/preview", json=six, headers=A).status_code == 422  # schema caps at 5

    # active job is visible to a re-attaching page, and only to its owner
    assert c.get(f"/projects/{pid}/training/active", headers=A).json() is None
    import threading
    import nocodeml_engine.api.app as appmod
    gate = threading.Event(); real = appmod.run_experiment
    monkeypatch.setattr(appmod, "run_experiment", lambda *a, **k: (gate.wait(10), real(*a, **k))[1])
    jid = c.post(f"/projects/{pid}/training", headers=A).json()["id"]
    act = c.get(f"/projects/{pid}/training/active", headers=A).json()
    assert act["id"] == jid and act["status"] in ("queued", "running")
    assert c.get(f"/projects/{pid}/training/active", headers=B).status_code == 404
    gate.set()
    for _ in range(300):
        if c.get(f"/projects/{pid}/training/active", headers=A).json() is None:
            break
        time.sleep(0.1)
    assert c.get(f"/projects/{pid}/training/{jid}", headers=A).json()["status"] == "succeeded"


def test_hyperparameter_key_order_does_not_break_experiment_recording(env, churn_df, churn_config):
    """Postgres jsonb re-orders object keys. A config with several hyperparameters/regularization keys
    must still hash identically after the round trip, or training fails to record its experiment."""
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    body = body_for(churn_config, ds["dataset_id"])
    # keys deliberately NOT in jsonb order (jsonb sorts by length, then alphabetically)
    body["models"] = [
        {"model_key": "random_forest", "hyperparameters": {"n_estimators": 30, "max_depth": 3, "bootstrap": True}},
        {"model_key": "logistic_regression", "regularization": {"type": "elasticnet", "strength": 0.5, "l1_ratio": 0.3}},
    ]
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    job = train_and_wait(c, pid)
    assert job["status"] == "succeeded", job
    exp = c.get(f"/projects/{pid}/experiments", headers=A).json()
    assert len(exp) == 1 and exp[0]["current"] is True       # and it is still recognised as current afterwards
