import io
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from nocodeml_engine.api.app import create_app
from nocodeml_engine.models.registry import XGBOOST_AVAILABLE

from .fake_supabase import FakeStore, FakeSupabase


@pytest.fixture
def env():
    store = FakeStore()
    users = {"tok-a": str(uuid.uuid4()), "tok-b": str(uuid.uuid4())}

    def factory(token):
        if token not in users:
            raise PermissionError("bad token")
        return FakeSupabase(store, users[token]), users[token]

    app = create_app(client_factory=factory, cors_origins=["http://localhost:3000"],
                     client_builder=lambda t: FakeSupabase(store, users[t]))
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
                                          "random_forest", "gradient_boosting",
                                          *(["xgboost"] if XGBOOST_AVAILABLE else [])}
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


def test_pdf_report_generation_storage_and_access(env, churn_df, churn_config):
    import io as _io
    from pypdf import PdfReader
    c, store, _ = env
    pid, ds = setup_project(c, churn_df)
    c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
    assert c.post(f"/projects/{pid}/experiments/1/report", headers=A).status_code == 404     # nothing trained yet
    train_and_wait(c, pid)

    r = c.post(f"/projects/{pid}/experiments/1/report", headers=A)
    assert r.status_code == 201, r.text
    art = r.json()
    assert art["kind"] == "report" and art["bucket"] == "reports" and art["storage_path"].endswith("report_exp1.pdf")
    pdf = store.objects[("reports", art["storage_path"])]
    assert pdf.startswith(b"%PDF") and art["size_bytes"] == len(pdf)
    text = "\n".join(p.extract_text() for p in PdfReader(_io.BytesIO(pdf)).pages)
    assert "Churn" in text and "customers.csv" in text and "Experiment #1" in text and "14. Final pipeline configuration" in text

    # listed with the other artifacts, downloadable through a signed link
    arts = c.get(f"/projects/{pid}/experiments/1/artifacts", headers=A).json()
    assert [a["kind"] for a in arts].count("report") == 1
    assert c.get(f"/projects/{pid}/artifacts/{art['id']}/url", headers=A).json()["url"].startswith("https://")

    # idempotent: asking again returns the same report; force=true replaces it (still exactly one)
    again = c.post(f"/projects/{pid}/experiments/1/report", headers=A)
    assert again.status_code == 200 and again.json()["id"] == art["id"]
    forced = c.post(f"/projects/{pid}/experiments/1/report?force=true", headers=A)
    assert forced.status_code == 201 and forced.json()["id"] != art["id"]
    kinds = [a["kind"] for a in c.get(f"/projects/{pid}/experiments/1/artifacts", headers=A).json()]
    assert kinds.count("report") == 1 and sum(1 for k in store.objects if k[0] == "reports") == 1

    # outdated runs say so
    body = body_for(churn_config, ds["dataset_id"]); body["preprocessing"]["scaling"] = "min_max"
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)
    stale = c.post(f"/projects/{pid}/experiments/1/report?force=true", headers=A).json()
    text2 = "\n".join(p.extract_text() for p in PdfReader(_io.BytesIO(store.objects[("reports", stale["storage_path"])])).pages)
    assert "pipeline was changed after this run" in text2

    # access control
    assert c.post(f"/projects/{pid}/experiments/1/report", headers=B).status_code == 404
    assert c.post(f"/projects/{pid}/experiments/99/report", headers=A).status_code == 404
    assert c.post(f"/projects/{pid}/experiments/1/report").status_code == 401


def test_feature_recommendations_and_preview_over_http(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    assert c.get(f"/projects/{pid}/pipeline/recommendations/features", headers=A).status_code == 409
    body = body_for(churn_config, ds["dataset_id"])
    c.put(f"/projects/{pid}/pipeline", json=body, headers=A)

    recs = c.get(f"/projects/{pid}/pipeline/recommendations/features", headers=A).json()
    inc = next(r for r in recs if r["id"] == "numeric_transform:income")
    assert inc["action"]["transform"] == "log" and "skewness" in inc["reason"]
    assert "numeric_transform:customer_id" not in {r["id"] for r in recs}      # identifier is dropped in preprocessing

    body["feature_engineering"] = {"numeric_transforms": [{"column": "income", "transform": "log"}], "date_features": []}
    pv = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert [f["name"] for f in pv["features"]] == ["income__log"] and pv["feature_issues"] == []
    assert len(pv["features"][0]["sample"]) == 5

    body["feature_engineering"] = {"numeric_transforms": [{"column": "age", "transform": "log"}],
                                   "date_features": [{"column": "gender", "extract": ["year"]}]}
    bad = c.post(f"/projects/{pid}/pipeline/preview", json=body, headers=A).json()
    assert bad["features"] == [] and any("does not look like a date" in i for i in bad["feature_issues"])
    assert c.get(f"/projects/{pid}/pipeline/recommendations/features", headers=B).status_code == 404


def test_chart_endpoints_return_small_aggregates_and_respect_access(env, churn_df, churn_config):
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    base = f"/projects/{pid}/datasets/{ds['dataset_id']}/charts"

    h = c.get(f"{base}/distribution", params={"column": "income", "bins": 12}, headers=A).json()
    assert len(h["counts"]) == 12 and sum(h["counts"]) == h["n"] == 400 and h["box"]["median"] > 0
    cat = c.get(f"{base}/categories", params={"column": "gender"}, headers=A).json()
    assert {i["label"] for i in cat["items"]} == {"M", "F"} and sum(i["count"] for i in cat["items"]) == 400
    corr = c.get(f"{base}/correlation", params={"target": "churn"}, headers=A).json()
    assert "churn" in corr["columns"] and "customer_id" in corr["columns"] and len(corr["matrix"]) == len(corr["columns"])
    assert c.get(f"{base}/missing", headers=A).json()["columns"][0]["column"] == "age"
    sc = c.get(f"{base}/scatter", params={"x": "age", "y": "income"}, headers=A).json()
    assert sc["n_shown"] == len(sc["x"]) == len(sc["y"]) <= 800

    auto = c.get(f"{base}/auto", params={"target": "churn"}, headers=A).json()
    assert auto[0]["type"] == "class_balance" and all(a["why"] for a in auto)
    assert "customer_id" not in [a["params"].get("column") for a in auto]
    assert len(str(auto)) < 60_000                                     # aggregates only, never the raw rows

    # clear errors
    assert c.get(f"{base}/distribution", params={"column": "gender"}, headers=A).status_code == 422
    r = c.get(f"{base}/distribution", params={"column": "nope"}, headers=A)
    assert r.status_code == 422 and "not found" in r.json()["detail"]
    assert c.get(f"{base}/distribution", params={"column": "age", "bins": 500}, headers=A).status_code == 422
    assert c.get(f"{base}/auto", params={"target": "nope"}, headers=A).status_code == 422
    assert c.get(f"{base}/scatter", params={"x": "age", "y": "age"}, headers=A).status_code == 422
    # access control
    for path, q in (("auto", {}), ("distribution", {"column": "age"}), ("missing", {}), ("correlation", {})):
        assert c.get(f"{base}/{path}", params=q, headers=B).status_code == 404, path
        assert c.get(f"{base}/{path}", params=q).status_code == 401, path


# ---- sign-in reliability: network blips are not "invalid token" -------------

class _Factory:
    """Counts calls; `script` is a list of outcomes (an Exception to raise, or 'ok')."""
    def __init__(self, store, script):
        self.store, self.script, self.calls = store, list(script), 0
        self.uid = str(uuid.uuid4())

    def __call__(self, token):
        self.calls += 1
        out = self.script.pop(0) if self.script else "ok"
        if isinstance(out, Exception):
            raise out
        return FakeSupabase(self.store, self.uid), self.uid


def _client(factory, built=None):
    def build(token):
        c = FakeSupabase(factory.store, factory.uid)
        if built is not None:
            built.append(c)
        return c
    return TestClient(create_app(client_factory=factory, cors_origins=[], client_builder=build))


def test_unreachable_auth_service_is_503_not_401_and_is_retried_once():
    store = FakeStore()
    blip = _Factory(store, [ConnectionError("network is down"), "ok"])
    with _client(blip) as c:
        assert c.get("/projects", headers=H("t")).status_code == 200          # one retry rescues a blip
        assert blip.calls == 2
    down = _Factory(store, [ConnectionError("down")] * 5)
    with _client(down) as c:
        r = c.get("/projects", headers=H("t"))
        assert r.status_code == 503 and "temporarily unreachable" in r.json()["detail"]
        assert down.calls == 2                                                # tried twice, then gave up
    class Timeout(Exception):                                                 # an unknown failure is also not "bad token"
        pass
    odd = _Factory(store, [Timeout("boom")] * 5)
    with _client(odd) as c:
        assert c.get("/projects", headers=H("t")).status_code == 503


def test_a_real_rejection_is_401_and_not_retried():
    store = FakeStore()
    for exc in (PermissionError("Invalid or expired access token"), type("AuthApiError", (Exception,), {"status": 401})("bad jwt")):
        f = _Factory(store, [exc] * 5)
        with _client(f) as c:
            assert c.get("/projects", headers=H("t")).status_code == 401
            assert f.calls == 1, "a definite 'no' must not be retried"
    rate = _Factory(store, [type("AuthApiError", (Exception,), {"status": 429})("slow down")] * 5)
    with _client(rate) as c:                                                   # rate-limited: the user did nothing wrong
        assert c.get("/projects", headers=H("t")).status_code == 503 and rate.calls == 2


def test_validated_tokens_are_cached_briefly_and_expire(monkeypatch):
    import nocodeml_engine.api.app as appmod
    clock = {"t": 1000.0}
    monkeypatch.setattr(appmod, "_now", lambda: clock["t"])
    f = _Factory(FakeStore(), [])
    with _client(f) as c:
        for _ in range(5):
            assert c.get("/projects", headers=H("tok-1")).status_code == 200
        assert f.calls == 1                                                    # one validation, four cache hits
        assert c.get("/projects", headers=H("tok-2")).status_code == 200 and f.calls == 2   # per-token
        clock["t"] += appmod.TOKEN_TTL_SECONDS - 1
        c.get("/projects", headers=H("tok-1")); assert f.calls == 2           # still trusted
        clock["t"] += 2
        c.get("/projects", headers=H("tok-1")); assert f.calls == 3           # expired -> asked again
        # a rejected token is never cached
        bad = _Factory(FakeStore(), [PermissionError("no")])
    with _client(bad) as c:
        assert c.get("/projects", headers=H("x")).status_code == 401
        assert c.get("/projects", headers=H("x")).status_code == 200 and bad.calls == 2


def test_token_cache_is_bounded():
    from nocodeml_engine.api.app import TokenCache
    cache = TokenCache(ttl=60, size=3)
    for i in range(10):
        cache.put(f"t{i}", f"u{i}")
    assert cache.get("t0") is None and cache.get("t9") == "u9" and len(cache._d) == 3


def test_clients_are_never_shared_between_requests():
    """Regression: sharing one Supabase client across threads broke its HTTP/2 connection (ReadError)."""
    f = _Factory(FakeStore(), [])
    built: list = []
    with _client(f, built) as c:
        for _ in range(4):
            assert c.get("/projects", headers=H("same-token")).status_code == 200
    assert f.calls == 1 and len(built) == 3 and len({id(x) for x in built}) == 3   # cache hits still get fresh clients


def test_upstream_network_errors_are_503_and_every_error_carries_cors_headers(env, monkeypatch):
    import httpx
    import nocodeml_engine.api.app as appmod
    c, *_ = env
    origin = {"Origin": "http://localhost:3000"}
    monkeypatch.setattr(appmod.ProjectService, "list_projects",
                        lambda self: (_ for _ in ()).throw(httpx.ConnectError("[Errno 54] Connection reset by peer")))
    r = c.get("/projects", headers={**A, **origin})
    assert r.status_code == 503 and "try again" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"    # visible to the browser
    assert "Errno" not in r.text                                                     # internals stay in the log

    monkeypatch.setattr(appmod.ProjectService, "list_projects", lambda self: (_ for _ in ()).throw(KeyError("secret-internal")))
    c2 = TestClient(c.app, raise_server_exceptions=False)
    r = c2.get("/projects", headers={**A, **origin})
    assert r.status_code == 500 and r.json()["detail"] == "Internal server error." and "secret" not in r.text
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_a_network_failure_during_training_gives_a_clear_job_error(env, churn_df, churn_config, monkeypatch):
    import httpx
    import nocodeml_engine.api.app as appmod
    c, *_ = env
    pid, ds = setup_project(c, churn_df)
    c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
    monkeypatch.setattr(appmod, "run_experiment", lambda *a, **k: (_ for _ in ()).throw(httpx.ReadError("reset")))
    job = train_and_wait(c, pid)
    assert job["status"] == "failed" and "Lost the connection to the database" in job["error"]
