import time
import uuid

import pytest
from fastapi.testclient import TestClient

from nocodeml_engine.api.app import create_app
from nocodeml_engine.api.jobs import JobBusy, JobManager
from nocodeml_engine.api.ratelimit import RateLimited, RateLimiter
from nocodeml_engine.training import TrainingTimeout, run_experiment

from .fake_supabase import FakeStore, FakeSupabase
from .test_api import A, B, body_for, csv_file, setup_project, train_and_wait


def make(**kw):
    store = FakeStore()
    users = {"tok-a": str(uuid.uuid4()), "tok-b": str(uuid.uuid4())}
    def factory(t):
        if t not in users:
            raise PermissionError("bad token")
        return FakeSupabase(store, users[t]), users[t]

    app = create_app(client_factory=factory,
                     client_builder=lambda t: FakeSupabase(store, users[t]), **kw)
    return TestClient(app)


def test_window_counts_expires_and_is_per_key():
    t = [0.0]
    rl = RateLimiter(clock=lambda: t[0])
    for _ in range(3):
        rl.check("x", "u1", 3, 10)
    with pytest.raises(RateLimited) as e:
        rl.check("x", "u1", 3, 10)
    assert e.value.retry_after == 10
    rl.check("x", "u2", 3, 10)       # another user is unaffected
    rl.check("y", "u1", 3, 10)       # another bucket is unaffected
    t[0] = 10.1                      # window rolled over
    rl.check("x", "u1", 3, 10)


def test_rejected_requests_do_not_extend_the_block():
    t = [0.0]
    rl = RateLimiter(clock=lambda: t[0])
    rl.check("x", "u", 1, 10)
    for i in range(1, 5):
        t[0] = float(i)
        with pytest.raises(RateLimited):
            rl.check("x", "u", 1, 10)
    t[0] = 10.1
    rl.check("x", "u", 1, 10)


def test_limiter_memory_is_bounded(monkeypatch):
    import nocodeml_engine.api.ratelimit as r
    monkeypatch.setattr(r, "MAX_KEYS", 5)
    rl = RateLimiter()
    for i in range(50):
        rl.check("x", f"k{i}", 1, 60)
    assert len(rl._hits) == 5


def test_heavy_endpoint_returns_429_with_retry_after_and_cors(churn_df):
    with make(cors_origins=["http://localhost:3000"], rate_limits={"upload": (2, 60)}) as c:
        pid = c.post("/projects", json={"name": "P", "task": "classification"}, headers=A).json()["id"]
        send = lambda h=A: c.post(f"/projects/{pid}/datasets", files=csv_file(churn_df),  # noqa: E731
                                  data={"target": "churn"}, headers={**h, "Origin": "http://localhost:3000"})
        assert send().status_code == 201 and send().status_code == 201
        r = send()
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1
        assert "wait" in r.json()["detail"]
        assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_limits_are_per_user(churn_df):
    with make(rate_limits={"upload": (1, 60)}) as c:
        pa, _ = setup_project(c, churn_df, A)
        pb, _ = setup_project(c, churn_df, B)    # B has its own budget
        again = c.post(f"/projects/{pa}/datasets", files=csv_file(churn_df), data={"target": "churn"}, headers=A)
        assert again.status_code == 429


def test_general_user_and_ip_limits():
    with make(rate_limits={"user": (3, 60)}) as c:
        codes = [c.get("/projects", headers=A).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        assert c.get("/projects", headers=B).status_code == 200
    with make(rate_limits={"ip": (2, 60)}) as c:
        assert [c.get("/models").status_code for _ in range(3)] == [200, 200, 429]
        assert c.get("/health").status_code == 200       # health checks are never limited


def test_bad_token_does_not_spend_a_users_budget():
    with make(rate_limits={"user": (1, 60)}) as c:
        assert c.get("/projects", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_full_queue_is_a_clear_503_and_not_a_crash():
    import threading
    jm = JobManager(max_workers=1, max_waiting=1)
    gate = threading.Event()
    jm.submit("p1", "u", lambda j: (gate.wait(5), 1)[1])   # occupies the worker
    time.sleep(0.1)
    jm.submit("p2", "u", lambda j: 2)                       # waits
    with pytest.raises(JobBusy):
        jm.submit("p3", "u", lambda j: 3)
    gate.set()
    jm.shutdown()


def test_training_stops_at_the_time_limit(churn_df, churn_config):
    with pytest.raises(TrainingTimeout):
        run_experiment(churn_df, churn_config, deadline=time.monotonic() - 1)


def test_timeout_surfaces_as_a_clear_failed_job(churn_df, churn_config):
    with make(train_timeout=-1) as c:
        pid, ds = setup_project(c, churn_df)
        c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
        job = train_and_wait(c, pid)
        assert job["status"] == "failed"
        assert "time limit" in job["error"]
        assert c.get(f"/projects/{pid}/experiments", headers=A).json() == []   # nothing half-saved


# ---- hardening --------------------------------------------------------------

def test_production_hides_docs_and_adds_strict_headers():
    with make(production=True, cors_origins=["https://app.example.com"]) as c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(path).status_code == 404
        h = c.get("/health").headers
        assert h["strict-transport-security"].startswith("max-age=")
        assert "frame-ancestors 'none'" in h["content-security-policy"]
    with make(production=False) as c:
        assert c.get("/docs").status_code == 200
        assert "strict-transport-security" not in c.get("/health").headers


def test_every_response_has_request_id_and_safe_headers_even_errors():
    with make() as c:
        for r in (c.get("/health"), c.get("/projects"), c.get("/nope")):      # ok, 401, 404
            assert len(r.headers["x-request-id"]) == 12
            assert r.headers["x-content-type-options"] == "nosniff"
            assert r.headers["cache-control"] == "no-store"
        assert c.get("/health").headers["x-request-id"] != c.get("/health").headers["x-request-id"]


def test_production_refuses_unsafe_cors_origins(monkeypatch):
    monkeypatch.delenv("NOCODEML_CORS_ORIGINS", raising=False)
    for bad in (None, ["*"], ["http://app.example.com"], ["https://ok.example.com", "http://localhost:3000"]):
        with pytest.raises(RuntimeError, match="CORS"):
            make(production=True, **({} if bad is None else {"cors_origins": bad}))
    make(production=True, cors_origins=["https://app.example.com"])           # fine


def test_rate_limit_uses_forwarded_address_only_when_proxy_is_trusted():
    hdr = lambda ip: {"X-Forwarded-For": f"6.6.6.6, {ip}"}  # noqa: E731   (first entry is attacker-controlled)
    with make(rate_limits={"ip": (2, 60)}, trust_proxy=True) as c:
        assert [c.get("/models", headers=hdr("1.1.1.1")).status_code for _ in range(3)] == [200, 200, 429]
        assert c.get("/models", headers=hdr("2.2.2.2")).status_code == 200      # a different real client
    with make(rate_limits={"ip": (2, 60)}, trust_proxy=False) as c:             # header ignored: all one client
        assert [c.get("/models", headers=hdr(f"9.9.9.{i}")).status_code for i in range(3)] == [200, 200, 429]


def test_cannot_delete_a_project_while_it_is_training(churn_df, churn_config, monkeypatch):
    import threading
    import nocodeml_engine.api.app as appmod
    gate = threading.Event()
    real = appmod.run_experiment
    monkeypatch.setattr(appmod, "run_experiment", lambda *a, **k: (gate.wait(10), real(*a, **k))[1])
    with make() as c:
        pid, ds = setup_project(c, churn_df)
        c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
        job = c.post(f"/projects/{pid}/training", headers=A).json()
        r = c.delete(f"/projects/{pid}", headers=A)
        assert r.status_code == 409 and "Training is still running" in r.json()["detail"]
        gate.set()
        for _ in range(200):
            if c.get(f"/projects/{pid}/training/{job['id']}", headers=A).json()["status"] != "running":
                break
            time.sleep(0.1)
        assert c.delete(f"/projects/{pid}", headers=A).status_code == 204
        assert c.get(f"/projects/{pid}", headers=A).status_code == 404
        assert c.delete(f"/projects/{pid}", headers=B).status_code == 404       # not anyone else's either
