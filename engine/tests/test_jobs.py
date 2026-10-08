import base64
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import nocodeml_engine.api.app as appmod
from nocodeml_engine.api.app import create_app, token_seconds_left
from nocodeml_engine.api.jobs import JobBusy, JobConflict, JobManager
from nocodeml_engine.persistence import ProjectService
from nocodeml_engine.training import TrainingCancelled

from .fake_supabase import FakeStore, FakeSupabase
from .test_api import A, B, body_for, csv_file, setup_project, train_and_wait


def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def svc_env():
    store, uid = FakeStore(), str(uuid.uuid4())
    svc = ProjectService(FakeSupabase(store, uid), uid)
    return store, uid, svc


def test_job_records_survive_a_restart_and_a_new_process_sees_them(svc_env):
    store, uid, svc = svc_env
    pid = svc.create_project("p")["id"]
    first = JobManager()
    job = first.submit(svc.jobs(), pid, lambda j, c: 7)
    assert wait_for(lambda: first.get(svc.jobs(), job.id, pid).status == "succeeded")
    first.shutdown()
    second = JobManager()                                    # "the API restarted": fresh memory, same database
    again = second.get(ProjectService(FakeSupabase(store, uid), uid).jobs(), job.id, pid)
    assert again.status == "succeeded" and again.experiment_number == 7 and again.finished_at
    assert [j.id for j in svc.jobs().recent(pid)] == [job.id]


def test_a_crashed_job_is_marked_failed_and_no_longer_blocks_training(svc_env):
    store, uid, svc = svc_env
    pid = svc.create_project("p")["id"]
    ghost = svc.jobs().create(pid)                           # a job some dead process left "running"
    svc.jobs().update(ghost.id, status="running")
    jm = JobManager(stale=60)
    assert jm.active_for(svc.jobs(), pid).status == "running"          # heartbeat is fresh: not reaped yet
    old = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
    svc.jobs().update(ghost.id, heartbeat_at=old)
    assert jm.active_for(svc.jobs(), pid) is None                      # silent for 5 minutes: reaped
    done = jm.get(svc.jobs(), ghost.id, pid)
    assert done.status == "failed" and "interrupted" in done.error and done.finished_at
    job = jm.submit(svc.jobs(), pid, lambda j, c: 1)                   # and a new training may start
    assert wait_for(lambda: jm.get(svc.jobs(), job.id, pid).status == "succeeded")
    jm.shutdown()


def test_a_slow_job_that_is_alive_here_is_never_reaped():
    store, uid = FakeStore(), str(uuid.uuid4())
    svc = ProjectService(FakeSupabase(store, uid), uid)
    pid = svc.create_project("p")["id"]
    gate = threading.Event()
    jm = JobManager(stale=0, heartbeat=0.05)                  # "stale" immediately: only liveness in this process protects it
    job = jm.submit(svc.jobs(), pid, lambda j, c: (gate.wait(10), 5)[1])
    assert wait_for(lambda: jm.get(svc.jobs(), job.id, pid).status == "running")
    time.sleep(0.2)
    assert jm.get(svc.jobs(), job.id, pid).status == "running"
    gate.set()
    assert wait_for(lambda: jm.get(svc.jobs(), job.id, pid).status == "succeeded")
    jm.shutdown()


def test_heartbeat_keeps_advancing_while_a_job_runs(svc_env):
    store, uid, svc = svc_env
    pid = svc.create_project("p")["id"]
    gate = threading.Event()
    jm = JobManager(heartbeat=0.05)
    job = jm.submit(svc.jobs(), pid, lambda j, c: (gate.wait(10), 1)[1])
    first = svc.jobs().last_heartbeat(job.id)
    assert wait_for(lambda: svc.jobs().last_heartbeat(job.id) > first)
    gate.set()
    jm.shutdown()


def test_one_active_job_per_project_even_across_two_processes(svc_env):
    store, uid, svc = svc_env
    pid = svc.create_project("p")["id"]
    gate = threading.Event()
    a, b = JobManager(), JobManager()                        # two API processes sharing one database
    a.submit(svc.jobs(), pid, lambda j, c: (gate.wait(10), 1)[1])
    with pytest.raises(JobConflict):
        b.submit(ProjectService(FakeSupabase(store, uid), uid).jobs(), pid, lambda j, c: 2)
    gate.set()
    a.shutdown(); b.shutdown()


def test_each_user_may_only_have_a_few_trainings_in_progress():
    store, uid = FakeStore(), str(uuid.uuid4())
    svc = ProjectService(FakeSupabase(store, uid), uid)
    pids = [svc.create_project(f"p{i}")["id"] for i in range(4)]
    gate = threading.Event()
    jm = JobManager(max_workers=4)
    for pid in pids[:3]:
        jm.submit(svc.jobs(), pid, lambda j, c: (gate.wait(10), 1)[1])
    with pytest.raises(JobBusy, match="3 trainings"):
        jm.submit(svc.jobs(), pids[3], lambda j, c: 1)
    gate.set()
    jm.shutdown()


def test_cancel_stops_a_running_job_and_a_waiting_job(svc_env):
    store, uid, svc = svc_env
    p1, p2 = svc.create_project("a")["id"], svc.create_project("b")["id"]
    started = threading.Event()

    def long_running(job, cancel):
        started.set()
        while not cancel.is_set():
            time.sleep(0.02)
        raise TrainingCancelled("Training was cancelled.")

    jm = JobManager(max_workers=1)
    running = jm.submit(svc.jobs(), p1, long_running)
    assert started.wait(5)
    waiting = jm.submit(svc.jobs(), p2, lambda j, c: 99)               # queued behind the single worker
    assert jm.get(svc.jobs(), waiting.id, p2).queue_position == 1
    jm.cancel(svc.jobs(), waiting.id, p2)
    jm.cancel(svc.jobs(), running.id, p1)
    assert wait_for(lambda: jm.get(svc.jobs(), running.id, p1).status == "cancelled")
    assert wait_for(lambda: jm.get(svc.jobs(), waiting.id, p2).status == "cancelled")
    r = jm.get(svc.jobs(), running.id, p1)
    assert "Nothing was saved" in r.error and r.experiment_number is None
    assert jm.get(svc.jobs(), waiting.id, p2).experiment_number is None   # never ran
    assert jm.cancel(svc.jobs(), "00000000-0000-0000-0000-000000000000", p1) is None
    jm.shutdown()


def test_a_cancel_asked_of_another_process_reaches_the_running_job(svc_env):
    store, uid, svc = svc_env
    pid = svc.create_project("p")["id"]
    seen = threading.Event()

    def fn(job, cancel):
        assert cancel.wait(10), "cancel never arrived"
        seen.set()
        raise TrainingCancelled("x")

    jm = JobManager(heartbeat=0.05)
    job = jm.submit(svc.jobs(), pid, fn)
    svc.jobs().update(job.id, cancel_requested=True)         # what another API process's cancel endpoint does
    assert seen.wait(5)
    assert wait_for(lambda: jm.get(svc.jobs(), job.id, pid).status == "cancelled")
    jm.shutdown()


# ---- over HTTP ---------------------------------------------------------------------------------------------

def make_client(**kw):
    store = FakeStore()
    users = {"tok-a": str(uuid.uuid4()), "tok-b": str(uuid.uuid4())}

    def factory(t):
        if t not in users:
            raise PermissionError("bad")
        return FakeSupabase(store, users[t]), users[t]

    app = create_app(client_factory=factory, client_builder=lambda t: FakeSupabase(store, users[t]), **kw)
    return TestClient(app), store


def test_cancel_over_http_and_job_history(churn_df, churn_config, monkeypatch):
    gate = threading.Event()

    def fake_run(df, cfg, deadline=None, cancel=None, **k):
        while not cancel.is_set():
            time.sleep(0.02)
        raise TrainingCancelled("Training was cancelled.")

    monkeypatch.setattr(appmod, "run_experiment", fake_run)
    c, store = make_client()
    with c:
        pid, ds = setup_project(c, churn_df)
        c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A)
        job = c.post(f"/projects/{pid}/training", headers=A).json()
        assert c.post(f"/projects/{pid}/training/{job['id']}/cancel", headers=B).status_code == 404   # not B's
        r = c.post(f"/projects/{pid}/training/{job['id']}/cancel", headers=A)
        assert r.status_code == 200
        assert wait_for(lambda: c.get(f"/projects/{pid}/training/{job['id']}", headers=A).json()["status"] == "cancelled")
        assert c.get(f"/projects/{pid}/training/active", headers=A).json() is None
        assert c.get(f"/projects/{pid}/experiments", headers=A).json() == []
        hist = c.get(f"/projects/{pid}/training", headers=A).json()
        assert [h["status"] for h in hist] == ["cancelled"]
        assert c.get(f"/projects/{pid}/training", headers=B).status_code == 404


def jwt_expiring_in(seconds):
    b64 = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{b64({'alg': 'none'})}.{b64({'exp': time.time() + seconds, 'sub': 'x'})}.sig"


def test_token_expiry_is_read_from_the_jwt_and_unreadable_tokens_are_ignored():
    assert 590 < token_seconds_left(jwt_expiring_in(600)) <= 600
    assert token_seconds_left("tok-a") is None and token_seconds_left("a.b.c") is None


def test_training_refuses_to_start_when_the_sign_in_would_expire_mid_run(churn_df, churn_config):
    store = FakeStore()
    uid = str(uuid.uuid4())
    soon, later = jwt_expiring_in(60), jwt_expiring_in(7200)
    app = create_app(client_factory=lambda t: (FakeSupabase(store, uid), uid), client_builder=lambda t: FakeSupabase(store, uid),
                     train_timeout=300)
    with TestClient(app) as c:
        H = lambda t: {"Authorization": f"Bearer {t}"}  # noqa: E731
        pid, ds = setup_project(c, churn_df, H(later))
        c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=H(later))
        r = c.post(f"/projects/{pid}/training", headers=H(soon))
        assert r.status_code == 401 and "about to expire" in r.json()["detail"]
        assert c.get(f"/projects/{pid}/training", headers=H(later)).json() == []          # nothing was queued
        assert train_and_wait(c, pid, H(later))["status"] == "succeeded"
