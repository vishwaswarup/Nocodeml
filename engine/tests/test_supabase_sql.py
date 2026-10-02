"""Tests the real migration SQL + RLS against a throwaway local Postgres.

Supabase's own pieces (auth.uid(), roles, storage schema) are stubbed in supabase_stub.sql.
Skipped automatically when Postgres binaries are unavailable.
"""
import json
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")
def _find_bin():
    """First directory that has initdb, pg_ctl and postgres together (avoids mixed versions)."""
    cands = sorted(Path("/opt/homebrew/opt").glob("postgresql*/bin")) + [
        Path(p).parent for p in [shutil.which("initdb")] if p]
    for d in cands:
        if all((d / n).exists() for n in ("initdb", "pg_ctl", "postgres")):
            return d
    return None


BIN = _find_bin()
pytestmark = pytest.mark.skipif(BIN is None, reason="postgres binaries not found")

MIGRATION = Path(__file__).parents[2] / "supabase" / "migrations" / "0001_schema.sql"
STUB = Path(__file__).parent / "supabase_stub.sql"


@pytest.fixture(scope="module")
def dsn(tmp_path_factory):
    d = tmp_path_factory.mktemp("pg")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    subprocess.run([BIN / "initdb", "-D", d / "data", "-U", "postgres", "--auth=trust"],
                   check=True, capture_output=True)
    subprocess.run([BIN / "pg_ctl", "-D", d / "data", "-o", f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories=",
                    "-w", "-l", d / "log", "start"], check=True, capture_output=True)
    dsn = f"host=127.0.0.1 port={port} dbname=postgres user=postgres"
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(STUB.read_text())
        c.execute(MIGRATION.read_text())
    yield dsn
    subprocess.run([BIN / "pg_ctl", "-D", d / "data", "-m", "immediate", "stop"], capture_output=True)


class Db:
    def __init__(self, dsn):
        self.c = psycopg.connect(dsn, autocommit=True)

    def as_user(self, uid):
        self.c.execute("reset role")
        self.c.execute("select set_config('request.jwt.claim.sub', %s, false)", [str(uid or "")])
        self.c.execute("set role authenticated")
        return self

    def admin(self):
        self.c.execute("reset role")
        return self

    def q(self, sql, *a):
        cur = self.c.execute(sql, a)
        return cur.fetchall() if cur.description else None


@pytest.fixture
def db(dsn):
    d = Db(dsn)
    yield d
    d.c.close()


@pytest.fixture
def users(db):
    a, b = uuid.uuid4(), uuid.uuid4()
    for u in (a, b):
        db.admin().q("insert into auth.users (id) values (%s)", u)
    return a, b


def new_project(db, uid, name="Churn"):
    return db.as_user(uid).q("insert into projects (name) values (%s) returning id", name)[0][0]


def add_version(db, pid, v=1, h="h1"):
    db.q("insert into pipeline_versions (project_id, version, config, config_hash) values (%s,%s,%s,%s)",
         pid, v, json.dumps({"x": v}), h)


def add_experiment(db, pid, n=1, v=1, h="h1", eid=None):
    db.q("insert into experiments (project_id, number, experiment_id, pipeline_version, config_hash, result)"
         " values (%s,%s,%s,%s,%s,'{}')", pid, n, eid or f"e{n}", v, h)


def fails(db, sql, *a, match=""):
    with pytest.raises(psycopg.Error, match=match):
        db.q(sql, *a)


def test_owner_defaults_and_isolation(db, users):
    a, b = users
    pid = new_project(db, a)
    assert db.q("select owner_id from projects where id=%s", pid)[0][0] == a
    add_version(db, pid)
    db.q("insert into datasets (project_id, name) values (%s,'d')", pid)

    db.as_user(b)  # another user sees nothing and can change nothing
    for t in ("projects", "datasets", "pipeline_versions", "experiments", "artifacts", "dataset_versions"):
        assert db.q(f"select count(*) from {t}")[0][0] == 0, t
    assert db.q("update projects set name='hacked' where id=%s returning id", pid) == []
    assert db.q("delete from projects where id=%s returning id", pid) == []
    fails(db, "insert into pipeline_versions (project_id, version, config, config_hash) values (%s,2,'{}','x')",
          pid, match="row-level security")
    fails(db, "insert into datasets (project_id, name) values (%s,'x')", pid, match="row-level security")
    fails(db, "insert into projects (owner_id, name) values (%s,'forged')", a, match="row-level security")
    db.admin()
    assert db.q("select name from projects where id=%s", pid)[0][0] == "Churn"


def test_unauthenticated_sees_nothing(db, users):
    pid = new_project(db, users[0])
    db.as_user(None)
    assert db.q("select count(*) from projects")[0][0] == 0
    fails(db, "insert into projects (name) values ('x')")  # owner_id null -> rejected


def test_pipeline_version_immutability_and_finalize(db, users):
    pid = new_project(db, users[0])
    add_version(db, pid, 1, "h1")
    fails(db, "update pipeline_versions set config='{}' where project_id=%s", pid, match="immutable")
    fails(db, "update pipeline_versions set config_hash='zz' where project_id=%s", pid, match="immutable")
    assert db.q("delete from pipeline_versions where project_id=%s returning 1", pid) == []  # no delete policy

    # finalize requires an experiment that ran this exact config
    fails(db, "update pipeline_versions set status='finalized', finalized_experiment_id='e1' "
              "where project_id=%s", pid, match="exact configuration")
    add_experiment(db, pid, 1, 1, "h1")
    db.q("update pipeline_versions set status='finalized', finalized_experiment_id='e1' where project_id=%s", pid)
    fails(db, "update pipeline_versions set status='draft', finalized_experiment_id=null where project_id=%s",
          pid, match="finalized pipeline versions are immutable")


def test_experiments_append_only_and_bound_to_config(db, users):
    pid = new_project(db, users[0])
    add_version(db, pid, 1, "h1")
    fails(db, "insert into experiments (project_id, number, experiment_id, pipeline_version, config_hash, result)"
              " values (%s,1,'e1',1,'WRONG','{}')", pid, match="does not match")
    fails(db, "insert into experiments (project_id, number, experiment_id, pipeline_version, config_hash, result)"
              " values (%s,1,'e1',9,'h1','{}')", pid)  # unknown version -> FK
    add_experiment(db, pid, 1, 1, "h1")
    assert db.q("update experiments set result='{\"edited\":1}' where project_id=%s returning 1", pid) == []
    assert db.q("delete from experiments where project_id=%s returning 1", pid) == []
    fails(db, "insert into experiments (project_id, number, experiment_id, pipeline_version, config_hash, result)"
              " values (%s,2,'e1',1,'h1','{}')", pid, match="unique")  # duplicate experiment_id


def test_updated_at_touched_and_cascade(db, users):
    a, _ = users
    pid = new_project(db, a)
    t0 = db.q("select updated_at from projects where id=%s", pid)[0][0]
    add_version(db, pid)
    assert db.q("select updated_at from projects where id=%s", pid)[0][0] > t0
    add_experiment(db, pid)
    db.q("delete from projects where id=%s", pid)  # owner can delete; history cascades
    db.admin()
    assert db.q("select count(*) from pipeline_versions where project_id=%s", pid)[0][0] == 0
    assert db.q("select count(*) from experiments where project_id=%s", pid)[0][0] == 0


def test_storage_policies(db, users):
    a, b = users
    pid = uuid.uuid4()
    mine, theirs = f"{a}/{pid}/data.csv", f"{b}/{pid}/data.csv"
    db.as_user(a)
    db.q("insert into storage.objects (bucket_id, name) values ('datasets', %s)", mine)
    fails(db, "insert into storage.objects (bucket_id, name) values ('datasets', %s)", theirs,
          match="row-level security")
    fails(db, "insert into storage.objects (bucket_id, name) values ('somebucket', %s)", mine,
          match="row-level security")
    db.admin().q("insert into storage.objects (bucket_id, name) values ('datasets', %s)", theirs)
    db.as_user(a)
    assert [r[0] for r in db.q("select name from storage.objects")] == [mine]
    assert db.q("delete from storage.objects where name=%s returning 1", theirs) == []
    db.admin()
    assert db.q("select file_size_limit from storage.buckets where id='datasets'")[0][0] == 104857600
    assert {r[0] for r in db.q("select id from storage.buckets")} == {
        "datasets", "models", "pipelines", "reports", "visualizations"}
