import uuid

import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.artifacts import export_artifacts
from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.persistence import ProjectService, StorageError
from nocodeml_engine.state import PipelineError, Section, VersionStatus
from nocodeml_engine.training import run_experiment

from .fake_supabase import FakeStore, FakeSupabase


@pytest.fixture
def store():
    return FakeStore()


def service(store, uid=None):
    uid = uid or str(uuid.uuid4())
    return ProjectService(FakeSupabase(store, uid), uid)


@pytest.fixture
def svc(store):
    return service(store)


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def config_for(project_id, churn_config):
    return churn_config.model_copy(update={"pipeline_id": project_id})


def test_project_lifecycle_and_restore(svc, churn_df, churn_config):
    assert svc.list_projects() == []
    p = svc.create_project("Customer Churn", task="classification")
    assert svc.list_projects()[0].status == "Empty"

    ds = svc.add_dataset(p["id"], "customers.csv", csv_bytes(churn_df), target="churn")
    assert ds.n_rows == 400 and ds.storage_path.startswith(f"{svc.user_id}/{p['id']}/")
    df = svc.load_dataset(ds.dataset_id)  # round-trips and passes the fingerprint check
    assert df.shape == churn_df.shape

    pipes = svc.pipelines(p["id"])
    pipes.create(config_for(p["id"], churn_config))
    exp = pipes.record_experiment(run_experiment(df, pipes.latest(p["id"]).config).result)
    assert svc.list_projects()[0].status == "Experimenting"
    assert svc.list_projects()[0].pipeline_version == "v1"

    # "leave and come back": a fresh service object restores the state
    state = service(svc.db.store, svc.user_id).open_project(p["id"])
    assert state.pipeline.version == 1 and [e.number for e in state.experiments] == [exp.number]
    assert state.datasets[0]["id"] == ds.dataset_id

    pipes.finalize(p["id"])
    summary = svc.list_projects()[0]
    assert (summary.status, summary.pipeline_version) == ("Finalized", "v1.0")


def test_versioning_roundtrips_through_repository(svc, churn_df, churn_config):
    p = svc.create_project("x")
    pipes = svc.pipelines(p["id"])
    pipes.create(config_for(p["id"], churn_config))
    e1 = pipes.record_experiment(run_experiment(churn_df, pipes.latest(p["id"]).config).result)
    pre = pipes.latest(p["id"]).config.preprocessing.model_copy(update={"drop_duplicates": True})
    v2, impact = pipes.update_section(p["id"], Section.PREPROCESSING, pre)
    assert v2.version == 2 and impact.stale_experiment_ids == [e1.result.experiment_id]
    assert pipes.current_experiments(p["id"]) == []
    assert isinstance(pipes.repo.get_version(p["id"], 1).config, PipelineConfig)
    assert pipes.repo.get_version(p["id"], 1).status is VersionStatus.DRAFT
    with pytest.raises(PipelineError, match="not found"):
        pipes.repo.get_version(p["id"], 9)


def test_users_cannot_see_each_others_projects(store, churn_df):
    a, b = service(store), service(store)
    pa = a.create_project("A's secret")
    ds = a.add_dataset(pa["id"], "d.csv", csv_bytes(churn_df))
    assert b.list_projects() == []
    with pytest.raises(StorageError, match="not found"):
        b.open_project(pa["id"])
    with pytest.raises(StorageError):
        b.load_dataset(ds.dataset_id)
    with pytest.raises(Exception):  # cannot write into A's project
        b.add_dataset(pa["id"], "evil.csv", csv_bytes(churn_df))


@pytest.mark.parametrize("name,data,msg", [
    ("data.xls", b"a,b\n1,2\n", "Only .csv, .xlsx and .parquet"),
    ("empty.csv", b"a,b\n", "no rows"),
    ("bad.csv", b"", "Could not read the CSV"),
])
def test_dataset_validation_rejects_and_stores_nothing(svc, store, name, data, msg):
    p = svc.create_project("p")
    with pytest.raises(StorageError, match=msg):
        svc.add_dataset(p["id"], name, data)
    assert store.objects == {} and store.tables["datasets"] == []


def test_dataset_size_limit(svc, monkeypatch):
    import nocodeml_engine.persistence.projects as m
    monkeypatch.setattr(m, "MAX_DATASET_BYTES", 10)
    p = svc.create_project("p")
    with pytest.raises(StorageError, match="limit"):
        svc.add_dataset(p["id"], "big.csv", b"a,b\n1,2\n3,4\n")


def test_filename_is_sanitised_and_versions_increment(svc, churn_df):
    p = svc.create_project("p")
    d1 = svc.add_dataset(p["id"], "../../etc/pass wd.csv", csv_bytes(churn_df))
    assert ".." not in d1.storage_path.split(svc.user_id, 1)[1] and d1.filename == "pass_wd.csv"
    d2 = svc.add_dataset(p["id"], "again.csv", csv_bytes(churn_df.head(50)), dataset_id=d1.dataset_id)
    assert (d1.version, d2.version) == (1, 2)
    assert len(svc.load_dataset(d1.dataset_id)) == 50  # latest by default
    assert len(svc.load_dataset(d1.dataset_id, version=1)) == 400


def test_tampered_dataset_is_detected(svc, store, churn_df):
    p = svc.create_project("p")
    d = svc.add_dataset(p["id"], "d.csv", csv_bytes(churn_df))
    store.objects[("datasets", d.storage_path)] = csv_bytes(churn_df.assign(age=0.0))
    with pytest.raises(StorageError, match="fingerprint"):
        svc.load_dataset(d.dataset_id)


def test_failed_row_insert_cleans_up_object(svc, store, churn_df, monkeypatch):
    p = svc.create_project("p")
    real = svc.db.table

    def boom(name):
        if name == "dataset_versions":
            raise RuntimeError("db down")
        return real(name)
    monkeypatch.setattr(svc.db, "table", boom)
    with pytest.raises(RuntimeError):
        svc.add_dataset(p["id"], "d.csv", csv_bytes(churn_df))
    assert store.objects == {}


def test_artifacts_upload_download_and_integrity(svc, store, churn_df, churn_config, tmp_path):
    p = svc.create_project("p")
    pipes = svc.pipelines(p["id"])
    cfg = config_for(p["id"], churn_config)
    pipes.create(cfg)
    run = run_experiment(churn_df, cfg)
    exp = pipes.record_experiment(run.result)
    export_artifacts(run, cfg, tmp_path)
    arts = svc.save_artifacts(p["id"], exp.number, tmp_path)
    kinds = {a["kind"] for a in arts}
    assert {"pipeline", "model", "configuration", "metrics", "other"} <= kinds
    pipe = next(a for a in arts if a["kind"] == "pipeline")
    assert pipe["bucket"] == "pipelines" and pipe["storage_path"].startswith(f"{svc.user_id}/")
    assert svc.download_artifact(pipe["id"]) == (tmp_path / pipe["storage_path"].split("/")[-1]).read_bytes()
    assert "sign" in svc.signed_url(pipe["id"])
    store.objects[("pipelines", pipe["storage_path"])] = b"tampered"
    with pytest.raises(StorageError, match="integrity"):
        svc.download_artifact(pipe["id"])
