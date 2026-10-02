"""End-to-end against a REAL Supabase project. Skipped unless these env vars are set:

    SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_TEST_EMAIL_A/PASSWORD_A, SUPABASE_TEST_EMAIL_B/PASSWORD_B

Use a throwaway project with 0001_schema.sql applied and two confirmed test users.
"""
import os

import pytest

from nocodeml_engine.persistence import (
    ProjectService, StorageError, SupabaseSettings, authenticated_client, sign_in_with_password,
)
from nocodeml_engine.training import run_experiment

REQUIRED = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_TEST_EMAIL_A", "SUPABASE_TEST_PASSWORD_A",
            "SUPABASE_TEST_EMAIL_B", "SUPABASE_TEST_PASSWORD_B"]
pytestmark = pytest.mark.skipif(any(k not in os.environ for k in REQUIRED),
                                reason="live Supabase env vars not set")


def svc_for(which):
    s = SupabaseSettings.from_env()
    token = sign_in_with_password(s, os.environ[f"SUPABASE_TEST_EMAIL_{which}"],
                                  os.environ[f"SUPABASE_TEST_PASSWORD_{which}"])
    client, uid = authenticated_client(s, token)
    return ProjectService(client, uid)


def test_full_flow_and_isolation(churn_df, churn_config):
    a, b = svc_for("A"), svc_for("B")
    p = a.create_project("live-test")
    try:
        ds = a.add_dataset(p["id"], "c.csv", churn_df.to_csv(index=False).encode(), target="churn")
        df = a.load_dataset(ds.dataset_id)
        pipes = a.pipelines(p["id"])
        cfg = churn_config.model_copy(update={"pipeline_id": p["id"]})
        pipes.create(cfg)
        pipes.record_experiment(run_experiment(df, cfg).result)
        pipes.finalize(p["id"])
        assert a.open_project(p["id"]).pipeline.status.value == "finalized"
        assert p["id"] not in [x.id for x in b.list_projects()]
        with pytest.raises(StorageError):
            b.open_project(p["id"])
    finally:
        a.db.table("projects").delete().eq("id", p["id"]).execute()
