"""Drive a running API over HTTP as a real Supabase user.

    # terminal 1 (from engine/):  .venv/bin/uvicorn nocodeml_engine.api.main:app --port 8000
    # terminal 2:                 .venv/bin/python smoke_api.py
"""
import io
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from nocodeml_engine.persistence import SupabaseSettings, sign_in_with_password  # noqa: E402

BASE = os.environ.get("API_URL", "http://127.0.0.1:8000")
s = SupabaseSettings.from_env()
tok = {w: sign_in_with_password(s, os.environ[f"SUPABASE_TEST_EMAIL_{w}"],
                                os.environ[f"SUPABASE_TEST_PASSWORD_{w}"]) for w in "AB"}
A, B = ({"Authorization": f"Bearer {tok[w]}"} for w in "AB")
c = httpx.Client(base_url=BASE, timeout=60)


def step(msg, r, expect):
    ok = r.status_code == expect
    print(f"{'✓' if ok else '✗'} {msg}  [{r.status_code}]")
    if not ok:
        print("   ", r.text[:400]); sys.exit(1)
    return r.json() if r.content else None


rng = np.random.default_rng(0); n = 300
df = pd.DataFrame({"customer_id": range(n), "age": rng.normal(40, 12, n),
                   "income": rng.lognormal(10, .5, n), "gender": rng.choice(["M", "F"], n)})
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-(0.05 * (df.age - 40) - 1)))).astype(int)
df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan

step("health", c.get("/health"), 200)
step("no token is rejected", c.get("/projects"), 401)
pid = step("create project", c.post("/projects", json={"name": "api-smoke"}, headers=A), 201)["id"]
try:
    ds = step("upload csv", c.post(f"/projects/{pid}/datasets", headers=A, data={"target": "churn"},
              files={"file": ("c.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}), 201)
    rows = step("raw rows page", c.get(f"/projects/{pid}/datasets/{ds['dataset_id']}/rows",
                params={"page_size": 5, "sort_by": "age"}, headers=A), 200)
    print("    total rows:", rows["total"])
    cfg = {"dataset": {"dataset_id": ds["dataset_id"], "version": 1, "target_column": "churn",
                       "task": "classification"},
           "preprocessing": {"drop_columns": ["customer_id"], "scaling": "standard",
                             "missing_values": [{"column": "age", "strategy": "median"}],
                             "encoding": [{"column": "gender", "strategy": "one_hot"}]},
           "split": {"method": "train_test", "stratify": True},
           "models": [{"model_key": "logistic_regression"}, {"model_key": "random_forest"}]}
    step("save pipeline", c.put(f"/projects/{pid}/pipeline", json=cfg, headers=A), 200)
    v = step("validate", c.post(f"/projects/{pid}/pipeline/validate", headers=A), 200)
    print("    valid:", v["valid"])
    job = step("start training", c.post(f"/projects/{pid}/training", headers=A), 202)
    while job["status"] in ("queued", "running"):
        time.sleep(0.5)
        job = c.get(f"/projects/{pid}/training/{job['id']}", headers=A).json()
    print("    job:", job["status"], "experiment", job["experiment_number"], job["warnings"])
    assert job["status"] == "succeeded", job
    for e in step("experiments", c.get(f"/projects/{pid}/experiments", headers=A), 200):
        print("    #%d quality=%s %s" % (e["number"], e["quality_score"],
              [(m["name"], round(m["value"], 3)) for m in e["models"]]))
    arts = step("artifacts", c.get(f"/projects/{pid}/experiments/1/artifacts", headers=A), 200)
    print("    files:", sorted(a["storage_path"].split("/")[-1] for a in arts))
    u = step("signed download url", c.get(f"/projects/{pid}/artifacts/{arts[0]['id']}/url", headers=A), 200)
    print("    downloadable:", httpx.get(u["url"]).status_code == 200)
    step("other user cannot open project", c.get(f"/projects/{pid}", headers=B), 404)
    step("finalize", c.post(f"/projects/{pid}/pipeline/finalize", headers=A), 200)
finally:
    step("delete project (+ stored files)", c.delete(f"/projects/{pid}", headers=A), 204)
print("\nAll good.")
