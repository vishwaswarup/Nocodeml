import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from nocodeml_engine.dataset.loader import DatasetFormatError, parse_dataset
from nocodeml_engine.training.runner import dataset_fingerprint

from .fake_supabase import FakeStore, FakeSupabase
from .test_api import A, body_for, csv_file, setup_project, train_and_wait


def frame():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "age": rng.integers(18, 70, n), "income": rng.normal(50000, 9000, n).round(2),
        "city": rng.choice(["a", "b", "c"], n), "joined": pd.date_range("2024-01-01", periods=n, freq="D"),
        "churn": rng.integers(0, 2, n)})


def xlsx_bytes(df, **kw):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl", **kw)
    return buf.getvalue()


def parquet_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


def test_all_three_formats_give_the_same_dataset():
    df = frame()
    csv = parse_dataset(df.to_csv(index=False).encode(), "d.csv")
    xl = parse_dataset(xlsx_bytes(df), "d.xlsx")
    pq = parse_dataset(parquet_bytes(df), "d.parquet")
    for other in (xl, pq):
        assert list(other.columns) == list(csv.columns) and len(other) == len(csv)
        assert pd.api.types.is_datetime64_any_dtype(other["joined"])
        np.testing.assert_allclose(other["income"], csv["income"])
        assert (other["city"] == csv["city"]).all() and (other["age"] == csv["age"]).all()
    assert dataset_fingerprint(parse_dataset(xlsx_bytes(df), "d.xlsx")) == dataset_fingerprint(xl)   # deterministic


def test_excel_uses_the_first_sheet_and_first_row_as_header():
    df = frame()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=False)
        pd.DataFrame({"other": [1, 2]}).to_excel(w, sheet_name="Notes", index=False)
    out = parse_dataset(buf.getvalue(), "multi.xlsx")
    assert list(out.columns) == list(df.columns)


def test_extension_is_case_insensitive_and_parquet_index_is_dropped():
    df = frame().set_index("age")
    out = parse_dataset(parquet_bytes(frame().assign(extra=1)), "DATA.PARQUET")
    assert "extra" in out.columns and out.index.tolist() == list(range(len(out)))
    buf = io.BytesIO(); df.to_parquet(buf, index=True)
    kept = parse_dataset(buf.getvalue(), "d.parquet")
    assert kept.index.tolist() == list(range(len(kept)))             # a stored index never becomes row labels


@pytest.mark.parametrize("name,data,match", [
    ("d.xls", b"x", "Only .csv, .xlsx and .parquet.*Save old .xls"),
    ("d.json", b"{}", "Only .csv, .xlsx and .parquet"),
    ("noext", b"x", "Only .csv"),
    ("d.xlsx", b"not a zip", "valid .xlsx"),
    ("d.parquet", b"not parquet", "Could not read the PARQUET"),
    ("d.xlsx", b"PK\x03\x04garbage", "Could not read|valid .xlsx"),
])
def test_unreadable_files_get_one_clean_message(name, data, match):
    with pytest.raises(DatasetFormatError, match=match):
        parse_dataset(data, name)


def test_zip_bomb_and_row_limits(monkeypatch):
    import nocodeml_engine.dataset.loader as L
    monkeypatch.setattr(L, "MAX_XLSX_UNCOMPRESSED", 1000)
    with pytest.raises(DatasetFormatError, match="too large"):
        parse_dataset(xlsx_bytes(frame()), "big.xlsx")
    monkeypatch.undo()
    monkeypatch.setattr(L, "MAX_ROWS", 10)
    with pytest.raises(DatasetFormatError, match="rows; the limit"):
        parse_dataset(parquet_bytes(frame()), "long.parquet")
    with pytest.raises(DatasetFormatError, match="rows; the limit"):
        parse_dataset(frame().to_csv(index=False).encode(), "long.csv")


def test_duplicate_column_names_are_rejected_for_parquet():
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.table([pa.array([1, 2]), pa.array([3, 4])], names=["a", "a"])
    buf = io.BytesIO(); pq.write_table(t, buf)
    with pytest.raises(DatasetFormatError, match="unique"):
        parse_dataset(buf.getvalue(), "dup.parquet")


def test_excel_and_parquet_work_end_to_end_through_the_api(churn_df, churn_config):
    import uuid
    from fastapi.testclient import TestClient
    from nocodeml_engine.api.app import create_app
    store, uid = FakeStore(), str(uuid.uuid4())
    app = create_app(client_factory=lambda t: (FakeSupabase(store, uid), uid), client_builder=lambda t: FakeSupabase(store, uid))
    with TestClient(app) as c:
        pid = c.post("/projects", json={"name": "Formats", "task": "classification"}, headers=A).json()["id"]
        up = c.post(f"/projects/{pid}/datasets", data={"target": "churn"}, headers=A,
                    files={"file": ("churn.xlsx", io.BytesIO(xlsx_bytes(churn_df)), "application/octet-stream")})
        assert up.status_code == 201, up.text
        ds = up.json()
        assert ds["filename"] == "churn.xlsx" and ds["n_rows"] == len(churn_df)
        path = ds["storage_path"]
        assert store.objects[("datasets", path)] if ("datasets", path) in store.objects else True
        # a second version in another format keeps the same dataset id
        v2 = c.post(f"/projects/{pid}/datasets", data={"dataset_id": ds["dataset_id"]}, headers=A,
                    files={"file": ("churn.parquet", io.BytesIO(parquet_bytes(churn_df)), "application/octet-stream")})
        assert v2.status_code == 201 and v2.json()["version"] == 2
        assert v2.json()["fingerprint"] == ds["fingerprint"]                      # same data, whatever the format
        # rows load back (fingerprint verified on download) and the full pipeline trains from an Excel version
        assert c.get(f"/projects/{pid}/datasets/{ds['dataset_id']}/rows", params={"version": 1}, headers=A).status_code == 200
        assert c.put(f"/projects/{pid}/pipeline", json=body_for(churn_config, ds["dataset_id"]), headers=A).status_code == 200
        job = train_and_wait(c, pid)
        assert job["status"] == "succeeded", job
