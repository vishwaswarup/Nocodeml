"""Projects, datasets (object storage + metadata) and artifacts on Supabase."""

from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from nocodeml_engine.dataset.loader import infer_datetimes
from nocodeml_engine.persistence.repository import SupabasePipelineRepository
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.state import PipelineService, VersionStatus
from nocodeml_engine.state.store import Experiment, PipelineVersion
from nocodeml_engine.training.runner import dataset_fingerprint

MAX_DATASET_BYTES = 100 * 1024 * 1024  # matches the `datasets` bucket limit (cloud training cap)

_ARTIFACT_KINDS = [  # (filename pattern, kind, bucket)
    (re.compile(r"^pipeline_.*\.pkl$"), "pipeline", "pipelines"),
    (re.compile(r"^model_.*\.pkl$"), "model", "models"),
    (re.compile(r"^configuration\.json$"), "configuration", "pipelines"),
    (re.compile(r"^metrics\.json$"), "metrics", "pipelines"),
    (re.compile(r"^report.*\.pdf$"), "report", "reports"),
]


class StorageError(RuntimeError):
    pass


class ProjectSummary(BaseModel):
    id: str
    name: str
    updated_at: str
    pipeline_version: str | None  # e.g. "v4" or "v1.0"
    status: str  # "Empty" | "Experimenting" | "Finalized"


class DatasetRecord(BaseModel):
    dataset_id: str
    version_id: str
    version: int
    filename: str
    storage_path: str
    n_rows: int
    n_columns: int
    size_bytes: int
    fingerprint: str


class ProjectState(BaseModel):
    project: dict
    datasets: list[dict]
    pipeline: PipelineVersion | None
    experiments: list[Experiment]  # only those valid for the current pipeline version


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "dataset.csv"
    return base[:120]


class ProjectService:
    def __init__(self, client, user_id: str):
        self.db = client
        self.user_id = str(user_id)

    # -- projects ----------------------------------------------------------

    def create_project(self, name: str, description: str = "", task: str | None = None) -> dict:
        # owner_id is NOT sent: the column defaults to auth.uid() and RLS enforces it.
        row = {"name": name, "description": description}
        if task:
            row["task"] = task
        return self.db.table("projects").insert(row).execute().data[0]

    def list_projects(self) -> list[ProjectSummary]:
        projects = self.db.table("projects").select("*").order("updated_at", desc=True).execute().data
        if not projects:
            return []
        versions = self.db.table("pipeline_versions").select("project_id,version,status").execute().data
        latest: dict[str, dict] = {}
        for v in versions:
            if v["project_id"] not in latest or v["version"] > latest[v["project_id"]]["version"]:
                latest[v["project_id"]] = v
        out = []
        for p in projects:
            v = latest.get(p["id"])
            if v is None:
                label, status = None, "Empty"
            elif v["status"] == "finalized":
                label, status = f"v{v['version']}.0", "Finalized"
            else:
                label, status = f"v{v['version']}", "Experimenting"
            out.append(ProjectSummary(id=p["id"], name=p["name"], updated_at=p["updated_at"],
                                      pipeline_version=label, status=status))
        return out

    def pipelines(self, project_id: str) -> PipelineService:
        return PipelineService(SupabasePipelineRepository(self.db))

    def open_project(self, project_id: str) -> ProjectState:
        """Restore everything needed to continue where the user left off."""
        rows = self.db.table("projects").select("*").eq("id", project_id).limit(1).execute().data
        if not rows:
            raise StorageError("Project not found")  # also what a non-owner sees (RLS)
        svc = self.pipelines(project_id)
        versions = svc.repo.list_versions(project_id)
        datasets = self.db.table("datasets").select("*").eq("project_id", project_id).execute().data
        return ProjectState(project=rows[0], datasets=datasets,
                            pipeline=versions[-1] if versions else None,
                            experiments=svc.current_experiments(project_id) if versions else [])

    # -- datasets ----------------------------------------------------------

    def add_dataset(self, project_id: str, filename: str, data: bytes,
                    target: str | None = None, dataset_id: str | None = None) -> DatasetRecord:
        """Validate, profile, upload and register a CSV. Pass dataset_id to add a new version."""
        if not filename.lower().endswith(".csv"):
            raise StorageError("Only .csv files are supported in this version.")
        if len(data) > MAX_DATASET_BYTES:
            raise StorageError(f"Dataset exceeds the {MAX_DATASET_BYTES // 2**20} MB limit.")
        try:
            df = infer_datetimes(pd.read_csv(io.BytesIO(data)))
        except Exception as e:  # malformed upload: surface a clean error, store nothing
            raise StorageError(f"Could not parse CSV: {e}") from e
        if df.empty:
            raise StorageError("The CSV contains no rows.")

        if dataset_id is None:
            dataset_id = self.db.table("datasets").insert(
                {"project_id": project_id, "name": safe_filename(filename)}).execute().data[0]["id"]
            version = 1
        else:
            existing = self.db.table("dataset_versions").select("version").eq(
                "dataset_id", dataset_id).execute().data
            version = max((r["version"] for r in existing), default=0) + 1

        fp = dataset_fingerprint(df)
        path = f"{self.user_id}/{project_id}/{dataset_id}/v{version}/{safe_filename(filename)}"
        profile = json.loads(json.dumps(profile_dataset(df, target).to_dict(), default=str))
        try:
            self.db.storage.from_("datasets").upload(path, data, {"content-type": "text/csv"})
        except Exception as e:
            raise StorageError(f"Upload failed: {e}") from e
        try:
            row = self.db.table("dataset_versions").insert({
                "project_id": project_id, "dataset_id": dataset_id, "version": version,
                "storage_path": path, "filename": safe_filename(filename), "size_bytes": len(data),
                "n_rows": len(df), "n_columns": df.shape[1], "fingerprint": fp,
                "profile": profile}).execute().data[0]
        except Exception:
            self.db.storage.from_("datasets").remove([path])  # no orphaned objects
            raise
        return DatasetRecord(dataset_id=dataset_id, version_id=row["id"], version=version,
                             filename=row["filename"], storage_path=path, n_rows=len(df),
                             n_columns=df.shape[1], size_bytes=len(data), fingerprint=fp)

    def load_dataset(self, dataset_id: str, version: int | None = None) -> pd.DataFrame:
        """Download a dataset version and verify it is byte-for-byte what was profiled."""
        q = self.db.table("dataset_versions").select("*").eq("dataset_id", dataset_id)
        rows = q.order("version", desc=True).execute().data
        if version is not None:
            rows = [r for r in rows if r["version"] == version]
        if not rows:
            raise StorageError("Dataset version not found")
        r = rows[0]
        df = infer_datetimes(pd.read_csv(io.BytesIO(
            self.db.storage.from_("datasets").download(r["storage_path"]))))
        if dataset_fingerprint(df) != r["fingerprint"]:
            raise StorageError("Dataset content does not match its recorded fingerprint.")
        return df

    # -- artifacts ---------------------------------------------------------

    def save_artifacts(self, project_id: str, experiment_number: int, export_dir: str | Path) -> list[dict]:
        out = []
        for f in sorted(Path(export_dir).iterdir()):
            match = next(((k, b) for rx, k, b in _ARTIFACT_KINDS if rx.match(f.name)), None)
            kind, bucket = match or ("other", "pipelines")
            data = f.read_bytes()
            path = f"{self.user_id}/{project_id}/exp{experiment_number}/{f.name}"
            self.db.storage.from_(bucket).upload(path, data)
            out.append(self.db.table("artifacts").insert({
                "project_id": project_id, "experiment_number": experiment_number, "kind": kind,
                "bucket": bucket, "storage_path": path, "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}).execute().data[0])
        return out

    def download_artifact(self, artifact_id: str) -> bytes:
        rows = self.db.table("artifacts").select("*").eq("id", artifact_id).limit(1).execute().data
        if not rows:
            raise StorageError("Artifact not found")
        a = rows[0]
        data = self.db.storage.from_(a["bucket"]).download(a["storage_path"])
        if a["sha256"] and hashlib.sha256(data).hexdigest() != a["sha256"]:
            raise StorageError("Artifact failed its integrity check.")
        return data

    def signed_url(self, artifact_id: str, expires_in: int = 300) -> str:
        """Short-lived download link. Raw object URLs are never exposed (buckets are private)."""
        rows = self.db.table("artifacts").select("*").eq("id", artifact_id).limit(1).execute().data
        if not rows:
            raise StorageError("Artifact not found")
        res = self.db.storage.from_(rows[0]["bucket"]).create_signed_url(rows[0]["storage_path"], expires_in)
        return res.get("signedURL") or res["signedUrl"]
