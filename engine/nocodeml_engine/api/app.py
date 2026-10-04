"""NoCodeML HTTP API.

Every request that touches user data carries `Authorization: Bearer <supabase access token>`.
The token is exchanged for a Supabase client acting as that user, so row-level security in the
database is the real authorization boundary; the checks here are for clean error messages.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Callable
from uuid import UUID

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse

from nocodeml_engine.api.jobs import Job, JobConflict, JobManager
from nocodeml_engine.api.schemas import (
    ExperimentSummary, PipelineConfigIn, PipelineSaved, ProjectCreate, RowsPage, ValidationReport,
)
from nocodeml_engine.artifacts import export_artifacts
from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.feature_engineering import FeatureEngineeringError
from nocodeml_engine.models import MODEL_REGISTRY, ModelConfigError
from nocodeml_engine.persistence import ProjectService, StorageError
from nocodeml_engine.persistence.projects import MAX_DATASET_BYTES
from nocodeml_engine.preprocessing import PreprocessingError
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.splitting import SplitError
from nocodeml_engine.state import PipelineError, VersionStatus
from nocodeml_engine.training import ConfigurationError, check_config, run_experiment

log = logging.getLogger("nocodeml.api")

ClientFactory = Callable[[str], tuple[object, str]]  # access_token -> (supabase client, user_id)

MAX_ROWS_PAGE = 200


class DataFrameCache:
    """Small LRU of verified dataset versions (immutable, so safe to cache). Key includes user."""

    def __init__(self, size: int = 6):
        self.size, self._d = size, OrderedDict()

    def get(self, key, loader):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        df = loader()
        self._d[key] = df
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return df


def _headline(model_result) -> dict:
    pm = model_result.primary_metric
    m = model_result.metrics.get("test") or model_result.metrics.get("cv") or {}
    return {"model_key": model_result.model_key, "name": model_result.name, "metric": pm,
            "value": m.get(pm)}


def create_app(client_factory: ClientFactory | None = None, cors_origins: list[str] | None = None,
               max_workers: int = 2) -> FastAPI:
    jobs = JobManager(max_workers=max_workers)
    cache = DataFrameCache()

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.shutdown()

    app = FastAPI(title="NoCodeML API", version="0.4.0", lifespan=lifespan)
    app.state.jobs = jobs

    if client_factory is None:
        from nocodeml_engine.persistence import SupabaseSettings, authenticated_client
        settings = SupabaseSettings.from_env()
        client_factory = lambda token: authenticated_client(settings, token)  # noqa: E731

    origins = cors_origins if cors_origins is not None else [
        o for o in os.environ.get("NOCODEML_CORS_ORIGINS", "http://localhost:3000").split(",") if o]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                       allow_headers=["Authorization", "Content-Type"])

    # ---- errors ----------------------------------------------------------

    def err(status: int, message: str, issues: list[str] | None = None):
        return JSONResponse(status_code=status, content={"detail": message, "issues": issues or []})

    @app.exception_handler(StorageError)
    async def _storage(_: Request, e: StorageError):
        msg = str(e)
        return err(404 if "not found" in msg.lower() else 400, msg)

    @app.exception_handler(PipelineError)
    async def _pipeline(_: Request, e: PipelineError):
        msg = str(e)
        return err(404 if "not found" in msg.lower() else 409, msg)

    @app.exception_handler(PreprocessingError)
    async def _prep(_: Request, e: PreprocessingError):
        return err(422, "Preprocessing configuration is not valid.", e.issues)

    @app.exception_handler(ConfigurationError)
    @app.exception_handler(SplitError)
    @app.exception_handler(ModelConfigError)
    @app.exception_handler(FeatureEngineeringError)
    async def _config(_: Request, e: Exception):
        return err(422, str(e), [str(e)])

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, e: Exception):
        log.exception("unhandled error")  # details stay in the server log, never in the response
        return err(500, "Internal server error.")

    # ---- dependencies ----------------------------------------------------

    bearer = HTTPBearer(auto_error=False)  # also gives /docs its "Authorize" button

    def current_service(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> ProjectService:
        if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
            raise HTTPException(401, "Missing bearer token.")
        try:
            client, user_id = client_factory(creds.credentials)
        except Exception:
            raise HTTPException(401, "Invalid or expired token.") from None
        return ProjectService(client, user_id)

    def project_ctx(project_id: UUID, svc: ProjectService = Depends(current_service)):
        pid = str(project_id)
        svc.get_project(pid)  # 404 for missing *and* foreign projects
        return svc, pid

    def load_df(svc: ProjectService, dataset_id: str, version: int | None = None) -> pd.DataFrame:
        versions = svc.dataset_versions(dataset_id)
        if not versions:
            raise StorageError("Dataset not found")
        row = versions[-1] if version is None else next(
            (v for v in versions if v["version"] == version), None)
        if row is None:
            raise StorageError("Dataset version not found")
        return cache.get((svc.user_id, row["storage_path"]),
                         lambda: svc.load_dataset(dataset_id, row["version"]))

    # ---- public ----------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/models")
    def models(task: str | None = Query(default=None, pattern="^(classification|regression)$")):
        """Registry metadata so the UI can render model-aware forms."""
        out = []
        for s in MODEL_REGISTRY.values():
            for t in s.tasks:
                if task and t.value != task:
                    continue
                reg = s.regularization[t]
                out.append({
                    "key": s.key, "name": s.name, "task": t.value,
                    "requires_scaling": s.requires_scaling, "cost": s.cost,
                    "interpretability": s.interpretability,
                    "hyperparameters": [
                        {"name": h.name, "kind": h.kind, "default": h.default,
                         "choices": list(h.choices), "min": h.min, "max": h.max,
                         "advanced": h.advanced, "description": h.description}
                        for h in s.hyperparameters],
                    "regularization": {"kind": reg.kind, "options": list(reg.options),
                                       "complexity_params": list(reg.complexity_params),
                                       "note": reg.note}})
        return out

    # ---- projects --------------------------------------------------------

    @app.post("/projects", status_code=201)
    def create_project(body: ProjectCreate, svc: ProjectService = Depends(current_service)):
        return svc.create_project(body.name, body.description, body.task)

    @app.get("/projects")
    def list_projects(svc: ProjectService = Depends(current_service)):
        return svc.list_projects()

    @app.get("/projects/{project_id}")
    def open_project(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.open_project(pid)

    @app.delete("/projects/{project_id}", status_code=204)
    def delete_project(ctx=Depends(project_ctx)):
        svc, pid = ctx
        svc.delete_project(pid)

    # ---- datasets --------------------------------------------------------

    @app.post("/projects/{project_id}/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), target: str | None = Form(default=None),
                             dataset_id: str | None = Form(default=None), ctx=Depends(project_ctx)):
        svc, pid = ctx
        data = await file.read(MAX_DATASET_BYTES + 1)  # never buffer more than the limit
        return svc.add_dataset(pid, file.filename or "dataset.csv", data, target, dataset_id)

    @app.get("/projects/{project_id}/datasets/{dataset_id}")
    def dataset_info(dataset_id: UUID, ctx=Depends(project_ctx)):
        """Versions of one dataset (metadata only; the profile has its own endpoint)."""
        svc, pid = ctx
        versions = [v for v in svc.dataset_versions(str(dataset_id)) if v["project_id"] == pid]
        if not versions:
            raise HTTPException(404, "Dataset not found.")
        keep = ("version", "filename", "n_rows", "n_columns", "size_bytes", "created_at")
        return {"dataset_id": str(dataset_id), "versions": [{k: v[k] for k in keep} for v in versions]}

    @app.get("/projects/{project_id}/datasets/{dataset_id}/profile")
    def dataset_profile(dataset_id: UUID, target: str | None = None, ctx=Depends(project_ctx)):
        svc, pid = ctx
        df = load_df(svc, str(dataset_id))
        try:
            prof = profile_dataset(df, target)
        except KeyError as e:
            raise HTTPException(422, str(e).strip("'\"")) from None
        return json.loads(json.dumps(prof.to_dict(), default=str))

    @app.get("/projects/{project_id}/datasets/{dataset_id}/rows", response_model=RowsPage)
    def dataset_rows(dataset_id: UUID, page: int = Query(1, ge=1),
                     page_size: int = Query(50, ge=1, le=MAX_ROWS_PAGE),
                     sort_by: str | None = None, descending: bool = False,
                     search: str | None = Query(default=None, max_length=200),
                     ctx=Depends(project_ctx)):
        """Paginated raw-data viewer. Never returns more than MAX_ROWS_PAGE rows."""
        svc, pid = ctx
        df = load_df(svc, str(dataset_id))
        if sort_by is not None and sort_by not in df.columns:
            raise HTTPException(422, f"Unknown column '{sort_by}'.")
        if search:
            mask = df.astype(str).apply(lambda c: c.str.contains(search, case=False, regex=False)).any(axis=1)
            df = df[mask]
        if sort_by:
            df = df.sort_values(sort_by, ascending=not descending, kind="stable", na_position="last")
        total = len(df)
        part = df.iloc[(page - 1) * page_size: page * page_size]
        rows = json.loads(part.to_json(orient="records", date_format="iso"))
        return RowsPage(columns=[{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
                        rows=rows, total=total, page=page, page_size=page_size)

    # ---- pipeline --------------------------------------------------------

    def full_config(pid: str, body: PipelineConfigIn) -> PipelineConfig:
        return PipelineConfig(pipeline_id=pid, **body.model_dump())

    @app.put("/projects/{project_id}/pipeline", response_model=PipelineSaved)
    def save_pipeline(body: PipelineConfigIn, ctx=Depends(project_ctx)):
        """Create the pipeline, or commit an edit. Edits create a new version and report impact."""
        svc, pid = ctx
        if not any(v["version"] == body.dataset.version
                   for v in svc.dataset_versions(body.dataset.dataset_id)):
            raise HTTPException(422, "dataset_id/version does not exist in this project.")
        pipes = svc.pipelines(pid)
        cfg = full_config(pid, body)
        if not pipes.repo.list_versions(pid):
            return PipelineSaved(version=pipes.create(cfg), created=True)
        v, impact = pipes.update(pid, cfg)
        return PipelineSaved(version=v, created=False, impact=impact)

    @app.get("/projects/{project_id}/pipeline")
    def get_pipeline(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.pipelines(pid).latest(pid)

    @app.get("/projects/{project_id}/pipeline/versions")
    def pipeline_versions(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.pipelines(pid).repo.list_versions(pid)

    @app.post("/projects/{project_id}/pipeline/validate", response_model=ValidationReport)
    def validate_pipeline(ctx=Depends(project_ctx)):
        """Dry run of the latest version: what would stop training, and why."""
        svc, pid = ctx
        cfg = svc.pipelines(pid).latest(pid).config
        issues = check_config(load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version), cfg)
        return ValidationReport(valid=not issues, issues=issues)

    @app.post("/projects/{project_id}/pipeline/finalize")
    def finalize(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.pipelines(pid).finalize(pid)

    # ---- training --------------------------------------------------------

    def train_job(svc: ProjectService, pid: str):
        def work(job: Job) -> int:
            pipes = svc.pipelines(pid)
            cfg = pipes.latest(pid).config
            df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
            run = run_experiment(df, cfg)
            exp = pipes.record_experiment(run.result)
            try:
                with tempfile.TemporaryDirectory() as d:
                    export_artifacts(run, cfg, d)
                    svc.save_artifacts(pid, exp.number, d)
            except Exception:  # noqa: BLE001
                log.exception("artifact upload failed")
                job.warnings.append("Results were saved, but uploading the downloadable "
                                    "artifacts failed.")
            return exp.number
        return work

    @app.post("/projects/{project_id}/training", status_code=202)
    def start_training(ctx=Depends(project_ctx)):
        """Queue training of the latest pipeline version. Poll the returned job."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        latest = pipes.latest(pid)
        if not latest.config.models:
            raise HTTPException(422, "Select at least one model before training.")
        try:
            return jobs.submit(pid, svc.user_id, train_job(svc, pid))
        except JobConflict as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/projects/{project_id}/training/{job_id}")
    def training_status(job_id: str, ctx=Depends(project_ctx)):
        svc, pid = ctx
        job = jobs.get(job_id, pid, svc.user_id)
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    # ---- experiments -----------------------------------------------------

    @app.get("/projects/{project_id}/experiments", response_model=list[ExperimentSummary])
    def list_experiments(ctx=Depends(project_ctx)):
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        if not pipes.repo.list_versions(pid):
            return []
        current = {e.number for e in pipes.current_experiments(pid)}
        return [ExperimentSummary(
            number=e.number, experiment_id=e.result.experiment_id,
            pipeline_version=e.pipeline_version, parent_number=e.parent_number,
            current=e.number in current, finished_at=e.result.finished_at,
            quality_score=e.result.quality.score,
            models=[_headline(m) for m in e.result.models])
            for e in pipes.repo.list_experiments(pid)]

    @app.get("/projects/{project_id}/experiments/{number}")
    def get_experiment(number: int, ctx=Depends(project_ctx)):
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        exp = next((e for e in pipes.repo.list_experiments(pid) if e.number == number), None)
        if exp is None:
            raise HTTPException(404, "Experiment not found.")
        return {"experiment": exp, "current": pipes.is_current(exp)}

    @app.post("/projects/{project_id}/experiments/{number}/duplicate")
    def duplicate_experiment(number: int, ctx=Depends(project_ctx)):
        svc, pid = ctx
        v, parent = svc.pipelines(pid).duplicate(pid, number)
        return {"version": v, "parent_number": parent}

    @app.get("/projects/{project_id}/compare")
    def compare(a: int, b: int, ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.pipelines(pid).compare(pid, a, b)

    @app.get("/projects/{project_id}/experiments/{number}/artifacts")
    def experiment_artifacts(number: int, ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.list_artifacts(pid, number)

    @app.get("/projects/{project_id}/artifacts/{artifact_id}/url")
    def artifact_url(artifact_id: UUID, ctx=Depends(project_ctx)):
        svc, pid = ctx
        if not any(a["id"] == str(artifact_id) for a in svc.list_artifacts(pid)):
            raise HTTPException(404, "Artifact not found.")
        return {"url": svc.signed_url(str(artifact_id)), "expires_in": 300}

    return app
