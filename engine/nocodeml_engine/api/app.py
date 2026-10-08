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
import threading
import time
import uuid as _uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Callable
from uuid import UUID

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse

from nocodeml_engine.api.jobs import Job, JobBusy, JobConflict, JobManager
from nocodeml_engine.api.ratelimit import RateLimited, RateLimiter
from nocodeml_engine.api.schemas import (
    ExperimentSummary, PipelineConfigIn, PipelineSaved, ProjectCreate, RowsPage, ValidationReport,
)
from nocodeml_engine.artifacts import export_artifacts
from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.feature_engineering import FeatureEngineeringError
from nocodeml_engine.models import MODEL_REGISTRY, ModelConfigError
from nocodeml_engine.models.registry import suggested_values, tunable_params
from nocodeml_engine.persistence import ProjectService, StorageError
from nocodeml_engine.persistence.projects import MAX_DATASET_BYTES
from nocodeml_engine.models import get_spec
from nocodeml_engine.preprocessing import PreprocessingError
from nocodeml_engine.preprocessing.preview import preview_models, preview_preprocessing
from nocodeml_engine.models import recommended_hyperparameters
from nocodeml_engine.preprocessing import prepare_frame
from nocodeml_engine.recommendations import recommend_features, recommend_preprocessing, recommend_split
from nocodeml_engine.reports import ReportContext, build_report
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.splitting import SplitError
from nocodeml_engine.state import PipelineError, VersionStatus
from nocodeml_engine import visualization as viz
from nocodeml_engine.colab.bundle import build_bundle
from nocodeml_engine.colab.importer import ResultsError, import_results
from nocodeml_engine.training import ConfigurationError, check_config, run_experiment

log = logging.getLogger("nocodeml.api")

ClientFactory = Callable[[str], tuple[object, str]]  # access_token -> (supabase client, user_id)

MAX_ROWS_PAGE = 200

# Requests allowed per rolling window. Generous for normal use, tight for the expensive operations.
IP_LIMIT = (600, 60)                 # everything, per client IP (stops floods before sign-in checks)
USER_LIMIT = (300, 60)               # everything signed-in, per user
HEAVY_LIMITS = {                     # name -> (requests, seconds), per user
    "upload": (10, 60),
    "train": (6, 60),
    "report": (6, 60),
    "preview": (60, 60),
}
PRODUCTION_ENV = "production"
EXPIRY_MARGIN_SECONDS = 120          # a sign-in must outlast the training time limit by this much to start a job
TRAIN_TIMEOUT_SECONDS = 600          # default cap on one training run (override: NOCODEML_TRAIN_TIMEOUT)


TOKEN_TTL_SECONDS = 15    # how long a validated token is trusted before Supabase is asked again
TOKEN_CACHE_SIZE = 512
_now = time.monotonic     # indirection so tests can move time


def token_seconds_left(token: str) -> float | None:
    """Seconds until a JWT access token expires, read from its `exp` claim (not verified: Supabase already did that).
    None if it can't be read (not a JWT, no exp)."""
    import base64
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(claims["exp"]) - time.time()
    except Exception:  # noqa: BLE001
        return None


class TokenCache:
    """Short-lived memo of validated access tokens -> user_id.

    Saves a network round trip to Supabase on every request. Only the *validation result* is cached:
    every request still gets its own client, because the Supabase client's HTTP/2 connections are not
    safe to share between threads (sharing one caused intermittent ReadErrors). The trade-off is
    explicit: a token revoked at Supabase (sign-out, deleted account) can be honoured for up to
    TOKEN_TTL_SECONDS longer.
    """

    def __init__(self, ttl: float = TOKEN_TTL_SECONDS, size: int = TOKEN_CACHE_SIZE):
        self.ttl, self.size, self._d, self._lock = ttl, size, OrderedDict(), threading.Lock()

    def get(self, token: str) -> str | None:
        with self._lock:
            hit = self._d.get(token)
            if hit and hit[1] > _now():
                self._d.move_to_end(token)
                return hit[0]
            self._d.pop(token, None)
            return None

    def put(self, token: str, user_id: str) -> None:
        with self._lock:
            self._d[token] = (user_id, _now() + self.ttl)
            self._d.move_to_end(token)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def _is_rejection(e: Exception) -> bool:
    """The auth service answered and said no (as opposed to not answering)."""
    if isinstance(e, PermissionError):
        return True
    status = getattr(e, "status", None) or getattr(e, "code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


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
               max_workers: int | None = None, client_builder: Callable[[str], object] | None = None,
               limiter: RateLimiter | None = None, train_timeout: float | None = None,
               rate_limits: dict | None = None, production: bool | None = None,
               trust_proxy: bool | None = None) -> FastAPI:
    """`client_factory(token)` validates a token and returns (client, user_id) (network call).
    `client_builder(token)` just builds a client for an already-validated token (no network call)."""
    if production is None:
        production = os.environ.get("NOCODEML_ENV", "").lower() == PRODUCTION_ENV
    if trust_proxy is None:
        trust_proxy = os.environ.get("NOCODEML_TRUST_PROXY", "").lower() in ("1", "true", "yes")
    if max_workers is None:     # how many trainings run at once: lower it on small servers (each needs RAM)
        max_workers = max(1, int(os.environ.get("NOCODEML_WORKERS", "2")))
    jobs = JobManager(max_workers=max_workers)
    cache = DataFrameCache(size=max(1, int(os.environ.get("NOCODEML_DATASET_CACHE", "6"))))
    tokens = TokenCache()
    limiter = limiter or RateLimiter()
    limits = {"ip": IP_LIMIT, "user": USER_LIMIT, **HEAVY_LIMITS, **(rate_limits or {})}
    timeout = train_timeout if train_timeout is not None else float(
        os.environ.get("NOCODEML_TRAIN_TIMEOUT", TRAIN_TIMEOUT_SECONDS))
    report_slots = threading.BoundedSemaphore(max_workers)  # PDF rendering is CPU/memory heavy: bounded like training

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.shutdown()

    # In production the interactive docs and the schema are not served (they advertise every endpoint).
    app = FastAPI(title="NoCodeML API", version="0.4.0", lifespan=lifespan,
                  docs_url=None if production else "/docs", redoc_url=None if production else "/redoc",
                  openapi_url=None if production else "/openapi.json")
    app.state.jobs = jobs

    if client_factory is None:
        from nocodeml_engine.persistence import SupabaseSettings, authenticated_client, build_client
        settings = SupabaseSettings.from_env()
        client_factory = lambda token: authenticated_client(settings, token)  # noqa: E731
        client_builder = client_builder or (lambda token: build_client(settings, token))
    if client_builder is None:  # without a cheap builder, every request validates (no caching benefit)
        client_builder = lambda token: client_factory(token)[0]  # noqa: E731

    # Must be registered before CORS so CORS wraps it: every error response then carries CORS headers
    # (otherwise the browser reports a generic "CORS" failure instead of the real message).
    @app.middleware("http")
    async def _catch_unexpected(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:  # noqa: BLE001
            log.exception("unhandled error")  # details stay in the server log, never in the response
            return JSONResponse(status_code=500, content={"detail": "Internal server error.", "issues": []})

    def client_ip(request: Request) -> str:
        """The caller's address. Behind a reverse proxy every request arrives from the proxy, so the real client is
        the *last* X-Forwarded-For entry (the one our own proxy appended; earlier ones can be forged by the caller).
        Only trusted when NOCODEML_TRUST_PROXY is set, otherwise a caller could pick their own address."""
        if trust_proxy:
            fwd = request.headers.get("x-forwarded-for", "")
            last = fwd.split(",")[-1].strip()
            if last:
                return last
        return request.client.host if request.client else "unknown"

    @app.middleware("http")
    async def _ip_limit(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path == "/health":
            return await call_next(request)
        ip = client_ip(request)
        try:
            limiter.check("ip", ip, *limits["ip"])
        except RateLimited as e:
            return _too_many(e)
        return await call_next(request)

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        """Request id on every response, one access-log line per request (no query string, no tokens), and
        conservative security headers. Registered last of the three, so it wraps the others and sees their responses."""
        rid = _uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - started) * 1000
        log.info("%s %s -> %s %.0fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"          # API responses are private to the signed-in user
        response.headers["Referrer-Policy"] = "no-referrer"
        if production:
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    origins = cors_origins if cors_origins is not None else [
        o.strip() for o in os.environ.get("NOCODEML_CORS_ORIGINS", "" if production else "http://localhost:3000").split(",")
        if o.strip()]
    if production:
        # Fail at startup rather than run with a browser policy that is wrong or wide open.
        bad = [o for o in origins if o == "*" or not o.startswith("https://")]
        if not origins or bad:
            raise RuntimeError("In production NOCODEML_CORS_ORIGINS must list the web app's https:// origin(s) "
                               f"(got {origins or 'nothing'}); '*' and http:// origins are not allowed.")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                       allow_headers=["Authorization", "Content-Type"], expose_headers=["Content-Disposition"])

    # ---- errors ----------------------------------------------------------

    def _too_many(e: RateLimited) -> JSONResponse:
        return JSONResponse(status_code=429, headers={"Retry-After": str(e.retry_after)},
                            content={"detail": f"You're going a bit fast. Please wait {e.retry_after} seconds and try again.",
                                     "issues": []})

    @app.exception_handler(RateLimited)
    async def _rate_limited(_: Request, e: RateLimited):
        return _too_many(e)

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

    @app.exception_handler(ResultsError)
    async def _results(_: Request, e: ResultsError):
        return err(422, str(e), e.issues)

    @app.exception_handler(PreprocessingError)
    async def _prep(_: Request, e: PreprocessingError):
        return err(422, "Preprocessing configuration is not valid.", e.issues)

    @app.exception_handler(viz.ChartError)
    async def _chart(_: Request, e: viz.ChartError):
        return err(422, str(e), [str(e)])

    @app.exception_handler(ConfigurationError)
    @app.exception_handler(SplitError)
    @app.exception_handler(ModelConfigError)
    @app.exception_handler(FeatureEngineeringError)
    async def _config(_: Request, e: Exception):
        return err(422, str(e), [str(e)])

    @app.exception_handler(httpx.TransportError)
    @app.exception_handler(ConnectionError)
    async def _upstream(_: Request, e: Exception):
        # The database/storage could not be reached (connection reset, timeout...). Transient: not a bug
        # and not the user's fault. The web app retries read requests automatically on 503.
        log.warning("upstream unavailable: %s: %s", type(e).__name__, str(e)[:200])
        return err(503, "Couldn't reach the database just now. Please try again in a moment.")

    # ---- dependencies ----------------------------------------------------

    bearer = HTTPBearer(auto_error=False)  # also gives /docs its "Authorize" button

    def authenticate(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> ProjectService:
        if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
            raise HTTPException(401, "Missing bearer token.")
        uid = tokens.get(creds.credentials)
        if uid:
            return ProjectService(client_builder(creds.credentials), uid)
        last: Exception | None = None
        for _attempt in range(2):                      # one retry: a network blip shouldn't sign anyone out
            try:
                client, user_id = client_factory(creds.credentials)
                tokens.put(creds.credentials, user_id)
                return ProjectService(client, user_id)
            except Exception as e:  # noqa: BLE001
                last = e
                if _is_rejection(e):
                    break
        log.warning("token validation failed: %s: %s", type(last).__name__, str(last)[:200])
        if last is not None and _is_rejection(last):
            raise HTTPException(401, "Invalid or expired token.") from None
        # The service could not be reached (or failed): that is not the user's fault and not a bad token.
        raise HTTPException(503, "The sign-in service is temporarily unreachable. Please try again in a moment.") from None

    def current_service(svc: ProjectService = Depends(authenticate)) -> ProjectService:
        limiter.check("user", svc.user_id, *limits["user"])
        return svc

    def heavy(name: str):
        """Dependency: count this request against the user's budget for an expensive operation."""
        def dep(svc: ProjectService = Depends(current_service)) -> None:
            limiter.check(name, svc.user_id, *limits[name])
        return dep

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
                    "tunable": [
                        {"name": h.name, "kind": h.kind, "default": h.default, "choices": list(h.choices),
                         "min": h.min, "max": h.max, "description": h.description, "suggested": suggested_values(h)}
                        for h in tunable_params(s).values() if h.name != "max_iter"],
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
        if jobs.active_for(svc.jobs(), pid):
            raise HTTPException(409, "Training is still running for this project. Wait for it to finish, then delete it.")
        svc.delete_project(pid)

    # ---- datasets --------------------------------------------------------

    @app.post("/projects/{project_id}/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), target: str | None = Form(default=None),
                             dataset_id: str | None = Form(default=None), ctx=Depends(project_ctx),
                             _rl=Depends(heavy("upload"))):
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

    # ---- charts (Section 0): small aggregated data, never raw rows --------------

    def chart_df(ctx, dataset_id: UUID):
        svc, _ = ctx
        return load_df(svc, str(dataset_id))

    @app.get("/projects/{project_id}/datasets/{dataset_id}/charts/auto")
    def charts_auto(dataset_id: UUID, target: str | None = None, ctx=Depends(project_ctx)):
        """A sensible default set of charts, each with the reason it was chosen."""
        df = chart_df(ctx, dataset_id)
        if target is not None and target not in df.columns:
            raise HTTPException(422, f"Unknown column '{target}'.")
        return viz.auto_charts(df, profile_dataset(df, target), target)

    @app.get("/projects/{project_id}/datasets/{dataset_id}/charts/distribution")
    def chart_distribution(dataset_id: UUID, column: str, bins: int = Query(30, ge=3, le=100), ctx=Depends(project_ctx)):
        return viz.histogram(chart_df(ctx, dataset_id), column, bins)

    @app.get("/projects/{project_id}/datasets/{dataset_id}/charts/categories")
    def chart_categories(dataset_id: UUID, column: str, top: int = Query(12, ge=3, le=30), ctx=Depends(project_ctx)):
        return viz.categories(chart_df(ctx, dataset_id), column, top)

    @app.get("/projects/{project_id}/datasets/{dataset_id}/charts/correlation")
    def chart_correlation(dataset_id: UUID, target: str | None = None, columns: list[str] | None = Query(default=None, max_length=30),
                          ctx=Depends(project_ctx)):
        return viz.correlation(chart_df(ctx, dataset_id), target, columns)

    @app.get("/projects/{project_id}/datasets/{dataset_id}/charts/missing")
    def chart_missing(dataset_id: UUID, ctx=Depends(project_ctx)):
        return viz.missing_overview(chart_df(ctx, dataset_id))

    @app.get("/projects/{project_id}/datasets/{dataset_id}/charts/scatter")
    def chart_scatter(dataset_id: UUID, x: str, y: str, ctx=Depends(project_ctx)):
        return viz.scatter(chart_df(ctx, dataset_id), x, y)

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

    @app.get("/projects/{project_id}/pipeline/recommendations/preprocessing")
    def preprocessing_recommendations(ctx=Depends(project_ctx)):
        """Deterministic suggestions with reasons. Nothing is applied: the UI lets the user choose."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        if not pipes.repo.list_versions(pid):
            raise HTTPException(409, "Save a target column in the Dataset section first.")
        cfg = pipes.latest(pid).config
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        profile = profile_dataset(df, cfg.dataset.target_column)
        scaling = (any(get_spec(m.model_key).requires_scaling for m in cfg.models) if cfg.models else None)
        recs = recommend_preprocessing(profile, cfg.dataset.target_column, scaling)
        return [r.__dict__ for r in recs]

    @app.get("/projects/{project_id}/pipeline/recommendations/split")
    def split_recommendations(ctx=Depends(project_ctx)):
        """How to split, with reasons. `optional` ones are shown but not ticked by default."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        if not pipes.repo.list_versions(pid):
            raise HTTPException(409, "Save a target column in the Dataset section first.")
        cfg = pipes.latest(pid).config
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        profile = profile_dataset(df, cfg.dataset.target_column)
        n_rows = len(prepare_frame(df, cfg.preprocessing, cfg.dataset.target_column)[0])
        return [r.__dict__ for r in recommend_split(profile, cfg.dataset.task, n_rows)]

    @app.get("/projects/{project_id}/pipeline/recommendations/features")
    def feature_recommendations(ctx=Depends(project_ctx)):
        """Feature-engineering suggestions with reasons (log of skewed columns, date parts)."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        if not pipes.repo.list_versions(pid):
            raise HTTPException(409, "Save a target column in the Dataset section first.")
        cfg = pipes.latest(pid).config
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        profile = profile_dataset(df, cfg.dataset.target_column)
        return [r.__dict__ for r in recommend_features(profile, cfg.dataset.target_column, cfg.preprocessing.drop_columns)]

    @app.get("/projects/{project_id}/pipeline/model-defaults")
    def model_defaults(ctx=Depends(project_ctx)):
        """For each selected model: the registry defaults and the values NoCodeML would recommend."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        if not pipes.repo.list_versions(pid):
            raise HTTPException(409, "Save a target column in the Dataset section first.")
        cfg = pipes.latest(pid).config
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        n_rows = len(prepare_frame(df, cfg.preprocessing, cfg.dataset.target_column)[0])
        n_features = max(1, df.shape[1] - 1)
        return {m.model_key: {
            "recommended": recommended_hyperparameters(m.model_key, n_rows, n_features),
            "defaults": {h.name: h.default for h in get_spec(m.model_key).hyperparameters},
            "n_rows": n_rows, "n_features": n_features} for m in cfg.models}

    @app.post("/projects/{project_id}/pipeline/validate-models")
    def validate_models(body: PipelineConfigIn, ctx=Depends(project_ctx), _rl=Depends(heavy("preview"))):
        """Instant check of each model's settings. Does not touch the dataset, so it stays fast."""
        _, pid = ctx
        return {"model_issues": preview_models(full_config(pid, body))}

    @app.post("/projects/{project_id}/pipeline/preview")
    def preview_pipeline(body: PipelineConfigIn, ctx=Depends(project_ctx), _rl=Depends(heavy("preview"))):
        """Validate an unsaved draft and show before/after data shape (preview only, nothing saved)."""
        svc, pid = ctx
        cfg = full_config(pid, body)
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        return preview_preprocessing(df, cfg)

    @app.post("/projects/{project_id}/pipeline/finalize")
    def finalize(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.pipelines(pid).finalize(pid)

    # ---- training --------------------------------------------------------

    def train_job(svc: ProjectService, pid: str):
        def work(job: Job, cancel: threading.Event) -> int:
            pipes = svc.pipelines(pid)
            cfg = pipes.latest(pid).config
            df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
            run = run_experiment(df, cfg, deadline=time.monotonic() + timeout, cancel=cancel)
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
    def start_training(ctx=Depends(project_ctx), _rl=Depends(heavy("train")),
                       creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
        """Queue training of the latest pipeline version. Poll the returned job."""
        svc, pid = ctx
        # The job saves its results with the caller's own token. If that token would expire mid-run, the results
        # could not be saved, so refuse up front with something the user can act on.
        left = token_seconds_left(creds.credentials) if creds else None
        if left is not None and left < timeout + EXPIRY_MARGIN_SECONDS:
            raise HTTPException(401, "Your sign-in is about to expire. Reload the page and try again.")
        pipes = svc.pipelines(pid)
        latest = pipes.latest(pid)
        if not latest.config.models:
            raise HTTPException(422, "Select at least one model before training.")
        try:
            return jobs.submit(svc.jobs(), pid, train_job(svc, pid))
        except JobConflict as e:
            raise HTTPException(409, str(e)) from None
        except JobBusy as e:
            raise HTTPException(503, str(e)) from None

    # ---- training on Google Colab ------------------------------------------------------------------

    MAX_RESULTS_BYTES = max(1, int(os.environ.get("NOCODEML_MAX_RESULTS_MB", "50"))) * 2**20

    @app.post("/projects/{project_id}/colab/bundle")
    def colab_bundle(ctx=Depends(project_ctx), _rl=Depends(heavy("train"))):
        """A zip with the prepared train/test files and the models to train, for the Colab notebook."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        cfg = pipes.latest(pid).config
        if not cfg.models:
            raise HTTPException(422, "Select at least one model before training.")
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        problems = check_config(df, cfg)
        if problems:
            return err(422, "The pipeline can't be trained yet.", problems)
        data, _manifest = build_bundle(df, cfg)
        return Response(content=data, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="nocodeml_bundle_v{cfg.version}.zip"'})

    @app.post("/projects/{project_id}/colab/results", status_code=201)
    async def colab_results(file: UploadFile = File(...), ctx=Depends(project_ctx), _rl=Depends(heavy("train"))):
        """Turn the predictions a Colab run produced into a normal experiment. All scoring happens here."""
        svc, pid = ctx
        raw = await file.read(MAX_RESULTS_BYTES + 1)
        if len(raw) > MAX_RESULTS_BYTES:
            raise HTTPException(413, f"The results file is larger than {MAX_RESULTS_BYTES // 2**20} MB.")
        try:
            results = json.loads(raw)
        except ValueError:
            raise HTTPException(422, "That file isn't valid JSON. Upload the nocodeml_results.json the notebook produced.") from None
        return await run_in_threadpool(_record_colab_results, svc, pid, results)

    def _record_colab_results(svc: ProjectService, pid: str, results):
        pipes = svc.pipelines(pid)
        latest = pipes.latest(pid)
        cfg = latest.config
        df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
        result = import_results(df, cfg, results)
        exp = pipes.record_experiment(result)
        return {"experiment_number": exp.number, "source": "colab"}

    @app.get("/projects/{project_id}/training/active")
    def active_training(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return jobs.active_for(svc.jobs(), pid)

    @app.get("/projects/{project_id}/training/{job_id}")
    def training_status(job_id: str, ctx=Depends(project_ctx)):
        svc, pid = ctx
        job = jobs.get(svc.jobs(), job_id, pid)
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    @app.post("/projects/{project_id}/training/{job_id}/cancel")
    def cancel_training(job_id: str, ctx=Depends(project_ctx)):
        """Ask a queued or running training to stop. It stops at its next step and nothing is saved."""
        svc, pid = ctx
        job = jobs.cancel(svc.jobs(), job_id, pid)
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    @app.get("/projects/{project_id}/training")
    def training_history(ctx=Depends(project_ctx)):
        svc, pid = ctx
        return svc.jobs().recent(pid)

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
            quality_score=e.result.quality.score, source=e.result.source,
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

    @app.post("/projects/{project_id}/experiments/{number}/report")
    def create_report(number: int, response: Response, force: bool = False, ctx=Depends(project_ctx),
                      _rl=Depends(heavy("report"))):
        """Generate (or return the existing) PDF report for an experiment. Deterministic, template-based."""
        svc, pid = ctx
        pipes = svc.pipelines(pid)
        exp = next((e for e in pipes.repo.list_experiments(pid) if e.number == number), None)
        if exp is None:
            raise HTTPException(404, "Experiment not found.")

        def existing():
            return next((a for a in svc.list_artifacts(pid, number) if a["kind"] == "report"), None)

        found = existing()
        if found and not force:
            return found
        if not report_slots.acquire(timeout=30):
            raise HTTPException(503, "The server is busy generating other reports. Try again in a moment.")
        try:
            version = pipes.repo.get_version(pid, exp.pipeline_version)
            cfg = version.config
            df = load_df(svc, cfg.dataset.dataset_id, cfg.dataset.version)
            ds_row = next((v for v in svc.dataset_versions(cfg.dataset.dataset_id) if v["version"] == cfg.dataset.version), None)
            label = f"v{version.version}.0 (finalized)" if version.status == VersionStatus.FINALIZED else f"v{version.version}"
            prof = profile_dataset(df, cfg.dataset.target_column)
            try:
                dataset_charts = viz.auto_charts(df, prof, cfg.dataset.target_column)
            except Exception:  # noqa: BLE001 - the report is still useful without its dataset charts
                log.exception("dataset charts for report failed")
                dataset_charts = []
            pdf = build_report(ReportContext(
                project_name=svc.get_project(pid)["name"], experiment_number=number, result=exp.result, config=cfg,
                pipeline_label=label, dataset_filename=(ds_row or {}).get("filename", "dataset.csv"),
                dataset_version=cfg.dataset.version, profile=prof.to_dict(),
                current=pipes.is_current(exp), extras={"dataset_charts": dataset_charts}))
        finally:
            report_slots.release()
        if found:  # force: replace
            svc.delete_artifact(found["id"])
        try:
            row = svc.save_artifact_bytes(pid, number, f"report_exp{number}.pdf", pdf, "report", "reports", "application/pdf")
        except Exception:
            again = existing()  # lost a race with a concurrent request: that one wins
            if again:
                return again
            raise
        response.status_code = 201
        return row

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
