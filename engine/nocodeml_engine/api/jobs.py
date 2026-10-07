"""Background training jobs.

MVP design: an in-process thread pool. Job *status* is in memory (lost on restart), but every
finished experiment is persisted in the database, so results are never lost. Replace with a
real queue (Celery/RQ) when training moves off the API process.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel, Field

MAX_TRACKED_JOBS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_network_error(e: BaseException) -> bool:
    try:
        import httpx
        if isinstance(e, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(e, (ConnectionError, TimeoutError))


class Job(BaseModel):
    id: str
    project_id: str
    owner_id: str
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    experiment_number: int | None = None
    error: str | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JobConflict(RuntimeError):
    pass


class JobBusy(RuntimeError):
    """Too many jobs are already waiting; the user should try again shortly."""


class JobManager:
    def __init__(self, max_workers: int = 2, max_waiting: int = 20):
        self._max_waiting = max_waiting
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, project_id: str, owner_id: str, fn: Callable[[Job], int]) -> Job:
        """fn(job) runs in a worker and returns the experiment number."""
        with self._lock:
            if any(j.project_id == project_id and j.status in ("queued", "running")
                   for j in self._jobs.values()):
                raise JobConflict("A training job is already running for this project.")
            waiting = sum(1 for j in self._jobs.values() if j.status == "queued")
            if waiting >= self._max_waiting:
                raise JobBusy("The training queue is full right now. Please try again in a few minutes.")
            job = Job(id=uuid.uuid4().hex, project_id=project_id, owner_id=owner_id,
                      created_at=_now())
            self._jobs[job.id] = job
            for old in list(self._jobs)[:-MAX_TRACKED_JOBS]:
                if self._jobs[old].status in ("succeeded", "failed"):
                    del self._jobs[old]
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], int]) -> None:
        job.status, job.started_at = "running", _now()
        try:
            job.experiment_number = fn(job)
            job.status = "succeeded"
        except Exception as e:  # noqa: BLE001 - surfaced to the user, see messages below
            job.status = "failed"
            issues = getattr(e, "issues", None)
            if issues:
                job.issues = list(issues)
            if _is_network_error(e):
                job.error = "Lost the connection to the database while training. Nothing was saved; please train again."
            else:
                job.error = str(e) if isinstance(e, (ValueError, RuntimeError)) else "Training failed unexpectedly."
        finally:
            job.finished_at = _now()

    def get(self, job_id: str, project_id: str, owner_id: str) -> Job | None:
        j = self._jobs.get(job_id)
        return j if j and j.project_id == project_id and j.owner_id == owner_id else None

    def active_for(self, project_id: str, owner_id: str) -> Job | None:
        """The queued/running job of this project, if any (lets a reloaded page re-attach)."""
        for j in self._jobs.values():
            if j.project_id == project_id and j.owner_id == owner_id and j.status in ("queued", "running"):
                return j
        return None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
