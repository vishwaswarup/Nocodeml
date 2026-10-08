"""Background training jobs.

Job *records* live in the database (`training_jobs`), so status survives an API restart and the database itself
guarantees one active job per project. The training itself still runs in this process, on a small thread pool.

How a crash is noticed: every queued/running job writes a heartbeat every few seconds. If a job's heartbeat has gone
quiet and this process isn't running it, whoever looks at it next marks it failed ("interrupted"), so nothing hangs
forever as "running". (Training can't be resumed after a crash because the API only ever holds the signed-in user's
short-lived token, never a master key; the user simply trains again.)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from nocodeml_engine.persistence.jobs import ACTIVE, Job, JobConflict, JobStore, now_iso  # noqa: F401 (re-exported)
from nocodeml_engine.training import TrainingCancelled

log = logging.getLogger("nocodeml.jobs")

HEARTBEAT_SECONDS = 10
STALE_SECONDS = 60          # silent this long and not running here => the process that held it is gone
MAX_ACTIVE_PER_USER = 3     # across all of a user's projects


def _is_network_error(e: BaseException) -> bool:
    try:
        import httpx
        if isinstance(e, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(e, (ConnectionError, TimeoutError))


class JobBusy(RuntimeError):
    """Too many jobs are already waiting (server-wide) or running for this user."""


class _Live:
    """What this process knows about a job it is holding."""

    def __init__(self, job: Job):
        self.job = job
        self.cancel = threading.Event()
        self.stop_heartbeat = threading.Event()


class JobManager:
    def __init__(self, max_workers: int = 2, max_waiting: int = 20, heartbeat: float = HEARTBEAT_SECONDS,
                 stale: float = STALE_SECONDS):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")
        self._max_waiting, self._heartbeat, self._stale = max_waiting, heartbeat, stale
        self._live: dict[str, _Live] = {}
        self._order: list[str] = []   # queued job ids, oldest first (for queue positions)
        self._lock = threading.Lock()

    # -- submitting ------------------------------------------------------------------------------------------

    def submit(self, store: JobStore, project_id: str, fn: Callable[[Job, threading.Event], int]) -> Job:
        """fn(job, cancel) runs in a worker thread, returns the experiment number, and should stop soon after
        `cancel` is set."""
        with self._lock:
            if len(self._order) >= self._max_waiting:
                raise JobBusy("The training queue is full right now. Please try again in a few minutes.")
            mine = sum(1 for lv in self._live.values() if lv.job.owner_id == store.user_id)
            if mine >= MAX_ACTIVE_PER_USER:
                raise JobBusy(f"You already have {MAX_ACTIVE_PER_USER} trainings in progress. Wait for one to finish.")
        self.reap(store, project_id)              # a crashed job must not block a new one
        job = store.create(project_id)            # raises JobConflict if one is already active (database-enforced)
        live = _Live(job)
        with self._lock:
            self._live[job.id] = live
            self._order.append(job.id)
        threading.Thread(target=self._beat, args=(store, live), daemon=True, name=f"beat-{job.id[:6]}").start()
        self._pool.submit(self._run, store, live, fn)
        return self._with_position(job)

    def _beat(self, store: JobStore, live: _Live) -> None:
        while not live.stop_heartbeat.wait(self._heartbeat):
            try:
                store.heartbeat(live.job.id)
                if store.cancel_requested(live.job.id):   # a cancel asked of any API process reaches this one here
                    live.cancel.set()
            except Exception as e:  # noqa: BLE001 - a missed beat is harmless; several in a row look like a crash
                log.warning("heartbeat failed for job %s: %s", live.job.id, e)

    def _run(self, store: JobStore, live: _Live, fn: Callable[[Job, threading.Event], int]) -> None:
        job = live.job
        try:
            with self._lock:
                if job.id in self._order:
                    self._order.remove(job.id)
            if live.cancel.is_set():
                self._finish(store, live, "cancelled", error="Cancelled before it started.")
                return
            job.started_at = now_iso()
            store.update(job.id, status="running", started_at=job.started_at)
            job.status = "running"
            try:
                job.experiment_number = fn(job, live.cancel)
                self._finish(store, live, "succeeded")
            except TrainingCancelled:
                self._finish(store, live, "cancelled", error="Cancelled. Nothing was saved.")
            except Exception as e:  # noqa: BLE001 - surfaced to the user, see messages below
                issues = getattr(e, "issues", None)
                if issues:
                    job.issues = list(issues)
                if _is_network_error(e):
                    msg = "Lost the connection to the database while training. Nothing was saved; please train again."
                elif isinstance(e, (ValueError, RuntimeError)):
                    msg = str(e)
                else:
                    log.exception("training job %s failed", job.id)
                    msg = "Training failed unexpectedly."
                self._finish(store, live, "failed", error=msg)
        except Exception:  # noqa: BLE001 - e.g. the database is unreachable while recording the outcome
            log.exception("could not record the outcome of job %s", job.id)
        finally:
            live.stop_heartbeat.set()
            with self._lock:
                self._live.pop(job.id, None)
                if job.id in self._order:
                    self._order.remove(job.id)

    def _finish(self, store: JobStore, live: _Live, status: str, error: str | None = None) -> None:
        job = live.job
        job.status, job.error, job.finished_at = status, error, now_iso()
        store.update(job.id, status=status, error=error, finished_at=job.finished_at,
                     experiment_number=job.experiment_number, issues=job.issues, warnings=job.warnings)

    # -- reading ---------------------------------------------------------------------------------------------

    def _with_position(self, job: Job) -> Job:
        with self._lock:
            pos = self._order.index(job.id) + 1 if job.id in self._order else None
        return job.model_copy(update={"queue_position": pos}) if job.status == "queued" else job

    def _is_live(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._live

    def reap(self, store: JobStore, project_id: str) -> None:
        """If the project's active job is orphaned (its process died), mark it failed."""
        job = store.active(project_id)
        if job is None or self._is_live(job.id):
            return
        beat = store.last_heartbeat(job.id)
        if beat is None or (datetime.now(timezone.utc) - beat).total_seconds() < self._stale:
            return
        log.warning("job %s was orphaned (no heartbeat since %s); marking it failed", job.id, beat)
        store.update(job.id, status="failed", finished_at=now_iso(), error=(
            "Training was interrupted (the server restarted or lost power). Nothing was saved; please train again."))

    def get(self, store: JobStore, job_id: str, project_id: str) -> Job | None:
        self.reap(store, project_id)
        job = store.get(job_id, project_id)
        return self._with_position(job) if job else None

    def active_for(self, store: JobStore, project_id: str) -> Job | None:
        """The queued/running job of this project, if any (lets a reloaded page re-attach)."""
        self.reap(store, project_id)
        job = store.active(project_id)
        return self._with_position(job) if job else None

    def cancel(self, store: JobStore, job_id: str, project_id: str) -> Job | None:
        job = store.get(job_id, project_id)
        if job is None:
            return None
        if job.status in ACTIVE:
            store.update(job.id, cancel_requested=True)          # visible to every API process
            with self._lock:
                live = self._live.get(job.id)
            if live:
                live.cancel.set()                                 # the running training notices at its next step
            job.cancel_requested = True
        return self._with_position(job)

    def shutdown(self) -> None:
        with self._lock:
            for lv in self._live.values():
                lv.stop_heartbeat.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
