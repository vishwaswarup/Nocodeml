"""Training-job records in the database (table `training_jobs`, row-level security like everything else)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

ACTIVE = ("queued", "running")
FINISHED = ("succeeded", "failed", "cancelled")


class Job(BaseModel):
    id: str
    project_id: str
    owner_id: str
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    experiment_number: int | None = None
    error: str | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cancel_requested: bool = False
    queue_position: int | None = None   # 1 = next to start; only known to the process holding the job


class JobConflict(RuntimeError):
    """A training job is already queued or running for this project."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_unique_violation(e: BaseException) -> bool:
    text = f"{getattr(e, 'code', '')} {e}".lower()
    return "23505" in text or "unique violation" in text or "duplicate key" in text


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


class JobStore:
    """Reads and writes jobs as the signed-in user: the database's row-level security decides what is visible."""

    def __init__(self, client, user_id: str):
        self.db, self.user_id = client, user_id

    def _t(self):
        return self.db.table("training_jobs")

    def _job(self, r: dict[str, Any]) -> Job:
        return Job(id=str(r["id"]), project_id=str(r["project_id"]), owner_id=self.user_id, status=r["status"],
                   created_at=r["created_at"], started_at=r.get("started_at"), finished_at=r.get("finished_at"),
                   experiment_number=r.get("experiment_number"), error=r.get("error"),
                   issues=list(r.get("issues") or []), warnings=list(r.get("warnings") or []),
                   cancel_requested=bool(r.get("cancel_requested")))

    def create(self, project_id: str) -> Job:
        try:
            row = self._t().insert({"project_id": project_id, "status": "queued", "heartbeat_at": now_iso()}).execute().data[0]
        except Exception as e:  # noqa: BLE001
            if is_unique_violation(e):
                raise JobConflict("A training job is already running for this project.") from None
            raise
        return self._job(row)

    def get(self, job_id: str, project_id: str) -> Job | None:
        rows = self._t().select("*").eq("id", job_id).eq("project_id", project_id).limit(1).execute().data
        return self._job(rows[0]) if rows else None

    def active(self, project_id: str) -> Job | None:
        for status in ACTIVE:
            rows = self._t().select("*").eq("project_id", project_id).eq("status", status).limit(1).execute().data
            if rows:
                return self._job(rows[0])
        return None

    def recent(self, project_id: str, limit: int = 10) -> list[Job]:
        rows = self._t().select("*").eq("project_id", project_id).order("created_at", desc=True).limit(limit).execute().data
        return [self._job(r) for r in rows]

    def update(self, job_id: str, **fields: Any) -> None:
        self._t().update(fields).eq("id", job_id).execute()

    def heartbeat(self, job_id: str) -> None:
        self.update(job_id, heartbeat_at=now_iso())

    def last_heartbeat(self, job_id: str) -> datetime | None:
        rows = self._t().select("heartbeat_at").eq("id", job_id).limit(1).execute().data
        return _parse(rows[0]["heartbeat_at"]) if rows and rows[0].get("heartbeat_at") else None

    def cancel_requested(self, job_id: str) -> bool:
        rows = self._t().select("cancel_requested").eq("id", job_id).limit(1).execute().data
        return bool(rows and rows[0].get("cancel_requested"))
