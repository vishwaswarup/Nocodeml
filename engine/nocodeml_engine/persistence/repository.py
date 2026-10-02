"""PipelineRepository backed by Supabase (PostgREST). pipeline_id == projects.id (uuid string)."""

from __future__ import annotations

from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.results import ExperimentResult
from nocodeml_engine.state.store import (
    Experiment, PipelineError, PipelineRepository, PipelineVersion, VersionStatus,
)


def _version_from_row(r: dict) -> PipelineVersion:
    return PipelineVersion(
        pipeline_id=str(r["project_id"]), version=r["version"], status=VersionStatus(r["status"]),
        config=PipelineConfig.model_validate(r["config"]), config_hash=r["config_hash"],
        created_at=r["created_at"], finalized_experiment_id=r.get("finalized_experiment_id"))


def _experiment_from_row(r: dict) -> Experiment:
    return Experiment(number=r["number"], pipeline_id=str(r["project_id"]),
                      pipeline_version=r["pipeline_version"], parent_number=r.get("parent_number"),
                      result=ExperimentResult.model_validate(r["result"]))


class SupabasePipelineRepository(PipelineRepository):
    def __init__(self, client):
        self.db = client

    def save_version(self, v: PipelineVersion) -> None:
        # Upsert: insert a new draft, or move an existing draft to finalized. The database
        # rejects any change to the configuration itself (see guard_pipeline_version).
        row = {"project_id": v.pipeline_id, "version": v.version, "status": v.status.value,
               "config": v.config.model_dump(mode="json"), "config_hash": v.config_hash,
               "finalized_experiment_id": v.finalized_experiment_id}
        self.db.table("pipeline_versions").upsert(row, on_conflict="project_id,version").execute()

    def get_version(self, pipeline_id: str, version: int) -> PipelineVersion:
        rows = (self.db.table("pipeline_versions").select("*").eq("project_id", pipeline_id)
                .eq("version", version).limit(1).execute().data)
        if not rows:
            raise PipelineError(f"Pipeline {pipeline_id} v{version} not found")
        return _version_from_row(rows[0])

    def list_versions(self, pipeline_id: str) -> list[PipelineVersion]:
        rows = (self.db.table("pipeline_versions").select("*").eq("project_id", pipeline_id)
                .order("version").execute().data)
        return [_version_from_row(r) for r in rows]

    def save_experiment(self, e: Experiment) -> None:
        self.db.table("experiments").insert({
            "project_id": e.pipeline_id, "number": e.number,
            "experiment_id": e.result.experiment_id, "pipeline_version": e.pipeline_version,
            "parent_number": e.parent_number, "config_hash": e.result.config_hash,
            "result": e.result.model_dump(mode="json")}).execute()

    def list_experiments(self, pipeline_id: str) -> list[Experiment]:
        rows = (self.db.table("experiments").select("*").eq("project_id", pipeline_id)
                .order("number").execute().data)
        return [_experiment_from_row(r) for r in rows]
