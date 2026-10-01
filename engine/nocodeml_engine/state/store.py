"""Pipeline state store.

`PipelineRepository` is the persistence boundary: Phase 3 provides a Supabase
implementation of the same interface. `InMemoryRepository` backs tests and the
local runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel

from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.results import ExperimentResult
from nocodeml_engine.state.versioning import (
    ChangeImpact, ConfigDiff, Section, analyze_change, diff_configs, section_value,
)
from nocodeml_engine.training.runner import config_hash


class PipelineError(RuntimeError):
    pass


class VersionStatus(str, Enum):
    DRAFT = "draft"
    FINALIZED = "finalized"


class PipelineVersion(BaseModel):
    pipeline_id: str
    version: int
    status: VersionStatus
    config: PipelineConfig
    config_hash: str
    created_at: str
    finalized_experiment_id: str | None = None

    @property
    def label(self) -> str:
        return f"v{self.version}.0" if self.status is VersionStatus.FINALIZED else f"v{self.version}"


class Experiment(BaseModel):
    number: int  # sequential per pipeline: "Experiment #12"
    pipeline_id: str
    pipeline_version: int
    parent_number: int | None = None  # set when duplicated from another experiment
    result: ExperimentResult


class ExperimentComparison(BaseModel):
    a: int
    b: int
    config_diff: list[ConfigDiff]
    metrics: dict[str, dict[str, dict[str, float | None]]]  # model -> metric -> {a, b, delta}


class PipelineRepository(ABC):
    @abstractmethod
    def save_version(self, v: PipelineVersion) -> None: ...
    @abstractmethod
    def get_version(self, pipeline_id: str, version: int) -> PipelineVersion: ...
    @abstractmethod
    def list_versions(self, pipeline_id: str) -> list[PipelineVersion]: ...
    @abstractmethod
    def save_experiment(self, e: Experiment) -> None: ...
    @abstractmethod
    def list_experiments(self, pipeline_id: str) -> list[Experiment]: ...


class InMemoryRepository(PipelineRepository):
    def __init__(self) -> None:
        self._versions: dict[tuple[str, int], PipelineVersion] = {}
        self._experiments: dict[str, list[Experiment]] = {}

    def save_version(self, v):
        self._versions[(v.pipeline_id, v.version)] = v.model_copy(deep=True)

    def get_version(self, pipeline_id, version):
        try:
            return self._versions[(pipeline_id, version)].model_copy(deep=True)
        except KeyError:
            raise PipelineError(f"Pipeline {pipeline_id} v{version} not found") from None

    def list_versions(self, pipeline_id):
        return sorted((v.model_copy(deep=True) for (p, _), v in self._versions.items()
                       if p == pipeline_id), key=lambda v: v.version)

    def save_experiment(self, e):
        self._experiments.setdefault(e.pipeline_id, []).append(e.model_copy(deep=True))

    def list_experiments(self, pipeline_id):
        return [e.model_copy(deep=True) for e in self._experiments.get(pipeline_id, [])]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineService:
    """Versioning rules:

    - Every edit that changes a section creates a new draft version; old versions are immutable.
    - Experiments are tied to the exact version (and config hash) they ran against.
    - A finalized version can never change; editing it forks a new draft.
    """

    def __init__(self, repo: PipelineRepository):
        self.repo = repo

    # -- versions ---------------------------------------------------------

    def create(self, config: PipelineConfig) -> PipelineVersion:
        if self.repo.list_versions(config.pipeline_id):
            raise PipelineError(f"Pipeline {config.pipeline_id} already exists")
        config = config.model_copy(update={"version": 1})
        v = PipelineVersion(pipeline_id=config.pipeline_id, version=1, status=VersionStatus.DRAFT,
                            config=config, config_hash=config_hash(config), created_at=_now())
        self.repo.save_version(v)
        return v

    def latest(self, pipeline_id: str) -> PipelineVersion:
        versions = self.repo.list_versions(pipeline_id)
        if not versions:
            raise PipelineError(f"Pipeline {pipeline_id} not found")
        return versions[-1]

    def update(self, pipeline_id: str, new_config: PipelineConfig) -> tuple[PipelineVersion, ChangeImpact]:
        """Commit an edited config. Returns the (possibly new) version and what it invalidated."""
        cur = self.latest(pipeline_id)
        impact = analyze_change(cur.config, new_config)
        if not impact.changed:
            return cur, impact
        nv = cur.version + 1
        new_config = new_config.model_copy(update={"pipeline_id": pipeline_id, "version": nv})
        v = PipelineVersion(pipeline_id=pipeline_id, version=nv, status=VersionStatus.DRAFT,
                            config=new_config, config_hash=config_hash(new_config), created_at=_now())
        self.repo.save_version(v)
        impact.stale_experiment_ids = [
            e.result.experiment_id for e in self.repo.list_experiments(pipeline_id)
            if e.result.config_hash == cur.config_hash]
        return v, impact

    def update_section(self, pipeline_id: str, section: Section, value) -> tuple[PipelineVersion, ChangeImpact]:
        """Replace one section of the latest config (value: the section's pydantic object/list)."""
        cur = self.latest(pipeline_id).config
        return self.update(pipeline_id, cur.model_copy(update={section.value: value}))

    # -- experiments ------------------------------------------------------

    def record_experiment(self, result: ExperimentResult, parent_number: int | None = None) -> Experiment:
        try:
            v = self.repo.get_version(result.pipeline_id, result.pipeline_version)
        except PipelineError:
            raise PipelineError("Experiment does not match any stored pipeline version") from None
        if v.config_hash != result.config_hash:
            raise PipelineError("Experiment does not match any stored pipeline version "
                                f"(v{v.version} configuration differs from what was run)")
        n = len(self.repo.list_experiments(result.pipeline_id)) + 1
        e = Experiment(number=n, pipeline_id=result.pipeline_id, pipeline_version=v.version,
                       parent_number=parent_number, result=result)
        self.repo.save_experiment(e)
        return e

    def current_experiments(self, pipeline_id: str) -> list[Experiment]:
        """Only experiments whose config matches the latest version: stale results are excluded."""
        h = self.latest(pipeline_id).config_hash
        return [e for e in self.repo.list_experiments(pipeline_id) if e.result.config_hash == h]

    def is_current(self, experiment: Experiment) -> bool:
        return experiment.result.config_hash == self.latest(experiment.pipeline_id).config_hash

    def duplicate(self, pipeline_id: str, number: int) -> tuple[PipelineVersion, int]:
        """Start a new draft from an experiment's exact configuration.

        Returns the draft version to edit and the parent experiment number.
        """
        exp = self._experiment(pipeline_id, number)
        src = self.repo.get_version(pipeline_id, exp.pipeline_version)
        latest = self.latest(pipeline_id)
        if latest.config_hash == src.config_hash:
            return latest, number
        v, _ = self.update(pipeline_id, src.config)
        return v, number

    def compare(self, pipeline_id: str, a: int, b: int) -> ExperimentComparison:
        ea, eb = self._experiment(pipeline_id, a), self._experiment(pipeline_id, b)
        ca = self.repo.get_version(pipeline_id, ea.pipeline_version).config
        cb = self.repo.get_version(pipeline_id, eb.pipeline_version).config
        metrics: dict = {}
        mb = {m.model_key: m for m in eb.result.models}
        for ma in ea.result.models:
            other = mb.get(ma.model_key)
            if other is None:
                continue
            sa = ma.metrics.get("test") or ma.metrics.get("cv") or {}
            sb = other.metrics.get("test") or other.metrics.get("cv") or {}
            metrics[ma.model_key] = {}
            for k, va in sa.items():
                vb = sb.get(k)
                if isinstance(va, float) and isinstance(vb, float) and k != "average":
                    metrics[ma.model_key][k] = {"a": va, "b": vb, "delta": vb - va}
        return ExperimentComparison(a=a, b=b, config_diff=diff_configs(ca, cb), metrics=metrics)

    # -- finalize ---------------------------------------------------------

    def finalize(self, pipeline_id: str) -> PipelineVersion:
        cur = self.latest(pipeline_id)
        if cur.status is VersionStatus.FINALIZED:
            raise PipelineError(f"{cur.label} is already finalized")
        current = self.current_experiments(pipeline_id)
        if not current:
            raise PipelineError("Run the pipeline before finalizing: no results match this version.")
        exp = current[-1]
        failed = [c.title for c in exp.result.quality.by_status("fail")]
        if failed:
            raise PipelineError(f"Cannot finalize: failed quality checks: {failed}")
        done = cur.model_copy(update={"status": VersionStatus.FINALIZED,
                                      "finalized_experiment_id": exp.result.experiment_id})
        self.repo.save_version(done)
        return done

    def _experiment(self, pipeline_id: str, number: int) -> Experiment:
        for e in self.repo.list_experiments(pipeline_id):
            if e.number == number:
                return e
        raise PipelineError(f"Experiment #{number} not found")
