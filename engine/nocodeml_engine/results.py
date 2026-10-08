"""Serializable experiment records (reproducibility, Section 56)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityCheck(BaseModel):
    id: str
    status: str  # "pass" | "warn" | "fail" | "na"
    title: str
    detail: str = ""
    model_key: str | None = None


class QualityReport(BaseModel):
    checks: list[QualityCheck] = Field(default_factory=list)
    score: float | None = None  # pipeline quality (0-100), NOT model performance

    def by_status(self, status: str) -> list[QualityCheck]:
        return [c for c in self.checks if c.status == status]


class ModelResult(BaseModel):
    model_key: str
    name: str
    hyperparameters: dict[str, Any]
    # split name ("train" | "validation" | "test" | "cv") -> metrics
    metrics: dict[str, dict[str, Any]]
    fold_primary_scores: list[float] = Field(default_factory=list)
    primary_metric: str
    baseline: dict[str, Any] = Field(default_factory=dict)
    n_rows_fitted: int = 0
    n_outlier_rows_removed: int = 0
    fit_seconds: float = 0.0
    # Hyperparameter search that chose this model's settings (None if it was trained once as configured).
    # In cross-validation the search is repeated inside every outer fold; this is the search on all of the data.
    search: dict[str, Any] | None = None


class ExperimentResult(BaseModel):
    experiment_id: str
    pipeline_id: str
    pipeline_version: int
    dataset_id: str
    dataset_version: int
    dataset_fingerprint: str
    config_hash: str
    task: str
    random_state: int
    started_at: str
    finished_at: str
    environment: dict[str, str]
    split: dict[str, Any]
    preparation_log: list[dict[str, Any]]
    models: list[ModelResult]
    quality: QualityReport
