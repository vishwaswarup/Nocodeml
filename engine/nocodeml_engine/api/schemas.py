from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nocodeml_engine.config import (
    DatasetRef, FeatureEngineeringConfig, ModelConfig, PreprocessingConfig, SplitConfig,
)
from nocodeml_engine.state import ChangeImpact, PipelineVersion


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    task: str | None = Field(default=None, pattern="^(classification|regression)$")


class PipelineConfigIn(BaseModel):
    """What the UI edits. pipeline_id and version are owned by the server."""
    dataset: DatasetRef
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: list[ModelConfig] = Field(default_factory=list, max_length=5)


class PipelineSaved(BaseModel):
    version: PipelineVersion
    created: bool  # True when this was the project's first pipeline
    impact: ChangeImpact | None = None  # what the edit invalidated (None on first save)


class ValidationReport(BaseModel):
    valid: bool
    issues: list[str]


class TrainingRequest(BaseModel):
    pass  # trains the project's latest pipeline version; reserved for future options


class RowsPage(BaseModel):
    columns: list[dict[str, str]]
    rows: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class ExperimentSummary(BaseModel):
    number: int
    experiment_id: str
    pipeline_version: int
    parent_number: int | None
    current: bool  # False once the pipeline changed after this run
    finished_at: str
    quality_score: float | None
    models: list[dict[str, Any]]  # key, name, headline metric
    source: str = "server"  # "server" or "colab"
