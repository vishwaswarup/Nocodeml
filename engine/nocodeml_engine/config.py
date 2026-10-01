"""The structured, versioned pipeline configuration.

This module is the single source of truth for an experiment. Every section of
the product (dataset, preprocessing, feature engineering, split, models,
regularization, hyperparameters) is represented here as data, never as code
the UI writes directly. The engine reconstructs the full sklearn pipeline
from one of these objects.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class MissingValueStrategy(str, Enum):
    DROP_ROWS = "drop_rows"
    DROP_COLUMN = "drop_column"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"


class EncodingStrategy(str, Enum):
    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    LABEL = "label"
    FREQUENCY = "frequency"
    TARGET = "target"


class ScalingStrategy(str, Enum):
    STANDARD = "standard"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    NONE = "none"


class OutlierStrategy(str, Enum):
    IQR = "iqr"
    Z_SCORE = "z_score"
    WINSORIZE = "winsorize"
    ISOLATION_FOREST = "isolation_forest"
    KEEP = "keep"


class MissingValueRule(BaseModel):
    column: str
    strategy: MissingValueStrategy
    constant_value: Any | None = None
    n_neighbors: int = 5


class EncodingRule(BaseModel):
    column: str
    strategy: EncodingStrategy


class OutlierRule(BaseModel):
    column: str
    strategy: OutlierStrategy = OutlierStrategy.KEEP
    threshold: float = 1.5  # IQR multiplier or z-score threshold


class PreprocessingConfig(BaseModel):
    drop_columns: list[str] = Field(default_factory=list)
    missing_values: list[MissingValueRule] = Field(default_factory=list)
    encoding: list[EncodingRule] = Field(default_factory=list)
    scaling: ScalingStrategy = ScalingStrategy.NONE
    scale_columns: list[str] = Field(default_factory=list)
    outliers: list[OutlierRule] = Field(default_factory=list)
    drop_duplicates: bool = False


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class NumericTransform(str, Enum):
    LOG = "log"
    SQRT = "sqrt"
    ABS = "abs"
    SQUARE = "square"


class DateFeature(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DAY_OF_WEEK = "day_of_week"
    QUARTER = "quarter"
    IS_WEEKEND = "is_weekend"


class NumericFeatureRule(BaseModel):
    column: str
    transform: NumericTransform


class DateFeatureRule(BaseModel):
    column: str
    extract: list[DateFeature] = Field(default_factory=list)


class FeatureEngineeringConfig(BaseModel):
    numeric_transforms: list[NumericFeatureRule] = Field(default_factory=list)
    date_features: list[DateFeatureRule] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class SplitMethod(str, Enum):
    TRAIN_TEST = "train_test"
    TRAIN_VAL_TEST = "train_val_test"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES = "time_series"


class SplitConfig(BaseModel):
    method: SplitMethod = SplitMethod.TRAIN_TEST
    test_size: float = 0.2
    validation_size: float = 0.0
    n_splits: int = 5
    stratify: bool = False
    # If set, rows are ordered by this column and holdout splits are chronological.
    time_column: str | None = None
    random_state: int = 42


# ---------------------------------------------------------------------------
# Models, regularization, hyperparameters
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    model_key: str  # must match a key in MODEL_REGISTRY
    # Model-aware regularization, e.g. {"type": "l2", "strength": 1.0}. The
    # registry validates the type against what the model actually supports.
    regularization: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    use_recommended_defaults: bool = True


# ---------------------------------------------------------------------------
# Pipeline config: the source of truth
# ---------------------------------------------------------------------------


class DatasetRef(BaseModel):
    dataset_id: str
    version: int = 1
    target_column: str
    task: TaskType


class PipelineConfig(BaseModel):
    pipeline_id: str
    version: int = 1
    dataset: DatasetRef
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: list[ModelConfig] = Field(default_factory=list)

    def bump_version(self) -> "PipelineConfig":
        """Return a new config object representing the next version.

        Callers use this whenever a section is edited, per the versioning
        rule: a config mutation always produces a new version rather than
        silently overwriting the one past results were computed against.
        """
        return self.model_copy(update={"version": self.version + 1})
