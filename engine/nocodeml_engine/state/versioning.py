"""Pipeline versioning, change-impact analysis, and experiment comparison (pure logic)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from nocodeml_engine.config import PipelineConfig


class Section(str, Enum):
    DATASET = "dataset"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    SPLIT = "split"
    MODELS = "models"  # model selection, regularization and hyperparameters


SECTION_ORDER = list(Section)
SECTION_LABEL = {
    Section.DATASET: "Dataset", Section.PREPROCESSING: "Preprocessing",
    Section.FEATURE_ENGINEERING: "Feature Engineering", Section.SPLIT: "Split",
    Section.MODELS: "Models / Regularization / Hyperparameters",
}


def section_value(config: PipelineConfig, section: Section) -> Any:
    v = getattr(config, section.value)
    return [m.model_dump(mode="json") for m in v] if section is Section.MODELS \
        else v.model_dump(mode="json")


def changed_sections(old: PipelineConfig, new: PipelineConfig) -> list[Section]:
    return [s for s in SECTION_ORDER if section_value(old, s) != section_value(new, s)]


class ChangeImpact(BaseModel):
    changed: list[Section]
    affected_sections: list[Section]  # the changed section(s) and everything downstream
    reruns_training: bool
    stale_experiment_ids: list[str] = Field(default_factory=list)
    message: str = ""

    @property
    def rerun_from(self) -> Section | None:
        return self.changed[0] if self.changed else None


def analyze_change(old: PipelineConfig, new: PipelineConfig) -> ChangeImpact:
    changed = changed_sections(old, new)
    if not changed:
        return ChangeImpact(changed=[], affected_sections=[], reruns_training=False,
                            message="No changes.")
    first = SECTION_ORDER.index(changed[0])
    affected = SECTION_ORDER[first:]
    labels = ", ".join(SECTION_LABEL[s] for s in changed)
    downstream = ", ".join(SECTION_LABEL[s] for s in affected if s not in changed)
    msg = f"{labels} changed."
    if downstream:
        msg += f" Affected downstream: {downstream}, Training, Evaluation."
    else:
        msg += " Training and Evaluation are affected."
    msg += " Previous results no longer apply to this pipeline."
    return ChangeImpact(changed=changed, affected_sections=affected, reruns_training=True,
                        message=msg)


class ConfigDiff(BaseModel):
    section: Section
    same: bool
    before: Any = None
    after: Any = None


def diff_configs(a: PipelineConfig, b: PipelineConfig) -> list[ConfigDiff]:
    out = []
    for s in SECTION_ORDER:
        va, vb = section_value(a, s), section_value(b, s)
        out.append(ConfigDiff(section=s, same=va == vb,
                              before=None if va == vb else va, after=None if va == vb else vb))
    return out
