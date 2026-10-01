import pytest

from nocodeml_engine.config import (
    MissingValueRule, MissingValueStrategy, ModelConfig, PreprocessingConfig, ScalingStrategy,
)
from nocodeml_engine.state import (
    InMemoryRepository, PipelineError, PipelineService, Section, VersionStatus,
)
from nocodeml_engine.training import run_experiment


@pytest.fixture
def svc(churn_config):
    s = PipelineService(InMemoryRepository())
    s.create(churn_config)
    return s


def run_latest(svc, df, pid="p1", parent=None):
    cfg = svc.latest(pid).config
    return svc.record_experiment(run_experiment(df, cfg).result, parent)


def test_edit_creates_version_and_invalidates_results(svc, churn_df):
    e1 = run_latest(svc, churn_df)
    assert e1.number == 1 and e1.pipeline_version == 1 and svc.is_current(e1)

    pre = svc.latest("p1").config.preprocessing.model_copy(update={"scaling": ScalingStrategy.MIN_MAX})
    v2, impact = svc.update_section("p1", Section.PREPROCESSING, pre)
    assert v2.version == 2 and impact.changed == [Section.PREPROCESSING]
    assert Section.SPLIT in impact.affected_sections and Section.DATASET not in impact.affected_sections
    assert impact.stale_experiment_ids == [e1.result.experiment_id]
    assert "no longer apply" in impact.message
    assert not svc.is_current(e1) and svc.current_experiments("p1") == []

    e2 = run_latest(svc, churn_df)
    assert e2.number == 2 and e2.pipeline_version == 2
    # old version is immutable and still retrievable
    assert svc.repo.get_version("p1", 1).config.preprocessing.scaling is ScalingStrategy.STANDARD


def test_noop_update_makes_no_version(svc):
    cur = svc.latest("p1")
    v, impact = svc.update("p1", cur.config)
    assert v.version == 1 and impact.changed == [] and len(svc.repo.list_versions("p1")) == 1


def test_finalize_rules(svc, churn_df):
    with pytest.raises(PipelineError, match="Run the pipeline"):
        svc.finalize("p1")
    e = run_latest(svc, churn_df)
    v = svc.finalize("p1")
    assert v.status is VersionStatus.FINALIZED and v.label == "v1.0"
    assert v.finalized_experiment_id == e.result.experiment_id
    with pytest.raises(PipelineError, match="already finalized"):
        svc.finalize("p1")
    # editing a finalized pipeline forks a draft; the finalized version is untouched
    v2, _ = svc.update_section("p1", Section.MODELS, [ModelConfig(model_key="knn")])
    assert v2.status is VersionStatus.DRAFT and v2.version == 2
    assert svc.repo.get_version("p1", 1).status is VersionStatus.FINALIZED


def test_duplicate_and_compare(svc, churn_df):
    e1 = run_latest(svc, churn_df)
    pre = svc.latest("p1").config.preprocessing.model_copy(update={
        "missing_values": [MissingValueRule(column="age", strategy=MissingValueStrategy.MEAN)]})
    svc.update_section("p1", Section.PREPROCESSING, pre)
    e2 = run_latest(svc, churn_df)

    draft, parent = svc.duplicate("p1", e1.number)  # back to experiment #1's config
    assert parent == 1 and draft.config.preprocessing == svc.repo.get_version("p1", 1).config.preprocessing
    e3 = run_latest(svc, churn_df, parent=parent)
    assert e3.number == 3 and e3.parent_number == 1
    assert e3.result.config_hash == e1.result.config_hash  # identical config => identical hash

    cmp = svc.compare("p1", 1, 2)
    diffs = {d.section: d.same for d in cmp.config_diff}
    assert diffs[Section.PREPROCESSING] is False and diffs[Section.MODELS] is True
    assert "f1" in cmp.metrics["random_forest"]
    d = cmp.metrics["random_forest"]["f1"]
    assert d["delta"] == pytest.approx(d["b"] - d["a"])


def test_experiment_must_match_a_version(svc, churn_df, churn_config):
    other = churn_config.model_copy(update={"preprocessing": PreprocessingConfig(
        drop_columns=["customer_id"],
        missing_values=[MissingValueRule(column="age", strategy=MissingValueStrategy.MODE)],
        encoding=churn_config.preprocessing.encoding)})
    with pytest.raises(PipelineError, match="does not match"):
        svc.record_experiment(run_experiment(churn_df, other).result)
