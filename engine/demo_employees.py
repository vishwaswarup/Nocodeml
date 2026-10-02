"""Run the engine on test_files/employees.csv: predict Salary, change one thing, compare.

    .venv/bin/python demo_employees.py
"""
from pathlib import Path

from nocodeml_engine.artifacts import export_artifacts
from nocodeml_engine.config import *  # noqa: F401,F403
from nocodeml_engine.dataset.loader import load_csv
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.state import InMemoryRepository, PipelineService, Section

here = Path(__file__).parent
df = load_csv(here / "test_files" / "employees.csv")
profile = profile_dataset(df, target="Salary")
print(f"Dataset: {profile.n_rows} rows x {profile.n_columns} cols | inferred task for Salary: "
      f"{profile.target['inferred_task']}")
for w in profile.warnings:
    print("  !", w.message)

# --- what a sensible user would choose -------------------------------------
# Target: Salary (regression). First Name is an unhelpful ~identifier, Last Login Time is a
# time-of-day with no date, so both are dropped. Start Date -> year + month. Categorical gaps
# filled with the mode, everything one-hot encoded, numerics standardised.
config = PipelineConfig(
    pipeline_id="employees", dataset=DatasetRef(dataset_id="emp", target_column="Salary",
                                                task=TaskType.REGRESSION),
    preprocessing=PreprocessingConfig(
        drop_columns=["First Name", "Last Login Time"],
        missing_values=[MissingValueRule(column=c, strategy=MissingValueStrategy.MODE)
                        for c in ("Gender", "Senior Management", "Team")],
        encoding=[EncodingRule(column=c, strategy=EncodingStrategy.ONE_HOT)
                  for c in ("Gender", "Senior Management", "Team")],
        scaling=ScalingStrategy.STANDARD),
    feature_engineering=FeatureEngineeringConfig(date_features=[
        DateFeatureRule(column="Start Date", extract=[DateFeature.YEAR, DateFeature.MONTH])]),
    split=SplitConfig(method=SplitMethod.K_FOLD, n_splits=5),
    models=[ModelConfig(model_key="linear_regression"),
            ModelConfig(model_key="ridge", regularization={"type": "l2", "strength": 1.0}),
            ModelConfig(model_key="random_forest"),
            ModelConfig(model_key="gradient_boosting")])

svc = PipelineService(InMemoryRepository())
svc.create(config)


def run_and_show(title):
    from nocodeml_engine.training import run_experiment
    run = run_experiment(df, svc.latest("employees").config)
    exp = svc.record_experiment(run.result)
    print(f"\n=== Experiment #{exp.number} on pipeline v{exp.pipeline_version}: {title} ===")
    print(f"{'model':22s} {'R2':>7s} {'RMSE':>9s} {'MAE':>9s} | baseline R2")
    for m in run.result.models:
        c = m.metrics["cv"]
        print(f"{m.name:22s} {c['r2']:7.3f} {c['rmse']:9.0f} {c['mae']:9.0f} | {m.baseline['r2']:.3f}")
    print(f"pipeline quality score: {run.result.quality.score}")
    icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}
    for c in run.result.quality.checks:
        if c.status != "pass":
            print(f"  {icon[c.status]} {c.title} - {c.detail}")
    return run


run1 = run_and_show("baseline choices (year+month from Start Date)")
export_artifacts(run1, svc.latest("employees").config, here / "exports" / "employees_v1")

# --- change ONE thing: stop using Start Date at all -------------------------------
v2, impact = svc.update_section("employees", Section.FEATURE_ENGINEERING, FeatureEngineeringConfig())
pre = v2.config.preprocessing.model_copy(update={
    "drop_columns": v2.config.preprocessing.drop_columns + ["Start Date"]})
v3, impact2 = svc.update_section("employees", Section.PREPROCESSING, pre)
print("\n>>> CHANGE: removed Start Date features")
print(">>>", impact2.message)
print(">>> experiments still valid for new version:", len(svc.current_experiments("employees")))
run2 = run_and_show("without Start Date")

cmp = svc.compare("employees", 1, 2)
print("\n=== COMPARISON #1 -> #2 (R2, CV) ===")
for model, ms in cmp.metrics.items():
    r = ms["r2"]
    print(f"{model:20s} {r['a']:.3f} -> {r['b']:.3f}  (delta {r['delta']:+.3f})")
print("sections that differ:", [d.section.value for d in cmp.config_diff if not d.same])
print("\nArtifacts saved in:", here / "exports" / "employees_v1")
