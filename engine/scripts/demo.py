"""Run the whole engine on a synthetic churn CSV and print what it produced.

    .venv/bin/python demo.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nocodeml_engine.artifacts import export_artifacts, load_pipeline
from nocodeml_engine.config import *  # noqa: F401,F403
from nocodeml_engine.dataset.loader import load_csv
from nocodeml_engine.profiling import profile_dataset
from nocodeml_engine.state import InMemoryRepository, PipelineService, Section
from nocodeml_engine.training import run_experiment

tmp = Path(tempfile.mkdtemp())

# 0. make a CSV like a user would upload
rng = np.random.default_rng(0)
n = 500
df = pd.DataFrame({"customer_id": range(1000, 1000 + n), "age": rng.normal(40, 12, n),
                   "income": rng.lognormal(10, 0.5, n), "gender": rng.choice(["M", "F"], n)})
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-(0.05 * (df.age - 40) - 1.8)))).astype(int)
df.loc[rng.choice(n, 25, replace=False), "age"] = np.nan
df.to_csv(tmp / "customers.csv", index=False)

# 1. Section 0: load + profile
data = load_csv(tmp / "customers.csv")
profile = profile_dataset(data, target="churn")
print("=== DATASET HEALTH ===")
print(f"{profile.n_rows} rows, {profile.n_columns} cols, {profile.missing_value_pct}% missing")
for w in profile.warnings:
    print(" !", w.message)

# 2. The pipeline config = single source of truth
config = PipelineConfig(
    pipeline_id="churn", dataset=DatasetRef(dataset_id="d1", target_column="churn",
                                            task=TaskType.CLASSIFICATION),
    preprocessing=PreprocessingConfig(
        drop_columns=["customer_id"],
        missing_values=[MissingValueRule(column="age", strategy=MissingValueStrategy.MEDIAN)],
        encoding=[EncodingRule(column="gender", strategy=EncodingStrategy.ONE_HOT)],
        scaling=ScalingStrategy.STANDARD),
    split=SplitConfig(method=SplitMethod.TRAIN_TEST, stratify=True),
    models=[ModelConfig(model_key="logistic_regression"), ModelConfig(model_key="random_forest"),
            ModelConfig(model_key="knn")])

# 3. Versioned state + run
svc = PipelineService(InMemoryRepository())
svc.create(config)
run = run_experiment(data, svc.latest("churn").config)
exp = svc.record_experiment(run.result)

print("\n=== RESULTS (held-out test set) ===")
for m in run.result.models:
    t = m.metrics["test"]
    print(f"{m.name:24s} acc={t['accuracy']:.3f} f1={t['f1']:.3f} auc={t['roc_auc']:.3f} "
          f"(baseline f1={m.baseline['f1']:.3f})")

print(f"\n=== PIPELINE HEALTH  (quality score {run.result.quality.score}) ===")
icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}
for c in run.result.quality.checks:
    print(f" {icon[c.status]} {c.title}")

# 4. Change something -> old results go stale
new_pre = config.preprocessing.model_copy(update={"scaling": ScalingStrategy.MIN_MAX})
v2, impact = svc.update_section("churn", Section.PREPROCESSING, new_pre)
print("\n=== MODIFY PIPELINE ===")
print(impact.message)
print("now on version", v2.version, "| results current for it:", len(svc.current_experiments("churn")))

# 5. Export and predict on raw rows
out = Path(__file__).resolve().parents[1] / "exports" / "demo"
paths = export_artifacts(run, config, out)
print("\n=== EXPORTED FILES ===")
for p in sorted(out.iterdir()):
    print(" ", p.name)
fp = load_pipeline(paths["pipeline_random_forest"])
print("\nPredictions for 3 raw rows (NaN age, string gender, id col included):",
      fp.predict(data.drop(columns=["churn"]).head(3)))
print("\nFull experiment record:", out / "metrics.json")
