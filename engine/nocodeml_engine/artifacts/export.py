"""Artifact export. The complete pipeline is the primary artifact, not just the estimator.

SECURITY: .pkl files are Python pickles (via joblib). Loading one executes
arbitrary code embedded in the file. Only load artifacts you created or fully
trust, and never load files uploaded by other users. `configuration.json` and
`metrics.json` are plain JSON and safe to inspect.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.training.runner import ExperimentRun

SECURITY_NOTE = __doc__


def export_artifacts(run: ExperimentRun, config: PipelineConfig, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, fp in run.pipelines.items():
        paths[f"pipeline_{key}"] = out / f"pipeline_{key}.pkl"
        joblib.dump(fp, paths[f"pipeline_{key}"])
        paths[f"model_{key}"] = out / f"model_{key}.pkl"
        joblib.dump(fp.model, paths[f"model_{key}"])
    paths["configuration"] = out / "configuration.json"
    paths["configuration"].write_text(config.model_dump_json(indent=2))
    paths["metrics"] = out / "metrics.json"
    paths["metrics"].write_text(run.result.model_dump_json(indent=2))
    paths["security"] = out / "SECURITY.txt"
    paths["security"].write_text(SECURITY_NOTE.strip() + "\n")
    return paths


def load_pipeline(path: str | Path):
    """Load a pipeline_*.pkl. WARNING: unpickling can execute arbitrary code; trusted files only."""
    return joblib.load(path)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
