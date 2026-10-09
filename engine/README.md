# NoCodeML engine

The Python side of NoCodeML: a deterministic ML engine plus the FastAPI service the website talks to.

    profile → preprocess → engineer features → split → train → evaluate → quality-check → export / report

Everything is driven by one versioned `PipelineConfig` ([`nocodeml_engine/config.py`](nocodeml_engine/config.py)); the engine
rebuilds the full scikit-learn pipeline from it, so the same data and config always give the same results.

## Set up and test

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"          # macOS: also `brew install libomp` for XGBoost
    .venv/bin/pytest                            # ~195 tests

Some tests are skipped when their prerequisites are missing: SQL tests need local PostgreSQL binaries, and
`tests/test_supabase_live.py` needs `SUPABASE_*` settings (see `../.env.example`).

## Run the API

    cp ../.env.example ../.env                  # fill in SUPABASE_URL and SUPABASE_ANON_KEY (anon key only)
    .venv/bin/uvicorn nocodeml_engine.api.main:app --port 8000

Open http://127.0.0.1:8000/health, or http://127.0.0.1:8000/docs for the interactive docs (disabled when
`NOCODEML_ENV=production`). Every data route needs `Authorization: Bearer <Supabase access token>`; the database's row-level
security does the real authorization. Settings for production and small servers are in [`../docs/deploy.md`](../docs/deploy.md).

## Package map (`nocodeml_engine/`)

| Module | What it does |
|---|---|
| `dataset/` | load CSV / Excel / Parquet safely, detect dates |
| `profiling/` | column types, missing values, outliers, identifiers, imbalance, correlations |
| `preprocessing/` | imputation, outliers, encoding, scaling, before/after preview |
| `feature_engineering/` | log transform, date parts, … (fitted on training rows only) |
| `splitting/` | train/test, train/val/test, (stratified) k-fold, time series; leak-free |
| `models/` | model registry: hyperparameters, regularization rules, recommended values |
| `training/` | fit pipelines, hyperparameter search, metric assembly, cancel and time limits |
| `evaluation/` · `quality/` | metrics, curves · "is the *experiment* sound?" checks |
| `recommendations/` | rule-based, explained suggestions for preprocessing, features and split |
| `reports/` · `artifacts/` | PDF report · exportable pipeline and metrics |
| `colab/` | data bundle for Google Colab, the notebook runner, and results import (server-side scoring) |
| `state/` · `persistence/` | pipeline versions, experiments, jobs; the Supabase-backed repository |
| `api/` | FastAPI app: auth, rate limits, jobs, endpoints |

## Key guarantees

- Fitted preprocessing (imputation, winsorizing, encoding, scaling, outlier bounds) is fit on train rows only, per fold.
- Invalid configs (unhandled NaNs, unencoded categoricals, L1/L2 on trees, more than 5 models) are rejected with explicit messages.
- Exported `pipeline_*.pkl` files accept raw rows. Pickles execute code on load: see `SECURITY.txt` in every export.
- The API holds no master key: it acts as the signed-in user.

## Scripts

`scripts/` has small developer helpers: `demo.py` and `demo_employees.py` (run the engine on sample data and compare two
variants), `smoke_api.py` (end-to-end check against a running API with test accounts), and `get_token.py`. Run them from
anywhere, e.g. `.venv/bin/python scripts/demo.py`; outputs go to `exports/` (git-ignored).
