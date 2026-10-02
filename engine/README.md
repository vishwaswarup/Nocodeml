# NoCodeML engine

Deterministic Python core: profile -> preprocess -> engineer -> split -> train -> evaluate -> quality-check -> export.
Everything is driven by a versioned `PipelineConfig` (`nocodeml_engine/config.py`).

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/pytest

## Phases
| # | Phase | Status |
|---|-------|--------|
| 1 | Core ML engine (profiler, preprocessing, FE, split, model registry, training, evaluation, quality, export) | done |
| 2 | Pipeline state (versions, change impact/staleness, experiments, finalize, compare) | done |
| 3 | Supabase (schema + RLS + storage policies, repository, project/dataset/artifact service) | done (live-project test not yet run) |
| 4 | FastAPI (auth, datasets, pipeline, background training, experiments, artifacts) | done, verified live |
| 5 | Next.js workspace | |
| 6 | Recommendation engine, PDF report, richer metrics | |
| 7 | UI/UX polish | |
| 8 | Local runner / deployment | |

## Key guarantees
- Fitted preprocessing (imputation, winsorizing, encoding, scaling, outlier bounds) is fit on train rows only, per fold.
- Invalid configs (unhandled NaNs, unencoded categoricals, L1/L2 on trees, >5 models) are rejected with explicit messages.
- Exported `pipeline_*.pkl` accepts raw rows. Pickles execute code on load: see `SECURITY.txt` in every export.

## Run the API
    set -a; source ../.env; set +a        # SUPABASE_URL, SUPABASE_ANON_KEY
    .venv/bin/uvicorn nocodeml_engine.api.main:app --reload --port 8000
    # interactive docs: http://127.0.0.1:8000/docs ; end-to-end check: .venv/bin/python smoke_api.py
All data routes need `Authorization: Bearer <Supabase access token>`; the database's row-level security does the real authorization.
