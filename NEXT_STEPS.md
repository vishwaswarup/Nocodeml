# NoCodeML: where we are and what to do next

_Last updated 2026-10-07._

## Done (all verified: 112 backend tests, 245 browser checks)
- Core ML engine, versioned pipelines, experiments, finalize, compare (Phases 1-2)
- Supabase: schema + row-level security + private storage, Google + email login (Phase 3)
- FastAPI: auth, datasets, pipeline, background training, results, downloads (Phase 4)
- Web app: landing page, login, projects, and **all 8 workspace sections**
  (Dataset + charts, Preprocessing, Features, Split, Models, Regularization, Training, Results)
- NoCodeML recommendations (preprocessing, features, split), pipeline-health quality checks
- PDF experiment report, signed downloads, experiment-vs-experiment comparison

## Do next, in this order
1. **Real job queue** (rate limiting + training time limit are DONE, 2026-10-07)
   - Done: per-user and per-IP rate limits (429 + Retry-After; uploads 10/min, training 6/min, reports 6/min,
     previews 60/min, 300/min overall per user, 600/min per IP), training time limit
     (`NOCODEML_TRAIN_TIMEOUT`, default 600s, checked between models/folds), queue cap (503 when 20 waiting).
   - Still to do: replace the in-memory thread pool with Celery/RQ + Redis so job status survives restarts,
     training runs off the API process, and the rate-limit counters are shared across API processes.
2. **Dataset charts in the PDF report** (class balance, correlations, distributions).
3. **Local runner** (spec section 27/47): train on the user's own machine so data never leaves it.
4. **Pre-launch hardening checklist**
   - Turn off `/docs` in production; set `NOCODEML_CORS_ORIGINS` to the real domain.
   - Google OAuth app: move from "Testing" to "In production" (consent screen -> Audience -> Publish).
   - Add the production URL to Supabase Authentication -> URL Configuration.
   - Error monitoring + logging; backups of the Supabase database.

## Backlog (not started)
- XGBoost (and LightGBM/CatBoost later)
- Hyperparameter search (GridSearchCV / RandomizedSearchCV, later Optuna)
- More feature engineering: polynomial, binning, power transforms, elapsed-time dates
- Clustering workflow (K-Means, DBSCAN, Agglomerative)
- Model deployment / generated prediction API
- Optional "explain my results" LLM feature (core app stays deterministic)
- Parquet / Excel upload
- Section 7 extras: learning curve, calibration curve, PR curve

## Housekeeping reminders
- **Rotate the Google OAuth client secret** (it was shown in a screenshot): Google Cloud ->
  Google Auth Platform -> Clients -> add a new secret, update it in Supabase (Authentication ->
  Providers -> Google), then delete the old one.
- Commit often: `git status`, then commit. `.env` and `web/.env.local` are git-ignored.
- Never put the Supabase `service_role` key in the repo or the web app.

## How to run it
```bash
# terminal 1: API
cd engine && set -a; source ../.env; set +a
.venv/bin/uvicorn nocodeml_engine.api.main:app --port 8000

# terminal 2: web app
cd web && npm run dev          # http://localhost:3000
```
Tests: `cd engine && .venv/bin/pytest` (backend). Browser checks used Playwright scripts kept outside the repo.
