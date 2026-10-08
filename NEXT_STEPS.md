# NoCodeML: where we are and what to do next

_Last updated 2026-10-07._

## Done (all verified: 163 backend tests, 245 browser checks)
- Core ML engine, versioned pipelines, experiments, finalize, compare (Phases 1-2)
- Supabase: schema + row-level security + private storage, Google + email login (Phase 3)
- FastAPI: auth, datasets, pipeline, background training, results, downloads (Phase 4)
- Delete project (UI + API, blocked while training), security headers, prod mode, DB advisor fixes
- Web app: landing page, login, projects, and **all 8 workspace sections**
  (Dataset + charts, Preprocessing, Features, Split, Models, Regularization, Training, Results)
- NoCodeML recommendations (preprocessing, features, split), pipeline-health quality checks
- PDF experiment report, signed downloads, experiment-vs-experiment comparison

## Do next, in this order
1. ~~Job queue~~ DONE 2026-10-08 as a *database-backed* queue (migration 0003): durable status, DB-enforced one active job per
   project, heartbeat + crash recovery, cancel, queue position, 3 active per user. Deliberately NOT Celery/Redis: workers
   would need a master key or the user's refresh token (the API only holds each user's short-lived token), and a single
   API process is enough for a private beta. Revisit (Redis + dedicated workers) if/when training load outgrows one box.
2. ~~Dataset charts in the PDF report~~ DONE 2026-10-07 (new section 3 "Dataset charts"; report is now 14 sections).
3. **Local runner** (spec section 27/47): train on the user's own machine so data never leaves it.
4. ~~Pre-launch hardening~~ code side DONE 2026-10-07. What remains is dashboard clicks: see **DEPLOY.md**
   (apply `supabase/migrations/0002_hardening.sql`, publish Google app, Supabase URL config, backups, monitoring).

## Backlog (not started)
- ~~XGBoost~~ DONE 2026-10-08 (also Gradient Boosting now supports classification). LightGBM/CatBoost later.
  macOS needs `brew install libomp` for XGBoost; without it the model is simply not offered.
- ~~Hyperparameter search~~ DONE 2026-10-08 (grid + random, nested inside training rows only; UI in Training, results card, PDF).
  Possible follow-ups: Optuna/Bayesian search, successive halving, tuning shown in Regularization section too, search progress bar.
- More feature engineering: polynomial, binning, power transforms, elapsed-time dates
- Clustering workflow (K-Means, DBSCAN, Agglomerative)
- Model deployment / generated prediction API
- Optional "explain my results" LLM feature (core app stays deterministic)
- ~~Parquet / Excel upload~~ DONE 2026-10-08 (.csv, .xlsx first sheet, .parquet; .xls refused on purpose)
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
