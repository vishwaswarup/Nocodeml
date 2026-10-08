# Going live: hardening checklist

Tick these off in order. Items marked **(you)** are clicks in a dashboard; the rest is already done in code.

## Already done in code
- [x] Rate limits per user and per IP; training time limit; queue cap (`engine/nocodeml_engine/api/`)
- [x] Training jobs stored in the database (survive restarts, one active job per project enforced by the database,
      heartbeat so a crashed job is marked failed, cancel button, per-user limit of 3 at once)
- [x] `NOCODEML_ENV=production` hides `/docs`, `/redoc`, `/openapi.json`; sends HSTS and a locked-down CSP
- [x] In production the API refuses to start unless `NOCODEML_CORS_ORIGINS` lists only `https://` origins
- [x] Every response has `X-Request-ID`, `nosniff`, `no-store`; one access-log line per request (no tokens, no query strings)
- [x] Web app sends `X-Frame-Options: DENY`, `nosniff`, a strict referrer policy and no `X-Powered-By`
- [x] Database: functions pinned/revoked from the public API, faster row-level-security policies, extra indexes
      (`supabase/migrations/0002_hardening.sql`, verified by `tests/test_supabase_sql.py`)

## 1. Apply the database migrations **(you)**
Supabase -> SQL Editor -> new query -> paste the whole file -> Run, in this order, each once:
1. `supabase/migrations/0002_hardening.sql` (done 2026-10-07)
2. `supabase/migrations/0003_training_jobs.sql` (durable training jobs: required by the current API code)

Then Advisors (Security) should list only "Leaked password protection" and the intentional `owns_project` note.

## 2. API environment (set these where the API runs)
```
NOCODEML_ENV=production
NOCODEML_CORS_ORIGINS=https://app.yourdomain.com
NOCODEML_TRUST_PROXY=1            # only if a proxy/load balancer sits in front (it must append X-Forwarded-For)
SUPABASE_URL=...                  SUPABASE_ANON_KEY=...
```
Run behind HTTPS (the host or a proxy terminates TLS). Never set the `service_role` key anywhere.
Run **one** API process for now: the rate-limit counters live in memory and training runs inside the API process.
(Job status itself is in the database, so a restart no longer loses it; a job running during a restart is marked
"interrupted" and the user trains again.)

### Container (optional)
`Dockerfile` (at the repository root) builds the API for any container host (Render, Railway, Fly, a VPS):
```
docker build -t nocodeml-api .
docker run --rm -p 8000:8000 -e SUPABASE_URL=... -e SUPABASE_ANON_KEY=... \
  -e NOCODEML_CORS_ORIGINS=https://app.yourdomain.com nocodeml-api
```
It runs in production mode, as a non-root user, with one worker. Built and tested 2026-10-08: starts and reports healthy;
refuses to start without https CORS origins; hides /docs; XGBoost, hyperparameter search, Parquet and the PDF report
all work inside it (image is about 1.9 GB, first build takes several minutes on a slow connection).

### Running on a free host (Render free plan: 512 MB RAM, sleeps when idle)
Measured inside a 512 MB container: the API alone needs ~260 MB; training 3 models on 5,000 rows peaks ~330 MB, on
50,000 rows (14 MB CSV) ~440 MB, right at the limit. So on a free host set:
```
NOCODEML_MAX_UPLOAD_MB=10        # also set NEXT_PUBLIC_MAX_UPLOAD_MB=10 on the website so it refuses early
NOCODEML_WORKERS=1               # one training at a time
NOCODEML_DATASET_CACHE=1         # keep only one dataset in memory
NOCODEML_TRAIN_TIMEOUT=300
```
A free server falls asleep after ~15 minutes without requests and needs up to a minute to wake: the website now waits
for it (reads retry for about a minute). To keep it awake, add a free uptime monitor (UptimeRobot, every 5 minutes) on
`https://YOUR-API/health`. Supabase's free plan also pauses a project after a week with no activity (one click to restore).
Vercel's free "Hobby" plan is for personal, non-commercial use.

## 3. Web app environment
`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL=https://api.yourdomain.com`

## 4. Sign-in settings **(you)**
- [ ] Supabase -> Authentication -> URL Configuration: Site URL = your production web URL; add
      `https://app.yourdomain.com/auth/callback` to Redirect URLs. Remove localhost entries once live (or keep them only in a dev project).
- [ ] Google Cloud -> Google Auth Platform -> Branding: App name `NoCodeML`; **User support email** = your own Google
      account (`vishwaswarup.756@gmail.com`; Google only accepts the owner's address or a Google Group you manage);
      **Developer contact** = `nocodemachinelearning@gmail.com`; Application home page = your site; Privacy policy link =
      `https://YOUR-DOMAIN/privacy`; Terms link = `https://YOUR-DOMAIN/terms`; Authorized domain = your domain.
- [ ] Google Auth Platform -> Audience: publish the app (Testing -> In production), otherwise only listed test users
      can sign in. Only the basic email/profile scopes are used, so no Google verification review is needed.
- [ ] (Owner decided NOT to rotate the Google client secret that was shown in a screenshot. Accepted risk; revisit if the
      screenshot was ever shared beyond the owner.)
- [ ] Authentication -> Providers -> Email: decide whether to keep email sign-in; if yes, turn on "Confirm email" and,
      on a plan that has it, **Leaked password protection** (Authentication -> Sign In / Providers -> password security).
- [ ] Delete the throw-away test accounts used for automated tests from the production project.

## 5. Monitoring and backups **(you)**
- [ ] Uptime check on `https://api.yourdomain.com/health` (UptimeRobot / BetterStack, free tier is fine).
- [ ] Ship the API's stdout log somewhere searchable; each line carries a request id that also appears in the
      response header, so a user's "it failed" can be matched to a log line.
- [ ] Error tracking (e.g. Sentry) for the web app and API: add when you pick a vendor.
- [ ] Supabase backups: Project Settings -> Database -> Backups (daily backups are on paid plans; on the free
      plan, schedule a `pg_dump` yourself). Storage files (datasets, artifacts) are **not** in database backups.
- [ ] Supabase -> Project Settings -> Billing/Usage: set a spend cap or alerts.

## 6. Final smoke test
Sign in with a real Google account -> create project -> upload CSV -> run all 8 sections -> generate the PDF ->
delete the project. Check `/docs` returns 404 and that a request from a different origin is blocked by the browser.

## Known gaps (not blockers for a small private beta)
- No Content-Security-Policy on the web app yet (needs a nonce setup tested on the real domain).
- Rate-limit counters are per process and training runs inside the API process (single API process; a separate worker
  fleet would need a token strategy because the API only holds each user's short-lived sign-in token).
- The privacy policy and terms (`/privacy`, `/terms`) are plain-language drafts written for this app, not legal advice.
  Have someone qualified review them before a public launch, and add a governing-law clause if you want one.
- No self-service account deletion: the policy promises deletion within 30 days on request by email.
- No account-deletion self-service (deleting a project works; deleting the account is done in the Supabase dashboard).
