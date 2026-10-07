# Going live: hardening checklist

Tick these off in order. Items marked **(you)** are clicks in a dashboard; the rest is already done in code.

## Already done in code
- [x] Rate limits per user and per IP; training time limit; queue cap (`engine/nocodeml_engine/api/`)
- [x] `NOCODEML_ENV=production` hides `/docs`, `/redoc`, `/openapi.json`; sends HSTS and a locked-down CSP
- [x] In production the API refuses to start unless `NOCODEML_CORS_ORIGINS` lists only `https://` origins
- [x] Every response has `X-Request-ID`, `nosniff`, `no-store`; one access-log line per request (no tokens, no query strings)
- [x] Web app sends `X-Frame-Options: DENY`, `nosniff`, a strict referrer policy and no `X-Powered-By`
- [x] Database: functions pinned/revoked from the public API, faster row-level-security policies, extra indexes
      (`supabase/migrations/0002_hardening.sql`, verified by `tests/test_supabase_sql.py`)

## 1. Apply the database hardening **(you)**
Supabase -> SQL Editor -> new query -> paste the whole of `supabase/migrations/0002_hardening.sql` -> Run.
Then Advisors (Security) should list only "Leaked password protection".

## 2. API environment (set these where the API runs)
```
NOCODEML_ENV=production
NOCODEML_CORS_ORIGINS=https://app.yourdomain.com
NOCODEML_TRUST_PROXY=1            # only if a proxy/load balancer sits in front (it must append X-Forwarded-For)
SUPABASE_URL=...                  SUPABASE_ANON_KEY=...
```
Run behind HTTPS (the host or a proxy terminates TLS). Never set the `service_role` key anywhere.
Run **one** API process for now: rate-limit counters and job status live in memory (see NEXT_STEPS.md, job queue).

## 3. Web app environment
`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL=https://api.yourdomain.com`

## 4. Sign-in settings **(you)**
- [ ] Supabase -> Authentication -> URL Configuration: Site URL = your production web URL; add
      `https://app.yourdomain.com/auth/callback` to Redirect URLs. Remove localhost entries once live (or keep them only in a dev project).
- [ ] Google Cloud -> Google Auth Platform -> Audience: publish the app (Testing -> In production), otherwise only
      listed test users can sign in. Add the production domain under Branding / Authorized domains.
- [ ] Rotate the Google OAuth client secret that was shown in a screenshot (new secret in Google Cloud, paste into
      Supabase -> Authentication -> Providers -> Google, delete the old one).
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
- Rate limits and job status are per process (single API process only until the job queue lands).
- No account-deletion self-service (deleting a project works; deleting the account is done in the Supabase dashboard).
