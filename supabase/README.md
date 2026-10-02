# Supabase setup

1. Create a project at supabase.com (or `supabase init && supabase start` locally).
2. Apply the schema: paste `migrations/0001_schema.sql` into the SQL editor, or `supabase db push`.
   It creates the tables, row-level security, immutability triggers, the five private storage
   buckets and their policies.
3. Authentication: Authentication -> Providers: enable Email, and Google / GitHub if wanted.
4. Copy the project URL and the **anon** key into the environment (never the service-role key):

       export SUPABASE_URL=https://xxxx.supabase.co
       export SUPABASE_ANON_KEY=eyJ...

## What the database enforces (independent of application code)
- Every row is owned via `projects.owner_id = auth.uid()`; other users see and change nothing.
- `pipeline_versions.config` can never change; a version can only go draft -> finalized, and only
  against an experiment that ran that exact config hash. Finalized versions are frozen.
- `experiments` are append-only and must match the config hash of the version they ran on.
- Storage paths must start with the caller's user id (`{user_id}/{project_id}/...`); buckets are private.
- `datasets` bucket is capped at 100 MB per object.

## Tests
- `tests/test_supabase_sql.py` runs the real migration against a throwaway local Postgres
  (Supabase's `auth.uid()`, roles and storage schema are stubbed in `tests/supabase_stub.sql`).
- `tests/test_persistence.py` tests the Python layer against an in-memory fake client.
- `tests/test_supabase_live.py` runs the full flow on a real project; skipped unless env vars are set.
