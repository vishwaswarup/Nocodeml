-- NoCodeML 0003: durable training jobs.
-- Job status used to live only in the API's memory (lost on every restart). It now lives here, behind the same
-- row-level security as everything else. A running job writes a heartbeat every few seconds; a job whose heartbeat
-- has gone quiet (its server died) is marked failed the next time its owner looks at it.

create table public.training_jobs (
  id                uuid primary key default gen_random_uuid(),
  project_id        uuid not null references public.projects (id) on delete cascade,
  status            text not null default 'queued'
                    check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
  created_at        timestamptz not null default now(),
  started_at        timestamptz,
  finished_at       timestamptz,
  heartbeat_at      timestamptz not null default now(),
  experiment_number integer,
  error             text,
  issues            jsonb not null default '[]',
  warnings          jsonb not null default '[]',
  cancel_requested  boolean not null default false
);
create index training_jobs_project_idx on public.training_jobs (project_id, created_at desc);

-- At most one queued/running job per project. The database enforces it, so it holds even with several API processes.
create unique index training_jobs_one_active on public.training_jobs (project_id)
  where status in ('queued', 'running');

alter table public.training_jobs enable row level security;
create policy training_jobs_select on public.training_jobs for select to authenticated using (public.owns_project(project_id));
create policy training_jobs_insert on public.training_jobs for insert to authenticated with check (public.owns_project(project_id));
create policy training_jobs_update on public.training_jobs for update to authenticated
  using (public.owns_project(project_id)) with check (public.owns_project(project_id));

revoke all on public.training_jobs from public, anon, authenticated;
grant select, insert, update on public.training_jobs to authenticated;   -- no delete: removed with its project

-- A finished job is history: it can't be reopened, renamed back to running, or have its outcome rewritten.
create function public.guard_training_job() returns trigger
language plpgsql set search_path = public, pg_temp as $$
begin
  if old.status in ('succeeded', 'failed', 'cancelled') then
    raise exception 'training job % is finished and can no longer change', old.id using errcode = 'check_violation';
  end if;
  if new.project_id <> old.project_id then
    raise exception 'training job cannot move to another project' using errcode = 'check_violation';
  end if;
  return new;
end $$;
revoke execute on function public.guard_training_job() from public, anon, authenticated;
create trigger training_jobs_guard before update on public.training_jobs
  for each row execute function public.guard_training_job();
