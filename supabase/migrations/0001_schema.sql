-- NoCodeML Phase 3: schema, row-level security, immutability rules, storage policies.
-- Ownership model: every row belongs to a project; a project belongs to one auth user.
-- Nothing is readable or writable without a matching auth.uid().

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

create table public.projects (
  id          uuid primary key default gen_random_uuid(),
  owner_id    uuid not null default auth.uid() references auth.users (id) on delete cascade,
  name        text not null check (char_length(btrim(name)) between 1 and 120),
  description text not null default '',
  task        text check (task in ('classification', 'regression')),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create index projects_owner_idx on public.projects (owner_id, updated_at desc);

create table public.datasets (
  id         uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects (id) on delete cascade,
  name       text not null,
  created_at timestamptz not null default now()
);
create index datasets_project_idx on public.datasets (project_id);

create table public.dataset_versions (
  id            uuid primary key default gen_random_uuid(),
  project_id    uuid not null references public.projects (id) on delete cascade,
  dataset_id    uuid not null references public.datasets (id) on delete cascade,
  version       integer not null check (version >= 1),
  storage_path  text not null,   -- object path inside the `datasets` bucket
  filename      text not null,
  size_bytes    bigint not null check (size_bytes >= 0),
  n_rows        integer not null,
  n_columns     integer not null,
  fingerprint   text not null,
  profile       jsonb not null default '{}'::jsonb,
  created_at    timestamptz not null default now(),
  unique (dataset_id, version),
  unique (storage_path)
);
create index dataset_versions_project_idx on public.dataset_versions (project_id);

-- One pipeline per project: pipeline_id in the engine == projects.id.
create table public.pipeline_versions (
  project_id               uuid not null references public.projects (id) on delete cascade,
  version                  integer not null check (version >= 1),
  status                   text not null default 'draft' check (status in ('draft', 'finalized')),
  config                   jsonb not null,
  config_hash              text not null,
  finalized_experiment_id  text,
  created_at               timestamptz not null default now(),
  primary key (project_id, version),
  check ((status = 'finalized') = (finalized_experiment_id is not null))
);

create table public.experiments (
  project_id        uuid not null references public.projects (id) on delete cascade,
  number            integer not null check (number >= 1),
  experiment_id     text not null,
  pipeline_version  integer not null,
  parent_number     integer,
  config_hash       text not null,
  result            jsonb not null,   -- full ExperimentResult (models, metrics, quality, environment)
  created_at        timestamptz not null default now(),
  primary key (project_id, number),
  unique (project_id, experiment_id),
  foreign key (project_id, pipeline_version) references public.pipeline_versions (project_id, version)
);

create table public.artifacts (
  id               uuid primary key default gen_random_uuid(),
  project_id       uuid not null references public.projects (id) on delete cascade,
  experiment_number integer,
  kind             text not null check (kind in ('pipeline', 'model', 'configuration', 'metrics', 'report', 'visualization', 'other')),
  bucket           text not null check (bucket in ('models', 'pipelines', 'reports', 'visualizations', 'datasets')),
  storage_path     text not null,
  size_bytes       bigint not null check (size_bytes >= 0),
  sha256           text,
  created_at       timestamptz not null default now(),
  unique (bucket, storage_path),
  foreign key (project_id, experiment_number) references public.experiments (project_id, number)
);
create index artifacts_project_idx on public.artifacts (project_id);

-- ---------------------------------------------------------------------------
-- Ownership helper (security definer so policies don't recurse through RLS)
-- ---------------------------------------------------------------------------

create function public.owns_project(pid uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.projects p where p.id = pid and p.owner_id = auth.uid());
$$;
revoke all on function public.owns_project(uuid) from public;
grant execute on function public.owns_project(uuid) to authenticated;

-- ---------------------------------------------------------------------------
-- Integrity triggers
-- ---------------------------------------------------------------------------

-- A pipeline version's configuration never changes. The only legal update is
-- draft -> finalized, and only against an experiment that ran this exact configuration.
create function public.guard_pipeline_version() returns trigger
language plpgsql as $$
begin
  if old.status = 'finalized' then
    raise exception 'finalized pipeline versions are immutable';
  end if;
  if new.config is distinct from old.config
     or new.config_hash <> old.config_hash
     or new.version <> old.version
     or new.project_id <> old.project_id then
    raise exception 'pipeline version configuration is immutable; create a new version';
  end if;
  if new.status = 'finalized' and not exists (
       select 1 from public.experiments e
       where e.project_id = new.project_id
         and e.experiment_id = new.finalized_experiment_id
         and e.pipeline_version = new.version
         and e.config_hash = new.config_hash) then
    raise exception 'can only finalize against an experiment run on this exact configuration';
  end if;
  return new;
end $$;
create trigger pipeline_versions_guard before update on public.pipeline_versions
  for each row execute function public.guard_pipeline_version();

-- An experiment must be recorded against the version (and config) it actually ran on.
create function public.guard_experiment() returns trigger
language plpgsql as $$
begin
  if not exists (select 1 from public.pipeline_versions v
                 where v.project_id = new.project_id and v.version = new.pipeline_version
                   and v.config_hash = new.config_hash) then
    raise exception 'experiment config_hash does not match pipeline version %', new.pipeline_version;
  end if;
  return new;
end $$;
create trigger experiments_guard before insert on public.experiments
  for each row execute function public.guard_experiment();

-- Keep projects.updated_at fresh ("Last updated: 2 hours ago").
create function public.touch_project() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  update public.projects set updated_at = now()
  where id = coalesce(new.project_id, old.project_id);
  return null;
end $$;
create function public.touch_self() returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end $$;

create trigger projects_touch before update on public.projects
  for each row execute function public.touch_self();
create trigger pv_touch after insert or update on public.pipeline_versions
  for each row execute function public.touch_project();
create trigger exp_touch after insert on public.experiments
  for each row execute function public.touch_project();
create trigger dsv_touch after insert on public.dataset_versions
  for each row execute function public.touch_project();

-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------

alter table public.projects          enable row level security;
alter table public.datasets          enable row level security;
alter table public.dataset_versions  enable row level security;
alter table public.pipeline_versions enable row level security;
alter table public.experiments       enable row level security;
alter table public.artifacts         enable row level security;

create policy projects_select on public.projects for select to authenticated using (owner_id = auth.uid());
create policy projects_insert on public.projects for insert to authenticated with check (owner_id = auth.uid());
create policy projects_update on public.projects for update to authenticated
  using (owner_id = auth.uid()) with check (owner_id = auth.uid());
create policy projects_delete on public.projects for delete to authenticated using (owner_id = auth.uid());

-- Mutable per-project data
create policy datasets_all on public.datasets for all to authenticated
  using (public.owns_project(project_id)) with check (public.owns_project(project_id));
create policy dataset_versions_all on public.dataset_versions for all to authenticated
  using (public.owns_project(project_id)) with check (public.owns_project(project_id));
create policy artifacts_all on public.artifacts for all to authenticated
  using (public.owns_project(project_id)) with check (public.owns_project(project_id));

-- History is append-only: versions can be created and finalized, experiments only created.
create policy pv_select on public.pipeline_versions for select to authenticated using (public.owns_project(project_id));
create policy pv_insert on public.pipeline_versions for insert to authenticated with check (public.owns_project(project_id));
create policy pv_update on public.pipeline_versions for update to authenticated
  using (public.owns_project(project_id)) with check (public.owns_project(project_id));
create policy exp_select on public.experiments for select to authenticated using (public.owns_project(project_id));
create policy exp_insert on public.experiments for insert to authenticated with check (public.owns_project(project_id));

-- ---------------------------------------------------------------------------
-- Privileges. Explicit and minimal, so this works whether or not the project has
-- "automatically expose new tables" enabled. RLS above still filters every row.
-- ---------------------------------------------------------------------------

revoke all on public.projects, public.datasets, public.dataset_versions,
              public.pipeline_versions, public.experiments, public.artifacts from anon, authenticated;
grant select, insert, update, delete on public.projects, public.datasets,
      public.dataset_versions, public.artifacts to authenticated;
grant select, insert, update on public.pipeline_versions to authenticated;  -- no delete: history is kept
grant select, insert         on public.experiments       to authenticated;  -- append-only

-- ---------------------------------------------------------------------------
-- Storage: private buckets; object path must start with the caller's user id:
--   {user_id}/{project_id}/...
-- ---------------------------------------------------------------------------

insert into storage.buckets (id, name, public, file_size_limit) values
  ('datasets',       'datasets',       false, 104857600),   -- 100 MB cloud-training limit
  ('models',         'models',         false, 524288000),
  ('pipelines',      'pipelines',      false, 524288000),
  ('reports',        'reports',        false, 52428800),
  ('visualizations', 'visualizations', false, 20971520)
on conflict (id) do nothing;

create policy nocodeml_objects_select on storage.objects for select to authenticated
  using (bucket_id in ('datasets','models','pipelines','reports','visualizations')
         and (storage.foldername(name))[1] = auth.uid()::text);
create policy nocodeml_objects_insert on storage.objects for insert to authenticated
  with check (bucket_id in ('datasets','models','pipelines','reports','visualizations')
              and (storage.foldername(name))[1] = auth.uid()::text);
create policy nocodeml_objects_update on storage.objects for update to authenticated
  using (bucket_id in ('datasets','models','pipelines','reports','visualizations')
         and (storage.foldername(name))[1] = auth.uid()::text);
create policy nocodeml_objects_delete on storage.objects for delete to authenticated
  using (bucket_id in ('datasets','models','pipelines','reports','visualizations')
         and (storage.foldername(name))[1] = auth.uid()::text);
