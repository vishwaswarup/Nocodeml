-- NoCodeML 0002: security and performance hardening (from Supabase's own database advisor).
-- Safe to run once on top of 0001. Changes no data and no behaviour, only how functions are exposed.

-- 1. Pin the search_path of the trigger functions, so a caller can't shadow tables or functions they use.
alter function public.touch_self()              set search_path = public, pg_temp;
alter function public.guard_pipeline_version()  set search_path = public, pg_temp;
alter function public.guard_experiment()        set search_path = public, pg_temp;

-- 2. Don't expose internal functions through the public API (/rest/v1/rpc/...).
--    Supabase grants EXECUTE to anon and authenticated by default; "revoke from public" doesn't undo that.
--    Trigger functions run as the table owner and need no caller privilege at all.
revoke execute on function public.touch_project()         from public, anon, authenticated;
revoke execute on function public.touch_self()            from public, anon, authenticated;
revoke execute on function public.guard_pipeline_version() from public, anon, authenticated;
revoke execute on function public.guard_experiment()      from public, anon, authenticated;
--    owns_project() is used inside row-level-security policies, which run as the signed-in user: that role keeps
--    EXECUTE, an anonymous visitor does not (it is useless to them: auth.uid() is null).
revoke execute on function public.owns_project(uuid) from public, anon;
grant  execute on function public.owns_project(uuid) to authenticated;
--    Supabase's own event-trigger helper (present on hosted projects only).
do $$ begin
  if to_regprocedure('public.rls_auto_enable()') is not null then
    revoke execute on function public.rls_auto_enable() from public, anon, authenticated;
  end if;
end $$;

-- 3. Evaluate auth.uid() once per query, not once per row.
drop policy projects_select on public.projects;
drop policy projects_insert on public.projects;
drop policy projects_update on public.projects;
drop policy projects_delete on public.projects;
create policy projects_select on public.projects for select to authenticated using (owner_id = (select auth.uid()));
create policy projects_insert on public.projects for insert to authenticated with check (owner_id = (select auth.uid()));
create policy projects_update on public.projects for update to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy projects_delete on public.projects for delete to authenticated using (owner_id = (select auth.uid()));

-- 4. Cover the composite foreign keys (faster joins and cascading deletes).
create index if not exists artifacts_project_experiment_idx   on public.artifacts   (project_id, experiment_number);
create index if not exists experiments_project_pipeline_idx   on public.experiments (project_id, pipeline_version);
