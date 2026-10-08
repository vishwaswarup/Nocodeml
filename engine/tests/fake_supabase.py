"""In-memory stand-in for the supabase-py client subset the persistence layer uses.

It mimics PostgREST semantics (unique keys, upsert, filters, ordering) and RLS-style
ownership filtering. Real RLS is verified separately against Postgres (test_supabase_sql.py).
"""
import copy
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

PKS = {"projects": ("id",), "datasets": ("id",), "dataset_versions": ("id",),
       "pipeline_versions": ("project_id", "version"), "experiments": ("project_id", "number"),
       "artifacts": ("id",), "training_jobs": ("id",)}
UNIQUE = {"experiments": [("project_id", "experiment_id")],
          "dataset_versions": [("storage_path",)], "artifacts": [("bucket", "storage_path")]}


def jsonb(v):
    """Postgres jsonb re-orders object keys: shorter keys first, then bytewise. Emulate that so tests
    catch any code that depends on key order surviving a database round trip."""
    if isinstance(v, dict):
        return {k: jsonb(v[k]) for k in sorted(v, key=lambda k: (len(k.encode()), k.encode()))}
    if isinstance(v, list):
        return [jsonb(x) for x in v]
    return v


JSONB_COLUMNS = ("config", "result", "profile")


class FakeStore:
    def __init__(self):
        self.tables = {t: [] for t in PKS}
        self.objects = {}  # (bucket, path) -> bytes


class _Query:
    def __init__(self, client, table):
        self.c, self.t = client, table
        self.op, self.payload, self.filters, self.ord, self.lim, self.conflict = "select", None, [], None, None, None

    def select(self, cols="*"): self.op = "select"; return self
    def insert(self, row): self.op, self.payload = "insert", row; return self
    def upsert(self, row, on_conflict=None): self.op, self.payload, self.conflict = "upsert", row, on_conflict; return self
    def update(self, row): self.op, self.payload = "update", row; return self
    def delete(self): self.op = "delete"; return self
    def eq(self, k, v): self.filters.append((k, v)); return self
    def order(self, col, desc=False): self.ord = (col, desc); return self
    def limit(self, n): self.lim = n; return self

    def _visible(self, row):
        owned = {p["id"] for p in self.c.store.tables["projects"] if p["owner_id"] == self.c.user_id}
        if self.t == "projects":
            return row["owner_id"] == self.c.user_id
        return row["project_id"] in owned

    def _match(self, row):
        return self._visible(row) and all(str(row.get(k)) == str(v) for k, v in self.filters)

    def execute(self):
        rows = self.c.store.tables[self.t]
        if self.op == "select":
            out = [copy.deepcopy(r) for r in rows if self._match(r)]
            if self.ord:
                out.sort(key=lambda r: r[self.ord[0]], reverse=self.ord[1])
            return SimpleNamespace(data=out[: self.lim] if self.lim else out)
        if self.op in ("insert", "upsert"):
            row = copy.deepcopy(self.payload)
            for col in JSONB_COLUMNS:
                if col in row:
                    row[col] = jsonb(row[col])
            if self.t == "projects":
                row.setdefault("owner_id", self.c.user_id)
                row["updated_at"] = datetime.now(timezone.utc).isoformat()
                assert row["owner_id"] == self.c.user_id, "row-level security violation"
            elif not self._visible(row):
                raise PermissionError("row-level security violation")
            if "id" in PKS[self.t]:
                row.setdefault("id", str(uuid.uuid4()))
            row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            row.setdefault("status", "draft") if self.t == "pipeline_versions" else None
            key = tuple(row[k] for k in PKS[self.t])
            for r in rows:
                if tuple(r[k] for k in PKS[self.t]) == key:
                    if self.op == "insert":
                        raise ValueError(f"duplicate key {key}")
                    r.update(row)
                    return SimpleNamespace(data=[copy.deepcopy(r)])
            if self.t == "training_jobs":    # partial unique index: one queued/running job per project
                row.setdefault("status", "queued")
                row.setdefault("heartbeat_at", datetime.now(timezone.utc).isoformat())
                row.setdefault("cancel_requested", False)
                if row["status"] in ("queued", "running") and any(
                        r["project_id"] == row["project_id"] and r["status"] in ("queued", "running") for r in rows):
                    raise ValueError('duplicate key value violates unique constraint "training_jobs_one_active" (23505)')
            for cols in UNIQUE.get(self.t, []):
                if any(all(r[c] == row[c] for c in cols) for r in rows):
                    raise ValueError(f"unique violation {cols}")
            rows.append(row)
            if self.t in ("pipeline_versions", "experiments", "dataset_versions"):
                for p in self.c.store.tables["projects"]:
                    if p["id"] == row["project_id"]:
                        p["updated_at"] = datetime.now(timezone.utc).isoformat()
            return SimpleNamespace(data=[copy.deepcopy(row)])
        hit = [r for r in rows if self._match(r)]
        if self.op == "update":
            for r in hit:
                if self.t == "training_jobs" and r["status"] in ("succeeded", "failed", "cancelled"):
                    raise ValueError(f"training job {r['id']} is finished and can no longer change")
                r.update(self.payload)
        else:
            for r in hit:
                rows.remove(r)
                if self.t == "projects":  # ON DELETE CASCADE
                    for t, rs in self.c.store.tables.items():
                        if t != "projects":
                            rs[:] = [x for x in rs if x.get("project_id") != r["id"]]
        return SimpleNamespace(data=[copy.deepcopy(r) for r in hit])


class _Bucket:
    def __init__(self, client, bucket): self.c, self.b = client, bucket

    def _check(self, path):
        if path.split("/")[0] != self.c.user_id:
            raise PermissionError("row-level security violation (storage)")

    def upload(self, path, data, opts=None):
        self._check(path)
        if (self.b, path) in self.c.store.objects:
            raise ValueError("object exists")
        self.c.store.objects[(self.b, path)] = bytes(data)

    def download(self, path):
        self._check(path)
        return self.c.store.objects[(self.b, path)]

    def remove(self, paths):
        for p in paths:
            self._check(p)
            self.c.store.objects.pop((self.b, p), None)

    def create_signed_url(self, path, expires):
        self._check(path)
        return {"signedURL": f"https://example.invalid/sign/{self.b}/{path}?exp={expires}"}


class FakeSupabase:
    def __init__(self, store, user_id):
        self.store, self.user_id = store, str(user_id)
        self.storage = SimpleNamespace(from_=lambda b: _Bucket(self, b))

    def table(self, name): return _Query(self, name)
