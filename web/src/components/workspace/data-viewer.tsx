"use client";

import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { DataTable } from "@/components/ui/data";
import { ErrorState, Skeleton } from "@/components/ui/feedback";
import { api } from "@/lib/api";
import { int } from "@/lib/format";
import type { RowsPage } from "@/lib/types";

const PAGE = 25;
const numeric = (dtype: string) => /int|float/.test(dtype);
const isDate = (dtype: string) => dtype.startsWith("datetime");
/** "2022-01-01T00:00:00.000" -> "2022-01-01"; keep the time only when it isn't midnight. */
const shortDate = (v: unknown) => {
  if (typeof v !== "string") return v;
  const [d, t = ""] = v.split("T");
  const time = t.replace(/\.0+$/, "").replace(/Z$/, "");
  return !time || time === "00:00:00" ? d : `${d} ${time.slice(0, 5)}`;
};

/** Server-paginated raw data viewer: never loads more than one page into the browser. */
export function DataViewer({ projectId, datasetId }: { projectId: string; datasetId: string }) {
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState<string | undefined>();
  const [desc, setDesc] = useState(false);
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [data, setData] = useState<RowsPage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => { setQuery(search); setPage(1); }, 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    let alive = true;
    const qs = new URLSearchParams({ page: String(page), page_size: String(PAGE) });
    if (sortBy) { qs.set("sort_by", sortBy); qs.set("descending", String(desc)); }
    if (query) qs.set("search", query);
    api<RowsPage>(`/projects/${projectId}/datasets/${datasetId}/rows?${qs}`)
      .then((d) => { if (alive) { setData(d); setError(null); } })
      .catch((e: Error) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [projectId, datasetId, page, sortBy, desc, query]);

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex h-9 w-full max-w-xs items-center gap-2 rounded-full bg-surface px-3.5 ring-1 ring-line focus-within:ring-fg/50">
          <Search className="size-4 text-fg-subtle" aria-hidden />
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search all columns"
            aria-label="Search rows" className="min-w-0 flex-1 bg-transparent text-[14px] outline-none placeholder:text-fg-subtle" />
        </div>
        {data && (
          <div className="flex items-center gap-2 text-[13px] text-fg-muted">
            <span className="tabular">
              {data.total === 0 ? "No rows" : `${int((page - 1) * PAGE + 1)}–${int(Math.min(page * PAGE, data.total))} of ${int(data.total)}`}
            </span>
            <button aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}
              className="inline-flex size-8 items-center justify-center rounded-full bg-surface text-fg disabled:opacity-30"><ChevronLeft className="size-4" /></button>
            <button aria-label="Next page" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}
              className="inline-flex size-8 items-center justify-center rounded-full bg-surface text-fg disabled:opacity-30"><ChevronRight className="size-4" /></button>
          </div>
        )}
      </div>
      {error ? <ErrorState title="Couldn't load rows" body={error} />
        : !data ? <Skeleton className="h-[420px] rounded-card" />
        : (
          <DataTable
            caption="Raw dataset rows"
            columns={data.columns.map((c) => ({ key: c.name, label: c.name, dtype: c.dtype.replace("64", "").replace("datetime[us]", "datetime"), numeric: numeric(c.dtype) }))}
            rows={data.rows.map((r) => {
              const out = { ...r };
              for (const c of data.columns) if (isDate(c.dtype)) out[c.name] = shortDate(r[c.name]);
              return out;
            })}
            sortBy={sortBy}
            descending={desc}
            onSort={(k) => { if (k === sortBy) setDesc(!desc); else { setSortBy(k); setDesc(false); } setPage(1); }}
          />
        )}
    </div>
  );
}
