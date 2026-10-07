"use client";

import { useEffect, useState } from "react";
import { api } from "./api";

/**
 * Fetch chart data for a URL (null = don't fetch). Keeps showing the previous result, flagged `stale`,
 * while the next one loads, so the page doesn't jump.
 */
export function useChart<T>(url: string | null) {
  const [loaded, setLoaded] = useState<{ url: string; data: T } | null>(null);
  const [failed, setFailed] = useState<{ url: string; message: string } | null>(null);
  useEffect(() => {
    if (!url) return;
    let alive = true;
    api<T>(url)
      .then((d) => { if (alive) setLoaded({ url, data: d }); })
      .catch((e: Error) => { if (alive) setFailed({ url, message: e.message }); });
    return () => { alive = false; };
  }, [url]);
  return {
    data: loaded?.data ?? null,
    stale: !!url && loaded?.url !== url && failed?.url !== url,
    error: failed && failed.url === url ? failed.message : null,
  };
}
