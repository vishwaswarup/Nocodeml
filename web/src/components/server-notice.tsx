"use client";

import { Loader2 } from "lucide-react";
import { useSyncExternalStore } from "react";
import { isSlow, subscribeSlow } from "@/lib/slow-notice";

/** A small, calm note shown only while a request is taking unusually long. */
export function ServerNotice() {
  const slow = useSyncExternalStore(subscribeSlow, isSlow, () => false);
  if (!slow) return null;
  return (
    <div role="status" aria-live="polite"
      className="glass fixed bottom-6 left-1/2 z-50 flex max-w-[calc(100vw-2rem)] -translate-x-1/2 items-center gap-3 rounded-full px-5 py-3 text-[14px]">
      <Loader2 className="size-4 shrink-0 animate-spin text-fg-muted" aria-hidden />
      <span>Still working. The server may be waking up after a quiet spell, which can take up to a minute.</span>
    </div>
  );
}
