import { clsx } from "clsx";
import { AlertCircle, Inbox } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "./button";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={clsx(
        "animate-shimmer rounded-[8px] bg-[length:200%_100%]",
        "bg-[linear-gradient(90deg,rgb(255_255_255/0.04)_0%,rgb(255_255_255/0.09)_50%,rgb(255_255_255/0.04)_100%)]",
        className,
      )}
    />
  );
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center rounded-card border border-dashed border-line-strong px-6 py-12 text-center">
      <span className="mb-4 inline-flex size-11 items-center justify-center rounded-full bg-surface-2 text-fg-muted">
        <Inbox className="size-5" />
      </span>
      <p className="text-[17px] text-fg">{title}</p>
      <p className="mt-1 max-w-sm text-[14px] text-fg-muted">{body}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorState({ title, body, issues, onRetry }: {
  title: string; body?: string; issues?: string[]; onRetry?: () => void;
}) {
  return (
    <div role="alert" className="rounded-card bg-fail/[0.07] p-5 ring-1 ring-fail/25">
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 size-5 shrink-0 text-fail" />
        <div className="flex-1">
          <p className="text-[15px] text-fg">{title}</p>
          {body && <p className="mt-1 text-[14px] text-fg-muted">{body}</p>}
          {issues && issues.length > 0 && (
            <ul className="mt-3 space-y-1.5">
              {issues.map((i) => (
                <li key={i} className="flex gap-2 text-[14px] text-fg-muted">
                  <span className="mt-2 size-1 shrink-0 rounded-full bg-fail" aria-hidden />
                  {i}
                </li>
              ))}
            </ul>
          )}
          {onRetry && <Button size="sm" variant="secondary" className="mt-4" onClick={onRetry}>Try again</Button>}
        </div>
      </div>
    </div>
  );
}
