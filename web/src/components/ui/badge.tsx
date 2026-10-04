import { clsx } from "clsx";
import { AlertTriangle, Check, Info, X } from "lucide-react";
import type { ReactNode } from "react";

/** Palette's "● 252 Matches made" pill. */
export function CountBadge({ value, children }: { value?: ReactNode; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-ember/10 py-1 pr-3.5 pl-1 text-[15px] font-medium text-ember">
      <span className="inline-flex items-center gap-1.5 rounded-full bg-ember px-2 py-0.5 text-white">
        <span className="size-1.5 rounded-full bg-white" aria-hidden />
        {value}
      </span>
      {children}
    </span>
  );
}

export function Tag({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={clsx("inline-flex items-center rounded-full bg-white/10 px-2.5 py-0.5 text-[13px] text-fg", className)}>
      {children}
    </span>
  );
}

export type Status = "pass" | "warn" | "fail" | "info";

const statusStyle: Record<Status, { cls: string; icon: ReactNode; label: string }> = {
  pass: { cls: "bg-pass/12 text-pass", icon: <Check className="size-3.5" strokeWidth={2.5} />, label: "Pass" },
  warn: { cls: "bg-warn/12 text-warn", icon: <AlertTriangle className="size-3.5" strokeWidth={2.5} />, label: "Warning" },
  fail: { cls: "bg-fail/12 text-fail", icon: <X className="size-3.5" strokeWidth={2.5} />, label: "Fail" },
  info: { cls: "bg-info/12 text-info", icon: <Info className="size-3.5" strokeWidth={2.5} />, label: "Info" },
};

/** Icon + colour + text: status is never conveyed by colour alone. */
export function StatusPill({ status, children }: { status: Status; children?: ReactNode }) {
  const s = statusStyle[status];
  return (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[13px] font-medium", s.cls)}>
      {s.icon}
      {children ?? s.label}
    </span>
  );
}

export function StatusIcon({ status }: { status: Status }) {
  const s = statusStyle[status];
  return (
    <span className={clsx("inline-flex size-6 shrink-0 items-center justify-center rounded-full", s.cls)} role="img" aria-label={s.label}>
      {s.icon}
    </span>
  );
}
