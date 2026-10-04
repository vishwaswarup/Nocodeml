import { clsx } from "clsx";
import type { HTMLAttributes } from "react";

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div {...rest} className={clsx("rounded-card bg-surface", className)} />;
}

/** The numbered square from Palette's "1 Contact Info" heading. */
export function StepNumber({ n, active = true }: { n: number | string; active?: boolean }) {
  return (
    <span
      className={clsx(
        "inline-flex size-[22px] shrink-0 items-center justify-center rounded-[6px] font-mono text-[12px] font-medium tabular",
        active ? "bg-fg text-on-light" : "bg-surface-3 text-fg-muted",
      )}
    >
      {n}
    </span>
  );
}

export function Eyebrow({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={clsx("font-mono text-[12px] tracking-[0.08em] text-fg-subtle uppercase", className)}>{children}</p>
  );
}
