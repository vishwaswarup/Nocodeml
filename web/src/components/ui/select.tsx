import { clsx } from "clsx";
import { ChevronDown } from "lucide-react";
import { useId, type SelectHTMLAttributes } from "react";

/** Native select (keyboard + screen-reader friendly), styled to match Field. */
export function Select({ label, hint, className, children, ...rest }:
  SelectHTMLAttributes<HTMLSelectElement> & { label: string; hint?: string }) {
  const id = useId();
  return (
    <div className={className}>
      <label htmlFor={id} className="mb-1.5 block text-[13px] text-fg-muted">{label}</label>
      <div className="relative">
        <select id={id} {...rest}
          className={clsx("h-10 w-full appearance-none rounded-control bg-surface-2 pr-9 pl-3 text-[15px] text-fg ring-1 ring-line outline-none",
            "transition-shadow focus:ring-fg/60")}>
          {children}
        </select>
        <ChevronDown className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-fg-subtle" aria-hidden />
      </div>
      {hint && <p className="mt-1.5 text-[12px] text-fg-subtle">{hint}</p>}
    </div>
  );
}
