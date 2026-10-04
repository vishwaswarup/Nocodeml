import { clsx } from "clsx";
import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const variants: Record<Variant, string> = {
  primary: "bg-fg text-on-light hover:bg-white/90",
  secondary: "bg-surface-3 text-fg hover:bg-[#474747]",
  ghost: "text-fg-muted hover:text-fg hover:bg-white/5",
  danger: "bg-fail/15 text-fail hover:bg-fail/25",
};
const sizes: Record<Size, string> = {
  sm: "h-8 px-3.5 text-[13px] gap-1.5",
  md: "h-10 px-5 text-[15px] gap-2",
  lg: "h-12 px-6 text-[17px] gap-2",
};

export function Button({
  variant = "primary", size = "md", loading = false, icon, className, children, disabled, ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant; size?: Size; loading?: boolean; icon?: ReactNode;
}) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={clsx(
        "inline-flex select-none items-center justify-center rounded-full font-medium whitespace-nowrap",
        "transition-[background-color,color,transform] duration-150 ease-out active:scale-[0.97]",
        "disabled:pointer-events-none disabled:opacity-40",
        variants[variant], sizes[size], className,
      )}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
}

/** Square control, as in Palette's form up/down arrows. */
export function IconButton({
  label, active = true, className, children, ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }) {
  return (
    <button
      {...rest}
      aria-label={label}
      title={label}
      className={clsx(
        "inline-flex size-9 items-center justify-center rounded-[8px] transition-colors duration-150",
        active ? "bg-fg text-on-light hover:bg-white/90" : "bg-surface-3 text-fg-subtle",
        "disabled:pointer-events-none",
        className,
      )}
    >
      {children}
    </button>
  );
}
