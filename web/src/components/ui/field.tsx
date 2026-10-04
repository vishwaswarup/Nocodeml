import { clsx } from "clsx";
import { useId, type InputHTMLAttributes } from "react";

/** Large, centred, underlined field: Palette's one-question form style. */
export function QuestionField({
  label, hint, error, className, ...rest
}: InputHTMLAttributes<HTMLInputElement> & { label: string; hint?: string; error?: string }) {
  const id = useId();
  return (
    <div className={clsx("w-full", className)}>
      <label htmlFor={id} className="mb-3 block text-center text-[19px] text-fg">{label}</label>
      <input
        id={id}
        {...rest}
        aria-invalid={!!error || undefined}
        aria-describedby={hint || error ? `${id}-d` : undefined}
        className={clsx(
          "w-full border-b bg-transparent pb-3 text-[26px] text-fg outline-none transition-colors placeholder:text-fg-subtle",
          error ? "border-fail" : "border-line-strong focus:border-fg",
        )}
      />
      {(hint || error) && (
        <p id={`${id}-d`} className={clsx("mt-2 text-[13px]", error ? "text-fail" : "text-fg-subtle")}>{error ?? hint}</p>
      )}
    </div>
  );
}

/** Compact field for the dense workspace. */
export function Field({
  label, hint, error, suffix, className, ...rest
}: InputHTMLAttributes<HTMLInputElement> & { label: string; hint?: string; error?: string; suffix?: string }) {
  const id = useId();
  return (
    <div className={className}>
      <label htmlFor={id} className="mb-1.5 block text-[13px] text-fg-muted">{label}</label>
      <div
        className={clsx(
          "flex h-10 items-center rounded-control bg-surface-2 px-3 ring-1 transition-shadow focus-within:ring-fg/60",
          error ? "ring-fail" : "ring-line",
        )}
      >
        <input
          id={id}
          {...rest}
          aria-invalid={!!error || undefined}
          className="min-w-0 flex-1 bg-transparent text-[15px] text-fg outline-none placeholder:text-fg-subtle tabular"
        />
        {suffix && <span className="pl-2 font-mono text-[12px] text-fg-subtle">{suffix}</span>}
      </div>
      {(hint || error) && <p className={clsx("mt-1.5 text-[12px]", error ? "text-fail" : "text-fg-subtle")}>{error ?? hint}</p>}
    </div>
  );
}
