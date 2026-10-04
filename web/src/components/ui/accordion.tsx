"use client";

import { clsx } from "clsx";
import { Plus } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useId, useState, type ReactNode } from "react";

/** Palette's FAQ rows: graphite card, round + that turns into x. */
export function AccordionItem({ title, children, defaultOpen = false }: {
  title: ReactNode; children: ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <div className="rounded-control bg-surface">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left text-[16px] text-fg"
      >
        {title}
        <motion.span
          animate={{ rotate: open ? 45 : 0 }}
          transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className={clsx("inline-flex size-7 shrink-0 items-center justify-center rounded-full", "bg-surface-3 text-fg")}
          aria-hidden
        >
          <Plus className="size-4" strokeWidth={2.25} />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 text-[15px] leading-relaxed text-fg-muted">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
