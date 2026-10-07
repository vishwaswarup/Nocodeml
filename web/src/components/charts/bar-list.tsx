import { ChartFrame, type TableView } from "./chart-frame";

export type BarItem = { label: string; value: number; detail?: string };

/**
 * Horizontal bars with the value at the tip: the right form for a handful of categories. One colour,
 * thin bars with a rounded data end, square at the baseline.
 */
export function BarList({ title, subtitle, items, max, format, footnote, table, ariaLabel }: {
  title: string; subtitle?: string; items: BarItem[]; max?: number; format: (v: number) => string;
  footnote?: string; table: TableView; ariaLabel: string;
}) {
  const top = max ?? Math.max(1e-9, ...items.map((i) => i.value));
  return (
    <ChartFrame title={title} subtitle={subtitle} table={table}>
      <ul role="img" aria-label={ariaLabel} className="space-y-2">
        {items.map((it) => (
          <li key={it.label} className="grid grid-cols-[minmax(0,9rem)_1fr_auto] items-center gap-3 text-[13px]" title={`${it.label}: ${format(it.value)}${it.detail ? ` (${it.detail})` : ""}`}>
            <span className="truncate text-right text-fg-muted">{it.label}</span>
            <span className="h-[18px] rounded-r-[4px] bg-sky" style={{ width: `${Math.max(1.5, (100 * it.value) / top)}%` }} />
            <span className="font-mono text-[12px] tabular">{format(it.value)}{it.detail && <span className="ml-1.5 text-fg-subtle">{it.detail}</span>}</span>
          </li>
        ))}
      </ul>
      {footnote && <p className="mt-3 text-[12px] leading-snug text-fg-subtle">{footnote}</p>}
    </ChartFrame>
  );
}
