const nf = new Intl.NumberFormat("en", { maximumFractionDigits: 3 });
const nf0 = new Intl.NumberFormat("en");

export const fmt = (v: unknown): string =>
  v === null || v === undefined ? "–" : typeof v === "number" ? nf.format(v) : String(v);
export const int = (v: number) => nf0.format(v);
export function bytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}
