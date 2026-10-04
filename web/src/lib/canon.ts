/**
 * Order-independent JSON for "is this the same as what's saved?" checks.
 *
 * The database stores configs as jsonb, which re-orders object keys, and the server fills in default
 * fields (e.g. constant_value: null). So a draft built in the browser and the same choices read back
 * from the server differ textually. This drops null/undefined fields and the default n_neighbors,
 * and sorts keys, so equal choices compare equal.
 */
export function canonical(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(canonical);
  if (v && typeof v === "object") {
    return Object.fromEntries(
      Object.entries(v as Record<string, unknown>)
        .filter(([k, x]) => x !== null && x !== undefined && !(k === "n_neighbors" && x === 5))
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, x]) => [k, canonical(x)]),
    );
  }
  return v;
}

export const sameJson = (a: unknown, b: unknown) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
