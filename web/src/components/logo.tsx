/** Wordmark with a gradient tile (Palette uses the same device; ours is a 2x2 "pipeline" grid). */
export function Logo({ size = 20 }: { size?: number }) {
  return (
    <span className="inline-flex items-center gap-2">
      <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden>
        <defs>
          <linearGradient id="ncml-g" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#fc6435" />
            <stop offset="0.63" stopColor="#ffb638" />
            <stop offset="1" stopColor="#80bdff" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="9" height="9" rx="2" fill="url(#ncml-g)" />
        <rect x="11" y="0" width="9" height="9" rx="2" fill="url(#ncml-g)" opacity="0.55" />
        <rect x="0" y="11" width="9" height="9" rx="2" fill="url(#ncml-g)" opacity="0.55" />
        <rect x="11" y="11" width="9" height="9" rx="2" fill="url(#ncml-g)" />
      </svg>
      <span className="text-[19px] font-semibold tracking-[-0.03em] text-fg">NoCodeML</span>
    </span>
  );
}
