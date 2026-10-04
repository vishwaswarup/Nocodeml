import Link from "next/link";
import { Logo } from "./logo";

/** Floating glass pill nav, after Palette's header. */
export function SiteNav({ links = [], cta }: { links?: { href: string; label: string }[]; cta?: React.ReactNode }) {
  return (
    <header className="pointer-events-none fixed inset-x-0 top-5 z-50 flex justify-center px-4">
      <div className="glass pointer-events-auto flex h-14 w-full max-w-[760px] items-center justify-between rounded-full pr-1.5 pl-6">
        <Link href="/" className="flex items-center gap-2" aria-label="NoCodeML home">
          <Logo />
        </Link>
        <nav className="flex items-center gap-1">
          {links.map((l) => (
            <Link key={l.href} href={l.href} className="hidden rounded-full px-3 py-2 text-[15px] text-fg-muted transition-colors hover:text-fg sm:block">
              {l.label}
            </Link>
          ))}
          {cta && <div className="ml-2">{cta}</div>}
        </nav>
      </div>
    </header>
  );
}
