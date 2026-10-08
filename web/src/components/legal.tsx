import Link from "next/link";
import type { ReactNode } from "react";
import { Logo } from "@/components/logo";
import { SITE } from "@/lib/site";

/** Shared frame for the privacy policy and terms: plain, readable, no marketing. */
export function LegalPage({ title, intro, children }: { title: string; intro: string; children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="flex h-16 items-center justify-between border-b border-line px-4 sm:px-8">
        <Link href="/" aria-label="NoCodeML home"><Logo /></Link>
        <Link href="/login" className="text-[14px] text-fg-muted hover:text-fg">Sign in</Link>
      </header>
      <main className="mx-auto max-w-[720px] px-4 py-14 sm:py-20">
        <h1 className="text-h1">{title}</h1>
        <p className="mt-3 text-[14px] text-fg-subtle">Last updated {SITE.updated}</p>
        <p className="mt-6 text-[17px] leading-relaxed text-fg-muted">{intro}</p>
        <div className="mt-10 space-y-10">{children}</div>
        <nav aria-label="Legal pages" className="mt-16 flex gap-6 border-t border-line pt-6 text-[14px] text-fg-muted">
          <Link className="hover:text-fg" href="/privacy">Privacy Policy</Link>
          <Link className="hover:text-fg" href="/terms">Terms of Use</Link>
          <Link className="hover:text-fg" href="/">Home</Link>
        </nav>
      </main>
    </div>
  );
}

export function LegalSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h2 className="text-h3">{title}</h2>
      <div className="mt-3 space-y-3 text-[16px] leading-relaxed text-[#c4c4c4]">{children}</div>
    </section>
  );
}

export function Bullets({ items }: { items: ReactNode[] }) {
  return (
    <ul className="list-disc space-y-1.5 pl-5 marker:text-fg-subtle">
      {items.map((t, i) => <li key={i}>{t}</li>)}
    </ul>
  );
}

export const ContactLink = () => (
  <a className="text-fg underline underline-offset-4 hover:no-underline" href={`mailto:${SITE.email}`}>{SITE.email}</a>
);
