import Link from "next/link";
import { SiteNav } from "@/components/site-nav";
import { Logo } from "@/components/logo";
import { AccordionItem } from "@/components/ui/accordion";
import { CountBadge } from "@/components/ui/badge";

const stages = [
  ["Explore", "Profile every column, flag identifiers, imbalance and gaps."],
  ["Prepare", "Impute, encode, scale. Fitted on training rows only."],
  ["Engineer", "Log transforms, date parts, explicit and reversible."],
  ["Split", "Stratified, chronological or k-fold, with reasons."],
  ["Model", "Up to five models with model-aware regularization."],
  ["Evaluate", "Task-aware metrics against a trivial baseline."],
  ["Check", "Leakage, overfitting and split hygiene, scored."],
  ["Export", "The whole pipeline, versioned and reproducible."],
] as const;

export default function Home() {
  return (
    <>
      <SiteNav
        links={[{ href: "/design", label: "Design system" }, { href: "#how", label: "How it works" }, { href: "#faq", label: "FAQ" }]}
        cta={<Link href="/design" className="inline-flex h-11 items-center rounded-full bg-fg px-5 text-[16px] font-medium text-on-light transition-colors hover:bg-white/90">Start building</Link>}
      />
      <main>
        <section className="flex flex-col items-center px-4 pt-44 pb-20 text-center">
          <CountBadge value="15">quality checks on every run</CountBadge>
          <h1 className="mt-6 max-w-[900px] text-[clamp(2.6rem,7vw,4.6rem)] leading-[1.03] font-medium tracking-[-0.035em]">
            Build machine-learning pipelines you can actually trust
          </h1>
          <p className="mt-6 max-w-[540px] text-[18px] leading-relaxed text-fg-muted">
            Go from a CSV to a trained, evaluated, exportable pipeline without code, with NoCodeML catching leakage,
            misleading metrics and bad splits along the way.
          </p>
          <div className="mt-9 flex gap-3">
            <Link href="/design" className="inline-flex h-12 items-center rounded-full bg-fg px-6 text-[17px] font-medium text-on-light transition-colors hover:bg-white/90">Start building</Link>
            <a href="#how" className="inline-flex h-12 items-center rounded-full bg-surface-3 px-6 text-[17px] font-medium transition-colors hover:bg-[#474747]">How it works</a>
          </div>
        </section>

        <section id="how" className="px-4 pb-28">
          <div className="mx-auto grid max-w-[1180px] gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {stages.map(([title, body], i) => (
              <div key={title} className="group rounded-card bg-surface p-5 transition-colors hover:bg-surface-2">
                <span className="font-mono text-[12px] text-fg-subtle tabular">0{i}</span>
                <p className="mt-10 text-[20px] tracking-[-0.02em]">{title}</p>
                <p className="mt-1.5 text-[14px] leading-snug text-fg-muted">{body}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="faq" className="mx-auto flex max-w-[620px] flex-col gap-2 px-4 pb-28">
          <h2 className="mb-6 text-center text-h1">Questions</h2>
          <AccordionItem title="Do I need to know Python?">No. Every step is a visual choice, and NoCodeML explains what each option does.</AccordionItem>
          <AccordionItem title="Does NoCodeML decide for me?">It recommends and explains; you approve, modify or ignore. Nothing is applied silently.</AccordionItem>
          <AccordionItem title="Where does my data go?">Into your private storage, readable only by your account. Local training is on the roadmap.</AccordionItem>
        </section>

        <section className="signature-gradient flex flex-col items-center px-4 py-28 text-center text-on-light">
          <h2 className="max-w-[560px] text-[clamp(2.2rem,5vw,3.4rem)] leading-[1.05] font-medium tracking-[-0.03em]">Ready to build your first pipeline?</h2>
          <p className="mt-4 text-[18px]">Upload a CSV and NoCodeML walks you through the rest.</p>
          <Link href="/design" className="mt-8 inline-flex h-12 items-center rounded-full bg-on-light px-6 text-[17px] font-medium text-fg transition-transform hover:scale-[1.02]">Start for free</Link>
        </section>
      </main>
      <footer className="bg-bg-deep px-4 py-16">
        <div className="mx-auto flex max-w-[1180px] flex-col justify-between gap-10 sm:flex-row">
          <div>
            <Logo />
            <p className="mt-3 max-w-[280px] text-[15px] text-fg-muted">Correct, reproducible, explainable ML pipelines without code.</p>
            <p className="mt-6 text-[14px] text-fg-subtle">© 2026</p>
          </div>
          <div className="flex gap-16 text-[15px]">
            <div className="flex flex-col gap-2.5"><span className="text-fg">Product</span><a className="text-fg-muted hover:text-fg" href="#how">How it works</a><Link className="text-fg-muted hover:text-fg" href="/design">Design system</Link></div>
            <div className="flex flex-col gap-2.5"><span className="text-fg">Project</span><a className="text-fg-muted hover:text-fg" href="#faq">FAQ</a></div>
          </div>
        </div>
      </footer>
    </>
  );
}
