import { Check } from "lucide-react";
import Link from "next/link";
import { SiteNav } from "@/components/site-nav";
import { Logo } from "@/components/logo";
import { AccordionItem } from "@/components/ui/accordion";
import { CountBadge } from "@/components/ui/badge";

/* What a researcher re-types for every experiment. Kept realistic on purpose. */
const boilerplate = `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

df = pd.read_csv("customers.csv").drop(columns=["customer_id"])
X, y = df.drop(columns="churn"), df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
])
model = Pipeline([("prep", prep),
                  ("rf", RandomForestClassifier(max_depth=10))])
model.fit(X_tr, y_tr)
print(f1_score(y_te, model.predict(X_te)))
# ...now copy all of this for SVM, KNN, a new max_depth...`;

const runs = [
  { name: "Random Forest", detail: "max_depth 10", f1: "0.381", best: true },
  { name: "Logistic Regression", detail: "L2 · C 1.0", f1: "0.214" },
  { name: "SVM", detail: "rbf · C 1.0", f1: "0.302" },
  { name: "KNN", detail: "k = 15", f1: "0.188" },
];

const steps = [
  ["Upload", "Drop in a CSV. Every column is profiled: types, gaps, outliers, identifiers, imbalance."],
  ["Configure", "Pick preprocessing and up to five models. Each choice comes with a reason, so you can move fast."],
  ["Compare", "Train once and see every model side by side, on the same split, against a baseline."],
  ["Tweak & re-run", "Change a hyperparameter or a preprocessing step, re-run, and compare with the last run."],
] as const;

const features = [
  "Up to 5 models trained and compared in one run",
  "Model-aware hyperparameters and regularization, with sensible defaults",
  "Every run versioned, so you can go back and compare any two",
  "Leakage, overfitting and baseline checks, done for you",
  "Export the full pipeline (preprocessing + model) as a file",
];

export default function Home() {
  return (
    <>
      <SiteNav
        links={[{ href: "#why", label: "Why" }, { href: "#how", label: "How it works" }, { href: "#faq", label: "FAQ" }]}
        cta={<Link href="/login" className="inline-flex h-11 items-center rounded-full bg-fg px-5 text-[16px] font-medium text-on-light transition-colors hover:bg-white/90">Start building</Link>}
      />
      <main>
        <section className="flex flex-col items-center px-4 pt-44 pb-24 text-center">
          <CountBadge value="5">models compared in one run</CountBadge>
          <h1 className="mt-6 max-w-[920px] text-[clamp(2.6rem,7vw,4.6rem)] leading-[1.03] font-medium tracking-[-0.035em]">
            Stop rewriting the same ML code for every experiment
          </h1>
          <p className="mt-6 max-w-[580px] text-[18px] leading-relaxed text-fg-muted">
            Upload a dataset, pick your models, tweak hyperparameters and compare the results side by side.
            The experiments you&apos;d script in Python, done in minutes in your browser.
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <Link href="/login" className="inline-flex h-12 items-center rounded-full bg-fg px-6 text-[17px] font-medium text-on-light transition-colors hover:bg-white/90">Start building</Link>
            <a href="#why" className="inline-flex h-12 items-center rounded-full bg-surface-3 px-6 text-[17px] font-medium transition-colors hover:bg-[#474747]">See the difference</a>
          </div>
        </section>

        <section id="why" className="scroll-mt-28 px-4 pb-28">
          <div className="mx-auto max-w-[1180px]">
            <h2 className="mx-auto max-w-[640px] text-center text-h1"><span className="block">Same experiment.</span> <span className="block text-fg-muted">A fraction of the effort.</span></h2>
            <div className="mt-12 grid gap-3 lg:grid-cols-[1.25fr_1fr]">
              <div className="flex flex-col overflow-hidden rounded-card bg-bg-deep ring-1 ring-line">
                <div className="flex items-center justify-between border-b border-line px-5 py-3">
                  <span className="text-[14px] text-fg-muted">Every experiment, by hand</span>
                  <span className="font-mono text-[12px] text-fg-subtle">experiment_v7_final2.py</span>
                </div>
                <pre className="flex-1 overflow-x-auto px-5 py-4 font-mono text-[12.5px] leading-[1.65] text-fg-muted"><code>{boilerplate}</code></pre>
              </div>
              <div className="flex flex-col rounded-card bg-surface p-6">
                <div className="flex items-center justify-between"><span className="text-[14px] text-fg-muted">The same thing on NoCodeML</span><span className="font-mono text-[11px] tracking-wider text-fg-subtle uppercase">example run</span></div>
                <ol className="mt-4 space-y-2.5 text-[16px]">
                  {["Choose median imputation, scaling and one-hot encoding", "Tick Random Forest, Logistic Regression, SVM and KNN", "Press Train"].map((t, i) => (
                    <li key={t} className="flex gap-3">
                      <span className="mt-0.5 inline-flex size-[22px] shrink-0 items-center justify-center rounded-[6px] bg-fg font-mono text-[12px] text-on-light">{i + 1}</span>
                      {t}
                    </li>
                  ))}
                </ol>
                <div className="mt-6 rounded-control bg-surface-2 p-1.5">
                  {runs.map((r) => (
                    <div key={r.name} className="flex items-center justify-between rounded-[8px] px-3 py-2.5 text-[14px] odd:bg-white/[0.03]">
                      <span>
                        {r.name} <span className="ml-1.5 font-mono text-[11.5px] text-fg-subtle">{r.detail}</span>
                      </span>
                      <span className="flex items-center gap-2 font-mono tabular">
                        {r.best && <span className="rounded-full bg-ember/12 px-2 py-0.5 font-sans text-[11px] text-ember">highest F1</span>}
                        {r.f1}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-auto pt-6 text-[13px] leading-snug text-fg-subtle">
                  Want to try max_depth 5 instead? Change one number and re-run. The previous run stays for comparison.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section id="how" className="scroll-mt-28 px-4 pb-28">
          <div className="mx-auto max-w-[1180px]">
            <h2 className="text-center text-h1">How it works</h2>
            <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {steps.map(([title, body], i) => (
                <div key={title} className="rounded-card bg-surface p-5 transition-colors hover:bg-surface-2">
                  <span className="font-mono text-[12px] text-fg-subtle tabular">0{i + 1}</span>
                  <p className="mt-10 text-[20px] tracking-[-0.02em]">{title}</p>
                  <p className="mt-1.5 text-[14px] leading-snug text-fg-muted">{body}</p>
                </div>
              ))}
            </div>
            <ul className="mx-auto mt-14 grid max-w-[860px] gap-x-10 gap-y-3 sm:grid-cols-2">
              {features.map((f) => (
                <li key={f} className="flex gap-3 text-[15px] text-fg-muted">
                  <Check className="mt-0.5 size-4 shrink-0 text-fg" strokeWidth={2.5} />
                  {f}
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section id="faq" className="mx-auto flex max-w-[620px] scroll-mt-28 flex-col gap-2 px-4 pb-28">
          <h2 className="mb-6 text-center text-h1">Questions</h2>
          <AccordionItem title="Do I need to know Python?">No. Every step is a visual choice. If you do know Python, NoCodeML saves you from re-writing the same pipeline every time you want to test an idea.</AccordionItem>
          <AccordionItem title="Which models can I compare?">Logistic Regression, KNN, Decision Tree, Random Forest and SVM for classification; Linear, Ridge, Lasso, Random Forest and Gradient Boosting for regression. Up to five per run.</AccordionItem>
          <AccordionItem title="Can I take the result back to my own code?">Yes. Download the trained pipeline (preprocessing and model together), its configuration and metrics.</AccordionItem>
          <AccordionItem title="Are the results trustworthy?">Preprocessing is fitted on training data only, every model is compared against a trivial baseline, and warnings flag leakage, imbalance and overfitting.</AccordionItem>
          <AccordionItem title="Where does my data go?">Into private storage that only your account can read. Training on your own machine is on the roadmap.</AccordionItem>
        </section>

        <section className="signature-gradient flex flex-col items-center px-4 py-28 text-center text-on-light">
          <h2 className="max-w-[600px] text-[clamp(2.2rem,5vw,3.4rem)] leading-[1.05] font-medium tracking-[-0.03em]">Spend your time on the research, not the boilerplate</h2>
          <p className="mt-4 text-[18px]">Upload a CSV and run your first comparison in minutes.</p>
          <Link href="/login" className="mt-8 inline-flex h-12 items-center rounded-full bg-on-light px-6 text-[17px] font-medium text-fg transition-transform hover:scale-[1.02]">Start for free</Link>
        </section>
      </main>
      <footer className="bg-bg-deep px-4 py-16">
        <div className="mx-auto flex max-w-[1180px] flex-col justify-between gap-10 sm:flex-row">
          <div>
            <Logo />
            <p className="mt-3 max-w-[300px] text-[15px] text-fg-muted">Run, compare and tweak ML experiments without writing code.</p>
            <p className="mt-6 text-[14px] text-fg-subtle">© 2026</p>
          </div>
          <div className="flex gap-16 text-[15px]">
            <div className="flex flex-col gap-2.5"><span className="text-fg">Product</span><a className="text-fg-muted hover:text-fg" href="#why">Why NoCodeML</a><a className="text-fg-muted hover:text-fg" href="#how">How it works</a></div>
            <div className="flex flex-col gap-2.5"><span className="text-fg">Project</span><a className="text-fg-muted hover:text-fg" href="#faq">FAQ</a><Link className="text-fg-muted hover:text-fg" href="/design">Design system</Link></div>
          </div>
        </div>
      </footer>
    </>
  );
}
