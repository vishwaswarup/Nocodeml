import type { Metadata } from "next";
import { Bullets, ContactLink, LegalPage, LegalSection } from "@/components/legal";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Terms of Use · NoCodeML",
  description: "The rules for using NoCodeML.",
};

export default function TermsPage() {
  return (
    <LegalPage title="Terms of Use"
      intro={`By creating an account or using ${SITE.product} you agree to these terms. ${SITE.product} is operated by ${SITE.operator} (“we”, “us”). If you do not agree, please do not use the service.`}>
      <LegalSection title="The service">
        <p>{SITE.product} lets you upload a dataset and run, compare and tweak machine-learning experiments in your browser. It is in <strong className="font-medium text-fg">beta</strong>: features can change, break or be removed, and we may limit usage (for example file sizes, request rates and training time) to keep it fair and stable.</p>
      </LegalSection>

      <LegalSection title="Your account">
        <p>You are responsible for your account and for what happens under it. Keep your sign-in details safe, give us accurate information, and tell us at <ContactLink /> if you think someone else has accessed your account. You must be old enough to enter into a contract where you live, and at least 16.</p>
      </LegalSection>

      <LegalSection title="Your data">
        <p>You keep all rights in the data you upload and in the results you produce. You give us permission only to store and process them as needed to run the service for you, as described in the <a className="text-fg underline underline-offset-4 hover:no-underline" href="/privacy">Privacy Policy</a>.</p>
        <p>You promise that you have the right to upload your data, and that doing so does not break the law or anyone&apos;s rights. Do not upload data you are not allowed to process, and do not use the service for regulated data such as protected health information.</p>
      </LegalSection>

      <LegalSection title="Acceptable use">
        <p>Do not:</p>
        <Bullets items={[
          "break the law, or use the service to harm, defraud or harass anyone;",
          "try to access other users' data, probe or attack the service, or get around its limits;",
          "overload the service, for example with automated bulk requests or deliberately expensive training runs;",
          "resell the service or present it as your own.",
        ]} />
        <p>We may suspend or remove accounts that break these rules, or that put the service or other users at risk.</p>
      </LegalSection>

      <LegalSection title="Results are not advice">
        <p>Machine-learning models can be wrong, biased or misleading, especially on small or unrepresentative data. {SITE.product} shows warnings and quality checks to help, but they cannot catch everything. Treat results as input to your own judgement. Do not rely on them alone for decisions about health, safety, legal rights, employment, credit or similar matters.</p>
      </LegalSection>

      <LegalSection title="No warranty">
        <p>The service is provided “as is” and “as available”, without promises that it will be uninterrupted, error-free or that data will never be lost. Please keep your own copy of anything important: download your models, reports and results, and keep your original datasets.</p>
      </LegalSection>

      <LegalSection title="Limit of liability">
        <p>To the extent the law allows, we are not liable for indirect or consequential loss, lost profits or lost data arising from your use of the service, and our total liability for any claim is limited to the amount you paid us in the 12 months before it (which, while the service is free, is zero). Nothing in these terms limits liability that cannot legally be limited.</p>
      </LegalSection>

      <LegalSection title="Ending your use">
        <p>You can stop at any time and delete your projects from the Projects page, or ask us to delete your account at <ContactLink />. We may suspend or end the service, or your access, for example for misuse or if we stop running the service. We will try to give notice where we reasonably can.</p>
      </LegalSection>

      <LegalSection title="Changes and contact">
        <p>We may update these terms. If a change is meaningful we will update the date at the top and, where practical, tell signed-in users; continuing to use the service after that means you accept the new terms. Questions: <ContactLink />.</p>
      </LegalSection>
    </LegalPage>
  );
}
