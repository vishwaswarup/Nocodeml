import type { Metadata } from "next";
import { Bullets, ContactLink, LegalPage, LegalSection } from "@/components/legal";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Privacy Policy · NoCodeML",
  description: "What NoCodeML collects, why, who can see it, and how to have it deleted.",
};

export default function PrivacyPage() {
  return (
    <LegalPage title="Privacy Policy"
      intro={`${SITE.product} is a tool for running and comparing machine-learning experiments in your browser. This page explains, in plain language, what information the service handles and what happens to it. It is operated by ${SITE.operator} (“we”, “us”).`}>
      <LegalSection title="What we collect">
        <Bullets items={[
          <><strong className="font-medium text-fg">Your account.</strong> When you sign in with Google we receive your email address, name and profile picture from Google. If you sign in with email we store your email address and a hashed password. Nothing else is requested from your Google account.</>,
          <><strong className="font-medium text-fg">Your data and work.</strong> The files you upload (CSV, Excel or Parquet datasets), and what you create from them: pipeline settings, experiment results, trained models, charts and PDF reports.</>,
          <><strong className="font-medium text-fg">Technical logs.</strong> Our servers log each request (time, page or endpoint, result, your IP address) so we can keep the service running, find faults and stop abuse. Logs do not contain your data or sign-in tokens.</>,
        ]} />
        <p>We do not use advertising or analytics trackers, and we do not set tracking cookies. Your browser stores a sign-in session so you stay signed in; that is the only thing kept on your device.</p>
      </LegalSection>

      <LegalSection title="How we use it">
        <Bullets items={[
          "To run the service: profile your dataset, train the models you ask for, show results, and produce downloads and reports.",
          "To keep your account secure and the service reliable (limits on request rates, error diagnosis).",
          "To contact you about the service if we need to, for example a security problem or a change to these terms.",
        ]} />
        <p>We do not sell your data or share it for advertising. We do not use your datasets or results to train any models of our own, and nobody reads your data except as needed to fix a fault you ask us to look at.</p>
      </LegalSection>

      <LegalSection title="Who can see your data">
        <p>Only you. Each user&apos;s projects, datasets and results are stored in private storage that is locked to their own account by the database itself, not just by our application. Other users cannot list or open your files.</p>
        <p>We rely on a small number of service providers to operate NoCodeML, and they process data on our behalf:</p>
        <Bullets items={[
          "Supabase: database, file storage and sign-in.",
          "Google: sign-in with Google.",
          "Our hosting providers for the website and the server that trains your models.",
        ]} />
        <p>If you choose to train on Google Colab, you download your prepared data from NoCodeML and upload it to your own Colab session, which Google runs under your Google account and its own terms. Only the predictions you upload back are stored with the experiment.</p>
        <p>We may also disclose information if the law requires it.</p>
      </LegalSection>

      <LegalSection title="Keeping and deleting your data">
        <p>We keep your data for as long as your account exists. You can delete any project at any time from the Projects page: this permanently removes its datasets, pipeline versions, experiments, trained models and reports. Deleted data may remain in the provider&apos;s encrypted backups for a limited time before it is overwritten.</p>
        <p>To delete your whole account, or to ask for a copy of what we hold about you, email <ContactLink />. We will act on it within 30 days.</p>
      </LegalSection>

      <LegalSection title="Security">
        <p>Connections to NoCodeML are encrypted in transit. Datasets and results sit in private storage with access limited to their owner. No system is perfectly secure, so please do not upload data you are not allowed to process, and avoid uploading personal data that is not needed for your analysis.</p>
      </LegalSection>

      <LegalSection title="Your data is your responsibility, too">
        <p>You must have the right to upload the data you upload. If it contains personal information about other people (for example health or customer records), you are responsible for having a lawful basis to process it, and for removing identifying details where you can. NoCodeML is not designed for regulated data such as protected health information, and should not be used for it.</p>
      </LegalSection>

      <LegalSection title="Your rights">
        <p>Depending on where you live, you may have the right to access, correct, export or delete your personal data, to object to how it is used, or to complain to your local data-protection authority. Email <ContactLink /> and we will help.</p>
      </LegalSection>

      <LegalSection title="Children">
        <p>NoCodeML is not meant for anyone under 16, and we do not knowingly collect their data.</p>
      </LegalSection>

      <LegalSection title="Changes and contact">
        <p>If we change this policy in a meaningful way we will update the date at the top and, where practical, tell signed-in users. Questions: <ContactLink />.</p>
      </LegalSection>
    </LegalPage>
  );
}
