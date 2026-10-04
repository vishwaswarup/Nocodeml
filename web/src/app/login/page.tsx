"use client";

import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";
import { useAuth } from "@/components/auth-provider";
import { GoogleIcon } from "@/components/google-icon";
import { Logo } from "@/components/logo";
import { Button } from "@/components/ui/button";
import { Field } from "@/components/ui/field";
import { supabase } from "@/lib/supabase";

export default function LoginPage() {
  const { session, loading } = useAuth();
  const router = useRouter();
  const [busy, setBusy] = useState<"google" | "email" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showEmail, setShowEmail] = useState(false);
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  useEffect(() => {
    if (!loading && session) router.replace("/projects");
  }, [loading, session, router]);

  const google = async () => {
    setBusy("google");
    setError(null);
    const { error } = await supabase().auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
    if (error) {
      setError(error.message);
      setBusy(null);
    }
    // on success the browser is already navigating to Google
  };

  const submitEmail = async (e: FormEvent) => {
    e.preventDefault();
    setBusy("email");
    setError(null);
    setNotice(null);
    const sb = supabase().auth;
    const { data, error } = mode === "signin"
      ? await sb.signInWithPassword({ email, password })
      : await sb.signUp({ email, password, options: { emailRedirectTo: `${window.location.origin}/auth/callback` } });
    setBusy(null);
    if (error) return setError(error.message);
    if (mode === "signup" && !data.session) setNotice("Check your inbox to confirm your email, then sign in.");
    // a successful sign-in updates the session; the effect above redirects
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-4 py-16">
      <Link href="/" aria-label="NoCodeML home" className="mb-12"><Logo size={24} /></Link>
      <div className="w-full max-w-[400px]">
        <h1 className="text-center text-[32px] leading-tight font-medium tracking-[-0.03em]">Welcome to NoCodeML</h1>
        <p className="mt-2 text-center text-[16px] text-fg-muted">Sign in to run and compare your experiments.</p>

        <div className="mt-10 flex flex-col gap-3">
          <Button size="lg" onClick={google} loading={busy === "google"} disabled={busy !== null}
            icon={<GoogleIcon />} className="w-full">
            Continue with Google
          </Button>

          {!showEmail && (
            <Button variant="ghost" size="md" onClick={() => setShowEmail(true)} className="w-full">
              Use email instead
            </Button>
          )}

          <AnimatePresence initial={false}>
            {showEmail && (
              <motion.form
                onSubmit={submitEmail}
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden"
              >
                <div className="my-4 flex items-center gap-3 text-[12px] text-fg-subtle">
                  <span className="h-px flex-1 bg-line" /> or <span className="h-px flex-1 bg-line" />
                </div>
                <div className="flex flex-col gap-3">
                  <Field label="Email" type="email" autoComplete="email" required value={email}
                    onChange={(e) => setEmail(e.target.value)} placeholder="you@university.edu" />
                  <Field label="Password" type="password" required minLength={6} value={password}
                    autoComplete={mode === "signin" ? "current-password" : "new-password"}
                    onChange={(e) => setPassword(e.target.value)} />
                  <Button type="submit" variant="secondary" loading={busy === "email"} disabled={busy !== null} className="mt-1 w-full">
                    {mode === "signin" ? "Sign in" : "Create account"}
                  </Button>
                  <button type="button" onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setError(null); setNotice(null); }}
                    className="text-[13px] text-fg-muted hover:text-fg">
                    {mode === "signin" ? "No account? Create one" : "Have an account? Sign in"}
                  </button>
                </div>
              </motion.form>
            )}
          </AnimatePresence>

          {error && <p role="alert" className="rounded-control bg-fail/10 px-4 py-3 text-[14px] text-fail">{error}</p>}
          {notice && <p role="status" className="rounded-control bg-pass/10 px-4 py-3 text-[14px] text-pass">{notice}</p>}
        </div>

        <p className="mt-10 text-center text-[13px] leading-relaxed text-fg-subtle">
          Your datasets stay private to your account.
        </p>
      </div>
    </main>
  );
}
