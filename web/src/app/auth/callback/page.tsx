"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";

/**
 * Google (via Supabase) redirects here with ?code=. The Supabase client exchanges it for a
 * session on load (PKCE, detectSessionInUrl); we wait for that, then go to the projects page.
 */
function Callback() {
  const router = useRouter();
  const params = useSearchParams();
  const providerError = params.get("error_description") || params.get("error");
  const [exchangeError, setError] = useState<string | null>(null);
  const error = providerError ?? exchangeError;

  useEffect(() => {
    if (providerError) return;
    const sb = supabase();
    let done = false;
    const finish = () => { if (!done) { done = true; router.replace("/projects"); } };
    const { data } = sb.auth.onAuthStateChange((event, session) => {
      if (session && (event === "SIGNED_IN" || event === "INITIAL_SESSION")) finish();
    });
    sb.auth.getSession().then(({ data: d, error: e }) => {
      if (e) setError(e.message);
      else if (d.session) finish();
    });
    const timeout = setTimeout(() => { if (!done) setError("Sign-in took too long. Please try again."); }, 15000);
    return () => { data.subscription.unsubscribe(); clearTimeout(timeout); };
  }, [router, providerError]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-4 text-center">
      {error ? (
        <>
          <p className="text-[17px]">Sign-in didn&apos;t complete</p>
          <p className="max-w-sm text-[14px] text-fg-muted">{error}</p>
          <Link href="/login" className="mt-2 inline-flex h-10 items-center rounded-full bg-fg px-5 text-[15px] font-medium text-on-light">Back to sign in</Link>
        </>
      ) : (
        <p className="text-[15px] text-fg-muted">Signing you in…</p>
      )}
    </main>
  );
}

export default function AuthCallback() {
  return (
    <Suspense fallback={<main className="flex min-h-screen items-center justify-center text-[15px] text-fg-muted">Signing you in…</main>}>
      <Callback />
    </Suspense>
  );
}
