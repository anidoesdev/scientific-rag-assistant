"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useAuth } from "../contexts/AuthContext";

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: object) => void;
          prompt: () => void;
        };
      };
    };
  }
}

/* ── Icons ──────────────────────────────────────────────────────────────── */

const AtomIcon = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <circle cx="12" cy="12" r="2" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    <path d="M2 12h20" />
  </svg>
);

const GoogleIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
  </svg>
);

/* ── Page ───────────────────────────────────────────────────────────────── */

export default function LoginPage() {
  const { login, user, isLoading } = useAuth();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const [gsiReady, setGsiReady] = useState(false);

  useEffect(() => {
    if (!isLoading && user) router.replace("/chat");
  }, [isLoading, user, router]);

  useEffect(() => {
    if (isLoading || user) return;

    const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
    if (!clientId) {
      setError("NEXT_PUBLIC_GOOGLE_CLIENT_ID is not configured.");
      return;
    }

    const handleCredential = async (response: { credential: string }) => {
      setSigningIn(true);
      setError(null);
      try {
        await login(response.credential);
        router.replace("/chat");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Sign-in failed");
        setSigningIn(false);
      }
    };

    function initGSI() {
      if (!window.google) return;
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleCredential,
      });
      setGsiReady(true);
    }

    if (window.google) {
      initGSI();
    } else {
      const script = document.querySelector(
        'script[src*="accounts.google.com/gsi/client"]'
      );
      if (script) {
        script.addEventListener("load", initGSI);
        return () => script.removeEventListener("load", initGSI);
      }
    }
  }, [isLoading, user, login, router]);

  function handleSignIn() {
    if (!window.google) return;
    window.google.accounts.id.prompt();
  }

  if (isLoading) return null;

  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-4">

      {/* Anime background */}
      <Image src="/img_bg.jpg" alt="" fill priority className="object-cover object-center" />
      <div className="absolute inset-0 bg-gradient-to-b from-amber-950/88 via-amber-900/80 to-amber-950/92" />

      {/* Top shimmer line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-amber-400/50 to-transparent" />

      <div className="relative z-10 w-full max-w-sm space-y-8 text-center">

        {/* Brand */}
        <div className="space-y-3">
          <div className="flex justify-center text-amber-300">
            <AtomIcon />
          </div>
          <h1 className="font-serif text-3xl font-bold tracking-tight text-white drop-shadow">
            Scientific RAG Assistant
          </h1>
          <p className="font-serif text-sm italic leading-relaxed text-white">
            Grounded answers from indexed research papers.
            <br />
            Sign in to continue.
          </p>
        </div>

        {/* Glass card */}
        <div className="rounded-2xl border border-white/15 bg-white/10 px-8 py-8 shadow-soft backdrop-blur-xl space-y-5">
          <div className="space-y-1">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-amber-200/90">
              Welcome, Researcher
            </p>
            <p className="font-serif text-sm text-white">
              Ask anything. Every answer is cited.
            </p>
          </div>

          <div className="h-px bg-white/10" />

          {signingIn ? (
            <div className="flex items-center justify-center gap-2.5 py-1 text-sm text-amber-100/85">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-amber-300 border-t-transparent" />
              <span className="font-serif italic">Signing in…</span>
            </div>
          ) : (
            <button
              onClick={handleSignIn}
              disabled={!gsiReady}
              className="group flex w-full items-center gap-3.5 rounded-xl border border-white/20 bg-white/12 px-4 py-3 backdrop-blur-sm transition-all hover:border-white/35 hover:bg-white/18 active:scale-[0.985] disabled:cursor-wait disabled:opacity-50"
            >
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm">
                <GoogleIcon />
              </span>
              <span className="flex-1 text-left">
                <span className="block font-serif text-[13.5px] italic text-white transition-colors">
                  Continue with Google
                </span>
              </span>
              <span className="shrink-0 text-white/35 transition-all group-hover:translate-x-0.5 group-hover:text-white/65">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </span>
            </button>
          )}

          {error && (
            <p className="rounded-lg border border-red-300/30 bg-red-900/30 px-4 py-2.5 text-xs text-red-300">
              {error}
            </p>
          )}
        </div>

        <p className="text-[11px] text-amber-200/65">
          Access is limited to signed-in users only.
        </p>
      </div>
    </div>
  );
}
