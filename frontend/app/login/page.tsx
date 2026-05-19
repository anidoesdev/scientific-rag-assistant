"use client";

import { useEffect, useState } from "react";
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

const AtomIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
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
    <div className="flex min-h-screen items-center justify-center bg-bg px-4">
      <div className="w-full max-w-xs space-y-8">

        {/* Brand */}
        <div className="space-y-2 text-center">
          <div className="flex justify-center text-accent">
            <AtomIcon />
          </div>
          <h1 className="font-serif text-2xl font-bold tracking-tight text-text">
            Scientific RAG Assistant
          </h1>
          <p className="font-serif text-sm italic text-muted/70">
            Grounded answers from indexed research.
          </p>
        </div>

        {/* Sign-in card */}
        <div className="rounded-2xl border border-stone-200 bg-white px-6 py-6 shadow-soft space-y-4">
          {signingIn ? (
            <div className="flex items-center justify-center gap-2.5 py-2 text-sm text-muted/70">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              <span className="font-serif italic">Signing in…</span>
            </div>
          ) : (
            <button
              onClick={handleSignIn}
              disabled={!gsiReady}
              className="flex w-full items-center gap-3 rounded-xl border border-stone-200 bg-white px-4 py-3 text-sm font-medium text-text/80 transition-all hover:border-stone-300 hover:bg-stone-50 active:scale-[0.985] disabled:cursor-wait disabled:opacity-50"
            >
              <GoogleIcon />
              <span>Continue with Google</span>
            </button>
          )}

          {error && (
            <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-600">
              {error}
            </p>
          )}
        </div>

        <p className="text-center text-[11px] text-muted/50">
          Sign-in required to access the assistant.
        </p>
      </div>
    </div>
  );
}
