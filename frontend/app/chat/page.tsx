"use client";

import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { UploadModal } from "../components/UploadModal";
import { useAuth } from "../contexts/AuthContext";

/* -- Types---------------------------------------------------------------- */

type Citation = {
  source_number: number;
  chunk_id: string;
  paper_id: string;
  file_name?: string | null;
  preview: string;
};

type AskResponse = {
  answer: string;
  unsupported: boolean;
  citations: Citation[];
  from_cache?: boolean;
  request_id?: string | null;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  meta?: AskResponse;
};

type Paper = {
  paper_id: string;
  file_name: string;
  is_session_upload: boolean;
};

/* -- Constants------------------------------------------------------------ */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/* -- Helpers-------------------------------------------------------------- */

function genId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

const SUGGESTED_QUESTIONS = [
  "What are sparse autoencoders used for in mechanistic interpretability?",
  "How do transformer attention mechanisms handle long-range dependencies?",
  "What are unidirectional error-correcting codes?",
  "How does retrieval-augmented generation improve factual accuracy?",
];

/* -- Helpers-------------------------------------------------------------- */

function prettifyTopic(raw: string): string {
  return raw
    .replace(/\.(pdf|txt|md)$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

/* -- Icons---------------------------------------------------------------- */

const AtomIcon = ({ size = 16 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <circle cx="12" cy="12" r="2" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    <path d="M2 12h20" />
  </svg>
);

const BookIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </svg>
);

const UploadIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const BoltIcon = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
  </svg>
);

const WarnIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4M12 17h.01" />
  </svg>
);

const MenuIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="3" y1="6" x2="21" y2="6" />
    <line x1="3" y1="12" x2="21" y2="12" />
    <line x1="3" y1="18" x2="21" y2="18" />
  </svg>
);

const PenIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
  </svg>
);

/* -- CitationFootnote------------------------------------------------------ */

function CitationFootnote({ c, msgId }: { c: Citation; msgId: string }) {
  const [expanded, setExpanded] = useState(false);
  const label = prettifyTopic(c.file_name || c.paper_id);

  return (
    <div key={`${msgId}-${c.chunk_id}`} className="flex gap-3 group">
      <span className="font-serif shrink-0 text-xs font-bold text-accent mt-0.5">
        [{c.source_number}]
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-text/80 truncate">{label}</p>
        <p
          className={`mt-0.5 text-xs leading-relaxed text-muted/80 transition-all ${
            expanded ? "" : "line-clamp-2"
          }`}
        >
          {c.preview}
        </p>
        {c.preview.length > 120 && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-0.5 text-[10px] text-accent/50 hover:text-accent transition-colors"
          >
            {expanded ? "collapse" : "expand"}
          </button>
        )}
      </div>
    </div>
  );
}

/* -- UserQuery------------------------------------------------------------ */

function UserQuery({ msg }: { msg: Message }) {
  return (
    <div className="msg-enter flex justify-end gap-2.5 py-2">
      <div className="max-w-[78%]">
        <p className="mb-1.5 text-right text-[10px] uppercase tracking-[0.15em] text-muted/65">Query</p>
        <div className="rounded-2xl rounded-tr-sm border border-accent/15 bg-accent/8 px-4 py-3">
          <p className="font-serif text-[15px] italic leading-relaxed text-text/85">
            &ldquo;{msg.text}&rdquo;
          </p>
        </div>
      </div>
      <div className="mt-6 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-accent/25 bg-accent/10 text-[10px] font-bold text-accent">
        Q
      </div>
    </div>
  );
}

/* -- AIResponse--------------------------------------------------------─-- */

function AIResponse({ msg }: { msg: Message }) {
  const citations = msg.meta?.citations ?? [];
  const unsupported = msg.meta?.unsupported ?? false;
  const fromCache = msg.meta?.from_cache ?? false;

  return (
    <div className="msg-enter space-y-4 py-2">
      {/* Decorative rule */}
      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-stone-200" />
        <div className="text-accent/40">
          <AtomIcon size={11} />
        </div>
        <div className="h-px flex-1 bg-stone-200" />
      </div>

      {/* Unsupported warning */}
      {unsupported && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <WarnIcon />
          <span className="font-serif italic">
            The indexed papers do not contain sufficient evidence to answer this question.
          </span>
        </div>
      )}

      {/* Answer body — serif, generous line height */}
      <div className="font-serif text-[15.5px] leading-[1.8] text-text/90 whitespace-pre-wrap">
        {msg.text}
      </div>

      {/* Cache badge */}
      {fromCache && (
        <div className="flex items-center gap-1.5 text-[10px] text-muted/70">
          <BoltIcon />
          <span className="uppercase tracking-widest">Cached response</span>
        </div>
      )}

      {/* Footnotes */}
      {citations.length > 0 && (
        <div className="border-t border-stone-200 pt-4 space-y-3">
          <p className="text-[10px] uppercase tracking-[0.18em] text-muted/70">
            References
          </p>
          <div className="space-y-3">
            {citations.map((c) => (
              <CitationFootnote
                key={`${msg.id}-${c.chunk_id}`}
                c={c}
                msgId={msg.id}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* -- Page--------------------------------------------------------------─-- */

export default function Page() {
  const { user, token, logout } = useAuth();
  const router = useRouter();

  const [question, setQuestion] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  const [paperCount, setPaperCount] = useState<number | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  const isInitial = messages.length === 0 && !loading;
  const sessionPapers = papers.filter((p) => p.is_session_upload);
  const indexedPapers = papers.filter((p) => !p.is_session_upload);

  const authHeaders: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

  // Load papers for everyone on mount
  useEffect(() => {
    refreshPaperCount();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh paper list when auth state changes
  useEffect(() => {
    refreshPaperCount();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  // Close profile dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (profileMenuRef.current && !profileMenuRef.current.contains(e.target as Node)) {
        setShowProfileMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  async function refreshPaperCount() {
    try {
      const res = await fetch(`${API_BASE}/api/papers`, { headers: authHeaders });
      if (res.ok) {
        const data: Paper[] = await res.json();
        setPapers(data);
        setPaperCount(data.length);
      }
    } catch { /* backend may not be running yet */ }
  }

  const canSend = useMemo(
    () => question.trim().length > 0 && !loading,
    [question, loading]
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function autoResize() {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }

  async function onSubmit(e?: FormEvent) {
    e?.preventDefault();
    if (!canSend) return;

    const userMessage: Message = { id: genId(), role: "user", text: question.trim() };
    setMessages((prev) => [...prev, userMessage]);
    const q = question.trim();
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setLoading(true);
    setSidebarOpen(false);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, k }),
      });
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = (await res.json()) as AskResponse;
      setMessages((prev) => [
        ...prev,
        { id: genId(), role: "assistant", text: data.answer, meta: data },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { id: genId(), role: "assistant", text: `Could not reach the backend: ${msg}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      onSubmit();
    }
  }

  const useSuggestion = useCallback((s: string) => {
    setQuestion(s);
    setTimeout(() => { textareaRef.current?.focus(); autoResize(); }, 0);
  }, []);

  /* -- Sidebar paper row-------------------------------------------------- */
  function SidebarPaperRow({ p }: { p: Paper }) {
    const label = prettifyTopic(p.file_name || p.paper_id);
    const [deleting, setDeleting] = useState(false);
    const [loadingPdf, setLoadingPdf] = useState(false);

    async function handleViewPdf(e: React.MouseEvent) {
      e.stopPropagation();
      setLoadingPdf(true);
      try {
        const res = await fetch(`${API_BASE}/api/papers/${p.paper_id}/pdf`, { headers: authHeaders });
        if (!res.ok) return;
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        window.open(url, "_blank");
      } finally {
        setLoadingPdf(false);
      }
    }

    async function handleDelete(e: React.MouseEvent) {
      e.stopPropagation();
      if (!window.confirm(`Remove "${label}" from the library?`)) return;
      setDeleting(true);
      try {
        await fetch(`${API_BASE}/api/papers/${p.paper_id}`, { method: "DELETE", headers: authHeaders });
        await refreshPaperCount();
      } finally {
        setDeleting(false);
      }
    }

    return (
      <div className={`group flex w-full items-start gap-2.5 rounded-lg px-3 py-2.5 transition-colors hover:bg-accent/8 ${
        p.is_session_upload ? "text-accent/80" : "text-text/70"
      }`}>
        <button
          onClick={() => { useSuggestion(`What are the key findings in "${label}"?`); setSidebarOpen(false); }}
          className="flex flex-1 items-start gap-2.5 text-left min-w-0"
        >
          <span className={`mt-0.5 shrink-0 ${p.is_session_upload ? "text-accent/50" : "text-muted/40"} group-hover:text-accent/60 transition-colors`}>
            <BookIcon />
          </span>
          <span className="text-xs leading-snug">{label}</span>
        </button>
        {user && <button
          onClick={handleViewPdf}
          disabled={loadingPdf}
          title="View PDF"
          className="shrink-0 mt-0.5 hidden group-hover:flex items-center justify-center w-4 h-4 rounded text-muted/40 hover:text-accent hover:bg-accent/10 transition-colors disabled:opacity-40"
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>}
        {user && <button
          onClick={handleDelete}
          disabled={deleting}
          title="Remove paper"
          className="shrink-0 mt-0.5 hidden group-hover:flex items-center justify-center w-4 h-4 rounded text-muted/40 hover:text-red-400 hover:bg-red-50 transition-colors disabled:opacity-40"
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>}
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">

      {/* ══ Header — journal masthead ════════════════════════════════════════ */}
      <header className="shrink-0 border-b border-stone-200/80 bg-bg/95 backdrop-blur-sm">
        {/* Top amber rule */}
        <div className="h-px bg-gradient-to-r from-transparent via-accent/55 to-transparent" />

        <div className="flex items-center justify-between px-5 py-3.5">
          {/* Left: hamburger (mobile) + brand */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen((v) => !v)}
              className="flex h-8 w-8 items-center justify-center rounded-md text-muted transition-colors hover:bg-stone-100 hover:text-text md:hidden"
            >
              <MenuIcon />
            </button>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-accent"><AtomIcon size={16} /></span>
                <h1 className="font-serif text-base font-bold tracking-tight text-text">
                  Scientific RAG Assistant
                </h1>
              </div>
              <p className="pl-6 text-[11px] italic text-muted/75">
                Grounded answers from indexed research
              </p>
            </div>
          </div>

          {/* Right: controls */}
          <div className="flex items-center gap-2.5">
            <Link
              href="/"
              className="hidden items-center gap-1.5 rounded-full border border-stone-200 px-3 py-1.5 text-xs text-muted/60 transition-colors hover:border-accent/30 hover:text-accent sm:flex"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 12H5M5 12l7 7M5 12l7-7" />
              </svg>
              Home
            </Link>
            <div className="hidden items-center gap-2 rounded-full border border-stone-200 bg-panel px-3 py-1.5 text-xs md:flex">
              <span className="uppercase tracking-widest text-muted/50" style={{ fontSize: "9px" }}>Top-K</span>
              <input
                type="range" min={1} max={10} value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="h-1 w-16 accent-accent"
              />
              <span className="w-3 text-center font-mono text-accent">{k}</span>
            </div>
            <button
              onClick={() => { if (!user) { router.push("/login"); return; } setShowUpload(true); }}
              className="flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-4 py-1.5 text-xs font-semibold text-accent transition-colors hover:bg-accent/18"
            >
              <UploadIcon />
              Upload Paper
            </button>
            {user ? (
              <div ref={profileMenuRef} className="relative">
                <button
                  onClick={() => setShowProfileMenu(v => !v)}
                  className="h-7 w-7 shrink-0 overflow-hidden rounded-full border border-stone-200 transition-opacity hover:opacity-75"
                >
                  {user.picture ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={user.picture} alt={user.name} className="h-full w-full object-cover" />
                  ) : (
                    <span className="flex h-full w-full items-center justify-center bg-stone-100 text-xs font-medium text-stone-500">
                      {(user.name || user.email).charAt(0).toUpperCase()}
                    </span>
                  )}
                </button>
                {showProfileMenu && (
                  <div className="absolute right-0 top-9 z-50 min-w-[180px] rounded-xl border border-stone-200 bg-white py-1 shadow-lg">
                    <div className="border-b border-stone-100 px-4 py-2.5">
                      <p className="text-xs font-semibold text-text truncate">{user.name}</p>
                      <p className="text-xs text-muted/60 truncate">{user.email}</p>
                    </div>
                    <button
                      onClick={() => { setShowProfileMenu(false); logout(); }}
                      className="flex w-full items-center gap-2 px-4 py-2 text-xs text-muted/70 transition-colors hover:bg-stone-50 hover:text-red-500"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                        <polyline points="16 17 21 12 16 7" />
                        <line x1="21" y1="12" x2="9" y2="12" />
                      </svg>
                      Sign out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <button
                onClick={() => router.push("/login")}
                className="rounded-full bg-accent px-4 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-accent/90"
              >
                Sign in
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ══ Body: sidebar + main ════════════════════════════════════════════ */}
      <div className="flex flex-1 overflow-hidden">

        {/* -- Sidebar overlay on mobile--------------------------------─-- */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-20 bg-stone-900/20 backdrop-blur-sm md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* -- Sidebar--------------------------------------------------─-- */}
        <aside
          className={`
            fixed inset-y-0 left-0 z-30 flex w-64 flex-col border-r border-stone-200 bg-panel pt-[57px] transition-transform duration-200
            md:relative md:inset-auto md:z-auto md:flex md:w-60 md:shrink-0 md:translate-x-0 md:pt-0
            ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          `}
        >
          {/* Sidebar header with amber accent */}
          <div className="relative overflow-hidden border-b border-stone-200 px-5 py-4">
            <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-accent/50 to-transparent" />
            <h2 className="font-serif text-sm font-semibold text-text">Paper Library</h2>
            <p className="mt-0.5 text-[11px] text-muted/60">
              {paperCount === null ? "Loading…" : paperCount === 0 ? "No papers indexed" : `${paperCount} paper${paperCount !== 1 ? "s" : ""} indexed`}
            </p>
          </div>

          {/* Papers list */}
          <div className="flex-1 overflow-y-auto px-2 py-3">
            {sessionPapers.length > 0 && (
              <div className="mb-4">
                <p className="px-3 mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-accent/60">
                  This Session
                </p>
                {sessionPapers.map((p) => <SidebarPaperRow key={p.paper_id} p={p} />)}
              </div>
            )}

            {indexedPapers.length > 0 && (
              <div>
                <p className="px-3 mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-muted/65">
                  Indexed
                </p>
                {indexedPapers.map((p) => <SidebarPaperRow key={p.paper_id} p={p} />)}
              </div>
            )}

            {papers.length === 0 && (
              <div className="px-3 py-8 text-center">
                <p className="font-serif text-xs italic text-muted/70">No papers indexed yet.</p>
                <p className="mt-1 text-[11px] text-muted/65">Upload a PDF to get started.</p>
              </div>
            )}
          </div>

          {/* Upload button */}
          <div className="border-t border-stone-200 p-4">
            <button
              onClick={() => { if (!user) { setSidebarOpen(false); router.push("/login"); return; } setShowUpload(true); setSidebarOpen(false); }}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-accent/30 py-2.5 text-xs font-medium text-accent/80 transition-colors hover:border-accent/50 hover:bg-accent/5 hover:text-accent"
            >
              <UploadIcon />
              Upload a paper
            </button>
          </div>
        </aside>

        {/* -- Main content------------------------------------------------ */}
        <div className="flex flex-1 flex-col overflow-hidden">

          {/* Scrollable content */}
          <main className="flex-1 overflow-y-auto">
            <div className="mx-auto max-w-2xl px-6 py-10 space-y-8">

              {/* -- Empty / welcome state----------------------------─-- */}
              {isInitial && (
                <div className="space-y-8">
                  {/* Anime mini-banner */}
                  <div className="relative -mx-6 -mt-10 h-44 overflow-hidden">
                    <Image src="/img_bg.jpg" alt="" fill className="object-cover object-[center_18%]" />
                    <div className="absolute inset-0 bg-gradient-to-b from-amber-950/55 via-amber-900/35 to-bg" />
                    <div className="relative z-10 flex h-full flex-col justify-end px-6 pb-5">
                      <p className="text-[10px] font-bold uppercase tracking-[0.28em] text-white drop-shadow">
                        Research Assistant · Papyrus
                      </p>
                    </div>
                  </div>

                  {/* Masthead block */}
                  <div className="space-y-3 border-b-2 border-stone-200 pb-8">
                    <h2 className="font-serif text-4xl font-bold leading-tight text-text">
                      Ask a scientific<br />question.
                    </h2>
                    <p className="max-w-md font-serif text-[15px] italic leading-relaxed text-muted/80">
                      Every answer is grounded in indexed research papers.
                      All claims are traceable to a specific source.
                    </p>
                  </div>

                  {/* Suggested queries */}
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted/70">
                      Try asking
                    </p>
                    <div className="grid gap-2.5 sm:grid-cols-2">
                      {SUGGESTED_QUESTIONS.map((s) => (
                        <button
                          key={s}
                          onClick={() => useSuggestion(s)}
                          className="group relative overflow-hidden rounded-xl border border-stone-200 bg-panel/50 px-4 py-3.5 text-left transition-all hover:border-accent/35 hover:shadow-soft"
                        >
                          <div className="absolute inset-0 bg-gradient-to-br from-amber-50/0 to-amber-50/0 transition-all group-hover:from-amber-50/80 group-hover:to-amber-100/40" />
                          <span className="relative font-serif text-sm italic leading-snug text-text/80 transition-colors group-hover:text-text">
                            &ldquo;{s}&rdquo;
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Low-paper notice */}
                  {paperCount !== null && paperCount < 3 && (
                    <div className="flex items-start gap-3 rounded-xl border border-amber-200/70 bg-amber-50/60 px-5 py-4">
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                        strokeWidth="2" strokeLinecap="round" className="mt-0.5 shrink-0 text-accent/70">
                        <circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" />
                      </svg>
                      <p className="font-serif text-sm leading-relaxed text-amber-900/80">
                        <span className="font-semibold not-italic">
                          {paperCount === 0 ? "No papers indexed yet." : `Only ${paperCount} paper${paperCount !== 1 ? "s" : ""} indexed.`}
                        </span>{" "}
                        Upload at least <strong>3–5 papers</strong> for reliable answers, or <strong>8–12</strong> for broader comparisons.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* -- Message thread-------------------------------------- */}
              {messages.map((msg) =>
                msg.role === "user"
                  ? <UserQuery key={msg.id} msg={msg} />
                  : <AIResponse key={msg.id} msg={msg} />
              )}

              {/* -- Loading------------------------------------------─-- */}
              {loading && (
                <div className="msg-enter space-y-4 py-2">
                  <div className="flex items-center gap-3">
                    <div className="h-px flex-1 bg-stone-200" />
                    <div className="text-accent/40"><AtomIcon size={11} /></div>
                    <div className="h-px flex-1 bg-stone-200" />
                  </div>
                  <div className="flex items-center gap-2 pl-1">
                    <span className="dot-bounce h-1.5 w-1.5 rounded-full bg-muted/40" style={{ animationDelay: "0ms" }} />
                    <span className="dot-bounce h-1.5 w-1.5 rounded-full bg-muted/40" style={{ animationDelay: "160ms" }} />
                    <span className="dot-bounce h-1.5 w-1.5 rounded-full bg-muted/40" style={{ animationDelay: "320ms" }} />
                    <span className="font-serif text-xs italic text-muted/70 ml-1">Searching papers…</span>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </main>

          {/* -- Compose bar--------------------------------------------─-- */}
          <div className="shrink-0 border-t border-stone-200/80 bg-bg/95 px-6 py-4 backdrop-blur-sm">
            <form onSubmit={onSubmit} className="mx-auto max-w-2xl">
              <div className="flex items-end gap-3 rounded-2xl border border-stone-200 bg-white px-5 py-3.5 shadow-soft transition-all focus-within:border-accent/40 focus-within:shadow-glow">
                <span className="mb-1 shrink-0 text-accent/30">
                  <PenIcon />
                </span>
                <textarea
                  ref={textareaRef}
                  rows={1}
                  value={question}
                  onChange={(e) => { setQuestion(e.target.value); autoResize(); }}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a scientific question…"
                  className="max-h-40 flex-1 resize-none bg-transparent font-serif text-[14.5px] italic leading-relaxed text-text outline-none placeholder:not-italic placeholder:font-sans placeholder:text-sm placeholder:text-muted/35"
                />
                <button
                  type="submit"
                  disabled={!canSend}
                  className="mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent text-white shadow-glow transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-20 disabled:shadow-none"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M12 19V5M5 12l7-7 7 7" />
                  </svg>
                </button>
              </div>
              <p className="mt-2 text-center text-[10px] text-muted/55">
                Ctrl + Enter to send · answers grounded in indexed papers only
              </p>
            </form>
          </div>

        </div>
      </div>

      {/* -- Upload modal------------------------------------------------─-- */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onDone={() => { refreshPaperCount(); }}
        />
      )}
    </div>
  );
}
