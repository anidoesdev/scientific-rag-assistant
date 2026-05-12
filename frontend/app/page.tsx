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
import { UploadModal } from "./components/UploadModal";

/* ── Types ──────────────────────────────────────────────────────────────── */

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

/* ── Constants ──────────────────────────────────────────────────────────── */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SUGGESTED_QUESTIONS = [
  "What are sparse autoencoders used for in mechanistic interpretability?",
  "How do transformer attention mechanisms handle long-range dependencies?",
  "What are unidirectional error-correcting codes?",
  "How does retrieval-augmented generation improve factual accuracy?",
];

/* ── Helpers ────────────────────────────────────────────────────────────── */

function prettifyTopic(raw: string): string {
  return raw
    .replace(/\.(pdf|txt|md)$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

/* ── Icons ──────────────────────────────────────────────────────────────── */

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

/* ── CitationFootnote ────────────────────────────────────────────────────── */

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

/* ── UserQuery ──────────────────────────────────────────────────────────── */

function UserQuery({ msg }: { msg: Message }) {
  return (
    <div className="msg-enter flex justify-end gap-3 py-2">
      <div className="max-w-[78%] text-right">
        <p className="mb-1 text-[10px] uppercase tracking-[0.15em] text-muted/50">Query</p>
        <p className="font-serif text-[15px] italic leading-relaxed text-text/80">
          &ldquo;{msg.text}&rdquo;
        </p>
      </div>
      <div className="mt-5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-accent/25 bg-accent/8 text-[10px] font-bold text-accent">
        Q
      </div>
    </div>
  );
}

/* ── AIResponse ─────────────────────────────────────────────────────────── */

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
        <div className="flex items-center gap-1.5 text-[10px] text-muted/50">
          <BoltIcon />
          <span className="uppercase tracking-widest">Cached response</span>
        </div>
      )}

      {/* Footnotes */}
      {citations.length > 0 && (
        <div className="border-t border-stone-200 pt-4 space-y-3">
          <p className="text-[10px] uppercase tracking-[0.18em] text-muted/50">
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

/* ── Page ───────────────────────────────────────────────────────────────── */

export default function Page() {
  const [question, setQuestion] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  const [paperCount, setPaperCount] = useState<number | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isInitial = messages.length === 0 && !loading;
  const sessionPapers = papers.filter((p) => p.is_session_upload);
  const indexedPapers = papers.filter((p) => !p.is_session_upload);

  useEffect(() => {
    async function init() {
      try {
        await fetch(`${API_BASE}/api/uploads/cleanup`, { method: "DELETE" });
      } catch { /* backend may not be running yet */ }
      refreshPaperCount();
    }
    init();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshPaperCount() {
    try {
      const res = await fetch(`${API_BASE}/api/papers`);
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

    const userMessage: Message = { id: crypto.randomUUID(), role: "user", text: question.trim() };
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
        { id: crypto.randomUUID(), role: "assistant", text: data.answer, meta: data },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", text: `Could not reach the backend: ${msg}` },
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

  /* ── Sidebar paper row ────────────────────────────────────────────────── */
  function SidebarPaperRow({ p }: { p: Paper }) {
    const label = prettifyTopic(p.file_name || p.paper_id);
    return (
      <button
        onClick={() => { useSuggestion(`What are the key findings in "${label}"?`); setSidebarOpen(false); }}
        className={`group flex w-full items-start gap-2.5 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-accent/8 ${
          p.is_session_upload ? "text-accent/80" : "text-text/70"
        }`}
      >
        <span className={`mt-0.5 shrink-0 ${p.is_session_upload ? "text-accent/50" : "text-muted/40"} group-hover:text-accent/60 transition-colors`}>
          <BookIcon />
        </span>
        <span className="text-xs leading-snug">{label}</span>
      </button>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">

      {/* ══ Header — journal masthead ════════════════════════════════════════ */}
      <header className="shrink-0 border-b border-stone-200 bg-bg">
        {/* Top amber rule */}
        <div className="h-0.5 bg-gradient-to-r from-transparent via-accent/60 to-transparent" />

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
              <p className="pl-6 text-[11px] italic text-muted/60">
                Grounded answers from indexed research
              </p>
            </div>
          </div>

          {/* Right: controls */}
          <div className="flex items-center gap-2.5">
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
              onClick={() => setShowUpload(true)}
              className="flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-4 py-1.5 text-xs font-semibold text-accent transition-colors hover:bg-accent/18"
            >
              <UploadIcon />
              Upload Paper
            </button>
          </div>
        </div>
      </header>

      {/* ══ Body: sidebar + main ════════════════════════════════════════════ */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Sidebar overlay on mobile ─────────────────────────────────── */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-20 bg-stone-900/20 backdrop-blur-sm md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* ── Sidebar ───────────────────────────────────────────────────── */}
        <aside
          className={`
            fixed inset-y-0 left-0 z-30 flex w-64 flex-col border-r border-stone-200 bg-panel pt-[57px] transition-transform duration-200
            md:relative md:inset-auto md:z-auto md:flex md:w-60 md:shrink-0 md:translate-x-0 md:pt-0
            ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          `}
        >
          {/* Sidebar header */}
          <div className="border-b border-stone-200 px-5 py-4">
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
                <p className="px-3 mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-muted/45">
                  Indexed
                </p>
                {indexedPapers.map((p) => <SidebarPaperRow key={p.paper_id} p={p} />)}
              </div>
            )}

            {papers.length === 0 && (
              <div className="px-3 py-8 text-center">
                <p className="font-serif text-xs italic text-muted/50">No papers indexed yet.</p>
                <p className="mt-1 text-[11px] text-muted/40">Upload a PDF to get started.</p>
              </div>
            )}
          </div>

          {/* Upload button */}
          <div className="border-t border-stone-200 p-4">
            <button
              onClick={() => { setShowUpload(true); setSidebarOpen(false); }}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-accent/30 py-2.5 text-xs font-medium text-accent/80 transition-colors hover:border-accent/50 hover:bg-accent/5 hover:text-accent"
            >
              <UploadIcon />
              Upload a paper
            </button>
          </div>
        </aside>

        {/* ── Main content ──────────────────────────────────────────────── */}
        <div className="flex flex-1 flex-col overflow-hidden">

          {/* Scrollable content */}
          <main className="flex-1 overflow-y-auto">
            <div className="mx-auto max-w-2xl px-6 py-10 space-y-8">

              {/* ── Empty / welcome state ─────────────────────────────── */}
              {isInitial && (
                <div className="space-y-10">
                  {/* Masthead block */}
                  <div className="space-y-3 border-b-2 border-stone-200 pb-8">
                    <p className="text-[10px] font-bold uppercase tracking-[0.25em] text-accent/70">
                      Research Assistant
                    </p>
                    <h2 className="font-serif text-4xl font-bold leading-tight text-text">
                      Ask a scientific<br />question.
                    </h2>
                    <p className="max-w-md font-serif text-[15px] leading-relaxed text-muted/80 italic">
                      Every answer is grounded in indexed research papers.
                      All claims are traceable to a specific source.
                    </p>
                  </div>

                  {/* Suggested queries */}
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted/50">
                      Try asking
                    </p>
                    <div className="grid gap-2.5 sm:grid-cols-2">
                      {SUGGESTED_QUESTIONS.map((s) => (
                        <button
                          key={s}
                          onClick={() => useSuggestion(s)}
                          className="group rounded-xl border border-stone-200 bg-white px-4 py-3.5 text-left transition-all hover:border-accent/30 hover:bg-amber-50/60 hover:shadow-sm"
                        >
                          <span className="font-serif text-sm italic text-text/70 group-hover:text-text/90 transition-colors leading-snug">
                            &ldquo;{s}&rdquo;
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Low-paper notice */}
                  {paperCount !== null && paperCount < 3 && (
                    <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50/70 px-5 py-4">
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

              {/* ── Message thread ────────────────────────────────────── */}
              {messages.map((msg) =>
                msg.role === "user"
                  ? <UserQuery key={msg.id} msg={msg} />
                  : <AIResponse key={msg.id} msg={msg} />
              )}

              {/* ── Loading ───────────────────────────────────────────── */}
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
                    <span className="font-serif text-xs italic text-muted/50 ml-1">Searching papers…</span>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </main>

          {/* ── Compose bar ─────────────────────────────────────────────── */}
          <div className="shrink-0 border-t border-stone-200 bg-bg px-6 py-4">
            <form onSubmit={onSubmit} className="mx-auto max-w-2xl">
              <div className="flex items-end gap-3 rounded-2xl border border-stone-200 bg-white px-5 py-3.5 shadow-sm transition-all focus-within:border-accent/40 focus-within:shadow-glow">
                <span className="mb-1 shrink-0 text-muted/35">
                  <PenIcon />
                </span>
                <textarea
                  ref={textareaRef}
                  rows={1}
                  value={question}
                  onChange={(e) => { setQuestion(e.target.value); autoResize(); }}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a scientific question…"
                  className="max-h-40 flex-1 resize-none bg-transparent font-serif text-[14.5px] italic leading-relaxed text-text outline-none placeholder:not-italic placeholder:font-sans placeholder:text-sm placeholder:text-muted/40"
                />
                <button
                  type="submit"
                  disabled={!canSend}
                  className="mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent text-white transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-25"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M12 19V5M5 12l7-7 7 7" />
                  </svg>
                </button>
              </div>
              <p className="mt-2 text-center text-[10px] text-muted/35">
                Ctrl + Enter to send · answers grounded in indexed papers only
              </p>
            </form>
          </div>

        </div>
      </div>

      {/* ── Upload modal ─────────────────────────────────────────────────── */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onDone={() => { refreshPaperCount(); }}
        />
      )}
    </div>
  );
}
