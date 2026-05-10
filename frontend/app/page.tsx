"use client";

import {
  FormEvent,
  KeyboardEvent,
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

function extractTopics(citations: Citation[] = []): string[] {
  const topics = new Set<string>();
  for (const c of citations) {
    const t = prettifyTopic(c.file_name || c.paper_id || c.chunk_id);
    if (t) topics.add(t);
    if (topics.size >= 4) break;
  }
  return Array.from(topics);
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

const ArrowUpIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
);

const WarnIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4M12 17h.01" />
  </svg>
);

const BoltIcon = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
  </svg>
);

const BookmarkIcon = () => (
  <svg width="9" height="9" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z" />
  </svg>
);

/* ── CitationCard ───────────────────────────────────────────────────────── */

function CitationCard({ c, msgId }: { c: Citation; msgId: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      key={`${msgId}-${c.chunk_id}`}
      className="rounded-xl border border-white/[0.07] bg-black/20 p-3 transition-all duration-200"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent/20 text-[10px] font-bold text-accent">
            {c.source_number}
          </span>
          <p className="truncate text-xs font-medium text-muted">
            {prettifyTopic(c.file_name || c.paper_id)}
          </p>
        </div>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="shrink-0 text-[10px] text-muted/50 transition-colors hover:text-muted"
        >
          {expanded ? "less" : "more"}
        </button>
      </div>

      <p
        className={`mt-2 text-sm leading-relaxed text-text/75 transition-all ${
          expanded ? "" : "line-clamp-2"
        }`}
      >
        {c.preview}
      </p>
    </div>
  );
}

/* ── Page ───────────────────────────────────────────────────────────────── */

export default function Page() {
  const [question, setQuestion] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages]   = useState<Message[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  const [paperCount, setPaperCount] = useState<number | null>(null);

  const bottomRef   = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isInitial = messages.length === 0 && !loading;

  /* Fetch paper count on mount and after successful upload */
  async function refreshPaperCount() {
    try {
      const res = await fetch(`${API_BASE}/api/papers`);
      if (res.ok) {
        const data = await res.json();
        setPaperCount(data.length);
      }
    } catch { /* backend may not be running yet */ }
  }

  useEffect(() => { refreshPaperCount(); }, []);

  const canSend = useMemo(
    () => question.trim().length > 0 && !loading,
    [question, loading]
  );

  /* Auto-scroll to bottom on new messages */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  /* Auto-resize textarea */
  function autoResize() {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }

  async function onSubmit(e?: FormEvent) {
    e?.preventDefault();
    if (!canSend) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: question.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const q = question.trim();
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setLoading(true);

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
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `Could not reach the backend: ${msg}`,
        },
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

  function useSuggestion(s: string) {
    setQuestion(s);
    setTimeout(() => {
      textareaRef.current?.focus();
      autoResize();
    }, 0);
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">

      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="flex shrink-0 items-center justify-between border-b border-white/[0.07] px-5 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
            <AtomIcon size={16} />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-text">
              Scientific RAG Assistant
            </h1>
            <p className="text-[11px] text-muted/70">
              Grounded answers from indexed papers
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Paper count badge */}
          {paperCount !== null && (
            <div className="hidden items-center gap-1.5 rounded-full border border-white/[0.07] bg-panel px-3 py-1.5 text-xs text-muted sm:flex">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
              </svg>
              <span>{paperCount} paper{paperCount !== 1 ? "s" : ""}</span>
            </div>
          )}

          {/* Upload button */}
          <button
            onClick={() => setShowUpload(true)}
            className="flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-3.5 py-1.5 text-xs font-medium text-accent transition hover:bg-accent/20"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            Upload
          </button>

          {/* K slider */}
          <div className="hidden items-center gap-2.5 rounded-full border border-white/[0.07] bg-panel px-3.5 py-1.5 md:flex">
            <span className="text-[10px] uppercase tracking-wider text-muted/60">Top-K</span>
            <input
              type="range" min={1} max={10} value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="h-1 w-20 accent-accent"
            />
            <span className="w-3 text-center font-mono text-xs text-accent">{k}</span>
          </div>
        </div>
      </header>

      {/* ── Chat area ────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
        <div className="mx-auto max-w-2xl space-y-5">

          {/* Welcome / empty state */}
          {isInitial && (
            <div className="flex flex-col items-center py-14 text-center">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/10 text-accent shadow-glow">
                <AtomIcon size={28} />
              </div>
              <h2 className="mb-2 text-xl font-semibold text-text">
                Ask a scientific question
              </h2>
              <p className="mb-7 max-w-sm text-sm leading-relaxed text-muted">
                Every answer is grounded in indexed research papers. All claims
                are traceable to a source.
              </p>
              <div className="grid w-full max-w-lg gap-2 sm:grid-cols-2">
                {SUGGESTED_QUESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => useSuggestion(s)}
                    className="rounded-xl border border-white/[0.07] bg-panel/70 px-4 py-3 text-left text-xs text-muted transition-all duration-150 hover:border-accent/25 hover:bg-accent/5 hover:text-text"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`msg-enter flex gap-3 ${
                msg.role === "user" ? "flex-row-reverse" : "flex-row"
              }`}
            >
              {/* Avatar */}
              <div
                className={`mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
                  msg.role === "user"
                    ? "bg-accent/25 text-accent"
                    : "bg-white/[0.07] text-muted"
                }`}
              >
                {msg.role === "user" ? "U" : "AI"}
              </div>

              {/* Content */}
              <div
                className={`flex max-w-[85%] flex-col gap-2.5 ${
                  msg.role === "user" ? "items-end" : "items-start"
                }`}
              >
                {/* Bubble */}
                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-gradient-to-br from-accent to-indigo-500 text-white shadow-soft"
                      : msg.meta?.unsupported
                      ? "border border-warn/20 bg-warn/5 text-text"
                      : "glass border border-white/[0.07] text-text"
                  }`}
                >
                  {/* Unsupported warning */}
                  {msg.meta?.unsupported && (
                    <div className="mb-2 flex items-center gap-1.5 text-[11px] text-warn">
                      <WarnIcon />
                      Insufficient evidence in indexed papers
                    </div>
                  )}

                  <p className="whitespace-pre-wrap">{msg.text}</p>

                  {/* Cache badge */}
                  {msg.meta?.from_cache && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] text-accent/60">
                      <BoltIcon />
                      Cached response
                    </div>
                  )}
                </div>

                {/* Topics + Citations */}
                {(msg.meta?.citations?.length ?? 0) > 0 && (
                  <div className="w-full space-y-2.5">
                    {/* Topic pills */}
                    <div className="flex flex-wrap gap-1.5">
                      {extractTopics(msg.meta!.citations).map((topic) => (
                        <span
                          key={`${msg.id}-${topic}`}
                          className="flex items-center gap-1 rounded-full border border-accent/20 bg-accent/[0.08] px-2.5 py-1 text-[11px] text-accent/85"
                        >
                          <BookmarkIcon />
                          {topic}
                        </span>
                      ))}
                    </div>

                    {/* Citation cards */}
                    <div className="space-y-1.5">
                      <p className="text-[10px] uppercase tracking-widest text-muted/50">
                        Sources
                      </p>
                      {msg.meta!.citations.map((c) => (
                        <CitationCard
                          key={`${msg.id}-${c.chunk_id}`}
                          c={c}
                          msgId={msg.id}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="msg-enter flex gap-3">
              <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/[0.07] text-[11px] font-semibold text-muted">
                AI
              </div>
              <div className="glass rounded-2xl border border-white/[0.07] px-4 py-3.5">
                <div className="flex items-center gap-1.5">
                  <span className="dot-bounce h-2 w-2 rounded-full bg-muted/50" style={{ animationDelay: "0ms" }} />
                  <span className="dot-bounce h-2 w-2 rounded-full bg-muted/50" style={{ animationDelay: "160ms" }} />
                  <span className="dot-bounce h-2 w-2 rounded-full bg-muted/50" style={{ animationDelay: "320ms" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── Input bar ────────────────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-white/[0.06] bg-bg/80 px-4 py-4 backdrop-blur-sm md:px-8">
        <form onSubmit={onSubmit} className="mx-auto max-w-2xl">
          <div className="glass flex items-end gap-3 rounded-2xl border border-white/[0.09] px-4 py-3 transition-colors duration-150 focus-within:border-accent/35">
            <textarea
              ref={textareaRef}
              rows={1}
              value={question}
              onChange={(e) => { setQuestion(e.target.value); autoResize(); }}
              onKeyDown={handleKeyDown}
              placeholder="Ask a scientific question…"
              className="max-h-40 flex-1 resize-none bg-transparent text-sm text-text outline-none placeholder:text-muted/40"
            />
            <button
              type="submit"
              disabled={!canSend}
              className="mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent text-bg transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-25"
            >
              <ArrowUpIcon />
            </button>
          </div>
          <p className="mt-2 text-center text-[10px] text-muted/35">
            Ctrl + Enter to send · Answers are grounded in indexed papers only
          </p>
        </form>
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
