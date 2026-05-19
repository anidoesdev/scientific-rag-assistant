"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Stats = {
  papers: number;
  chunks: number;
  avg_chunks_per_paper: number;
  embedding_model: string;
  embedding_dims: number;
  generation_model: string;
  reranker_model: string;
  retrieval_candidate_k: number;
  retrieval_final_k: number;
  similarity_threshold: number;
  cache_ttl_seconds: number;
};

type Health = {
  status: string;
  uptime_seconds: number;
  checks: Record<string, { status: string }>;
};

type StepChat =
  | { type: "query"; query: string }
  | { type: "embed"; query: string; vectorPreview: string }
  | { type: "retrieve"; query: string; candidates: { src: "dense" | "keyword"; sim?: string; text: string }[] }
  | { type: "fuse"; query: string; items: { rank: number; score: string; text: string }[] }
  | { type: "filter"; query: string; items: { sim: string; text: string; kept: boolean }[] }
  | { type: "rerank"; query: string; items: { score: number; text: string; kept: boolean }[] }
  | { type: "answer"; query: string; answer: string; citations: { n: number; paper: string }[] }
  | { type: "cache"; query: string; answer: string; cacheKey: string; ttl: string };

const QUERY = "What are unidirectional error correcting codes?";


const PIPELINE: { step: number; id: string; label: string; badge: string; description: string; detail: string; chat: StepChat }[] = [
  {
    step: 1, id: "query", label: "User Query", badge: "Input",
    description: "Natural language question submitted through the chat interface.",
    detail: "plain text → pipeline entry point",
    chat: { type: "query", query: QUERY },
  },
  {
    step: 2, id: "embed", label: "Query Embedding", badge: "OpenAI",
    description: "The query is converted into a dense 1536-dimensional vector by OpenAI's embedding model.",
    detail: "text-embedding-3-small · 1536 dims · cosine space",
    chat: {
      type: "embed", query: QUERY,
      vectorPreview: "[0.021, −0.034, 0.189, 0.002, −0.117, 0.063, 0.044, −0.091 …] × 1536 dims",
    },
  },
  {
    step: 3, id: "dual", label: "Dual Retrieval", badge: "Parallel",
    description: "Dense vector search (pgvector cosine) and keyword search (PostgreSQL ILIKE) run simultaneously, each returning 20 candidates.",
    detail: "pgvector cosine similarity + ILIKE · 20 candidates each",
    chat: {
      type: "retrieve", query: QUERY,
      candidates: [
        { src: "dense", sim: "0.87", text: "Unidirectional codes detect all errors where bit-flips go in one direction…" },
        { src: "keyword", text: "…t-EC codes, also called t-UEC codes, are used in memory systems…" },
        { src: "dense", sim: "0.81", text: "A code is t-unidirectional error correcting if it can correct t errors…" },
      ],
    },
  },
  {
    step: 4, id: "rrf", label: "RRF Fusion", badge: "Fusion",
    description: "Reciprocal Rank Fusion merges both result sets into a single ranked list, rewarding chunks that rank highly in both searches.",
    detail: "score = Σ 1/(rank + 60) · duplicates collapsed",
    chat: {
      type: "fuse", query: QUERY,
      items: [
        { rank: 1, score: "0.031", text: "Unidirectional codes detect all errors where…" },
        { rank: 2, score: "0.016", text: "…t-EC codes, also called t-UEC codes…" },
        { rank: 3, score: "0.013", text: "A code is t-unidirectional error correcting…" },
      ],
    },
  },
  {
    step: 5, id: "filter", label: "Threshold Filter", badge: "Filter",
    description: "Chunks whose cosine similarity falls below the threshold are pruned, removing noisy or unrelated results.",
    detail: "threshold = 0.30 cosine similarity",
    chat: {
      type: "filter", query: QUERY,
      items: [
        { sim: "0.87", text: "Unidirectional codes detect all errors…", kept: true },
        { sim: "0.51", text: "A code is t-unidirectional error correcting…", kept: true },
        { sim: "0.21", text: "Memory systems often require robust error…", kept: false },
      ],
    },
  },
  {
    step: 6, id: "rerank", label: "LLM Reranker", badge: "AI Score",
    description: "gpt-4o-mini independently scores each remaining chunk 0–10 for relevance to the specific query.",
    detail: "Model: gpt-4o-mini · min passing score = 4",
    chat: {
      type: "rerank", query: QUERY,
      items: [
        { score: 9, text: "Unidirectional codes detect all errors…", kept: true },
        { score: 7, text: "A code is t-unidirectional error correcting…", kept: true },
        { score: 2, text: "Memory systems often require robust error…", kept: false },
      ],
    },
  },
  {
    step: 7, id: "generate", label: "Answer Generation", badge: "Generate",
    description: "gpt-4o-mini synthesises a grounded answer using only the top-K reranked chunks, with numbered inline citations.",
    detail: "Top-K = 5 chunks · grounded generation · cited sources",
    chat: {
      type: "answer", query: QUERY,
      answer: "Unidirectional error correcting codes detect and correct errors where all bit-flips go in the same direction [1]. They are widely used in memory systems and storage devices where asymmetric faults are common [2].",
      citations: [
        { n: 1, paper: "error_correcting_codes.pdf" },
        { n: 2, paper: "memory_fault_models.pdf" },
      ],
    },
  },
  {
    step: 8, id: "cache", label: "Redis Cache", badge: "Cache",
    description: "The complete response is stored in Redis. Identical queries are served instantly without any LLM calls.",
    detail: "TTL = 3600 s · key = hash(query + K)",
    chat: {
      type: "cache", query: QUERY,
      answer: "Unidirectional error correcting codes detect and correct errors where all bit-flips go in the same direction [1]…",
      cacheKey: "ask:3f8a2c1b…",
      ttl: "3600 s",
    },
  },
];

const TECH_STACK = [
  {
    name: "FastAPI", version: "0.136",
    role: "Async Python API with automatic OpenAPI docs, dependency injection, and CORS middleware.",
    color: "#059669",
  },
  {
    name: "PostgreSQL + pgvector", version: "pg16 + 0.8",
    role: "Primary store for chunk text and 1536-dim embeddings with an IVFFlat index for fast cosine similarity search.",
    color: "#4F46E5",
  },
  {
    name: "Redis", version: "7-alpine",
    role: "In-memory answer cache with a 1-hour TTL. Eliminates repeated LLM calls for identical queries.",
    color: "#DC2626",
  },
  {
    name: "OpenAI", version: "gpt-4o-mini",
    role: "Dual role: text-embedding-3-small produces 1536-dim chunk embeddings; gpt-4o-mini handles LLM reranking and answer generation.",
    color: "#B45309",
  },
];


export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/stats`).then(r => r.json()).then(setStats).catch(() => {});
    fetch(`${API_BASE}/health`).then(r => r.json()).then(setHealth).catch(() => {});
  }, []);

  useEffect(() => {
    if (paused) return;
    const id = setInterval(() => setActiveStep(s => (s + 1) % PIPELINE.length), 3000);
    return () => clearInterval(id);
  }, [paused]);

  const uptime = (() => {
    const s = health?.uptime_seconds;
    if (!s) return "—";
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  })();

  const allOk = health?.status === "ok";
  const checking = !health;

  return (
    <div className="min-h-screen bg-bg font-sans text-text">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 glass border-b border-stone-200/70 px-6 py-3">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="text-accent">
              <circle cx="12" cy="12" r="2" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              <path d="M2 12h20" />
            </svg>
            <span className="font-serif text-base font-bold text-text">Papyrus</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 sm:flex">
              {["database", "openai", "redis"].map(k => (
                <span key={k} title={k} className={`h-2 w-2 rounded-full transition-colors ${
                  checking ? "animate-pulse bg-stone-300" :
                  health?.checks[k]?.status === "ok" ? "bg-success" : "bg-warn"
                }`} />
              ))}
              <span className="text-xs text-muted/50">
                {checking ? "Checking…" : allOk ? "All systems up" : "Degraded"}
              </span>
            </div>
            <Link
              href="/chat"
              className="flex items-center gap-1.5 rounded-full bg-accent px-4 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-accent/90"
            >
              Try the Assistant
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-24 pt-14">

        {/* ── Hero ──────────────────────────────────────────────────────── */}
        <section className="relative mb-16 overflow-hidden rounded-3xl">
          {/* Anime background image */}
          <Image
            src="/img_bg.jpg"
            alt=""
            fill
            priority
            className="object-cover object-center"
          />
          {/* Dark overlay — top opaque for text, fades to page bg at bottom */}
          <div className="absolute inset-0 bg-gradient-to-b from-amber-950/85 via-amber-900/60 to-[#F7F4EF]" />

          {/* Content */}
          <div className="relative z-10 px-6 pb-24 pt-16 text-center sm:px-12">
            <p className="mb-3 text-[11px] font-bold uppercase tracking-[0.22em] text-amber-200/90">
              Scientific RAG Assistant
            </p>
            <h1 className="font-serif text-4xl font-bold text-white drop-shadow-lg sm:text-5xl lg:text-6xl">
              Ask the Papers.
            </h1>
            <p className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-amber-100/85">
              A multi-stage retrieval pipeline combining dense vector search, keyword matching,
              reciprocal rank fusion, and LLM reranking to surface the most relevant evidence
              from the indexed paper corpus.
            </p>

            <div className="mt-10 flex flex-col items-center gap-5">
              <Link
                href="/chat"
                className="inline-flex items-center gap-2.5 rounded-full bg-white px-8 py-3.5 text-sm font-bold text-amber-900 shadow-lg transition-all hover:scale-[1.05] hover:shadow-xl"
              >
                Try the Assistant
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
              {stats && (
                <div className="flex flex-wrap justify-center gap-2.5">
                  {([
                    { v: stats.papers,                  l: "papers indexed" },
                    { v: stats.chunks.toLocaleString(), l: "vector chunks"  },
                    { v: `${stats.embedding_dims}-dim`, l: "embeddings"     },
                    { v: `up ${uptime}`,                l: "uptime"         },
                  ] as { v: string | number; l: string }[]).map(({ v, l }) => (
                    <span key={l} className="rounded-full border border-white/20 bg-white/10 px-4 py-1.5 text-sm backdrop-blur-sm">
                      <span className="font-bold text-white">{v}</span>
                      <span className="ml-1.5 text-amber-100/80">{l}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Stat cards ────────────────────────────────────────────────── */}
        {stats && (
          <section className="mb-16 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Papers Indexed"       value={stats.papers}                          sub="default corpus"                       accent="from-amber-500 to-orange-500" />
            <StatCard label="Vector Chunks"        value={stats.chunks.toLocaleString()}          sub={`~${stats.avg_chunks_per_paper} per paper`} accent="from-indigo-500 to-violet-500" />
            <StatCard label="Retrieval Candidates" value={stats.retrieval_candidate_k}            sub="per search method"                    accent="from-emerald-500 to-teal-500" />
            <StatCard label="Final Top-K"          value={stats.retrieval_final_k}               sub={`threshold ${stats.similarity_threshold}`}  accent="from-rose-500 to-pink-500"  />
          </section>
        )}

        {/* ── Pipeline timeline ─────────────────────────────────────────── */}
        <section className="mb-16">
          <SectionHeader
            label="Retrieval Pipeline"
            title="8-Stage Query Processing"
            description="Every query flows through this sequence. Click any step to explore — or watch the example query transform live as it moves through each stage."
          />

          <div className="mt-10">
            {PIPELINE.map((step, i) => (
              <div key={step.id} className="flex gap-4 sm:gap-6">

                {/* Spine: circle + connector */}
                <div className="flex flex-col items-center">
                  <button
                    onClick={() => { setActiveStep(i); setPaused(true); }}
                    className={`relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border-2 transition-all duration-300 focus:outline-none ${
                      i === activeStep
                        ? "border-accent bg-accent text-white shadow-glow scale-110"
                        : i < activeStep
                        ? "border-accent/40 bg-accentSoft/40 text-accent"
                        : "border-stone-200 bg-white text-muted/60 hover:border-accent/30"
                    }`}
                  >
                    {i < activeStep ? (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    ) : (
                      <span className="text-[11px] font-bold">{step.step}</span>
                    )}
                    {i === activeStep && (
                      <span className="absolute inset-0 rounded-full bg-accent animate-ping opacity-20 pointer-events-none" />
                    )}
                  </button>
                  {i < PIPELINE.length - 1 && (
                    <div className={`w-0.5 flex-1 min-h-[12px] my-1 rounded-full transition-colors duration-500 ${
                      i < activeStep ? "bg-accent/40" : "bg-stone-200"
                    }`} />
                  )}
                </div>

                {/* Card */}
                <div
                  className={`flex-1 mb-3 cursor-pointer overflow-hidden rounded-2xl border transition-all duration-500 ${
                    i === activeStep
                      ? "border-accent/30 bg-white shadow-glow"
                      : i < activeStep
                      ? "border-accent/15 bg-accentSoft/8 hover:border-accent/25"
                      : "border-stone-200 bg-panel/40 hover:border-stone-300"
                  }`}
                  style={{ maxHeight: i === activeStep ? "800px" : "56px" }}
                  onClick={() => { setActiveStep(i); setPaused(true); }}
                >
                  {/* Header row — always visible */}
                  <div className="flex h-14 items-center justify-between px-5">
                    <div className="flex items-center gap-2.5">
                      <span className={`text-sm font-semibold transition-colors ${
                        i === activeStep ? "text-accent" :
                        i < activeStep ? "text-text/75" : "text-muted/65"
                      }`}>
                        {step.label}
                      </span>
                      <span className={`rounded px-1.5 py-px text-[9px] font-bold uppercase tracking-wide transition-all ${
                        i === activeStep ? "bg-accent text-white" :
                        i < activeStep ? "bg-accent/20 text-accent/80" :
                        "bg-stone-200/80 text-muted/65"
                      }`}>
                        {step.badge}
                      </span>
                    </div>
                    {i < activeStep && <span className="text-[10px] font-medium text-accent/70">Complete</span>}
                    {i > activeStep && <span className="text-[10px] text-muted/55">Pending</span>}
                    {i === activeStep && <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />}
                  </div>

                  {/* Expanded content */}
                  {i === activeStep && (
                    <div className="msg-enter border-t border-accent/10 px-5 pb-5 pt-4">
                      <p className="mb-4 text-sm text-muted/70">{step.description}</p>
                      <MiniChatWindow chat={step.chat} />
                      <code className="mt-3 inline-block rounded-md bg-accent/10 px-2.5 py-1 text-xs text-accent">
                        {step.detail}
                      </code>
                    </div>
                  )}
                </div>

              </div>
            ))}
          </div>

          <div className="mt-1 flex items-center justify-between">
            <p className="text-xs text-muted/70">Click any step · auto-advances every 3 s</p>
            <button
              onClick={() => setPaused(p => !p)}
              className="rounded-full border border-stone-200 px-3 py-1 text-xs text-muted/60 transition-colors hover:border-accent/30 hover:text-accent"
            >
              {paused ? "▶ Resume" : "⏸ Pause"}
            </button>
          </div>
        </section>

        {/* ── Retrieval deep dive ───────────────────────────────────────── */}
        <section className="mb-16">
          <SectionHeader
            label="Retrieval Strategy"
            title="Dual Search + RRF Fusion"
            description="Two fundamentally different retrieval methods run in parallel then combined with Reciprocal Rank Fusion — each compensates for the other's blind spots."
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-stone-200 bg-panel/60 p-5">
              <div className="mb-3 flex items-center gap-2">
                <span className="font-semibold text-text">Dense Vector Search</span>
                <span className="rounded bg-indigo-100 px-2 py-0.5 text-xs font-semibold text-indigo-700">pgvector</span>
              </div>
              <p className="mb-3 text-sm text-muted/70">
                The query vector is compared against all chunk embeddings using cosine similarity.
                Captures <strong className="text-text/80">semantic meaning</strong> — finds conceptually
                related chunks even when exact keywords don&apos;t appear.
              </p>
              <code className="block rounded-lg bg-bg px-3 py-2 text-xs text-muted/70 break-all">
                ORDER BY embedding &lt;=&gt; $query_vec LIMIT 20
              </code>
            </div>
            <div className="rounded-2xl border border-stone-200 bg-panel/60 p-5">
              <div className="mb-3 flex items-center gap-2">
                <span className="font-semibold text-text">Keyword Search</span>
                <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700">ILIKE</span>
              </div>
              <p className="mb-3 text-sm text-muted/70">
                Exact and partial term matching against raw chunk text. Reliable for
                <strong className="text-text/80"> proper nouns, acronyms, and technical terms</strong> that
                embeddings may not capture precisely.
              </p>
              <code className="block rounded-lg bg-bg px-3 py-2 text-xs text-muted/70 break-all">
                WHERE text ILIKE &apos;%query_term%&apos; LIMIT 20
              </code>
            </div>
          </div>
          <div className="mt-4 rounded-2xl border border-stone-200 bg-panel/60 px-6 py-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex-1">
                <p className="mb-1 font-semibold text-text">Reciprocal Rank Fusion</p>
                <p className="text-sm text-muted/70">
                  Each method returns its own ranked list. RRF assigns a score to every chunk based on
                  its rank in each list, then merges them. A chunk that ranks highly in <em>both</em> lists
                  scores highest. The constant <code className="rounded bg-stone-200 px-1 text-xs">k=60</code> smooths
                  top-rank dominance.
                </p>
              </div>
              <div className="shrink-0 rounded-xl bg-accentSoft/60 px-5 py-3 text-center font-mono text-sm text-accent">
                <div>RRF(d) = Σ</div>
                <div className="mt-0.5 border-t border-accent/30 pt-0.5">k + rank<sub>i</sub>(d)</div>
                <div className="mt-1 text-xs text-accent/80">k = 60</div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Tech stack ────────────────────────────────────────────────── */}
        <section className="mb-16">
          <SectionHeader
            label="Technology"
            title="Production Stack"
            description="Each component was selected for performance, correctness, and minimal operational overhead on a single-node deployment."
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {TECH_STACK.map(t => (
              <div key={t.name} className="rounded-2xl border border-stone-200 bg-panel/60 p-5 transition-shadow hover:shadow-soft">
                <div className="mb-2 flex items-start justify-between gap-2">
                  <span className="font-semibold text-text">{t.name}</span>
                  <span className="shrink-0 rounded bg-stone-200/80 px-2 py-0.5 text-xs text-muted/70">{t.version}</span>
                </div>
                <div className="mb-3 h-0.5 w-8 rounded-full" style={{ backgroundColor: t.color }} />
                <p className="text-sm text-muted/70">{t.role}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Evaluation ────────────────────────────────────────────────── */}
        <section className="mb-16">
          <SectionHeader
            label="Evaluation"
            title="Retrieval Quality Metrics"
            description="The pipeline is evaluated against a curated set of query–paper pairs using standard information retrieval metrics."
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-stone-200 bg-panel/60 p-5">
              <p className="mb-1 text-lg font-bold text-text">Hit@K</p>
              <code className="mb-3 block rounded-lg bg-accentSoft/50 px-3 py-2 text-xs text-accent">
                1 if expected_paper ∈ top-K results, else 0
              </code>
              <p className="text-sm text-muted/70">
                Fraction of queries where the expected paper appears in the top-K retrieved chunks.
                Measures whether the retriever surfaces the correct source at all.
              </p>
            </div>
            <div className="rounded-2xl border border-stone-200 bg-panel/60 p-5">
              <p className="mb-1 text-lg font-bold text-text">MRR</p>
              <code className="mb-3 block rounded-lg bg-accentSoft/50 px-3 py-2 text-xs text-accent">
                1/N · Σ 1/rank(expected_paper)
              </code>
              <p className="text-sm text-muted/70">
                Mean Reciprocal Rank rewards retrievers that surface the correct paper at a higher rank.
                A score of 1.0 means the expected paper is always ranked first.
              </p>
            </div>
          </div>
          
        </section>

        {/* ── Bottom CTA ────────────────────────────────────────────────── */}
        <section className="relative overflow-hidden rounded-3xl text-center">
          <Image src="/img_bg.jpg" alt="" fill className="object-cover object-top" />
          <div className="absolute inset-0 bg-amber-950/80" />
          <div className="relative z-10 px-8 py-14">
            <h2 className="font-serif text-3xl font-bold text-white">Ready to explore the research?</h2>
            <p className="mx-auto mt-3 max-w-md text-sm text-amber-100/85">
              Ask any question across {stats?.papers ?? "20"} indexed papers. Grounded answers with full citations.
            </p>
            <Link
              href="/chat"
              className="mt-8 inline-flex items-center gap-2.5 rounded-full bg-white px-9 py-3.5 text-sm font-bold text-amber-900 shadow-lg transition-all hover:scale-[1.05] hover:shadow-xl"
            >
              Open the Assistant
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </section>

      </main>
    </div>
  );
}

/* ── Shared sub-components ────────────────────────────────────────────────── */

function StatCard({ label, value, sub, accent = "from-accent to-amber-500" }: { label: string; value: string | number; sub: string; accent?: string }) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-stone-200 bg-panel/60 px-5 py-5 transition-all hover:-translate-y-1 hover:shadow-soft">
      <div className={`absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r ${accent}`} />
      <p className="text-xs text-muted/80">{label}</p>
      <p className="mt-1.5 text-2xl font-bold text-text">{value}</p>
      <p className="mt-0.5 text-xs text-muted/65">{sub}</p>
    </div>
  );
}

function SectionHeader({ label, title, description }: { label: string; title: string; description: string }) {
  return (
    <div>
      <p className="mb-1 text-xs font-semibold uppercase tracking-widest text-accent">{label}</p>
      <h2 className="text-2xl font-semibold text-text">{title}</h2>
      <p className="mt-1.5 max-w-2xl text-sm text-muted/85">{description}</p>
    </div>
  );
}

/* ── Mini chat window (pipeline demo) ────────────────────────────────────── */

function MiniChatWindow({ chat }: { chat: StepChat }) {
  return (
    <div className="overflow-hidden rounded-xl border border-stone-200 bg-bg shadow-soft">
      {/* macOS-style title bar */}
      <div className="flex items-center gap-3 border-b border-stone-200 bg-panel/80 px-3 py-2">
        <div className="flex gap-1.5">
          <div className="h-2.5 w-2.5 rounded-full bg-red-300/80" />
          <div className="h-2.5 w-2.5 rounded-full bg-amber-300/80" />
          <div className="h-2.5 w-2.5 rounded-full bg-green-300/80" />
        </div>
        <div className="flex items-center gap-1.5">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="text-accent/60">
            <circle cx="12" cy="12" r="2" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            <path d="M2 12h20" />
          </svg>
          <span className="font-serif text-[11px] font-bold text-text/75">Scientific RAG Assistant</span>
        </div>
        <div className="ml-auto h-0.5 w-10 rounded-full bg-gradient-to-r from-transparent via-accent/35 to-transparent" />
      </div>

      {/* Chat body */}
      <div className="space-y-3 px-4 py-3">
        {/* User query bubble */}
        <div className="flex justify-end gap-2">
          <div className="max-w-[80%] text-right">
            <p className="mb-1 text-[8px] uppercase tracking-[0.15em] text-muted/65">Query</p>
            <p className="font-serif text-[12px] italic leading-relaxed text-text/90">
              &ldquo;{chat.query}&rdquo;
            </p>
          </div>
          <div className="mt-4 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-accent/25 bg-accent/8 text-[8px] font-bold text-accent">
            Q
          </div>
        </div>

        {/* Divider + AI area */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="h-px flex-1 bg-stone-200" />
            <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="text-accent/35">
              <circle cx="12" cy="12" r="2" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              <path d="M2 12h20" />
            </svg>
            <div className="h-px flex-1 bg-stone-200" />
          </div>
          <MiniChatContent chat={chat} />
        </div>
      </div>
    </div>
  );
}

function MiniChatContent({ chat }: { chat: StepChat }) {
  switch (chat.type) {
    case "query":
      return (
        <div className="flex items-center gap-1.5 pl-0.5 py-1">
          <span className="dot-bounce h-1 w-1 rounded-full bg-muted/40" style={{ animationDelay: "0ms" }} />
          <span className="dot-bounce h-1 w-1 rounded-full bg-muted/40" style={{ animationDelay: "160ms" }} />
          <span className="dot-bounce h-1 w-1 rounded-full bg-muted/40" style={{ animationDelay: "320ms" }} />
          <span className="ml-1 font-serif text-[11px] italic text-muted/70">Searching papers…</span>
        </div>
      );

    case "embed":
      return (
        <div className="space-y-1.5">
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted/70">Vector embedding</p>
          <code className="block break-all rounded-lg bg-accent/8 px-3 py-2 font-mono text-[10px] leading-relaxed text-accent/80">
            {chat.vectorPreview}
          </code>
        </div>
      );

    case "retrieve":
      return (
        <div className="space-y-1.5">
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted/70">
            {chat.candidates.length} candidates retrieved
          </p>
          {chat.candidates.map((c, i) => (
            <div key={i} className="flex items-start gap-2 rounded-lg bg-stone-50 px-2.5 py-2">
              <span className={`mt-px shrink-0 rounded px-1.5 py-px text-[8px] font-bold uppercase ${
                c.src === "dense" ? "bg-indigo-100 text-indigo-600" : "bg-emerald-100 text-emerald-600"
              }`}>
                {c.src === "dense" ? `sim ${c.sim}` : "keyword"}
              </span>
              <p className="line-clamp-1 flex-1 text-[11px] leading-snug text-text/85">{c.text}</p>
            </div>
          ))}
        </div>
      );

    case "fuse":
      return (
        <div className="space-y-1.5">
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted/70">RRF merged ranking</p>
          {chat.items.map((item) => (
            <div key={item.rank} className="flex items-center gap-2 rounded-lg bg-stone-50 px-2.5 py-2">
              <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-accent/15 text-[8px] font-bold text-accent">
                {item.rank}
              </span>
              <span className="shrink-0 rounded bg-amber-100 px-1.5 py-px font-mono text-[8px] text-amber-700">
                {item.score}
              </span>
              <p className="line-clamp-1 flex-1 text-[11px] leading-snug text-text/85">{item.text}</p>
            </div>
          ))}
        </div>
      );

    case "filter":
      return (
        <div className="space-y-1.5">
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted/70">Threshold = 0.30</p>
          {chat.items.map((item, i) => (
            <div key={i} className={`flex items-center gap-2 rounded-lg px-2.5 py-2 ${item.kept ? "bg-stone-50" : "bg-red-50/40"}`}>
              <span className={`shrink-0 text-[10px] font-bold ${item.kept ? "text-emerald-500" : "text-red-400"}`}>
                {item.kept ? "✓" : "✗"}
              </span>
              <span className="shrink-0 rounded bg-stone-200 px-1.5 py-px font-mono text-[8px] text-muted/70">
                {item.sim}
              </span>
              <p className={`line-clamp-1 flex-1 text-[11px] leading-snug ${item.kept ? "text-text/65" : "text-muted/40 line-through"}`}>
                {item.text}
              </p>
            </div>
          ))}
        </div>
      );

    case "rerank":
      return (
        <div className="space-y-1.5">
          <p className="text-[9px] font-semibold uppercase tracking-[0.15em] text-muted/70">LLM relevance scores</p>
          {chat.items.map((item, i) => (
            <div key={i} className={`flex items-center gap-2 rounded-lg px-2.5 py-2 ${item.kept ? "bg-stone-50" : "bg-red-50/40"}`}>
              <span className={`shrink-0 text-[10px] font-bold ${item.kept ? "text-emerald-500" : "text-red-400"}`}>
                {item.kept ? "✓" : "✗"}
              </span>
              <span className={`shrink-0 rounded px-1.5 py-px text-[8px] font-bold ${
                item.score >= 7 ? "bg-emerald-100 text-emerald-700" :
                item.score >= 4 ? "bg-amber-100 text-amber-700" :
                "bg-red-100 text-red-600"
              }`}>
                {item.score}/10
              </span>
              <p className={`line-clamp-1 flex-1 text-[11px] leading-snug ${item.kept ? "text-text/65" : "text-muted/40 line-through"}`}>
                {item.text}
              </p>
            </div>
          ))}
        </div>
      );

    case "answer":
      return (
        <div className="space-y-2.5">
          <p className="font-serif text-[12.5px] leading-relaxed text-text/85">{chat.answer}</p>
          <div className="space-y-1.5 border-t border-stone-200 pt-2">
            <p className="text-[8px] font-semibold uppercase tracking-[0.18em] text-muted/65">References</p>
            {chat.citations.map((c) => (
              <div key={c.n} className="flex items-start gap-1.5">
                <span className="shrink-0 font-serif text-[9px] font-bold text-accent">[{c.n}]</span>
                <span className="truncate text-[10px] text-muted/75">{c.paper}</span>
              </div>
            ))}
          </div>
        </div>
      );

    case "cache":
      return (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor" className="text-accent/60">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
            </svg>
            <span className="text-[9px] font-semibold uppercase tracking-widest text-muted/70">Cached response</span>
          </div>
          <p className="font-serif text-[12.5px] leading-relaxed text-text/85">{chat.answer}</p>
          <div className="rounded-lg bg-accent/8 px-2.5 py-1.5">
            <code className="font-mono text-[9px] text-accent/85">
              {chat.cacheKey} · TTL {chat.ttl} · &lt;5 ms
            </code>
          </div>
        </div>
      );
  }
}
