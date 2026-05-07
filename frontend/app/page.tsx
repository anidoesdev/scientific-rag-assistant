"use client";

import { FormEvent, useMemo, useState } from "react";

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

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  for (const citation of citations) {
    const base = citation.file_name || citation.paper_id || citation.chunk_id;
    const topic = prettifyTopic(base);
    if (topic) topics.add(topic);
    if (topics.size >= 4) break;
  }
  return Array.from(topics);
}

export default function Page() {
  const [question, setQuestion] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Ask any scientific question. I answer using only indexed paper evidence."
    }
  ]);

  const canSend = useMemo(
    () => question.trim().length > 0 && !loading,
    [question, loading]
  );

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSend) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: question.trim()
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentQuestion = question.trim();
    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: currentQuestion, k })
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data = (await res.json()) as AskResponse;

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer,
          meta: data
        }
      ]);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown network error";
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `Could not reach backend: ${message}`
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="relative mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-10 md:px-8">
      <div className="pointer-events-none absolute -left-20 top-0 h-72 w-72 rounded-full bg-accent/20 blur-3xl" />
      <div className="pointer-events-none absolute -right-10 top-20 h-72 w-72 rounded-full bg-fuchsia-500/10 blur-3xl" />

      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Scientific RAG Assistant</p>
        <h1 className="mt-2 bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-3xl font-semibold tracking-tight text-transparent md:text-5xl">
          Beautiful, grounded research answers
        </h1>
        <p className="mt-3 max-w-2xl text-sm text-muted md:text-base">
          Ask a question and get concise answers with transparent source-backed
          topics and citation previews.
        </p>
      </header>

      <section className="glass shadow-soft relative flex-1 rounded-3xl p-4 md:p-7">
        <div className="mb-6 max-h-[62vh] space-y-4 overflow-y-auto pr-1">
          {messages.map((msg) => (
            <article
              key={msg.id}
              className={`rounded-2xl border p-4 md:p-5 ${
                msg.role === "user"
                  ? "ml-auto max-w-[85%] border-accent/30 bg-gradient-to-br from-accentSoft/50 to-indigo-500/10"
                  : "mr-auto max-w-[95%] border-white/10 bg-panel/90"
              }`}
            >
              <p className="mb-2 text-xs uppercase tracking-wide text-muted">
                {msg.role === "user" ? "You" : "Assistant"}
              </p>
              <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>

              {msg.meta?.citations?.length ? (
                <div className="mt-4 space-y-2 border-t border-white/10 pt-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-muted">
                      Based on topics
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {extractTopics(msg.meta.citations).map((topic) => (
                        <span
                          key={`${msg.id}-${topic}`}
                          className="rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-slate-200"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs uppercase tracking-wide text-muted">
                    Citations
                  </p>
                  {msg.meta.citations.map((c) => (
                    <div
                      key={`${msg.id}-${c.chunk_id}`}
                      className="rounded-xl border border-white/10 bg-black/20 p-3"
                    >
                      <p className="text-xs text-muted/90">
                        Source {c.source_number} · {prettifyTopic(c.file_name || c.paper_id)}
                      </p>
                      <p className="mt-1 text-sm text-text/90">{c.preview}</p>
                    </div>
                  ))}
                </div>
              ) : null}
            </article>
          ))}

          {loading ? (
            <div className="mr-auto rounded-xl border border-white/10 bg-panel/90 p-4 text-sm text-muted">
              Thinking...
            </div>
          ) : null}
        </div>

        <form onSubmit={onSubmit} className="space-y-3 border-t border-white/10 pt-4">
          <div className="flex items-center justify-between">
            <label className="block text-sm text-muted" htmlFor="k">
              Top citations
            </label>
            <span className="rounded-full border border-white/10 px-2 py-1 text-xs text-muted">
              k = {k}
            </span>
          </div>
          <input
            id="k"
            type="range"
            min={1}
            max={10}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-full accent-accent"
          />

          <div className="flex gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What are sparse autoencoders used for in transformer interpretability?"
              className="flex-1 rounded-xl border border-white/10 bg-panel px-4 py-3 text-sm outline-none ring-accent transition placeholder:text-muted/80 focus:ring-2"
            />
            <button
              type="submit"
              disabled={!canSend}
              className="rounded-xl bg-gradient-to-br from-accent to-indigo-500 px-5 py-3 text-sm font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Ask
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
