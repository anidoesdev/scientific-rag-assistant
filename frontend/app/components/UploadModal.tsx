"use client";

import { useCallback, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/* ── Icons ──────────────────────────────────────────────────────────────── */

const UploadIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const CheckIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const CloseIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <path d="M18 6 6 18M6 6l12 12" />
  </svg>
);

/* ── Types ──────────────────────────────────────────────────────────────── */

type UploadResult = { paper_id: string; filename: string; chunks: number };
type State = "idle" | "uploading" | "done" | "error";

/* ── Component ──────────────────────────────────────────────────────────── */

interface Props {
  onClose: () => void;
  onDone: () => void;
}

export function UploadModal({ onClose, onDone }: Props) {
  const [state, setState]   = useState<State>("idle");
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError]   = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ── Upload ─────────────────────────────────────────────────────────── */

  async function startUpload(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are accepted."); return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setError("File too large (max 50 MB)."); return;
    }

    setError(null);
    setState("uploading");

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? `Upload failed (${res.status})`);
      setResult(body as UploadResult);
      setState("done");
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setState("error");
    }
  }

  /* ── Drag & drop ────────────────────────────────────────────────────── */

  const onDragOver  = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragging(true); }, []);
  const onDragLeave = useCallback(() => setDragging(false), []);
  const onDrop      = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) startUpload(file);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Render ─────────────────────────────────────────────────────────── */

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center p-4 sm:items-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-md rounded-3xl border border-white/[0.09] bg-panel p-6 shadow-soft">

        {/* Header */}
        <div className="mb-5 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-text">Upload a Paper</h2>
            <p className="text-xs text-muted/60">PDF only · max 50 MB · cleared on session restart</p>
          </div>
          <button onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted/50 transition-colors hover:text-text">
            <CloseIcon />
          </button>
        </div>

        {/* ── Idle: drop zone ── */}
        {(state === "idle" || state === "error") && (
          <>
            <div
              onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 transition-all duration-150 ${
                dragging
                  ? "scale-[1.01] border-accent bg-accent/10"
                  : "border-white/10 bg-white/[0.02] hover:border-accent/40 hover:bg-accent/[0.04]"
              }`}
            >
              <div className={`flex h-11 w-11 items-center justify-center rounded-xl transition-colors ${
                dragging ? "bg-accent/20 text-accent" : "bg-white/[0.06] text-muted"
              }`}>
                <UploadIcon />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-text">
                  {dragging ? "Release to upload" : "Drop a PDF here"}
                </p>
                <p className="mt-0.5 text-xs text-muted/50">or click to browse</p>
              </div>
              <input ref={fileInputRef} type="file" accept=".pdf" className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) startUpload(f); }} />
            </div>

            {error && (
              <p className="mt-3 rounded-xl border border-red-500/20 bg-red-500/[0.08] px-4 py-2.5 text-xs text-red-400">
                {error}
              </p>
            )}
          </>
        )}

        {/* ── Uploading: spinner ── */}
        {state === "uploading" && (
          <div className="flex flex-col items-center gap-4 py-10">
            <div className="relative flex h-14 w-14 items-center justify-center">
              <div className="absolute inset-0 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              <div className="h-8 w-8 rounded-full bg-accent/10 text-accent flex items-center justify-center">
                <UploadIcon />
              </div>
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-text">Processing…</p>
              <p className="mt-1 text-xs text-muted/60">
                Chunking → embedding → indexing into pgvector
              </p>
            </div>
          </div>
        )}

        {/* ── Done ── */}
        {state === "done" && result && (
          <div className="flex flex-col items-center gap-4 py-6">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/15 text-success">
              <CheckIcon />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-text">{result.filename}</p>
              <p className="mt-1 text-xs text-muted/60">
                {result.chunks} chunks indexed and ready to query
              </p>
            </div>
            <div className="flex w-full gap-2 pt-1">
              <button
                onClick={() => { setState("idle"); setResult(null); setError(null); }}
                className="flex-1 rounded-xl border border-white/10 py-2.5 text-xs text-muted transition hover:border-white/20 hover:text-text"
              >
                Upload another
              </button>
              <button
                onClick={onClose}
                className="flex-1 rounded-xl bg-accent py-2.5 text-xs font-medium text-bg transition hover:brightness-110"
              >
                Start querying ↗
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
