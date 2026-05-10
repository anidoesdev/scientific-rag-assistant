from pathlib import Path
from typing import Any, Dict, List
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.services.chunker import TEXT_SPLITTER, build_paper_text, load_paper
from app.services.embedder import embed_texts

DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"


def _already_indexed_sources(session: Session) -> set:
    rows = session.execute(sql_text("SELECT DISTINCT source FROM chunks")).all()
    return {row[0] for row in rows if row[0]}


def _next_paper_idx(session: Session) -> int:
    rows = session.execute(sql_text("SELECT DISTINCT paper_id FROM chunks")).all()
    max_idx = 0
    for (pid,) in rows:
        if not pid:
            continue
        try:
            idx = int(pid.split("_")[1])
            max_idx = max(max_idx, idx)
        except (IndexError, ValueError):
            pass
    return max_idx + 1


def _ingest_pdf(pdf_path: Path, paper_id: str, session: Session) -> Dict[str, Any]:
    from langchain_classic.schema import Document

    paper_record = load_paper(pdf_path, paper_id)
    paper_text = build_paper_text(paper_record)
    if not paper_text:
        return {
            "file": pdf_path.name,
            "paper_id": paper_id,
            "chunks": 0,
            "skipped": True,
            "reason": "empty text after cleaning",
        }

    base_doc = Document(
        page_content=paper_text,
        metadata={"paper_id": paper_id, "file_name": pdf_path.name, "source": str(pdf_path)},
    )
    chunk_docs = TEXT_SPLITTER.split_documents([base_doc])
    total_for_paper = len(chunk_docs)

    texts = [doc.page_content for doc in chunk_docs]
    embeddings = embed_texts(texts)

    insert_stmt = sql_text(
        """
        INSERT INTO chunks
            (chunk_id, paper_id, file_name, source, chunk_index, total_chunks_for_paper, text, embedding)
        VALUES
            (:chunk_id, :paper_id, :file_name, :source, :chunk_index, :total_chunks_for_paper, :text,
             CAST(:embedding AS vector))
        ON CONFLICT (chunk_id) DO NOTHING
        """
    )
    for chunk_idx, (doc, emb) in enumerate(zip(chunk_docs, embeddings), start=1):
        session.execute(
            insert_stmt,
            {
                "chunk_id": f"{paper_id}_chunk_{chunk_idx:04d}",
                "paper_id": paper_id,
                "file_name": pdf_path.name,
                "source": str(pdf_path),
                "chunk_index": chunk_idx,
                "total_chunks_for_paper": total_for_paper,
                "text": doc.page_content,
                "embedding": f"[{','.join(map(str, emb))}]",
            },
        )
    session.commit()

    return {"file": pdf_path.name, "paper_id": paper_id, "chunks": total_for_paper}


def ingest_single(pdf_path: Path, session: Session) -> Dict[str, Any]:
    """Ingest one PDF. Skips if the source path is already indexed."""
    indexed = _already_indexed_sources(session)
    if str(pdf_path) in indexed:
        return {"file": pdf_path.name, "skipped": True, "reason": "already indexed"}

    paper_id = f"paper_{_next_paper_idx(session):03d}"
    return _ingest_pdf(pdf_path, paper_id, session)


def ingest_all_raw(session: Session) -> Dict[str, List]:
    """Ingest all PDFs in RAW_DIR that are not yet indexed."""
    indexed = _already_indexed_sources(session)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))

    ingested: List[Dict] = []
    skipped: List[str] = []
    failed: List[Dict] = []

    for pdf_path in pdfs:
        if str(pdf_path) in indexed:
            skipped.append(pdf_path.name)
            continue

        paper_id = f"paper_{_next_paper_idx(session):03d}"
        try:
            result = _ingest_pdf(pdf_path, paper_id, session)
            ingested.append(result)
        except Exception as exc:
            failed.append({"file": pdf_path.name, "reason": str(exc)})

    return {"ingested": ingested, "skipped": skipped, "failed": failed}
