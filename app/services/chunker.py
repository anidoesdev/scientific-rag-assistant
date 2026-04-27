from pathlib import Path
import json
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("./data")

RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = PARSED_DIR / "chunks.jsonl"



def load_papers() -> List[Dict[str,Any]]:
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    all_papers: List[Dict[str,Any]] = []
    
    for paper_idx, pdf_path in enumerate(pdf_files,start=1):
        loader = PyMuPDFLoader(file_path=str(pdf_path))
        page_docs = list(loader.lazy_load()) 

        paper_id = f"paper_{paper_idx:03d}"
        
        paper_record: Dict[str,Any] = {
            "paper_id": paper_id,
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "total_pages_loaded": len(page_docs),
            "pages": []
        }
        
        for page_idx, page_doc in enumerate(page_docs):
            page_record = {
                "page_number": page_doc.metadata.get("page",page_idx),
                "total_pages": page_doc.metadata.get("total_pages"),
                "source": page_doc.metadata.get("source"),
                "text": page_doc.page_content or "",
                "raw_metadata": page_doc.metadata,
            }
            paper_record["pages"].append(page_record)
        all_papers.append(paper_record)
    
    print(f"Loaded {len(all_papers)} papers from {RAW_DIR}")
    return all_papers

def clean_page_text(text: str) -> str:
    """Light text cleanup for PDF-extracted content."""
    if not text:
        return ""

    # Remove NUL bytes first
    t = text.replace("\x00", "")

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple blank lines
    lines = [line.strip() for line in t.split("\n")]
    cleaned_lines = []
    blank_streak = 0

    for line in lines:
        if line == "":
            blank_streak += 1
            if blank_streak == 1:
                cleaned_lines.append("")
        else:
            blank_streak = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def build_paper_text(paper: Dict[str,Any]) -> str:
    # merged cleaned page texts into single paper-level string.
    cleaned_pages: List[str] = []
    
    for page in paper["pages"]:
        cleaned = clean_page_text(page["text"])
        if not cleaned:
            continue
        cleaned_pages.append(cleaned)
    paper_text = "\n\n".join(cleaned_pages).strip()
    return paper_text

def chunk_papers() -> None:
    papers = load_papers()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150
    )
    
    total_chunks = 0
    
    with CHUNKS_PATH.open("w", encoding="utf-8") as f_out:
        for paper in papers:
            paper_id=paper["paper_id"]
            file_name = paper["file_name"]
            source = paper["file_path"]
            
            paper_text = build_paper_text(paper)
            if not paper_text:
                print(f"[WARN] Paper {paper_id} ({file_name}) has empty cleaned text; skipping")
                continue
            
            from langchain_classic.schema import Document
            
            base_doc = Document(
                page_content=paper_text,
                metadata = {
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "source": source,
                }
            )
            chunk_docs = text_splitter.split_documents([base_doc])
            
            total_for_paper = len(chunk_docs)
            
            print(f"{paper_id}: {file_name} -> {total_for_paper} chunks")
            
            for chunk_idx, doc in enumerate(chunk_docs, start=1):
                chunk_id = f"{paper_id}_chunk_{chunk_idx:04d}"
                
                chunk_obj = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "source": source,
                    "chunk_index": chunk_idx,
                    "total_chunks_for_paper": total_for_paper,
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                
                f_out.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                total_chunks += 1
    print(f"Finished chunking: {total_chunks} chunks written to {CHUNKS_PATH}")

if __name__ == "__main__":
    chunk_papers()