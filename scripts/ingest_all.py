"""
CLI: ingest all PDFs in data/raw/ that are not yet indexed.

Usage
-----
    python scripts/ingest_all.py            # ingest everything new
    python scripts/ingest_all.py --dry-run  # list files that would be ingested

This script calls the same pipeline used by POST /api/ingest and POST /api/upload —
no duplicated logic.
"""
import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.services.pipeline import RAW_DIR, _already_indexed_sources, ingest_all_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs from data/raw/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested without actually processing them.",
    )
    args = parser.parse_args()

    if args.dry_run:
        with SessionLocal() as session:
            indexed = _already_indexed_sources(session)

        pdfs = sorted(RAW_DIR.glob("*.pdf"))
        new = [p for p in pdfs if str(p) not in indexed]

        if not new:
            print("✓ All PDFs in data/raw/ are already indexed.")
        else:
            print(f"Would ingest {len(new)} file(s):")
            for p in new:
                print(f"  • {p.name}")
        return

    # ── Real ingestion ────────────────────────────────────────────────────────
    print("Scanning data/raw/ …")
    with SessionLocal() as session:
        result = ingest_all_raw(session)

    ingested = result["ingested"]
    skipped  = result["skipped"]
    failed   = result["failed"]

    if ingested:
        print(f"\n✓ Ingested {len(ingested)} paper(s):")
        for r in ingested:
            print(f"  • {r['file']}  →  {r['chunks']} chunks  ({r['paper_id']})")

    if skipped:
        print(f"\n– Skipped {len(skipped)} already-indexed file(s):")
        for name in skipped:
            print(f"  • {name}")

    if failed:
        print(f"\n✗ Failed {len(failed)} file(s):")
        for r in failed:
            print(f"  • {r['file']}: {r['reason']}")
        sys.exit(1)

    if not ingested and not failed:
        print("✓ Nothing new to ingest.")


if __name__ == "__main__":
    main()
