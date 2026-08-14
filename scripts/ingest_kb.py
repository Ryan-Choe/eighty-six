"""Build the Chroma index from data/kb/. Knowledge-base docs land on Day 3;
until then this is a no-op so `make seed` works end to end."""

import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

if __name__ == "__main__":
    docs = list(KB_DIR.rglob("*.md"))
    if not docs:
        print("kb ingest: no docs yet (KB authoring is a Day-3 task) — skipping")
        sys.exit(0)
    print(f"kb ingest: found {len(docs)} docs — indexing not wired yet")
