"""Build the Chroma index from data/kb/. Part of `make seed`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eightysix.rag import CHROMA_DIR, build_index

if __name__ == "__main__":
    n = build_index()
    print(f"kb ingest: {n} chunks indexed into {CHROMA_DIR.name}/")
