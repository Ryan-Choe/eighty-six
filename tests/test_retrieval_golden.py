"""The retrieval golden set, run keyless as tests.

The same JSON that uploads to LangSmith via `make eval` runs here against the
local index, so a retrieval regression fails the build instead of waiting for
someone to read an experiment. Only local MiniLM embeddings -- no API keys.
"""

import json
from pathlib import Path

import pytest

from eightysix.rag import CHROMA_DIR, KB_DIR, retrieve

REPO = Path(__file__).resolve().parent.parent
GOLDEN = json.loads(
    (REPO / "evals" / "datasets" / "retrieval-v1.json").read_text()
)["examples"]


def test_every_golden_query_retrieves_its_chunk():
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        pytest.skip("Chroma index not built -- run `make seed` first")
    misses = []
    for example in GOLDEN:
        query = example["inputs"]["query"]
        want = {"source": example["outputs"]["source"],
                "section": example["outputs"]["section"]}
        got = [{"source": d.metadata["source"], "section": d.metadata["section"]}
               for d in retrieve(query, None, k=6)]   # the production k
        if want not in got:
            misses.append(f"  {query!r}\n    wanted {want['source']} § {want['section']}, "
                          f"top hits: {[(c['source'], c['section']) for c in got[:3]]}")
    assert not misses, "golden queries missing their chunk:\n" + "\n".join(misses)


def test_golden_set_covers_every_kb_doc():
    # a doc nobody queries is a doc whose retrieval can silently rot
    covered = {e["outputs"]["source"] for e in GOLDEN}
    assert covered == {p.stem for p in KB_DIR.rglob("*.md")}
