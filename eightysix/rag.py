"""Chroma index over the knowledge base.

The vector store holds only prose a human would write in sentences:
supplier terms and kitchen policies. Anything that fits in a WHERE clause
(prices, pack sizes, minimums as numbers) lives in SQLite instead, and the
columns win when the two disagree.
"""

import os
from functools import lru_cache
from pathlib import Path

# quiet the HF hub before anything imports transformers: progress bars and
# rate-limit warnings would otherwise print in the middle of the demo
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:  # belt over the env-var suspenders: some bars ignore the env vars
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:  # pragma: no cover - cosmetic only
    pass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "eighty_six_kb"

# Docs are authored with one idea per ## section, so heading-based splitting
# gives retrieval-sized chunks with zero tuning — and the heading doubles as
# the citation.
_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "doc_title"), ("##", "section")]
)


@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    # cached: loading MiniLM takes ~2s and retrieval runs several times a turn
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@lru_cache(maxsize=1)
def _store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def load_chunks(kb_dir: Path = KB_DIR) -> list[Document]:
    chunks: list[Document] = []
    for path in sorted(kb_dir.rglob("*.md")):
        doc_type = "supplier" if path.parent.name == "suppliers" else "policy"
        for chunk in _SPLITTER.split_text(path.read_text()):
            title = chunk.metadata.get("doc_title", "")
            section = chunk.metadata.get("section", title)
            # the splitter strips headers OUT of the text, so "Minimum order
            # is $250" carries no clue it's Roma's minimum -- three suppliers
            # have lookalike sections and embeddings can't tell them apart.
            # Re-inject the context so the chunk embeds (and reads) as whose
            # terms it is.
            chunk.page_content = f"{title} - {section}:\n{chunk.page_content}"
            chunk.metadata.update(
                source=path.stem, section=section, doc_type=doc_type,
            )
            chunks.append(chunk)
    return chunks


def build_index() -> int:
    """Rebuild the persisted index from data/kb/. Returns the chunk count."""
    chunks = load_chunks()
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.reset_collection()  # rebuild-from-scratch keeps `make seed` idempotent
    if chunks:
        store.add_documents(chunks)
    return len(chunks)


def retrieve(query: str, doc_type: str | None, k: int) -> list[Document]:
    # no default k on purpose: the right k is a call-site decision with an
    # eval behind it. Whole-KB queries run at 6 (policy_qa says why); the old
    # default of 4 was just the library's, and 4 is the k the lookalike-
    # sections bug lived at.
    flt = {"doc_type": doc_type} if doc_type else None
    return _store().similarity_search(query, k=k, filter=flt)


# Supplier display names -> KB file stems. Explicit map because
# "Valco Cash & Carry" does not normalize to "valco_cash_carry" by string
# munging, and a silent mismatch would quietly drop that supplier's terms.
SUPPLIER_DOCS = {
    "Roma Foods": "roma_foods",
    "Valco Cash & Carry": "valco_cash_carry",
    "Cascade Produce": "cascade_produce",
}


def retrieve_supplier_terms(supplier: str, k: int = 3) -> list[Document]:
    """Terms for ONE supplier, filtered by source doc — not similarity-ranked
    across all suppliers. Similarity alone let Roma's chunks outrank a
    supplier's own terms sheet, which would bury the pickup-only trap."""
    stem = SUPPLIER_DOCS.get(supplier)
    if stem is None:
        return retrieve(f"{supplier} delivery terms", "supplier", k=k)
    return _store().similarity_search(
        "delivery schedule pickup order cutoff minimum fees",
        k=k,
        filter={"$and": [{"doc_type": "supplier"}, {"source": stem}]},
    )
