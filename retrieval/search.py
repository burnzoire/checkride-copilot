"""
BM25 search over the pre-built chunks.json index.

Usage:
    from retrieval.search import search
    results = search("formation lights switch location", top_k=2)
    # [{"page": 42, "text": "...", "score": 3.14}, ...]

The index is loaded once and cached for the process lifetime.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

CHUNKS_PATH = Path(__file__).parent / "chunks.json"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _load_index():
    from rank_bm25 import BM25Okapi

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk index not found at {CHUNKS_PATH}. "
            "Run: python -m retrieval.ingest"
        )
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25   = BM25Okapi(corpus)
    return bm25, chunks


def search(query: str, top_k: int = 2) -> list[dict]:
    """Return top_k chunks most relevant to query, each with page + text + score."""
    bm25, chunks = _load_index()
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        range(len(chunks)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for i in ranked:
        if scores[i] > 0:
            results.append({**chunks[i], "score": float(scores[i])})
    return results
