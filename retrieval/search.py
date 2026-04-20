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

# Maps natural-language pilot verbs to NATOPS switch-action vocabulary.
# BM25 fails when the question says "open" but the manual says "EXTEND".
_EXPANSIONS: dict[str, list[str]] = {
    "open":      ["extend", "open"],
    "extend":    ["extend"],
    "retract":   ["retract", "close", "stow"],
    "close":     ["retract", "close"],
    "stow":      ["retract", "stow"],
    "deploy":    ["extend", "deploy", "lower", "down"],
    "raise":     ["retract", "raise", "up"],
    "lower":     ["lower", "extend", "down"],
    "activate":  ["arm", "on", "enable", "set"],
    "enable":    ["arm", "on", "enable", "set"],
    "turn":      ["switch", "set", "on", "off", "bright"],
    "start":     ["start", "engine", "crank", "ignition"],
    "use":       ["use", "employ", "fire", "launch", "release"],
    "fire":      ["release", "fire", "pickle"],
    "firing":    ["release", "fire", "pickle"],
    "fired":     ["release", "fire", "pickle"],
    "launch":    ["release", "fire", "pickle"],
    "launching": ["release", "fire", "pickle"],
    "launched":  ["release", "fire", "pickle"],
    "release":   ["release", "pickle", "drop"],
    "releasing": ["release", "pickle", "drop"],
    "press":     ["press", "depress", "pickle", "button"],
    "pressing":  ["press", "depress", "pickle", "button"],
    "pressed":   ["press", "depress", "pickle", "button"],
    "shoot":     ["release", "fire", "pickle"],
    "shooting":  ["release", "fire", "pickle"],
    "refuel":    ["refueling", "probe", "extend", "retract"],
    "probe":     ["probe", "refueling", "extend", "retract"],
    "trouble":   ["problem", "issue", "unable", "fail", "check"],
    "problem":   ["problem", "issue", "unable", "fail", "check"],
    "issue":     ["problem", "issue", "unable", "fail", "check"],
    "nothing":   ["fail", "unable", "check"],
    "works":     ["arm", "ready", "set"],
    "working":   ["arm", "ready", "set"],
}

# Chunks that are clearly intro/overview/TOC — high term density but no
# actionable content. Penalise them by zeroing their score.
_INTRO_RE = re.compile(
    r"INTRODUCTION|TABLE OF CONTENTS|PART \d+ \u2013|PART \d+ -",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _expand_query(query: str) -> list[str]:
    """Return an expanded token list by replacing pilot verbs with NATOPS synonyms."""
    tokens = _tokenize(query)
    expanded: list[str] = []
    for t in tokens:
        expanded.extend(_EXPANSIONS.get(t, [t]))
    return expanded


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


# Weapon/system keywords that anchor a query to a specific section family.
# If the query mentions one of these, chunks from matching sections get a score boost.
_SECTION_ANCHORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmaverick|agm.?65\b",    re.I), "MAVERICK"),
    (re.compile(r"\bwalleye|agm.?62\b",     re.I), "WALLEYE"),
    (re.compile(r"\bslam|agm.?84\b",        re.I), "SLAM"),
    (re.compile(r"\bharpoon|agm.?84a\b",    re.I), "HARPOON"),
    (re.compile(r"\bjdam|gbu.?3[18]\b",     re.I), "JDAM"),
    (re.compile(r"\bjsow|agm.?154\b",       re.I), "JSOW"),
    (re.compile(r"\bharm|agm.?88\b",        re.I), "HARM"),
    (re.compile(r"\bsidewinder|aim.?9\b",   re.I), "SIDEWINDER"),
    (re.compile(r"\bamraam|aim.?120\b",     re.I), "AMRAAM"),
    (re.compile(r"\btacan\b",               re.I), "TACAN"),
    (re.compile(r"\bils\b",                 re.I), "ILS"),
    (re.compile(r"\bcarrier|case\s*i|case\s*1|ball|groove|iflols|lso\b", re.I), "LANDING"),
    (re.compile(r"\brefuel|probe\b",        re.I), "REFUELING"),
]

_SECTION_BOOST = 4.0   # additive score bonus for chunks in the matched section


def _section_affinity(query: str) -> str | None:
    """Return the section keyword to boost, or None if no weapon anchor found."""
    for pattern, keyword in _SECTION_ANCHORS:
        if pattern.search(query):
            return keyword
    return None


def search(query: str, top_k: int = 2, min_score: float = 14.0) -> list[dict]:
    """Return top_k chunks most relevant to query, each with page + text + score.

    Chunks that are intro/overview sections are excluded — they score high on
    term frequency but contain no actionable switch-level information.
    When the query names a specific weapon/system, chunks from that section
    receive a score boost to prevent cross-weapon contamination.
    min_score filters out low-confidence hits that actively mislead the LLM.
    """
    bm25, chunks = _load_index()
    tokens = _expand_query(query)
    scores = bm25.get_scores(tokens)

    affinity = _section_affinity(query)

    masked_scores = []
    for i in range(len(chunks)):
        s = scores[i]
        if _INTRO_RE.search(chunks[i]["text"]):
            s = 0.0
        elif affinity and affinity in chunks[i].get("section", "").upper():
            s += _SECTION_BOOST
        masked_scores.append(s)

    ranked = sorted(
        range(len(chunks)), key=lambda i: masked_scores[i], reverse=True
    )[:top_k * 4]

    results = []
    seen_pages: set[int] = set()
    for i in ranked:
        if masked_scores[i] < min_score:
            continue
        page = chunks[i]["page"]
        if any(abs(page - p) <= 1 for p in seen_pages):
            continue
        seen_pages.add(page)
        results.append({**chunks[i], "score": float(masked_scores[i])})
        if len(results) >= top_k:
            break

    return results
