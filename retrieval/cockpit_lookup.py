"""
Fast switch-location lookup for F/A-18C cockpit queries.

Intercepts "where is X" questions before BM25. Returns the console area
and panel for a named switch, or None if no match.

Usage:
    from retrieval.cockpit_lookup import lookup
    result = lookup("probe switch")
    # {"canonical": "Refueling Probe Control Switch",
    #  "area": "left_console", "area_label": "Left Console",
    #  "panel": "probe_hook", "panel_label": "Probe / Hook Panel",
    #  "positions": ["RETRACT", "EXTEND", "EMERG EXTD"]}
"""

import json
import re
from functools import lru_cache
from pathlib import Path

_INDEX_PATH = Path(__file__).parent.parent / "data" / "airframes" / "fa18c" / "switches.json"

# Queries that are likely asking for a location/identity.
# Match these before running BM25 so we get a precise answer.
_WHERE_RE = re.compile(
    r"\b(where|which|what\s+(console|panel|side)|location|located|find|how\s+do\s+i\s+find)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    """Return flat list of switch records: {canonical, alternates, area, area_label, panel, panel_label, positions}."""
    data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    records: list[dict] = []
    for area_key, area in data["areas"].items():
        for panel_key, panel in area["panels"].items():
            for sw in panel["switches"]:
                records.append({
                    "canonical":        sw["canonical"],
                    "alternates":       sw.get("alternates", []),
                    "area":             area_key,
                    "area_label":       area["label"],
                    "panel":            panel_key,
                    "panel_label":      panel["label"],
                    "positions":        sw.get("positions", []),
                    "spoken":           sw.get("spoken", sw.get("positions", [])),
                    "spoken_location":  sw.get("spoken_location"),
                    "dcs_actions":      sw.get("dcs_actions"),
                })
    return records


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9/]+", text.lower()))


def lookup(query: str) -> dict | None:
    """
    Find the best-matching switch for a query string.

    Returns a result dict if confidence is high enough, else None.
    Matching strategy:
      1. Exact substring match on any alternate name or canonical name (case-insensitive)
      2. Token overlap score — best match above threshold wins
    """
    q_lower = query.lower()
    records = _load()

    # Pass 1: exact substring match on any name
    for rec in records:
        names = [rec["canonical"]] + rec["alternates"]
        for name in names:
            if name.lower() in q_lower:
                return rec

    # Pass 2: token overlap (Jaccard-ish)
    q_tokens = _tokenize(query)
    if not q_tokens:
        return None

    best_rec, best_score = None, 0.0
    for rec in records:
        names = [rec["canonical"]] + rec["alternates"]
        for name in names:
            n_tokens = _tokenize(name)
            if not n_tokens:
                continue
            overlap = len(q_tokens & n_tokens)
            score = overlap / len(n_tokens)
            if overlap >= 2 and score > best_score:
                best_score, best_rec = score, rec

    if best_score >= 0.5:
        return best_rec
    return None


def suggest(query: str) -> dict | None:
    """
    Loose match — returns the best candidate switch even with weak overlap.
    Used as a 'did you mean X?' hint when BM25 returns LOW confidence.
    Won't fire on generic words like 'switch' alone (min 1 specific token).

    Skips tokens that are too common to be discriminating.
    """
    _STOPWORDS = {"the", "a", "an", "is", "it", "on", "in", "at", "to", "do",
                  "i", "my", "me", "where", "what", "which", "how", "can",
                  "find", "turn", "use", "that", "this", "its", "there",
                  "select", "set", "put", "get", "go", "need", "want",
                  "switch", "panel", "button", "handle", "lever", "control"}

    q_tokens = _tokenize(query) - _STOPWORDS
    if not q_tokens:
        return None

    records = _load()
    q_lower = query.lower()

    # Pass 1: exact substring (same as lookup)
    for rec in records:
        names = [rec["canonical"]] + rec["alternates"]
        for name in names:
            if name.lower() in q_lower:
                return rec

    # Pass 2: any single meaningful token overlap
    best_rec, best_overlap = None, 0
    for rec in records:
        names = [rec["canonical"]] + rec["alternates"]
        for name in names:
            n_tokens = _tokenize(name) - _STOPWORDS
            overlap = len(q_tokens & n_tokens)
            if overlap > best_overlap:
                best_overlap, best_rec = overlap, rec

    return best_rec if best_overlap >= 1 else None


def format_location(result: dict) -> str:
    """Return a brief human-readable location string."""
    area = result["area_label"]
    panel = result["panel_label"]
    positions = result.get("positions", [])
    pos_str = " / ".join(positions) if positions else ""
    canonical = result["canonical"]
    if pos_str:
        return f"{canonical} — {area}, {panel}. Positions: {pos_str}."
    return f"{canonical} — {area}, {panel}."


def is_location_query(text: str) -> bool:
    """Heuristic: does this text look like a 'where is X' question?"""
    return bool(_WHERE_RE.search(text))
