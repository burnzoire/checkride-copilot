"""
Post-STT correction pass for military aviation terminology.

Whisper transcribes phonetically and has no knowledge of military designations,
so it consistently mishears them the same way. This module maps those known
misreadings back to the correct terms before the transcript reaches the LLM.

Add new entries whenever you spot a recurring misread in the logs.
"""

import re

# Each entry: (compiled regex, replacement string)
# Replacements run in order — put more specific patterns before general ones.
_RULES: list[tuple[re.Pattern, str]] = [
    # ── Missile designations ────────────────────────────────────────────────
    # AGM-XX: Whisper hears "age em", "a g m", "aim", "in age em"
    (re.compile(r"\bin\s+age\s+em[-\s]?(\d+)\b",       re.I), r"AGM-\1"),
    (re.compile(r"\bage\s+em[-\s]?(\d+)\b",             re.I), r"AGM-\1"),
    (re.compile(r"\ba\.?g\.?m\.?[-\s]?(\d+)\b",         re.I), r"AGM-\1"),
    # AIM-XX: "aim", "a i m"
    (re.compile(r"\ba\.?i\.?m\.?[-\s]?(\d+\w*)\b",      re.I), r"AIM-\1"),
    # AMRAAM: "a metric", "am ram", "aim ram", "aim raam"
    (re.compile(r"\b(a\s+metric|am\s*r?aam|aim\s+r?aam|aim\s+ram)\b", re.I), "AMRAAM"),
    # Sidewinder: occasionally "side winder"
    (re.compile(r"\bside\s+winder\b",                   re.I), "Sidewinder"),
    # Maverick: colloquial plural or mishear → canonical name + designation for BM25
    (re.compile(r"\bmavericks\b",                        re.I), "Maverick AGM-65"),
    (re.compile(r"\bmetrics\b",                          re.I), "Maverick AGM-65"),
    (re.compile(r"\bfabrics\b",                          re.I), "Maverick AGM-65"),
    (re.compile(r"\bmatrix\b",                           re.I), "Maverick AGM-65"),

    # ── Avionics / systems ───────────────────────────────────────────────────
    # TGP: "tea gee pee", "t g p", "t.g.p"
    (re.compile(r"\b(tea\s*gee\s*pee|t[\s.]g[\s.]p\.?)\b", re.I), "TGP"),
    # HUD: "hyud", "h u d", "h.u.d"
    (re.compile(r"\b(hyud|h[\s.]u[\s.]d\.?)\b",             re.I), "HUD"),
    # MFD: "m f d", "m.f.d"
    (re.compile(r"\bm[\s.]f[\s.]d\.?\b",                     re.I), "MFD"),
    # NATOPS: "nay tops", "nay taps"
    (re.compile(r"\bnay\s*t[ao]ps?\b",                   re.I), "NATOPS"),
    # TACAN: "tay can", "taken"
    (re.compile(r"\b(tay\s*can|takin|taken)\b",          re.I), "TACAN"),
    # ILS: "i l s"
    (re.compile(r"\bi\.?l\.?s\.?\b",                     re.I), "ILS"),
    # FLIR / TGP pod names
    (re.compile(r"\bat\s*flair\b",                       re.I), "ATFLIR"),
    (re.compile(r"\bat\s*flir\b",                        re.I), "ATFLIR"),

    # ── Refueling probe ──────────────────────────────────────────────────────
    # "refuel pro", "refuel probe", "re-fuel probe" → "refueling probe"
    (re.compile(r"\brefuel(?:ing)?\s+pro\b",              re.I), "refueling probe"),
    (re.compile(r"\brefuel\s+probe\b",                    re.I), "refueling probe"),

    # ── OBOGS / oxygen ───────────────────────────────────────────────────────
    # Whisper hears "o-bogs" → "bug", "oh bogs", "obogs"
    (re.compile(r"\b(bug\s+switch|oh\s*bogs|o[\s-]bogs)\b", re.I), "OBOGS"),

    # ── Common acronym mishears ──────────────────────────────────────────────
    # FOV (Field of View) → "FOB" (Forward Operating Base)
    (re.compile(r"\bF\.?O\.?B\.?\b",                      re.I), "FOV"),
    # JDAM → "Jay Dam", "J-Dam"
    (re.compile(r"\bj[-\s]dam\b",                         re.I), "JDAM"),
    # CCRP → "C C R P", CCIP → "C C I P"
    (re.compile(r"\bc\.?c\.?r\.?p\.?\b",                  re.I), "CCRP"),
    (re.compile(r"\bc\.?c\.?i\.?p\.?\b",                  re.I), "CCIP"),
    # SMS → "S M S"
    (re.compile(r"\bs\.?m\.?s\.?\b",                      re.I), "SMS"),

    # ── Targeting / seeker terms ─────────────────────────────────────────────
    # "seeker" → "soccer", "seaker", "seeker" (small.en phonetic mishear)
    (re.compile(r"\bsoccer\b",                            re.I), "seeker"),
    (re.compile(r"\bseaker\b",                            re.I), "seeker"),
    # "uncage" → "on cage", "un-cage", "uncaged" → "on caged"
    (re.compile(r"\bon\s+caged?\b",                       re.I), r"uncage"),
    (re.compile(r"\bun[-\s]caged?\b",                     re.I), r"uncaged"),
    # "slaved" → "slaved" is usually correct; "slave" → sometimes "sleeve"
    (re.compile(r"\bsleeve\s+(?:the\s+)?maverick\b",      re.I), "slave the Maverick"),
    # "TGP" mishears: "tea gee pee" handled above; also "t-g-p"
    (re.compile(r"\bt[-\s]g[-\s]p\b",                     re.I), "TGP"),
    # "IR" → "eye are", "i.r."
    (re.compile(r"\beye\s+are\b",                         re.I), "IR"),
    (re.compile(r"\bi\.r\.\b",                            re.I), "IR"),

    # ── Cockpit controls ─────────────────────────────────────────────────────
    # AOA / AoA: "a o a", "angle of attack" is fine
    (re.compile(r"\ba\.?o\.?a\.?\b",                     re.I), "AoA"),
    # UFC: "u f c"
    (re.compile(r"\bu\.?f\.?c\.?\b",                     re.I), "UFC"),
    # DDI: "d d i"
    (re.compile(r"\bd\.?d\.?i\.?\b",                     re.I), "DDI"),
    # MPCD: "m p c d"
    (re.compile(r"\bm\.?p\.?c\.?d\.?\b",                 re.I), "MPCD"),
]


def correct(text: str) -> str:
    """Apply all correction rules to a raw Whisper transcript."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
