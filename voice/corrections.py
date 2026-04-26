"""
Post-STT correction pass for military aviation terminology.

Whisper transcribes phonetically and has no knowledge of military designations,
so it consistently mishears them the same way. This module maps those known
misreadings back to the correct terms before the transcript reaches the LLM.

Add new entries whenever you spot a recurring misread in the logs.
"""

import re
import json
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

# CVN hull number-word table — used to normalise spoken numbers after "CVN".
# e.g. "CVN seventy three" → "CVN-73"
_CVN_HULL_NUM_WORDS: dict[str, str] = {
    "sixty eight": "68",  "sixty nine": "69",
    "seventy":     "70",
    "seventy one":   "71", "seventy two":   "72", "seventy three": "73",
    "seventy four":  "74", "seventy five":  "75", "seventy six":   "76",
    "seventy seven": "77", "seventy eight": "78", "seventy nine":  "79",
}
_CVN_NUM_WORD_RE = re.compile(
    r"\bCVN\s*-?\s*(" +
    "|".join(re.escape(k) for k in sorted(_CVN_HULL_NUM_WORDS, key=len, reverse=True)) +
    r")\b",
    re.I,
)

# Each entry: (compiled regex, replacement — string or callable).
# Replacements run in order — put more specific patterns before general ones.
_RULES: list[tuple] = [
    # ── Carrier comms phraseology ───────────────────────────────────────────
    # "call the ball" often misheard as "cull the bull" or similar.
    (re.compile(r"\b(cull|coal|call)\s+the\s+(bull|ball)\b", re.I), "call the ball"),
    # Spelled-out letters: "C V N" → "CVN" (must run before hull-number rules)
    (re.compile(r"\bC\s+V\s+N\b", re.I), "CVN"),
    # Parakeet mishears "CVN" as "Syvian" (phonetic drift on the consonant cluster)
    # e.g. "Syvian seventy three" → "CVN seventy three" → (num_word rule) → "CVN-73"
    (re.compile(r"\bSyvian\b", re.I), "CVN"),
    # CVN hull numbers: Parakeet occasionally mishears "CVN" leading letter
    # e.g. "CBN-74" → "CVN-74", "CDN-71" → "CVN-71"
    (re.compile(r"\b[CcGg][BbDdNn][NnMm][-\s]?(\d{1,3})\b"), r"CVN-\1"),
    # CVN spoken number words: "CVN seventy three" → "CVN-73"
    (_CVN_NUM_WORD_RE, lambda m: f"CVN-{_CVN_HULL_NUM_WORDS[m.group(1).lower()]}"),

    # ── Missile designations ────────────────────────────────────────────────
    # GBU-XX: Whisper occasionally hears "GPU" for "GBU"
    (re.compile(r"\bg\.?p\.?u\.?[-\s]?(\d+\w*)\b",      re.I), r"GBU-\1"),
    # AGM-XX: Whisper hears "age em", "a g m", "aim", "in age em"
    (re.compile(r"\bin\s+age\s+em[-\s]?(\d+)\b",       re.I), r"AGM-\1"),
    (re.compile(r"\bage\s+em[-\s]?(\d+)\b",             re.I), r"AGM-\1"),
    (re.compile(r"\ba\.?g\.?m\.?[-\s]?(\d+)\b",         re.I), r"AGM-\1"),
    # AIM-XX: "aim", "a i m"
    (re.compile(r"\ba\.?i\.?m\.?[-\s]?(\d+\w*)\b",      re.I), r"AIM-\1"),
    # AMRAAM: "a metric", "am ram", "aim ram", "aim raam", "MRAM", "m-ram"
    (re.compile(r"\b(a\s+metric|am\s*r?aam|aim\s+r?aam|aim\s+ram|m[-\s]?ram)\b", re.I), "AMRAAM"),
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
    # TACAN: "tay can", "taken", "tackan", "dukhan", "tekken", "tchakken", "tacon", "t-can"
    # Parakeet-specific: "tachahan", "takan", "take can"
    # Multi-word Parakeet mishears of "TACAN frequency": "attack-end frequency", "take care frequency"
    (re.compile(r"\battack[\s-]end\s+(freq\w*|channel)\b", re.I), r"TACAN \1"),
    (re.compile(r"\btake\s+care\s+(freq\w*|channel)\b",    re.I), r"TACAN \1"),
    (re.compile(r"\b(tay\s*can|takin|taken|tackan|taccan|tac\s*on|t[\-\s]can|dukhan|tekken|tchakken|t[ae]k[ae]n|tachahan|takan|take\s*can|tackhand|tekhan|tack\s+and)\b", re.I), "TACAN"),
    # ILS: "i l s"
    (re.compile(r"\bi\.?l\.?s\.?\b",                     re.I), "ILS"),
    # Countermeasures: "chaffin" → "chaff and", "measures" → "countermeasures"
    (re.compile(r"\bchaffin\b",                            re.I), "chaff and"),
    (re.compile(r"\b(kind of\s+)?measures\s+dispense",       re.I), r"\1countermeasures dispense"),
    (re.compile(r"\bdispense\s+measures\b",                   re.I), "dispense countermeasures"),

    # HOTAS: spoken as letters or phonetic spellings
    (re.compile(r"\bh[\s.\-]?o[\s.\-]?t[\s.\-]?a[\s.\-]?s\b", re.I), "HOTAS"),
    (re.compile(r"\b(hotass|ho\s*tas|hoe\s*tas)\b",        re.I), "HOTAS"),
    (re.compile(r"\bhotez\b",                                 re.I), "HOTAS"),
    (re.compile(r"\bhotaz\b",                                 re.I), "HOTAS"),
    # "HOTAS" often misheard as "hotels" in short phrases like "on my HOTAS"
    (re.compile(r"\b(my|your|the|on|with|use)\s+hotels\b",  re.I), r"\1 HOTAS"),
    # FLIR / TGP pod names
    (re.compile(r"\bat\s*flair\b",                       re.I), "ATFLIR"),
    (re.compile(r"\bat\s*flir\b",                        re.I), "ATFLIR"),
    # "flare switch/pod/sensor" → "FLIR switch/pod/sensor"
    # Disambiguates FLIR (sensor) from flare (countermeasure) by trailing context.
    # "flare" alone (e.g. "dispense flares") is intentionally left untouched.
    (re.compile(r"\bfl(?:are|ay|elia|eir|eer|ear)\s+(switch|pod|sensor|display|page|button)\b", re.I), r"FLIR \1"),
    # Whisper phonetic mishears for standalone FLIR when no trailing noun
    # e.g. "the FLIR" heard as "the flay", "the felia"
    (re.compile(r"\b(flay|felia|fleir|flear)\b",            re.I), "FLIR"),

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

    # ── Radio frequency bands (phonetic alphabet) ──────────────────────────────
    # "Uniform" (phonetic U) → UHF
    (re.compile(r"\b(?:uniform|u\.?h\.?f\.?)\s+(?:freq|frequency|freak|channel)\b", re.I), "UHF frequency"),
    (re.compile(r"\buniform\b(?!\s+ground)", re.I), "UHF"),
    # "Victor" (phonetic V) → VHF
    (re.compile(r"\b(?:victor|v\.?h\.?f\.?)\s+(?:freq|frequency|freak|channel)\b", re.I), "VHF frequency"),
    (re.compile(r"\bvictor\s+(?:high|hi)\b", re.I), "VHF_HI"),
    (re.compile(r"\bvictor\s+(?:low)\b", re.I), "VHF_LOW"),
    (re.compile(r"\bvictor\b", re.I), "VHF"),
]

_AIRFIELDS_PATH = Path(__file__).parent.parent / "data" / "airfields_by_map.json"


@lru_cache(maxsize=1)
def _airfield_alias_to_canonical() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    try:
        data = json.loads(_AIRFIELDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return alias_map

    for airfields in data.get("maps", {}).values():
        for entry in airfields:
            canonical = str(entry.get("canonical", "")).strip()
            if not canonical:
                continue
            canonical_key = canonical.lower()
            alias_map[canonical_key] = canonical
            for alias in entry.get("aliases", []) or []:
                key = str(alias).strip().lower()
                if key:
                    alias_map[key] = canonical
    return alias_map


def _best_airfield_name(raw: str) -> str | None:
    alias_map = _airfield_alias_to_canonical()
    if not alias_map:
        return None

    candidate = raw.strip().lower()
    candidate = re.sub(r"^(?:nearby|the|a|an|at|to|from|for)\s+", "", candidate)
    candidate = re.sub(r"^(?:nearby|the|a|an|at|to|from|for)\s+", "", candidate)
    if not candidate:
        return None

    exact = alias_map.get(candidate)
    if exact:
        return exact

    choices = list(alias_map.keys())
    matches = get_close_matches(candidate, choices, n=1, cutoff=0.72)
    if not matches:
        return None
    return alias_map[matches[0]]


_AIRFIELD_SUFFIX_RE = re.compile(
    r"\b([a-z][a-z\-']*(?:\s+[a-z][a-z\-']*){0,2})\s+"
    r"(traffic|tower|airfield|airport)\b",
    re.I,
)


def _normalize_airfield_names(text: str) -> str:
    """Normalize probable airfield names near radio/landing words."""
    lowered = text.lower()
    if not re.search(r"\b(land|landing|inbound|radio|traffic|tower|airfield|airport|divert)\b", lowered):
        return text

    def _replace(match: re.Match) -> str:
        place = match.group(1)
        suffix = match.group(2)
        prefix_match = re.match(r"^((?:(?:nearby|the|a|an|at|to|from|for)\s+)*)", place, re.I)
        prefix = prefix_match.group(1) if prefix_match else ""
        core_place = place[len(prefix):] if prefix else place
        canonical = _best_airfield_name(core_place)
        if not canonical:
            return match.group(0)
        return f"{prefix}{canonical} {suffix}"

    return _AIRFIELD_SUFFIX_RE.sub(_replace, text)


def correct(text: str) -> str:
    """Apply all correction rules to a raw Whisper transcript."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return _normalize_airfield_names(text)
