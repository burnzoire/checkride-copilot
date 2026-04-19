#!/usr/bin/env python3
"""
Extract numbered procedures from DCS FA-18C Hornet Guide PDF.

For each TOC leaf section, scans the page range for numbered step lists
(lines like "1. Do this") and emits structured procedure entries.

Run:
    python -m data.airframes.fa18c.build_procedures

Output: data/airframes/fa18c/procedures.json
"""

import json
import re
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    sys.exit("pip install pymupdf")


PDF_PATH  = Path(".training/DCS FA-18C Hornet Guide.pdf")
OUT_DIR   = Path("data/airframes/fa18c/procedures")
INDEX_PATH = OUT_DIR / "index.json"

# Must have at least this many steps to emit as a procedure
MIN_STEPS = 3

# TOC titles to skip — not actionable procedures
_SKIP_RE = re.compile(
    r"^(introduction|overview|section structure|lingo|terminology|"
    r"controls setup|display\b|symbology|specifications|"
    r"table of contents|disclaimer|video tutorial|resources|tips|"
    r"my (weapons|sensors)|information display|pen vs fan|"
    r"doppler|annunciator|hafu|plid|bullseye|general landing|"
    r"lso communications|performance spec|armament matrix|"
    r"section \d|function breakdown)",
    re.IGNORECASE,
)

# Lines that look like step numbers: "1.", "2.", "10." etc.
_STEP_LINE_RE = re.compile(r"^\s*(\d{1,2})\.\s+(.{5,})$")

# Category inference from section path
_CATEGORY_MAP = [
    (re.compile(r"start.?up",          re.I), "normal_ops"),
    (re.compile(r"takeoff|catapult",   re.I), "normal_ops"),
    (re.compile(r"landing|recovery|icls|acls|case (i|ii|iii)", re.I), "normal_ops"),
    (re.compile(r"refuel",             re.I), "normal_ops"),
    (re.compile(r"engine relight|fire suppress|emergency", re.I), "emergency"),
    (re.compile(r"fuel dump",          re.I), "systems"),
    (re.compile(r"autopilot|afcs|atc", re.I), "systems"),
    (re.compile(r"waypoint|markpoint|tacan|navigation|hsi|adf", re.I), "navigation"),
    (re.compile(r"radar|rws|tws|acm|sttt|aacq|map mode|exp\d|gmt|sea mode", re.I), "sensors"),
    (re.compile(r"targeting pod|litening|atflir|tgp|lasing|lst", re.I), "sensors"),
    (re.compile(r"jhmcs|helmet",       re.I), "sensors"),
    (re.compile(r"datalink|iff|plid",  re.I), "systems"),
    (re.compile(r"countermeasure|rwr|jammer|aspj", re.I), "defense"),
    (re.compile(r"radio|arc.210",      re.I), "comms"),
    (re.compile(r"mk.82|mk.20|rocket|gun|ccip|ccrp|unguided", re.I), "weapons"),
    (re.compile(r"gbu|jdam|jsow|paveway|laser.guided", re.I), "weapons"),
    (re.compile(r"agm.65|maverick",    re.I), "weapons"),
    (re.compile(r"agm.88|harm",        re.I), "weapons"),
    (re.compile(r"agm.84|harpoon|slam", re.I), "weapons"),
    (re.compile(r"aim.9|sidewinder",   re.I), "weapons"),
    (re.compile(r"aim.7|sparrow",      re.I), "weapons"),
    (re.compile(r"aim.120|amraam",     re.I), "weapons"),
    (re.compile(r"walleye|tald|jettison", re.I), "weapons"),
]


def _infer_category(title: str, ancestors: list[str]) -> str:
    full = " ".join(ancestors + [title])
    for pattern, cat in _CATEGORY_MAP:
        if pattern.search(full):
            return cat
    return "general"


def _slug(title: str) -> str:
    # Strip leading section numbers like "2.7.1 - " before slugifying.
    clean = re.sub(r"^[\d.\s\u2013\u2014\-]+", "", title).strip()
    s = re.sub(r"[^a-z0-9]+", "_", (clean or title).lower()).strip("_")
    return s[:64]


def _clean(text: str) -> str:
    # collapse runs of whitespace, strip leading/trailing
    return re.sub(r"\s+", " ", text).strip()


# Words that look like acronyms but are button states, directions, common English
# words, DCS keybinds, or other noise found all-caps in the PDF.
_TERM_NOISE = frozenset([
    # Directions / states
    "ON", "OFF", "IN", "OUT", "UP", "DOWN", "LEFT", "RIGHT", "AFT", "FWD",
    "OPEN", "CLOSE", "CLOSED", "OPEN", "LOCK", "LOCKED", "UNLOCK", "UNLOCKED",
    "ARMED", "DISARMED", "SAFE", "READY", "STANDBY", "STBY", "OPERATE",
    "ENGAGED", "DISENGAGED", "SPREAD", "FOLD", "FOLDED",
    # Common English words that appear in ALL-CAPS in this PDF
    "SET", "NOT", "ALL", "NOW", "YES", "NO", "MAX", "MIN", "OK", "GO",
    "TO", "DO", "IF", "OR", "AND", "THE", "FOR", "BUT", "AS", "BY",
    "GET", "ADD", "NEW", "USE", "RUN", "SEE", "TRY", "NOTE",
    "PRESS", "CLICK", "HOLD", "WAIT", "DONE", "PUSH", "PULL",
    "NORMAL", "MANUAL", "AUTO", "AUTOMATIC",
    "GROUND", "CARRIER", "FIELD", "SHORE", "LAUNCH", "LAND",
    "FIRE", "FUEL", "POWER", "HEAT", "HALF", "FULL", "RESET",
    "STOP", "START", "IDLE", "MASTER", "PROCEDURE", "MENU",
    "TEST", "STEP", "NEXT", "BACK", "RETURN", "ENTER", "EXIT",
    "ROLL", "PITCH", "YAW", "TRIM", "LIGHTS", "CANOPY",
    "WING", "WINGS", "HOOK", "FLAPS",
    # DCS keybinds
    "RALT", "LALT", "LSHIFT", "RSHIFT", "LCTRL", "RCTRL",
    "SPACE", "DEL", "INS", "HOME", "END", "PUSHED", "PULLED",
    # Single/double letters and Fn keys handled by length filter
    "FA", "PC", "XP", "LVL", "HDG",
    # Plain English words that appear ALL_CAPS in this PDF
    "ALIGN", "ALIGNING", "RADAR", "CAUTION", "PITOT", "STROBE",
    "DISPENSER", "UNLK", "BINGO", "PBIT", "QUAL", "SUPT",
    "ENT", "BLK", "GND", "GRND", "RDR", "RDY", "PWR", "AIC",
    "FAILURES", "ABOUT", "CASE", "SPEED", "FORGET", "FLARE",
    "RECOVERY", "CLR", "COMM1", "IFLOS",
    # Noise specific to this PDF
    "DCS", "WOW", "TGT", "NXT", "WPT", "STD", "ATT", "ALT",
    "TIMEUFC", "STORES", "DISPLAYS", "FIELD", "POSITION", "SPREAD",
    "COMM", "BLEED", "CANOPY", "LANDING", "PARK", "DAY", "NGT",
    "BRT", "HALF", "CLOSE", "BYPASS",
    # Shore/altitude terms not avionics acronyms
    "ALTITUDE", "GAIN", "HIGH", "LOW", "MIDDLE", "RETRACTED",
    "NORM", "COOL",   # button labels, not acronyms
])

# Pattern for inline expansions: "SMS (Stores Management System)"
# Expansion must be ≥8 chars and start with a capital letter.
_INLINE_EXPANSION_RE = re.compile(
    r"\b([A-Z][A-Z0-9/\-]{1,8})\s+\(([A-Z][^)]{7,50})\)"
)
# Standalone acronyms (2–10 uppercase chars, may contain digits, hyphens, slashes)
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:[/\-][A-Z0-9]+)*)\b")


def _extract_terminology(steps: list[dict]) -> list[str]:
    """Return sorted unique terminology items found in procedure step text."""
    full_text = " ".join(s["action"] for s in steps)

    terms: dict[str, str] = {}  # acronym → expansion (empty if none found)

    # Inline expansions take priority
    for m in _INLINE_EXPANSION_RE.finditer(full_text):
        acro, expansion = m.group(1), m.group(2).strip()
        if acro in _TERM_NOISE:
            continue
        # Skip button-state descriptions (not definitions)
        if re.search(
            r"\b(click|right|left|middle|upper|lower|position|boxed|should be|two times|"
            r"up\b|down\b|high\b|low\b)\b",
            expansion, re.I,
        ):
            continue
        # Skip expansions that end mid-word (truncated by line break)
        if expansion.endswith(("-", "/")):
            continue
        terms[acro] = expansion

    # Standalone acronyms (only add if not already captured with expansion)
    for m in _ACRONYM_RE.finditer(full_text):
        acro = m.group(1)
        if acro in _TERM_NOISE:
            continue
        # Require ≥3 chars for unconfirmed acronyms; 2-char only via inline expansion
        if len(acro) < 3:
            continue
        # Skip Fn keys (F1–F12)
        if re.fullmatch(r"F\d{1,2}", acro):
            continue
        # Skip pure numbers
        if re.fullmatch(r"\d+", acro):
            continue
        # Skip slash-joined compound labels (e.g. AFT/FWD/LEFT/RIGHT, LANDING/TAXI, DX/DY)
        if "/" in acro and not re.search(r"\d", acro):  # allow A/G, A/A, AGM-65F/G
            continue
        # Skip inflected English words masquerading as acronyms
        if len(acro) >= 6 and re.search(r"(ING|MENT|TION|ATED|NESS|IGHT|IGHT|CHED)$", acro):
            continue
        if acro not in terms:
            terms[acro] = ""

    # Format: "SMS – Stores Management System" or just "DDI"
    result = []
    for acro, expansion in sorted(terms.items()):
        result.append(f"{acro}: {expansion}" if expansion else acro)
    return result


# Step formats found in this PDF:
#   "1.\n<text>"   — single-digit: number alone on a line, text on the next line(s)
#   "10. <text>"   — multi-digit: number and text on the same line
_STEP_ALONE_RE  = re.compile(r"^\s*(\d{1,2})\.\s*$")
_STEP_INLINE_RE = re.compile(r"^\s*(\d{1,2})\.\s+(.{4,})$")

# Lines that are clearly noise: bare page numbers, section headers, figure labels.
# Must NOT match step-number lines like "1." or "10. text".
_NOISE_RE = re.compile(
    r"^("
    r"\d{1,3}(?!\.)$"          # bare page number (digit(s) not followed by a period)
    r"|F/A-18C"
    r"|HORNET"
    r"|PART \d+"
    r"|\d+[a-c]$"              # figure labels like "15a", "15b"
    r")",
    re.IGNORECASE,
)

# Lines that are running section headers from PDF footers/headers — stop absorbing.
# e.g. "2.7.1 – AGM-65F/G MAVERICK (IR-MAVF, ...)"
_SECTION_HDR_LINE_RE = re.compile(r"^\d+\.\d[\d.]*\s*[–\-\u2013\u2014]\s*[A-Z]")

# Inline section-number header that bleeds into action text mid-string.
_INLINE_SECTION_RE = re.compile(r"\s+\d+\.\d[\d.]*\s*[–\-\u2013\u2014]\s*[A-Z]")

# ALL-CAPS running section title at the end of text (PDF footer), e.g. "START-UP PROCEDURE".
# Requires ≥ 2 all-caps words separated by space/hyphen.
_CAPS_SECTION_TAIL_RE = re.compile(r"\s+[A-Z]{2,}(?:[\s/\-][A-Z]{2,})+\s*$")

# Known figure-caption phrases (appear after action text or as standalone lines).
_FIGURE_PHRASE_RE = re.compile(
    r"\s+Alignment Time Remaining"
    r"|\s+Press two times\b",
    re.IGNORECASE,
)

# Lowercase common words used in real sentences (articles, preps, verbs, conjunctions).
_SENTENCE_WORDS = frozenset(
    "a an the is are was were will be been being have has had do does did not but and or if "
    "when then with for from on in at to of by as its this that these those which you your we "
    "our it its can may should must would could set press click select".split()
)


def _is_label_sequence(fragment: str) -> bool:
    """Return True if `fragment` looks like a figure annotation (label sequence, not prose)."""
    words = re.findall(r"[a-zA-Z]+", fragment)
    if len(words) < 3:
        return False
    # ALL_CAPS words are button-state indicators (IN, OFF, OUT, ARM), not prose words.
    prose_words = sum(1 for w in words if w.lower() in _SENTENCE_WORDS and not w.isupper())
    # If <20% of words are common prose words, it's a label sequence.
    return prose_words / len(words) < 0.20


def _strip_noise_tail(text: str) -> str:
    """Remove figure captions and running-header bleed-in from the end of a step."""
    # Cut at an inline section header (e.g. " 2.7.1 – AGM-65F/G...")
    m = _INLINE_SECTION_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    # Cut at known figure-caption phrases
    m = _FIGURE_PHRASE_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    # Cut at ALL-CAPS running footer (e.g. "  START-UP PROCEDURE")
    m = _CAPS_SECTION_TAIL_RE.search(text)
    if m:
        text = text[:m.start()].strip()
    # Cut figure-label sequences that follow a sentence-ending period
    period_idx = text.rfind(". ")
    if period_idx > 20:
        tail = text[period_idx + 2:]
        if _is_label_sequence(tail):
            text = text[:period_idx + 1].strip()
    return text


def _extract_steps(pages: list[fitz.Page]) -> list[dict]:
    """
    Find numbered step lists across a page range.

    Handles two formats used in this PDF:
      - Single-digit: "1." alone on a line, text on next line
      - Multi-digit:  "10. text on same line"
    """
    lines = []
    for page in pages:
        for line in page.get_text().splitlines():
            line = line.strip()
            if line and not _NOISE_RE.match(line):
                lines.append(line)

    def _is_step_start(line: str) -> tuple[int, str] | None:
        m = _STEP_INLINE_RE.match(line)
        if m:
            return int(m.group(1)), _clean(m.group(2))
        m = _STEP_ALONE_RE.match(line)
        if m:
            return int(m.group(1)), ""   # text comes on next line
        return None

    def _is_noise(line: str) -> bool:
        # Skip very short lines, figure references, all-caps labels
        if len(line) < 4:
            return True
        if re.match(r"^\d{1,2}[a-c]?$", line):
            return True
        return False

    # Parse into (num, text) pairs
    best_steps: list[dict] = []
    steps:      list[dict] = []
    expected = 1
    i = 0

    while i < len(lines):
        result = _is_step_start(lines[i])
        if result is not None:
            num, text = result
            if num == 1:
                # New list starting — if what we built is longer, keep it
                if len(steps) > len(best_steps):
                    best_steps = steps
                steps = []
                expected = 1

            if num == expected:
                if not text and i + 1 < len(lines):
                    # Single-digit format: text is on the next line
                    i += 1
                    text = _clean(lines[i])

                # Absorb continuation lines until next step or noise boundary
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if _is_step_start(nxt) is not None:
                        break
                    if _SECTION_HDR_LINE_RE.match(nxt):
                        break  # running section header from PDF footer
                    if _is_noise(nxt):
                        j += 1
                        continue
                    # Bullet sub-items: absorb into step detail
                    if re.match(r"^[•\-\*◆]", nxt):
                        text += " " + _clean(nxt.lstrip("•-*◆ "))
                    elif re.match(r"^Note:", nxt, re.I):
                        text += " (" + _clean(nxt) + ")"
                    else:
                        text += " " + _clean(nxt)
                    j += 1

                steps.append({"num": num, "action": _strip_noise_tail(text.strip())})
                expected = num + 1
                i = j
                continue
        i += 1

    if len(steps) > len(best_steps):
        best_steps = steps

    return best_steps


def _make_alternates(title: str, ancestors: list[str]) -> list[str]:
    alts = []
    # Strip section numbers like "2.5.1 - "
    clean = re.sub(r"^[\d\.\s\-]+", "", title).strip()
    if clean and clean.lower() != title.lower():
        alts.append(clean)
    # Abbreviation: take words longer than 3 chars
    words = re.findall(r"[A-Za-z]{4,}", title)
    if len(words) >= 2:
        alts.append(" ".join(words[:4]).lower())
    # Weapon names from ancestors
    for a in ancestors:
        m = re.search(r"(AGM-\d+\w*|AIM-\d+\w*|GBU-\d+\w*|MK-\d+|M61)", a, re.I)
        if m:
            alts.append(m.group(1).upper() + " " + clean)
            break
    return list(dict.fromkeys(a for a in alts if a))


def build(doc: fitz.Document) -> dict:
    toc = doc.get_toc()
    n_pages = len(doc)

    # Build list of (level, title, page_0indexed) with next_page computed
    entries = []
    for i, (level, title, page) in enumerate(toc):
        next_page = toc[i + 1][2] if i + 1 < len(toc) else n_pages + 1
        entries.append({
            "level": level,
            "title": title,
            "page":  page - 1,          # fitz is 0-indexed
            "end":   next_page - 1,
            "idx":   i,
        })

    procedures = {}

    for idx, entry in enumerate(entries):
        title = entry["title"]

        if _SKIP_RE.search(title):
            continue

        # Collect ancestor titles for context
        ancestors = []
        lvl = entry["level"]
        for prev in reversed(entries[:idx]):
            if prev["level"] < lvl:
                ancestors.insert(0, prev["title"])
                lvl = prev["level"]
                if prev["level"] == 1:
                    break

        page_start = entry["page"]
        page_end   = min(entry["end"], n_pages)
        if page_end <= page_start:
            page_end = page_start + 1

        pages = [doc[p] for p in range(page_start, min(page_end, n_pages))]
        steps = _extract_steps(pages)

        if len(steps) < MIN_STEPS:
            continue

        category = _infer_category(title, ancestors)
        alts     = _make_alternates(title, ancestors)
        key      = _slug(title)
        # Deduplicate keys
        if key in procedures:
            key += f"_{page_start}"

        # Best-effort section label from ancestors + title
        section_path = " > ".join(a for a in ancestors[-2:] + [title])

        canonical  = re.sub(r"^[\d\.\s\-]+", "", title).strip() or title
        step_dicts = [{"num": s["num"], "action": s["action"]} for s in steps]
        procedures[key] = {
            "key":         key,
            "canonical":   canonical,
            "alternates":  alts,
            "category":    category,
            "terminology": _extract_terminology(step_dicts),
            "source": {
                "doc":     "DCS FA-18C Hornet Guide",
                "page":    page_start + 1,
                "section": section_path,
            },
            "step_count": len(step_dicts),
            "steps":      step_dicts,
        }

    return {"airframe": "FA-18C", "procedures": procedures}


def _write_split(data: dict) -> None:
    """Write one JSON file per procedure (grouped by category) plus an index."""
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    index_entries = []
    for key, proc in data["procedures"].items():
        cat = proc["category"]
        cat_dir = OUT_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        proc_path = cat_dir / f"{key}.json"
        proc_path.write_text(json.dumps(proc, indent=2, ensure_ascii=False), encoding="utf-8")

        # Index entry: metadata only (no steps)
        index_entries.append({
            "key":        key,
            "canonical":  proc["canonical"],
            "alternates": proc["alternates"],
            "category":   cat,
            "source":     proc["source"],
            "step_count": proc["step_count"],
            "path":       f"{cat}/{key}.json",
        })

    index = {"airframe": data["airframe"], "procedures": index_entries}
    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    print(f"Opening {PDF_PATH} ...")
    doc = fitz.open(str(PDF_PATH))
    print(f"  {len(doc)} pages, {len(doc.get_toc())} TOC entries")

    data = build(doc)
    count = len(data["procedures"])
    print(f"  Extracted {count} procedures")

    _write_split(data)
    print(f"  Written {count} files to {OUT_DIR}/")
    print(f"  Index: {INDEX_PATH}")

    from collections import Counter
    cats = Counter(p["category"] for p in data["procedures"].values())
    for cat, n in sorted(cats.items()):
        print(f"    {cat:15s} {n}")


if __name__ == "__main__":
    main()
