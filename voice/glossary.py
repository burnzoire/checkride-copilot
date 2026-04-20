"""
Aviation glossary — maps written abbreviations/phrases to spoken forms.

Applied in tts._normalize() before synthesis so Kokoro hears natural speech.
Two mechanisms:
  GLOSSARY  — phrase/abbreviation → spoken replacement (longest match first)
  _WORD_CAPS — all-caps words that are normal English words, not acronyms;
               lowercased so the TTS doesn't spell them out letter by letter.
"""

from __future__ import annotations
import re

# ── Phrase / abbreviation substitutions ──────────────────────────────────────
# Keys are matched case-insensitively at word boundaries.
# Longer phrases must come before shorter ones (sort by length — see apply()).

GLOSSARY: dict[str, str] = {
    # Recovery cases
    "CASE III":  "case three",
    "CASE II":   "case two",
    "CASE I":    "case one",
    "CASE IV":   "case four",

    # AoA
    "AoA":       "A-O-A",
    "AOA":       "A-O-A",

    # Carrier approach / ACLS
    "ACLS":      "A-C-L-S",
    "ICLS":      "I-C-L-S",
    "ACL":       "A-C-L",
    "IFLOLS":    "eye-flols",
    "LSO":       "L-S-O",
    "PAPI":      "pappy",
    "UTM":       "U-T-M",
    "SUPT":      "S-U-P-T",

    # Autopilot / throttle
    "CPLD":      "coupled",      # must come before CPL
    "CPL":       "coupled",
    "ATC":       "A-T-C",
    "T/C":       "traffic and control",
    "T/R":       "transmit/receive",
    "D/L":       "datalink",
    "AP":        "A-P",

    # Display / nav pages
    "HSI":       "H-S-I",
    "OSB":       "O-S-B",
    "CRS":       "C-R-S",
    "TCN":       "T-C-N",
    "RDY":       "ready",
    "RDR":       "radar",

    # Abbreviations found in procedure voiced steps
    "CHKLST":    "checklist",
    "LND":       "landing",
    "CHK":       "check",
    "ENT":       "enter",
    "ROD":       "rate of descent",
    "CMD":       "commanded",

    # Weapon systems
    "AMRAAM":    "am-ram",
    "JDAM":      "jay-dam",
    "HARM":      "harm",
    "JSOW":      "jay-sow",
    "SLAM":      "slam",
    "AIM-9":     "aim nine",
    "AIM-7":     "aim seven",
    "AIM-120":   "aim one-twenty",
    "MFUZ":      "m-fuze",
    "EFUZ":      "e-fuze",
    "INST":      "instant",
    "DLY1":      "delay one",
    "DLY2":      "delay two",

    # Sensors / avionics
    "FCR":       "F-C-R",
    "TWS":       "T-W-S",
    "STT":       "S-T-T",
    "RWS":       "R-W-S",
    "VSR":       "V-S-R",
    "TCS":       "T-C-S",
    "FLIR":      "fleer",
    "TGP":       "T-G-P",
    "ATFLIR":    "at-fleer",
    "HUD":       "hud",
    "DDI":       "D-D-I",
    "MPCD":      "M-P-C-D",
    "AMPCD":     "A-M-P-C-D",
    "UFC":       "U-F-C",
    "MFD":       "M-F-D",
    "SMS":       "S-M-S",
    "BIT":       "built-in test",
    "TOO":       "T-O-O",
    "STBY":      "standby",

    # Navigation / comms
    "TACAN":     "tay-can",
    "ILS":       "I-L-S",
    "INS":       "I-N-S",
    "GPS":       "G-P-S",
    "IFF":       "I-F-F",
    "BRC":       "base recovery course",
    "NATOPS":    "nay-tops",

    # EW
    "RWR":       "R-W-R",
    "TEWS":      "T-E-W-S",
    "ECM":       "E-C-M",
    "DECM":      "D-E-C-M",

    # Flight / procedures
    "FCS":       "F-C-S",
    "FPAS":      "F-P-A-S",
    "OBOGS":     "oh-bogs",
    "MCG":       "M-C-G",
    "SAS":       "S-A-S",
    "NWS":       "nose wheel steering",
    "WOW":       "weight on wheels",
    "HOTAS":     "ho-taz",

    # Directions / references
    "FWD":       "forward",

    # HOTAS already handled in tts._normalize
}

# ── All-caps words that are normal English / aviation position words ──────────
# Kokoro spells these out when it sees them fully capitalised.
_WORD_CAPS: frozenset[str] = frozenset({
    # Position / state labels
    "DOWN", "UP", "FORWARD", "AFT", "LEFT", "RIGHT",
    "ON", "OFF", "IN", "OUT", "INBOARD", "OUTBOARD",
    "ARM", "ARMED", "SAFE",
    "OPEN", "CLOSED", "CLOSE",
    "EXTEND", "RETRACT",
    "AUTO", "MAN", "MANUAL",
    "NORM", "NORMAL",
    "EMER", "EMERG", "EMERGENCY",
    "SELECT", "SELECTED",
    "PRESS", "HOLD", "RELEASE",
    "SET", "RESET",
    "FULL", "FOLD",
    "IDLE", "MIL",
    # Common English words written in caps in docs
    "AND", "OR", "THE", "TO", "WITH", "FOR", "AT",
    "NOT", "NO", "YES",
    "CONFIRM", "CHECK",
    "NOTE", "CAUTION", "WARNING",
    "ENGAGE", "DISENGAGE",
    "ENABLE", "DISABLE",
    "PRIMARY", "SECONDARY",
    "STANDBY", "STBY",
    # Carrier procedure words
    "DATA", "TEST", "FAIL", "LOCK", "TILT",
    "ALTITUDE", "MODE", "SPEED", "POWER",
    "CARRIER", "APPROACH", "FINAL",
})

# Pre-compile: longest keys first so "CASE III" matches before "CASE"
_GLOSSARY_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
    for k, v in sorted(GLOSSARY.items(), key=lambda x: -len(x[0]))
]

_CAPS_WORD_RE = re.compile(r'\b([A-Z]{2,})\b')


def apply(text: str) -> str:
    """Apply glossary substitutions then fix all-caps normal words."""
    for pattern, replacement in _GLOSSARY_RE:
        text = pattern.sub(replacement, text)

    def _fix_caps(m: re.Match) -> str:
        w = m.group(1)
        return w.lower() if w in _WORD_CAPS else w

    return _CAPS_WORD_RE.sub(_fix_caps, text)
