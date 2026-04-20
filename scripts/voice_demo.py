#!/usr/bin/env python3
"""
Voice demo: PTT → STT → Ollama (with live state context) → TTS spoken reply.

Requires:
  - collector running:  python -m collector.main
  - Ollama running:     ollama serve
  - Piper model at:     models/piper/en_US-ryan-medium.onnx

Usage:
    python scripts/voice_demo.py
    python scripts/voice_demo.py --ptt caps_lock
    python scripts/voice_demo.py --ptt caps_lock --mic 1
    python scripts/voice_demo.py --no-speak        # print reply only, no audio
    python scripts/voice_demo.py --model mistral:7b

PTT key names: caps_lock, scroll_lock, f13, or any single character.
"""

import argparse
import ctypes
import difflib
import json
import random
import re
import sys
import threading
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger

from diagnostic.action_map import normalize_action
from diagnostic.rules import get_rules_for_action
from diagnostic.engine import diagnose_action
from voice import stt, tts

try:
    from retrieval.search import search as _rag_search
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False

try:
    from retrieval.cockpit_lookup import lookup as _cockpit_lookup, suggest as _cockpit_suggest, format_location, is_location_query
    from retrieval.keybinds import describe_switch as _describe_keybinds
    _LOOKUP_AVAILABLE = True
except Exception:
    _LOOKUP_AVAILABLE = False


COLLECTOR_URL   = "http://127.0.0.1:7779/state"
OLLAMA_URL      = "http://localhost:11434/api/chat"
DEFAULT_MODEL   = "qwen2.5:7b"
MAX_HISTORY     = 10    # max message pairs to keep (older turns dropped)
STATE_FRESH_MS  = 5000  # treat state as live only if updated within this window

_FEW_SHOT = """\

Examples:

[Reference confidence: HIGH]
Refueling Probe Control Switch EXTEND / RETRACT / EMERGENCY EXTENDED
Student: How do I open the refueling probe?
Set the Refueling Probe Control Switch to EXTEND.

[Reference confidence: HIGH]
Set FORMATION LIGHTS to BRT
Student: How do I turn on formation lights?
FORMATION LIGHTS switch to BRT — left console, exterior lights panel.

[Reference confidence: MEDIUM]
Set PROBE switch to EXTEND (step 8 of AAR procedure)
Student: Where is the probe switch?
The probe switch is on the left console. What are you trying to do — extend or retract?

[Reference confidence: LOW — no relevant procedure found]
Student: What's the trick to catching the wire?
That's technique, and it varies. What's happening — are you high, fast, or both?

[Reference confidence: LOW — no relevant procedure found]
Student: Can you tell me which console to look at?
Start with the left console — most of the fuel and secondary systems live there.
"""

_SYSTEM_CORE = """\
You are the Checkride Copilot — an F/A-18C Hornet instructor pilot (IP) riding
backseat on intercom. You have 2,000+ hours in the jet. Your job is to train,
not to recite. You explain the WHY, flag common errors, and speak the way a
real IP would in a brief or on hot mic.

Voice and tone:
- Direct, calm, authoritative. Never robotic or encyclopaedic.
- Do not address the student as "Pilot". Speak naturally, as to a colleague.
- Do not apologise or hedge unnecessarily.
- Never mention manuals, documents, or references — you are the knowledge.
- Never invent switch names, system names, acronyms, positions, or mechanisms.
  If you do not have reliable information, say so and ask a clarifying question.
  Inventing a plausible-sounding name is worse than admitting uncertainty.

Calibrate verbosity from the [Reference confidence] block:
HIGH  — answer in 1–2 sentences. Include exact figures, switch names, positions.
MEDIUM — answer what you know; ask ONE focused follow-up if something is unclear.
LOW   — no reference data was found. Do NOT attempt to answer from general knowledge.
        Ask what the pilot is trying to accomplish. Never invent a system or switch name.

If a [Closest cockpit match] block is present, open with "Did you mean X?" using
the exact switch name, then give its location.

This is a voice interface. Responses must be speakable — no bullet lists,
no asterisks, no markdown. Short sentences. Under 40 words unless the question
genuinely requires depth.
"""

SYSTEM_PROMPT_INFLIGHT = _SYSTEM_CORE + _FEW_SHOT
SYSTEM_PROMPT_GROUND   = _SYSTEM_CORE + _FEW_SHOT


# ── Procedure index ───────────────────────────────────────────────────────────
# index.json holds lightweight metadata (no steps). Individual proc files are
# loaded on demand when a procedure is activated.

_PROC_INDEX: list[dict] = []                # entries from index.json (no steps)
_PROC_CACHE: dict[str, dict] = {}           # key → full proc (steps loaded lazily)
_SECTION_NUM_TO_PROC: dict[str, str] = {}   # "2.7.1" → proc key
_PROC_WORDS: dict[str, frozenset[str]] = {} # proc key → keyword set
_PROC_DIR = Path(__file__).parent.parent / "data" / "airframes" / "fa18c" / "procedures"


def _load_proc(key: str) -> dict | None:
    """Load a single procedure file into _PROC_CACHE and return it."""
    if key in _PROC_CACHE:
        return _PROC_CACHE[key]
    # Find the index entry to get the path
    for entry in _PROC_INDEX:
        if entry["key"] == key:
            proc_path = _PROC_DIR / entry["path"]
            try:
                proc = json.loads(proc_path.read_text(encoding="utf-8"))
                _PROC_CACHE[key] = proc
                return proc
            except Exception as e:
                logger.warning(f"Could not load procedure {key}: {e}")
                return None
    return None


def _init_proc_index() -> None:
    index_path = _PROC_DIR / "index.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        _PROC_INDEX.extend(data.get("procedures", []))
    except Exception as e:
        logger.warning(f"Could not load procedures index: {e}")
        return

    for entry in _PROC_INDEX:
        key = entry["key"]
        section = entry.get("source", {}).get("section", "")
        last_part = section.split(">")[-1] if ">" in section else section
        m = re.search(r"(\d+(?:\.\d+)+)", last_part)
        if m:
            _SECTION_NUM_TO_PROC[m.group(1)] = key

        words: set[str] = set()
        for text in [entry["canonical"]] + entry.get("alternates", []):
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                if len(w) >= 3:
                    words.add(w)
        _PROC_WORDS[key] = frozenset(words)

_init_proc_index()


# ── Disambiguation groups ──────────────────────────────────────────────────────
# Loaded once; each group defines a tree of clarifying questions that routes to
# a specific procedure key before steps begin.

_DISAMBIG_GROUPS: list[dict] = []
_DISAMBIG_KEYWORDS: list[tuple[frozenset[str], dict]] = []  # (kw_set, group)

def _init_disambig() -> None:
    path = Path(__file__).parent.parent / "data" / "airframes" / "fa18c" / "proc_groups.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _DISAMBIG_GROUPS.extend(data.get("groups", []))
    except Exception as e:
        logger.warning(f"Could not load proc_groups.json: {e}")
        return
    for group in _DISAMBIG_GROUPS:
        kws = frozenset(w for kw in group["keywords"]
                        for w in re.findall(r"[a-z0-9]+", kw.lower()))
        _DISAMBIG_KEYWORDS.append((kws, group))

_init_disambig()


def _match_disambig_group(transcript: str) -> dict | None:
    """Return the disambiguation group whose keywords best match the transcript."""
    tokens = frozenset(re.findall(r"[a-z0-9]+", transcript.lower()))
    best_score, best_group = 0, None
    for kws, group in _DISAMBIG_KEYWORDS:
        score = len(tokens & kws)
        if score > best_score:
            best_score, best_group = score, group
    return best_group if best_score > 0 else None


def _match_disambig_option(transcript: str, options: list[dict]) -> dict | None:
    """Return the option node whose keywords best match the transcript."""
    tokens = frozenset(re.findall(r"[a-z0-9]+", transcript.lower()))
    best_score, best_opt = 0, None
    for opt in options:
        kws = frozenset(w for kw in opt["keywords"]
                        for w in re.findall(r"[a-z0-9]+", kw.lower()))
        score = len(tokens & kws)
        if score > best_score:
            best_score, best_opt = score, opt
    return best_opt if best_score > 0 else None


_PROC_REQUEST_RE = re.compile(
    r"\b("
    # "how do/can/should/would I <verb>" — all modal forms, not just "do"
    r"how\s+(do|can|should|would)\s+i\s+(fire|launch|release|drop|employ|arm\s+up|set\s+up|"
    r"start\s+up|execute|deploy|shoot|pickle|engage|lase|refuel|use|apply|target)|"
    r"(show|take|walk|talk|run|slip|step)\s+me\s+(through|over)|"
    r"step\s+by\s+step|what\s+are\s+the\s+steps|one\s+step\s+at\s+a\s+time|"
    r"procedure\s+for|run\s+me\s+through|talk\s+me\s+through|"
    r"i\s+need\s+to\s+(fire|launch|release|drop|employ|shoot|engage)|"
    r"how\s+to\s+(fire|launch|release|drop|employ|shoot|use|apply|engage)|"
    r"can\s+you\s+(show|walk|take|run|slip)\s+me"
    r")\b",
    re.IGNORECASE,
)

def _is_proc_request(transcript: str) -> bool:
    """True only when the pilot explicitly asks to be walked through a procedure."""
    return bool(_PROC_REQUEST_RE.search(transcript))


def _find_proc_for_section(section_str: str) -> str | None:
    """Map a RAG chunk section label ('2.7.1: AGM-65F/G MAVERICK') to a proc key."""
    m = re.match(r"(\d+(?:\.\d+)*)\s*[:\-]", section_str)
    if m:
        return _SECTION_NUM_TO_PROC.get(m.group(1))
    return None


def _normalize_phrase(text: str) -> str:
    """Lowercase and normalize text for exact phrase matching."""
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _find_proc_by_exact_phrase(transcript: str) -> str | None:
    """Return a procedure key when transcript contains an exact known phrase.
    Uses canonical names, alternates, key text, and canonical parenthetical text.
    """
    query = _normalize_phrase(transcript)
    if not query:
        return None

    matches: list[tuple[int, str]] = []
    for entry in _PROC_INDEX:
        key = entry["key"]
        phrases: set[str] = {
            entry.get("canonical", ""),
            *(entry.get("alternates", []) or []),
            key.replace("_", " "),
        }

        # Parenthetical canonical text often contains compact names like
        # "Case I Recovery" that users speak verbatim.
        canonical = entry.get("canonical", "")
        for paren in re.findall(r"\(([^)]+)\)", canonical):
            phrases.add(paren)

        for phrase in phrases:
            p = _normalize_phrase(phrase)
            if len(p) < 6:
                continue

            alias_phrases = {p}
            alias_swaps = (
                ("case i", "case 1"), ("case i", "case one"),
                ("case ii", "case 2"), ("case ii", "case two"),
                ("case iii", "case 3"), ("case iii", "case three"),
            )
            for a, b in alias_swaps:
                if a in p:
                    alias_phrases.add(p.replace(a, b))
                if b in p:
                    alias_phrases.add(p.replace(b, a))

            for ap in alias_phrases:
                if ap in query:
                    matches.append((len(ap), key))

    if not matches:
        return None

    matches.sort(reverse=True)
    top_len = matches[0][0]
    top_keys = {k for n, k in matches if n == top_len}
    if len(top_keys) != 1:
        return None
    return next(iter(top_keys))


_PROC_STOPWORDS = frozenset({"how", "the", "what", "when", "where", "why", "for",
                              "and", "can", "you", "use", "do", "a", "to", "i",
                              "fire", "help", "want", "need", "me"})

def _find_proc_by_keywords(transcript: str) -> str | None:
    """Return best-matching procedure key from transcript word overlap, or None.
    Returns None on ties — caller falls through to RAG disambiguation."""
    query = frozenset(re.findall(r"[a-z0-9]+", transcript.lower())) - _PROC_STOPWORDS
    if not query:
        return None
    scores: list[tuple[int, str]] = [
        (len(query & words), key)
        for key, words in _PROC_WORDS.items()
        if len(query & words) >= 1
    ]
    if not scores:
        return None
    scores.sort(reverse=True)
    # Only return if there's a clear winner with no tie at the top
    if len(scores) >= 2 and scores[0][0] == scores[1][0]:
        return None
    return scores[0][1]


def _clean_step_text(text: str) -> str:
    """Strip OCR artefacts from extracted PDF step text."""
    # Remove non-printable characters (OCR noise), normalize dashes
    text = re.sub(r"[^\x20-\x7E\u2013\u2014\u2018\u2019]", " ", text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _maneuver_group_start(steps: list, step_idx: int) -> int:
    """Return index of the first step in the maneuver group containing step_idx."""
    if steps[step_idx].get("pace") != "maneuver":
        return step_idx
    i = step_idx
    while i > 0 and steps[i - 1].get("pace") == "maneuver":
        i -= 1
    return i


def _maneuver_group_end(steps: list, step_idx: int) -> int:
    """Return index of the last step in a maneuver group starting at step_idx."""
    if steps[step_idx].get("pace") != "maneuver":
        return step_idx
    i = step_idx
    while i + 1 < len(steps) and steps[i + 1].get("pace") == "maneuver":
        i += 1
    return i


def _format_step(proc_key: str, step_idx: int) -> str:
    """Format a procedure step for TTS. Maneuver-paced steps are joined as one block."""
    proc  = _load_proc(proc_key)
    if not proc:
        return _pick("proc_unavailable")
    steps = proc["steps"]
    step  = steps[step_idx]
    if step.get("pace") == "maneuver":
        parts = []
        i = step_idx
        while i < len(steps) and steps[i].get("pace") == "maneuver":
            parts.append(_clean_step_text(steps[i].get("voiced") or steps[i]["action"]))
            i += 1
        return " ".join(parts)
    return _clean_step_text(step.get("voiced") or step["action"])


def _format_proc_intro(proc_key: str) -> str:
    """Name the procedure, then deliver step 1 immediately."""
    proc  = _load_proc(proc_key)
    if not proc:
        return _pick("proc_unavailable")
    step1 = _clean_step_text(proc["steps"][0].get("voiced") or proc["steps"][0]["action"])
    total_steps = len(proc["steps"])
    return f"{proc['canonical']}. {total_steps} steps total. First: {step1}"


# ─────────────────────────────────────────────────────────────────────────────

def _fetch_state() -> dict | None:
    """Return state only if fresh (pilot is in an active jet). None otherwise."""
    try:
        r = httpx.get(COLLECTOR_URL, timeout=1.5)
        r.raise_for_status()
        state = r.json()
        if state.get("data_age_ms", 99999) > STATE_FRESH_MS:
            return None
        return state
    except Exception:
        return None


def _state_summary(state: dict) -> str:
    """One-paragraph plain-English summary of cockpit state for the LLM."""
    lines = [
        f"Aircraft: {state.get('airframe_normalized','?').upper()}",
        f"Altitude: {state.get('altitude_ft', 0):.0f} ft  Heading: {state.get('heading_deg', 0):.0f}°  Speed: {state.get('airspeed_kts', 0):.0f} kts",
        f"Master Arm: {state.get('master_arm','?')}  Battery: {state.get('battery_switch','?')}",
        f"Radar: {'ON' if state.get('radar_powered') else 'OFF'} ({state.get('radar_mode','?')})",
        f"TGP: {'ON' if state.get('tgp_powered') else 'OFF'}  Lasing: {state.get('tgp_lasing','?')}  Tracking: {state.get('tgp_tracking','?')}",
        f"Gear: {state.get('gear_position','?')}  Flaps: {state.get('flaps_position','?')}  Hook: {state.get('hook_position','?')}",
        f"Engine L: {state.get('engine_left_state','?')}  Engine R: {state.get('engine_right_state','?')}",
    ]
    ws = state.get("weapon_stations", [])
    if ws:
        stores = ", ".join(f"{w['quantity']}×{w['store_type']}" for w in ws)
        lines.append(f"Stores: {stores}")
    else:
        lines.append("Stores: none")
    return "\n".join(lines)


def _diagnostic_note(text: str, state: dict) -> str:
    """Run diagnostic engine if the query mentions a known action; return findings."""
    keywords = {
        "aim": "fire_aim120", "amraam": "fire_aim120", "120": "fire_aim120",
        "sidewinder": "fire_aim9x", "9x": "fire_aim9x",
        "maverick": "release_agm65", "agm": "release_agm65", "65": "release_agm65",
        "gbu": "release_gbu12", "bomb": "release_gbu12",
        "laser": "lase_target", "lase": "lase_target",
        "tgp": "tgp_track", "track": "tgp_track",
        "engine": "start_engine", "start": "start_engine",
    }
    lower = text.lower()
    action = next((v for k, v in keywords.items() if k in lower), None)
    if not action:
        return ""

    action_norm = normalize_action(action)
    if not get_rules_for_action(action_norm):
        return ""

    result = diagnose_action(action_norm, state)
    if result["ok"]:
        return f"Diagnostic: {action_norm} — all conditions met, clear to execute."

    blockers = result.get("blockers", [])
    if blockers:
        fixes = "; ".join(b.get("fix_description") or b.get("message", "") for b in blockers[:2])
        return f"Diagnostic: {action_norm} — BLOCKED. Fix: {fixes}"
    return ""


_AVIATION_RE = re.compile(
    r"\b(aim|amraam|sidewinder|maverick|agm|gbu|bomb|laser|tgp|radar|engine|"
    r"gear|flap|hook|light|switch|panel|console|throttle|hud|mfd|sensor|fuel|"
    r"hydraulic|speed|altitude|heading|start|master|arm|fire|release|track|"
    r"lock|probe|refuel|refueling|tacan|ils|jdam|harm|harpoon|"
    r"weapon|weapons|missile|select|target|engage|stick|hotas|"
    r"delivery|ccrp|ccip|mk[-\s]?\d+|procedure|approach|intercept|landing|"
    r"carrier|recovery|catapult|trap|bolter|bingo|fence|marshal)\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(r"[?]|\b(where|what|how|which|when|why|can\s+you|do\s+i|should\s+i)\b", re.I)

# Questions asking for a list/overview of options — should not trigger variant disambiguation.
_ENUMERATION_RE = re.compile(
    r"\b(what\s+(type|kind|sort|variant|version)s?\s+(of|are|do)|"
    r"(list|tell\s+me\s+about)\s+(all|the)\s+|"
    r"all\s+(the\s+)?(type|kind|variant|version)s?|"
    r"what.{0,30}\bavailable\b|"
    r"what.{0,30}\bexist\b|"
    r"what.{0,30}\bdo\s+(we|i|you)\s+have\b)\b",
    re.IGNORECASE,
)

def _is_pure_social(text: str) -> bool:
    """True if the utterance is a short acknowledgment with no question and no aviation content."""
    words = text.split()
    if len(words) > 6:
        return False
    if _QUESTION_RE.search(text):
        return False
    if _AVIATION_RE.search(text):
        return False
    return True

# Meta / conversational patterns that should skip RAG entirely.
# The model must answer these from conversation history alone.
_SKIP_RAG_RE = re.compile(
    r"(you\s+(just\s+)?said"
    r"|i\s+thought\s+you"
    r"|why\s+did\s+you"
    r"|what.{0,30}(got|have|has)\s+to\s+do"
    r"|doesn.?t\s+make\s+sense"
    r"|can\s+you\s+tell\s+me\s+(which|where|what|how)"
    r"|you\s+tell\s+me"
    # Bare referential follow-ons — no standalone aviation content; must use history.
    # BM25 augmentation with history pollutes these queries and picks wrong procs.
    r"|show\s+me\s+(how|that|it)"
    r"|walk\s+me\s+through\s+(it|that)"
    r"|how\s+do\s+(i|you)\s+do\s+that"
    r"|how\s+does\s+that\s+work"
    r"|tell\s+me\s+(more|how\s+to)"
    r"|and\s+how(\s+do\s+i)?"
    r")",
    re.IGNORECASE,
)

# Social openers that a pilot might prepend to a real question.
# Strip these before building the BM25 query so they don't dilute signal.
_SOCIAL_PREFIX_RE = re.compile(
    r"^\s*(perfect[.,!]?\s+|great[.,!]?\s+|awesome[.,!]?\s+|excellent[.,!]?\s+|"
    r"nice[.,!]?\s+|solid[.,!]?\s+|cool[.,!]?\s+|"
    r"thanks[.,!]?\s+|thank\s+you[.,!]?\s+|cheers[.,!]?\s+|"
    r"roger[.,!]?\s+|copy[.,!]?\s+|wilco[.,!]?\s+|"
    r"ok[.,!]?\s+|okay[.,!]?\s+|alright[.,!]?\s+|"
    r"got\s+it[.,!]?\s+|understood[.,!]?\s+|right[.,!]?\s+)",
    re.IGNORECASE,
)


def _build_search_query(transcript: str, history: list[dict]) -> str:
    """Build the BM25 query: strip social prefixes, augment short queries with history."""
    # Strip social openers ("Thanks. Where is the switch?" → "Where is the switch?")
    core = _SOCIAL_PREFIX_RE.sub("", transcript).strip() or transcript

    words = core.split()
    has_keywords = bool(_AVIATION_RE.search(core))

    # Long queries with aviation terms: search as-is (already specific enough)
    if len(words) >= 6 and has_keywords:
        return core

    # Short or keyword-sparse: pick the prior message with the most aviation
    # keyword hits — that's the topic anchor BM25 needs.
    def _kw_count(text: str) -> int:
        return len(_AVIATION_RE.findall(text))

    def _extract(msg: dict) -> str:
        t = msg["content"]
        if msg["role"] == "user":
            idx = t.rfind("Pilot: ")
            t = t[idx + 7:].strip() if idx != -1 else t
        return t.strip()

    # Limit augmentation to recent turns: last 3 turns AND within 90 seconds.
    # A long pause signals a topic switch even if the message count is low.
    now = _time.time()
    best_text, best_score = "", 0
    for msg in history[-6:]:
        if now - msg.get("ts", now) > 90:
            continue
        t = _extract(msg)
        s = _kw_count(t)
        if s > best_score:
            best_score, best_text = s, t

    if best_text:
        return f"{best_text} {core}"
    return core


# Preamble patterns the model uses to introduce a list it's about to give.
# These lines end with ":" — the stop tokens ate the list that followed.
# Only strip lines that actually end with a colon (the list-intro marker).
_PREAMBLE_RE = re.compile(
    r"^(to\s+\w[^,]*,?\s*)?(here\s+are|you'?ll\s+need\s+to|"
    r"follow\s+these|the\s+\w+\s+steps?(\s+are)?|"
    r"here'?s\s+how|the\s+following|maintain\s+the\s+following|"
    r"as\s+follows)[^:]*:\s*$",
    re.IGNORECASE,
)


_LEAKED_TAG_RE = re.compile(r"(\.\s*Reference confidence[^.]*\.?|\s*\[Switch[^\]]*\][^.]*\.?)\s*$", re.IGNORECASE)
_DOC_REF_RE = re.compile(
    r"\b(according to|as per|per|based on|from|in)\s+(the\s+)?(natops|manual|guide|reference|documentation|procedure\s+guide)\b[^.,]*[.,]?\s*",
    re.IGNORECASE,
)
_FILLER_TAIL_RE = re.compile(
    r"[.,!]?\s*(let me know\b|feel free\b|don'?t hesitate\b|if you (need|have|require)\b|"
    r"hope that (helps|clarifies)\b|is there anything else\b|further (assistance|questions)\b|"
    r"happy to help\b|here to help\b).*$",
    re.IGNORECASE,
)

# Sycophantic openers the LLM produces despite the persona prompt.
_SYCOPHANTIC_RE = re.compile(
    r"^\s*(great(\.| job)?|excellent(\.| job)?|perfect(\.)?|well done(\.)?|"
    r"good (job|work|answer|response|question)(\.)?|"
    r"you'?re doing (well|great)(\.)?|"
    r"that'?s (correct|right|great|good)(\.)?|"
    r"absolutely(\.)?|certainly(\.)?|of course(\.)?|"
    r"sure(,| thing)?(\.)?)\s*",
    re.IGNORECASE,
)


_BOLD_RE   = re.compile(r"\*{1,2}(.+?)\*{1,2}")
_CURLY_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u201a\u201b]")
_CURLY_QUOTE_RE      = re.compile(r"[\u201c\u201d\u201e\u201f]")

def _clean_reply(text: str) -> str:
    """Strip leaked tags, sycophantic openers, bold markers, and curly quotes."""
    text = _DOC_REF_RE.sub("", text).strip()
    text = _LEAKED_TAG_RE.sub(".", text).strip()
    text = _BOLD_RE.sub(r"\1", text)
    text = _CURLY_APOSTROPHE_RE.sub("'", text)
    text = _CURLY_QUOTE_RE.sub('"', text)
    text = _SYCOPHANTIC_RE.sub("", text).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not _PREAMBLE_RE.match(l)]
    return " ".join(lines).strip()


def _ask_ollama(
    transcript: str,
    state: dict | None,
    model: str,
    history: list[dict],
    low_streak: int = 0,
) -> tuple[str, str, str | None]:
    """Returns (reply, confidence, proc_key).
    proc_key is set when RAG unambiguously identifies a known procedure —
    caller should enter procedure mode and not use the reply text."""
    in_jet    = state is not None
    sys_prompt = SYSTEM_PROMPT_INFLIGHT if in_jet else SYSTEM_PROMPT_GROUND

    is_meta = bool(_SKIP_RAG_RE.search(transcript)) or _is_pure_social(transcript)
    ref_block = "" if is_meta else "[Reference confidence: LOW — no relevant procedure found. Do not invent system names, acronyms, or switch names. Ask what the pilot is trying to do.]\n\n"
    confidence = "MEDIUM" if is_meta else "LOW"

    # Pure social acks (no question, no aviation) have nothing to answer.
    # Return a canned acknowledgment instead of letting the LLM hallucinate suggestions.
    # Never short-circuit an explicit procedure request — even if the phrasing is short.
    if _is_pure_social(transcript) and not _is_proc_request(transcript):
        return _pick("roger"), "MEDIUM", None

    # If the previous assistant turn was a disambiguation question and the
    # current query is too vague to resolve it, re-ask rather than hallucinate.
    # Exception: if the user said a NATO phonetic word (Delta, Echo, Golf…),
    # treat it as a variant answer and let RAG resolve it with history context.
    _variant_resolved = False
    if is_meta and history:
        last_reply = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        if last_reply.startswith("Which variant"):
            tl = transcript.lower()
            nato_words = [w.lower() for w in _NATO.values()]
            # Exact match first; then fuzzy match for single-word responses
            # so Whisper mishears like "Gulf"/"Gold" resolve "Golf".
            exact = any(w in tl for w in nato_words)
            fuzzy = (not exact and len(tl.split()) <= 2 and
                     any(difflib.get_close_matches(w, nato_words, n=1, cutoff=0.72)
                         for w in tl.split()))
            if exact or fuzzy:
                is_meta = False   # phonetic heard → route through RAG with history
                _variant_resolved = True
            else:
                history.append({"role": "user",      "content": transcript,   "ts": _time.time()})
                history.append({"role": "assistant",  "content": last_reply,   "ts": _time.time()})
                return last_reply, "MEDIUM", None

    # Cockpit lookup: if this looks like a location query, check the switch
    # index first — this gives a precise, authoritative answer without BM25.
    # Try bare transcript first; if no match, try the history-augmented query
    # so "Where is the switch?" resolves via prior turn context ("probe switch").
    if _LOOKUP_AVAILABLE and not is_meta and is_location_query(transcript):
        hit = _cockpit_lookup(transcript)
        if not hit:
            augmented = _build_search_query(transcript, history)
            if augmented != transcript:
                hit = _cockpit_lookup(augmented)
        if hit:
            canonical = hit["canonical"]
            area      = hit["area_label"]
            panel     = hit["panel_label"]
            # Use pre-written spoken_location if available, else build from parts.
            if hit.get("spoken_location"):
                reply = hit["spoken_location"]
            else:
                spoken = hit.get("spoken") or hit.get("positions", [])
                if spoken:
                    if len(spoken) == 1:
                        pos_str = spoken[0]
                    elif len(spoken) == 2:
                        pos_str = f"{spoken[0]} or {spoken[1]}"
                    else:
                        pos_str = ", ".join(spoken[:-1]) + f", or {spoken[-1]}"
                    reply = f"{canonical} — {area}, {panel}. {pos_str}."
                else:
                    reply = f"{canonical} — {area}, {panel}."
            dcs_actions = hit.get("dcs_actions")
            if dcs_actions:
                kb = _describe_keybinds(dcs_actions)
                if kb:
                    reply = reply.rstrip(".") + f". {kb}"
            history.append({"role": "user",      "content": transcript, "ts": _time.time()})
            history.append({"role": "assistant",  "content": reply,      "ts": _time.time()})
            return reply, "HIGH", None

    # Direct keyword match: for clear procedure requests that may score LOW on
    # BM25 (e.g. "how do I use the Maverick"), try matching against proc keywords
    # before running RAG. This bypasses BM25 weakness on generic verbs like "use".
    if not is_meta and _is_proc_request(transcript):
        # Exact phrase matches should take precedence over disambiguation groups.
        exact_proc = _find_proc_by_exact_phrase(transcript)
        if exact_proc and exact_proc in _PROC_WORDS:
            history.append({"role": "user",      "content": transcript, "ts": _time.time()})
            history.append({"role": "assistant",  "content": f"[PROC:{exact_proc}]", "ts": _time.time()})
            return "", "HIGH", exact_proc

        # Check for disambiguation group — but if the transcript already names a
        # specific option (e.g. "CCIP mode"), resolve it immediately without asking.
        dg = _match_disambig_group(transcript)
        if dg:
            opt = _match_disambig_option(transcript, dg["options"])
            if opt and opt["proc_key"] in _PROC_WORDS:
                history.append({"role": "user",      "content": transcript, "ts": _time.time()})
                history.append({"role": "assistant",  "content": f"[PROC:{opt['proc_key']}]", "ts": _time.time()})
                return "", "HIGH", opt["proc_key"]
            return "", "HIGH", f"__DISAMBIG__:{dg['id']}"

        kw_proc = _find_proc_by_keywords(transcript)
        if kw_proc and kw_proc in _PROC_WORDS:
            history.append({"role": "user",      "content": transcript, "ts": _time.time()})
            history.append({"role": "assistant",  "content": f"[PROC:{kw_proc}]", "ts": _time.time()})
            return "", "HIGH", kw_proc

    # After 2 consecutive LOW turns, drop the score floor so we inject the
    # best available content even if it's a weak hit, and tell the model to
    # give the procedure instead of asking yet another clarifying question.
    fallback_mode = (not is_meta) and (low_streak >= 2)
    min_score = 8.0 if fallback_mode else 12.0
    is_ambiguous = False
    if _RAG_AVAILABLE and not is_meta:
        search_query = _build_search_query(transcript, history)
        hits = _rag_search(search_query, top_k=3, min_score=min_score)
        if hits:
            top_score = hits[0]["score"]
            confidence = "HIGH" if top_score >= 18 else "MEDIUM" if top_score >= 14 else "LOW"
            # Disambiguation: if hits span multiple distinct sections, ask which variant.
            sections = list(dict.fromkeys(h.get("section", "") for h in hits if h.get("section")))
            # Collect variant names — only sections with a designation letter qualify.
            # Sections like "How To Use Markpoints" return None and are excluded.
            # Enumeration questions ("what types are available?") always skip disambiguation
            # and let the LLM enumerate from the combined hits.
            is_enumeration = bool(_ENUMERATION_RE.search(transcript))
            seen_v: set[str] = set()
            variants: list[str] = []
            if len(sections) >= 2 and confidence != "HIGH" and not is_enumeration and not _variant_resolved and _is_proc_request(transcript):
                for s in sections:
                    v = _variant_name(s)
                    if v is not None and v not in seen_v:
                        seen_v.add(v)
                        variants.append(v)

            if len(variants) >= 2:
                if len(variants) == 2:
                    q = f"Which variant? {variants[0]}, or {variants[1]}?"
                else:
                    q = "Which variant? " + ", ".join(variants[:-1]) + f", or {variants[-1]}?"
                history.append({"role": "user",      "content": transcript})
                history.append({"role": "assistant",  "content": q})
                return q, "MEDIUM", None

            # Single unambiguous section — enter proc mode only when the pilot
            # explicitly asks to be walked through something (not a general question),
            # AND the detected procedure is actually relevant to the transcript keywords
            # (prevents e.g. "how do I use Mavericks" → markpoints via BM25 title match).
            if confidence in ("HIGH", "MEDIUM") and sections and not is_meta and _is_proc_request(transcript):
                proc_key = _find_proc_for_section(sections[0])
                if proc_key and proc_key in _PROC_WORDS:
                    # Sanity check against the section label (not proc keywords) so
                    # "AGM-65F Missile IR Seeker Only" correctly matches "maverick" via
                    # the section string "2.7.1: AGM-65F/G MAVERICK".
                    query_words   = frozenset(re.findall(r"[a-z0-9]+", transcript.lower())) - _PROC_STOPWORDS
                    section_words = frozenset(re.findall(r"[a-z0-9]+", sections[0].lower())) - _PROC_STOPWORDS
                    if query_words & section_words:
                        history.append({"role": "user",      "content": transcript, "ts": _time.time()})
                        history.append({"role": "assistant",  "content": f"[PROC:{proc_key}]", "ts": _time.time()})
                        return "", confidence, proc_key

            excerpts = "\n---\n".join(h["text"] for h in hits)
            label = confidence
            if fallback_mode:
                label = f"{confidence} — give the best available procedure, do not ask again"
            ref_block = f"[Reference confidence: {label}]\n{excerpts}\n\n"
        else:
            # BM25 found nothing — try cockpit index as a disambiguation hint,
            # but only for short identity-style queries, not "how do I" procedural ones.
            _HOW_DO_I_RE = re.compile(r"\b(how\s+(do|can|should)\s+i|what\s+do\s+i|do\s+i\s+need)\b", re.I)
            is_procedural = bool(_HOW_DO_I_RE.search(transcript))
            if _LOOKUP_AVAILABLE and not is_procedural:
                aug = _build_search_query(transcript, history)
                suggestion = _cockpit_suggest(aug)
                if suggestion:
                    loc = format_location(suggestion)
                    ref_block = f"[Reference confidence: LOW — no procedure found]\n[Closest cockpit match: {loc}]\n\n"
            if fallback_mode:
                ref_block = "[Reference confidence: LOW — no procedure found. Do not invent a procedure or system name. Tell the pilot you don't have that procedure and ask what they're trying to accomplish.]\n\n"

    # Token budget and temperature scale with confidence.
    # LOW is lowest — no reference material means the model must follow instructions
    # tightly (don't invent), not improvise freely. High temp at LOW = hallucination.
    temperature  = {"HIGH": 0.1,  "MEDIUM": 0.25, "LOW": 0.15}.get(confidence, 0.25)
    num_predict  = {"HIGH": 120,  "MEDIUM": 100,  "LOW": 60}.get(confidence, 100)
    if is_meta:
        temperature, num_predict = 0.45, 40

    if in_jet:
        diag_note = _diagnostic_note(transcript, state)
        user_msg  = f"{ref_block}[Cockpit state]\n{_state_summary(state)}"
        if diag_note:
            user_msg += f"\n\n[Diagnostic]\n{diag_note}"
        user_msg += f"\n\nPilot: {transcript}"
    else:
        user_msg = f"{ref_block}Pilot: {transcript}"

    messages = [{"role": "system", "content": sys_prompt}]
    messages += history[-MAX_HISTORY * 2:]          # keep last N turns
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {
            "num_predict": num_predict,
            "temperature": temperature,
            "repeat_penalty": 1.15,
            "stop": ["[Reference", "[Switch",
                     "let me know", "feel free", "further assistance", "if you need", "if you have"],
        },
    }

    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=15.0)
        r.raise_for_status()
        reply = _clean_reply(r.json()["message"]["content"].strip())
        history.append({"role": "user",      "content": user_msg, "ts": _time.time()})
        history.append({"role": "assistant",  "content": reply,    "ts": _time.time()})
        return reply, confidence, None
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return _pick("llm_unavailable"), "LOW", None


# Windows virtual-key codes for common PTT choices.
# GetAsyncKeyState is used to poll key state without installing a hook,
# avoiding the pynput hook install/uninstall race that causes segfaults.
_PTT_VK: dict[str, int] = {
    "caps_lock": 0x14, "scroll_lock": 0x91, "num_lock": 0x90,
    **{f"f{n}": 0x6B + n for n in range(13, 25)},   # F13-F24 → 0x7C-0x87
}

def _ptt_vk(key_str: str) -> int | None:
    k = key_str.lower()
    if k in _PTT_VK:
        return _PTT_VK[k]
    if len(k) == 1:
        return ord(k.upper())
    return None

def _key_held(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


_NATO = {
    'A': 'Alpha',   'B': 'Bravo',    'C': 'Charlie', 'D': 'Delta',
    'E': 'Echo',    'F': 'Foxtrot',  'G': 'Golf',    'H': 'Hotel',
    'I': 'India',   'J': 'Juliet',   'K': 'Kilo',    'L': 'Lima',
    'M': 'Mike',    'N': 'November', 'O': 'Oscar',   'P': 'Papa',
    'Q': 'Quebec',  'R': 'Romeo',    'S': 'Sierra',  'T': 'Tango',
    'U': 'Uniform', 'V': 'Victor',   'W': 'Whiskey', 'X': 'X-ray',
    'Y': 'Yankee',  'Z': 'Zulu',
}

_VARIANT_LETTER_RE = re.compile(r"\b\d+([A-Z](?:/[A-Z])*)\b", re.I)

def _variant_name(section: str) -> str | None:
    """Extract variant letter(s) from a section title and expand to NATO phonetic.
    Returns None if the section has no variant letter — it is not a variant entry."""
    m = _VARIANT_LETTER_RE.search(section)
    if not m:
        return None
    letters = m.group(1).upper().split("/")
    return "/".join(_NATO.get(l, l) for l in letters)


# Splits "Sentence. 2. Next step." but not "Preamble: 1. First step." so that
# a context preamble stays attached to step 1.
_STEP_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=\d+\.\s)')

# Words the pilot can say to advance to the next step.
_CONTINUE_RE = re.compile(
    r"^\s*(check|done|ok|okay|next|go\s+(on|ahead)|continue|ready|"
    r"roger|copy|affirm|affirmative|wilco|proceed|confirm(ed)?|yes|yep|go)\s*[.,!]?\s*$",
    re.IGNORECASE,
)

# Repeat the current step.
_REPEAT_RE = re.compile(
    r"^\s*(what|huh|say\s+again|come\s+again|repeat|again|"
    r"what\s+(did\s+you\s+(say|just\s+say)|was\s+that)|"
    r"repeat\s+(the\s+)?(last(\s+step)?|that)|"
    r"i\s+(didn.?t|did\s+not)\s+(hear|catch)\s+(that|you))\s*[?,!.]?\s*$",
    re.IGNORECASE,
)

# Restart the current procedure from step 1.
_RESTART_RE = re.compile(
    r"\b(start\s+(from\s+)?(the\s+)?(start|beginning|top)|"
    r"restart|from\s+the\s+(top|start|beginning)|"
    r"back\s+to\s+(the\s+)?(start|beginning|top)|"
    r"again\s+from\s+the\s+(start|top|beginning))\b",
    re.IGNORECASE,
)

# Abandon the current procedure.
_CANCEL_PROC_RE = re.compile(
    r"^\s*(cancel|stop|forget\s+it|never\s*mind|that.?s\s+(all|it|fine|good))\s*[.,!]?\s*$",
    re.IGNORECASE,
)



def _split_steps(reply: str) -> list[str]:
    """Return individual steps if reply is a numbered procedure, else []."""
    parts = _STEP_SPLIT_RE.split(reply.strip())
    parts = [p.strip() for p in parts if p.strip()]
    # Require at least the second chunk to start with a step number
    if len(parts) >= 2 and re.match(r'\d+\.', parts[1]):
        return parts
    return []


def _step_tts(step: str) -> str:
    """Strip inline step numbers for cleaner TTS ('1. Do X' → 'Do X')."""
    return re.sub(r'\b\d+\.\s+', '', step).strip()


_FILLERS: dict[str, list[str]] = {
    "MEDIUM": ["Hmm. ", "Right. ", "Let me think. "],
    "LOW":    ["Hmm... ", "Uh... ", "Well... "],
}

_CANNED: dict[str, list[str]] = {
    "roger": [
        "Roger.", "Copy.", "Wilco.", "Understood.", "Got it.",
    ],
    "say_again": [
        "Say again?", "Come again?", "Didn't catch that — say again.",
        "Say that again?", "Missed that one.",
    ],
    "were_you_asking": [
        "Were you asking —", "Did you mean —", "Are you asking about —",
        "Just to check —", "Is this about —",
    ],
    "not_following": [
        "Sorry, not following that one.", "Can't get a handle on that — try me again.",
        "Not sure what you're after. Come again?", "Lost you there.",
        "Say again — didn't get that.",
    ],
    "proc_unavailable": [
        "Procedure unavailable.", "Can't pull up that procedure right now.",
        "That one's not available.", "Procedure not found.",
    ],
    "llm_unavailable": [
        "LLM unavailable.", "Stand by — system issue.", "No joy on that right now.",
        "Comms issue — try again.", "System's not responding.",
    ],
}

def _pick(key: str) -> str:
    return random.choice(_CANNED[key])

def _voiced(reply: str, conf: str) -> str:
    """Optionally prepend a natural filler for non-trivial queries."""
    if conf not in _FILLERS or len(reply.split()) < 8:
        return reply
    if random.random() < 0.4:
        return random.choice(_FILLERS[conf]) + reply
    return reply


def run(ptt_key: str, mic_device: int | None, speak: bool, model: str) -> None:
    logger.info("Pre-loading Whisper ...")
    stt._get_model()

    if speak:
        logger.info("Pre-loading TTS ...")
        tts.prewarm()

    parsed_ptt = stt._parse_key(ptt_key)
    ptt_vk     = _ptt_vk(ptt_key)

    print(f"\nCheckride Copilot")
    print(f"PTT : {ptt_key.upper()}  |  Model : {model}  |  Audio : {'on' if speak else 'off'}")
    print(f"Hold [{ptt_key.upper()}] and speak.  Ctrl-C to quit.\n")

    history:             list[dict] = []
    low_streak:          int        = 0
    active_proc:         str | None = None
    active_step:         int        = 0
    disambiguation_node: dict|None  = None

    while True:
        try:
            _ctx_terms: list[str] | None = None
            if active_proc:
                _ap = _load_proc(active_proc)
                if _ap:
                    _ctx_terms = _ap.get("terminology") or None
            transcript = stt.listen_once(ptt_key=ptt_key, device=mic_device,
                                         context_terms=_ctx_terms)
            if not transcript:
                print("  (nothing heard)")
                continue

            print(f"  You  : {transcript!r}")
            stripped = transcript.strip()

            # ── Cancel active procedure or disambiguation ────────────────────
            if _CANCEL_PROC_RE.fullmatch(stripped) and (active_proc or disambiguation_node):
                active_proc = None
                active_step = 0
                disambiguation_node = None
                reply = _pick("roger")
                conf  = "HIGH"
                print("  [PROC] cancelled")

            # ── Repeat the current step ──────────────────────────────────────
            elif active_proc and _REPEAT_RE.fullmatch(stripped):
                _p = _load_proc(active_proc)
                steps = _p["steps"] if _p else []
                group_start = _maneuver_group_start(steps, active_step)
                active_step = group_start
                reply = _format_step(active_proc, active_step)
                conf  = "HIGH"
                print(f"  [PROC] repeat step {active_step + 1}/{len(steps)}")

            # ── Restart active procedure from step 1 ─────────────────────────
            elif active_proc and _RESTART_RE.search(stripped):
                active_step = 0
                reply = _format_step(active_proc, 0)
                conf  = "HIGH"
                _p = _load_proc(active_proc)
                print(f"  [PROC] restart → step 1/{len(_p['steps']) if _p else '?'}")

            # ── Advance to next step ──────────────────────────────────────────
            elif active_proc and _CONTINUE_RE.fullmatch(stripped):
                _p = _load_proc(active_proc)
                steps = _p["steps"] if _p else []
                # Skip past any maneuver group that was just delivered
                group_end = _maneuver_group_end(steps, active_step)
                active_step = group_end + 1
                if active_step >= len(steps):
                    reply = f"{_p['canonical'] if _p else active_proc} complete."
                    active_proc = None
                    active_step = 0
                    print("  [PROC] complete")
                else:
                    reply = _format_step(active_proc, active_step)
                    print(f"  [PROC] step {active_step + 1}/{len(steps)}")
                conf = "HIGH"

            # ── Disambiguation in progress ────────────────────────────────────
            elif disambiguation_node:
                opt = _match_disambig_option(stripped, disambiguation_node["options"])
                if opt:
                    if "proc_key" in opt:
                        # Terminal node — enter the procedure
                        active_proc = opt["proc_key"]
                        active_step = 0
                        disambiguation_node = None
                        reply = _format_proc_intro(active_proc)
                        _p = _load_proc(active_proc)
                        print(f"  [DISAMBIG→PROC] {active_proc} ({len(_p['steps']) if _p else '?'} steps)")
                    else:
                        # Intermediate node — ask next question
                        disambiguation_node = opt
                        reply = opt["question"]
                        print(f"  [DISAMBIG] → {opt['label']!r}: {reply!r}")
                else:
                    reply = f"{_pick('say_again')} {disambiguation_node['question']}"
                    print(f"  [DISAMBIG] no match, re-asking")
                conf = "HIGH"

            # ── General query (Q&A or new procedure) ─────────────────────────
            else:
                state = _fetch_state()
                mode  = "IN-JET" if state else "GROUND"
                reply, conf, proc_key = _ask_ollama(transcript, state, model, history, low_streak)
                new_streak = (low_streak + 1) if conf == "LOW" else 0
                print(f"  [{mode}] conf={conf} streak={new_streak}")

                if proc_key and proc_key.startswith("__DISAMBIG__:"):
                    gid = proc_key.split(":", 1)[1]
                    dg = next((g for g in _DISAMBIG_GROUPS if g["id"] == gid), None)
                    if dg:
                        disambiguation_node = dg
                        reply = dg["question"]
                        conf  = "HIGH"
                        print(f"  [DISAMBIG] enter group '{gid}'")
                elif proc_key:
                    active_proc = proc_key
                    active_step = 0
                    _p = _load_proc(proc_key)
                    reply = _format_proc_intro(proc_key)
                    conf  = "HIGH"
                    print(f"  [PROC] enter '{proc_key}' ({len(_p['steps']) if _p else '?'} steps)")
                elif conf == "LOW":
                    # Clear questions (question word + ≥ 3 words) are intelligible even when
                    # RAG scores LOW — let the LLM reply through without streak interception.
                    _clear_q = bool(_QUESTION_RE.search(transcript)) and len(transcript.split()) >= 3
                    if _clear_q:
                        pass  # use LLM reply as-is; streak still increments
                    elif new_streak == 1:
                        reply = _pick("say_again")
                    elif new_streak == 2:
                        reply = f"{_pick('were_you_asking')} {reply}?" if reply else _pick("say_again")
                    else:
                        reply = _pick("not_following")
                        new_streak = 0
                elif not reply:
                    reply = _pick("say_again")

                low_streak = new_streak

            print(f"  Reply: {reply}\n")

            # ── TTS ──────────────────────────────────────────────────────────
            if speak:
                stop_tts = threading.Event()
                tts_done = threading.Event()

                def _interrupt_watch():
                    while not tts_done.is_set():
                        if ptt_vk and _key_held(ptt_vk):
                            stop_tts.set()
                            return
                        _time.sleep(0.04)

                threading.Thread(target=_interrupt_watch, daemon=True).start()
                try:
                    tts.speak(_voiced(reply, conf), stop_event=stop_tts)
                finally:
                    tts_done.set()

        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt",      default="caps_lock",
                        help="PTT key (default: caps_lock)")
    parser.add_argument("--mic",      type=int, default=None,
                        help="PyAudio input device index")
    parser.add_argument("--no-speak", action="store_true",
                        help="Print reply without audio")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    run(ptt_key=args.ptt, mic_device=args.mic, speak=not args.no_speak, model=args.model)
