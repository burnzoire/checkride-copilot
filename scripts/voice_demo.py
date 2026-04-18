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
    _LOOKUP_AVAILABLE = True
except Exception:
    _LOOKUP_AVAILABLE = False


COLLECTOR_URL   = "http://127.0.0.1:7779/state"
OLLAMA_URL      = "http://localhost:11434/api/chat"
DEFAULT_MODEL   = "qwen2.5:7b"
MAX_HISTORY     = 10    # max message pairs to keep (older turns dropped)
STATE_FRESH_MS  = 5000  # treat state as live only if updated within this window

_FEW_SHOT = """\

Examples of correct responses:

[Reference confidence: HIGH]
Refueling Probe Control Switch EXTEND / RETRACT / EMERGENCY EXTENDED
Pilot: How do I open the refueling probe?
Set the Refueling Probe Control Switch to EXTEND.

[Reference confidence: HIGH]
Set FORMATION LIGHTS to BRT
Pilot: How do I turn on formation lights?
Set FORMATION LIGHTS to BRT.

[Reference confidence: MEDIUM]
Set PROBE switch to EXTEND (step 8 of AAR procedure)
Pilot: Where is the probe switch?
The PROBE switch — which console are you looking at?

[Reference confidence: LOW — no relevant procedure found]
Pilot: What's the trick to catching the wire?
I'm not solid on that one. What's giving you trouble?

[Reference confidence: LOW — no relevant procedure found]
Pilot: Can you tell me which console to look at?
Not certain of that location — start with the left console, that's where the fuel system switches sit.
"""

_SYSTEM_CORE = """\
You are Checkride Copilot, an experienced F/A-18C instructor on intercom. You know this aircraft's systems and procedures intimately.
Calm, direct, never robotic. Do not address the pilot as "Pilot". Do not apologise.
Never mention documents, manuals, guides, or references — you are the expert, speak from knowledge.
Never invent physical mechanisms, sensory cues, keyboard shortcuts, or locations you are not certain of.
A [Reference confidence] block precedes each question — use it to calibrate certainty:
HIGH: answer in 1-2 sentences. Include all relevant figures, switch names, and settings exactly. No preamble.
MEDIUM: answer what you know, ask ONE clarifying question if needed.
LOW: be honest that you're not certain, ask what's giving them trouble.
[Closest cockpit match] block: say "Did you mean X?" using the exact switch name, then state its location.
"""

SYSTEM_PROMPT_INFLIGHT = _SYSTEM_CORE + _FEW_SHOT
SYSTEM_PROMPT_GROUND   = _SYSTEM_CORE + _FEW_SHOT


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
    r"lock|probe|refuel|refueling|tacan|ils|jdam|harm|harpoon)\b",
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
    r"|you\s+tell\s+me)",
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


def _clean_reply(text: str) -> str:
    """Strip dangling list preambles, leaked tags, document references, and LLM filler tails."""
    text = _DOC_REF_RE.sub("", text).strip()
    text = _FILLER_TAIL_RE.sub("", text).strip()
    text = _LEAKED_TAG_RE.sub(".", text).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not _PREAMBLE_RE.match(l)]
    return " ".join(lines).strip()


def _ask_ollama(
    transcript: str,
    state: dict | None,
    model: str,
    history: list[dict],
    low_streak: int = 0,
) -> str:
    in_jet    = state is not None
    sys_prompt = SYSTEM_PROMPT_INFLIGHT if in_jet else SYSTEM_PROMPT_GROUND

    is_meta = bool(_SKIP_RAG_RE.search(transcript)) or _is_pure_social(transcript)
    ref_block = "" if is_meta else "[Reference confidence: LOW — no relevant procedure found]\n\n"
    confidence = "LOW"

    # If the previous assistant turn was a disambiguation question and the
    # current query is too vague to resolve it, re-ask rather than hallucinate.
    # Exception: if the user said a NATO phonetic word (Delta, Echo, Golf…),
    # treat it as a variant answer and let RAG resolve it with history context.
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
            else:
                history.append({"role": "user",      "content": transcript,   "ts": _time.time()})
                history.append({"role": "assistant",  "content": last_reply,   "ts": _time.time()})
                return last_reply, "MEDIUM"

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
            spoken    = hit.get("spoken") or hit.get("positions", [])
            if spoken:
                if len(spoken) == 2:
                    pos_str = f"{spoken[0]} or {spoken[1]}"
                else:
                    pos_str = ", ".join(spoken[:-1]) + f", or {spoken[-1]}"
                reply = f"{canonical} is on the {area}, {panel}. It goes {pos_str}."
            else:
                reply = f"{canonical} is on the {area}, {panel}."
            history.append({"role": "user",      "content": transcript, "ts": _time.time()})
            history.append({"role": "assistant",  "content": reply,      "ts": _time.time()})
            return reply, "HIGH"

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
            if len(sections) >= 2 and confidence != "HIGH" and not is_enumeration:
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
                return q, "MEDIUM"

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
                ref_block = "[Reference confidence: LOW — give the best available procedure from your knowledge, do not ask again]\n\n"

    # Token budget and temperature scale with confidence.
    temperature  = {"HIGH": 0.1,  "MEDIUM": 0.25, "LOW": 0.45}.get(confidence, 0.25)
    num_predict  = {"HIGH": 120,  "MEDIUM": 100,  "LOW": 100}.get(confidence, 100)
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
        return reply, confidence
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return "LLM unavailable.", "LOW"


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
    r"roger|copy|affirm|wilco|proceed|confirmed?|yes|yep|go)\s*[.,!]?\s*$",
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

    history:       list[dict] = []
    low_streak:    int        = 0
    pending_steps: list[str]  = []   # remaining steps in an active procedure

    while True:
        try:
            transcript = stt.listen_once(ptt_key=ptt_key, device=mic_device)
            if not transcript:
                print("  (nothing heard)")
                continue

            print(f"  You  : {transcript!r}")

            # ── Step continuation ────────────────────────────────────────────
            if pending_steps and _CONTINUE_RE.fullmatch(transcript.strip()):
                step  = pending_steps.pop(0)
                reply = _step_tts(step)
                conf  = "HIGH"
                print(f"  [STEP] {len(pending_steps)} remaining")
            else:
                # Topic changed — discard pending steps
                if pending_steps:
                    pending_steps.clear()

                state = _fetch_state()
                mode  = "IN-JET" if state else "GROUND"
                reply, conf = _ask_ollama(transcript, state, model, history, low_streak)
                low_streak = (low_streak + 1) if conf == "LOW" else 0
                print(f"  [{mode}] conf={conf} streak={low_streak}")

                # Enter step mode if reply contains a numbered procedure
                steps = _split_steps(reply)
                if steps:
                    pending_steps = steps[1:]
                    reply = _step_tts(steps[0])
                    print(f"  [STEP MODE] {len(steps)} steps, speaking step 1/{len(steps)}")

            print(f"  Reply: {reply}\n")

            # ── TTS ─────────────────────────────────────────────────────────
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
                    voiced = _voiced(reply, conf)
                    filler_m = re.match(r'^((?:Hmm|Uh|Well|Right|Let me think)[.,!]*\.?\s+)', voiced, re.I)
                    if filler_m:
                        tts.speak(filler_m.group(1).strip(), stop_event=stop_tts)
                        if not stop_tts.is_set():
                            tts.speak(voiced[filler_m.end():], stop_event=stop_tts)
                    else:
                        tts.speak(voiced, stop_event=stop_tts)
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
