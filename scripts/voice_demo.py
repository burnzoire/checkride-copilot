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
import json
import re
import sys
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


COLLECTOR_URL   = "http://127.0.0.1:7779/state"
OLLAMA_URL      = "http://localhost:11434/api/chat"
DEFAULT_MODEL   = "mistral:7b"
MAX_HISTORY     = 10    # max message pairs to keep (older turns dropped)
STATE_FRESH_MS  = 5000  # treat state as live only if updated within this window

_FEW_SHOT = """\

Examples of correct responses — follow this pattern exactly:

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
My reference mentions a PROBE switch in the AAR procedure but doesn't give a panel location — which console are you looking at?

[Reference confidence: LOW — no relevant procedure found]
Pilot: What's the trick to catching the wire?
My reference doesn't cover that. Can you describe what's giving you trouble?

[Reference confidence: LOW — no relevant procedure found]
Pilot: Can you tell me which console to look at?
I don't have that location in my reference — try the left console first, it's where the fuel system switches usually sit.
"""

SYSTEM_PROMPT_INFLIGHT = """\
You are Checkride Copilot, an F/A-18C NATOPS advisor on intercom. Calm, direct, experienced — never robotic.
Do not address the pilot as "Pilot". Do not apologise. Do not invent information.
A [Reference confidence] block precedes each question. It contains excerpts from the NATOPS guide.
HIGH confidence: repeat the exact switch name and setting from the reference. Nothing else.
MEDIUM/LOW confidence: state what the reference does or doesn't cover, then ask ONE clarifying question.
Never invent physical mechanisms, sensory cues, keyboard shortcuts, or console locations not in the reference.
""" + _FEW_SHOT

SYSTEM_PROMPT_GROUND = """\
You are Checkride Copilot, an F/A-18C NATOPS advisor. Calm, direct, experienced — never robotic.
Do not address the pilot as "Pilot". Do not apologise. Do not invent information.
A [Reference confidence] block precedes each question. It contains excerpts from the NATOPS guide.
HIGH confidence: repeat the exact switch name and setting from the reference. Nothing else.
MEDIUM/LOW confidence: state what the reference does or doesn't cover, then ask ONE clarifying question.
Never invent physical mechanisms, sensory cues, keyboard shortcuts, or console locations not in the reference.
""" + _FEW_SHOT


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
    r"\b(aim|amraam|sidewinder|gbu|bomb|laser|tgp|radar|engine|gear|flap|hook|"
    r"light|switch|panel|console|throttle|hud|mfd|sensor|fuel|hydraulic|"
    r"speed|altitude|heading|start|master|arm|fire|release|track|lock|probe|"
    r"refuel|refueling)\b",
    re.IGNORECASE,
)

# Meta / conversational patterns that should skip RAG entirely.
# The model must answer these from conversation history alone.
# Pure social/meta utterances — skip RAG, model replies from conversation history.
_SKIP_RAG_RE = re.compile(
    r"(you\s+(just\s+)?said"
    r"|i\s+thought\s+you"
    r"|why\s+did\s+you"
    r"|what.{0,30}(got|have|has)\s+to\s+do"
    r"|doesn.?t\s+make\s+sense"
    r"|can\s+you\s+tell\s+me\s+(which|where|what|how)"   # pilot bouncing model's question back
    r"|you\s+tell\s+me"
    r"|^\s*(thanks|thank\s+you|roger|copy|wilco|good|ok|okay|got\s+it|"
    r"okay\s+got\s+it|never\s+mind|stand\s+by|affirm|negative|say\s+again)[\s.!?]*$)",
    re.IGNORECASE,
)

# Social openers that a pilot might prepend to a real question.
# Strip these before building the BM25 query so they don't dilute signal.
_SOCIAL_PREFIX_RE = re.compile(
    r"^\s*(thanks[.,!]?\s+|thank\s+you[.,!]?\s+|roger[.,!]?\s+|"
    r"copy[.,!]?\s+|ok[.,!]?\s+|okay[.,!]?\s+|alright[.,!]?\s+|"
    r"got\s+it[.,!]?\s+|understood[.,!]?\s+)",
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

    best_text, best_score = "", 0
    for msg in history:
        t = _extract(msg)
        s = _kw_count(t)
        if s > best_score:
            best_score, best_text = s, t

    if best_text:
        return f"{best_text} {core}"
    return core

    return core


# Preamble patterns the model uses to introduce a list it's about to give.
# These lines end with ":" — the stop tokens ate the list that followed.
# Only strip lines that actually end with a colon (the list-intro marker).
_PREAMBLE_RE = re.compile(
    r"^(to\s+\w[^,]*,?\s*)?(here\s+are|you'?ll\s+need\s+to|"
    r"follow\s+these|the\s+\w+\s+steps?(\s+are)?|"
    r"here'?s\s+how)[^:]*:\s*$",
    re.IGNORECASE,
)


def _clean_reply(text: str) -> str:
    """Strip dangling list preambles left after stop tokens eat the list."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if not _PREAMBLE_RE.match(l)]
    return " ".join(lines).strip()


def _ask_ollama(
    transcript: str,
    state: dict | None,
    model: str,
    history: list[dict],
) -> str:
    in_jet    = state is not None
    sys_prompt = SYSTEM_PROMPT_INFLIGHT if in_jet else SYSTEM_PROMPT_GROUND

    is_meta = bool(_SKIP_RAG_RE.search(transcript))
    ref_block = "" if is_meta else "[Reference confidence: LOW — no relevant procedure found]\n\n"
    confidence = "LOW"
    if _RAG_AVAILABLE and not is_meta:
        search_query = _build_search_query(transcript, history)
        hits = _rag_search(search_query, top_k=2)
        if hits:
            top_score = hits[0]["score"]
            confidence = "HIGH" if top_score >= 18 else "MEDIUM" if top_score >= 14 else "LOW"
            excerpts = "\n---\n".join(h["text"] for h in hits)
            ref_block = f"[Reference confidence: {confidence}]\n{excerpts}\n\n"

    # Token budget and temperature scale with confidence.
    # HIGH: quote the switch name — 25 tokens is plenty, keep it tight.
    # LOW/meta: natural question or social reply needs more room.
    temperature  = {"HIGH": 0.1,  "MEDIUM": 0.25, "LOW": 0.45}.get(confidence, 0.25)
    num_predict  = {"HIGH": 30,   "MEDIUM": 55,   "LOW": 65}.get(confidence, 55)
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
            "stop": ["\n1", "\n2", "\n-", "\n•", "\n*", "\nStep", "steps:", "following steps", "[Reference"],
        },
    }

    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=15.0)
        r.raise_for_status()
        reply = _clean_reply(r.json()["message"]["content"].strip())
        # Append this turn to history for next call
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant",  "content": reply})
        return reply
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return "LLM unavailable."


def run(ptt_key: str, mic_device: int | None, speak: bool, model: str) -> None:
    logger.info("Pre-loading Whisper ...")
    stt._get_model()

    print(f"\nCheckride Copilot")
    print(f"PTT : {ptt_key.upper()}  |  Model : {model}  |  Audio : {'on' if speak else 'off'}")
    print(f"Hold [{ptt_key.upper()}] and speak.  Ctrl-C to quit.\n")

    history: list[dict] = []   # grows with each turn; trimmed inside _ask_ollama

    while True:
        try:
            transcript = stt.listen_once(ptt_key=ptt_key, device=mic_device)
            if not transcript:
                print("  (nothing heard)")
                continue

            print(f"  You  : {transcript!r}")

            state = _fetch_state()
            mode  = "IN-JET" if state else "GROUND"
            reply = _ask_ollama(transcript, state, model, history)
            print(f"  [{mode}]")

            print(f"  Reply: {reply}\n")

            if speak:
                tts.speak(reply)

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
