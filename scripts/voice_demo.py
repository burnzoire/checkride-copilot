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

SYSTEM_PROMPT_INFLIGHT = """\
You are Checkride Copilot, a laconic but friendly F/A-18C NATOPS advisor speaking over intercom to a pilot in the jet.
Relevant excerpts from the NATOPS guide are provided before each question under [Reference]. Use them as your primary source.
Live cockpit state is also provided — use it when the question is about current systems or a specific action.
Speak like an experienced WSO: direct, calm, conversational. No jargon beyond NATOPS terms.
Keep answers to 1-2 short sentences. Never list steps or use bullet points — prose only.
Answer about the weapon or system the pilot named, not whatever happens to be loaded.
Give exact values from the reference (switch names, locations, settings). Do not paraphrase vaguely.
If the reference doesn't cover the question, say so plainly — do not invent an answer.
If the question is unclear or cut off, ask the pilot to say again in plain words.
For social/conversational inputs (thanks, roger, etc.), respond naturally and briefly.
"""

SYSTEM_PROMPT_GROUND = """\
You are Checkride Copilot, a laconic but friendly F/A-18C NATOPS advisor.
Relevant excerpts from the NATOPS guide are provided before each question under [Reference]. Use them as your primary source.
Speak like an experienced WSO: direct, calm, conversational.
Keep answers to 1-2 short sentences. Never list steps or use bullet points — prose only.
Give exact values from the reference (switch names, locations, settings). Do not paraphrase vaguely.
If the reference doesn't cover the question, say so plainly — do not invent an answer.
For social/conversational inputs (thanks, roger, etc.), respond naturally and briefly.
"""


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
    r"speed|altitude|heading|start|master|arm|fire|release|track|lock)\b",
    re.IGNORECASE,
)


def _build_search_query(transcript: str, history: list[dict]) -> str:
    """For short or low-keyword queries, prepend the most recent topic from history."""
    words = transcript.split()
    has_keywords = bool(_AVIATION_RE.search(transcript))
    if len(words) >= 6 or has_keywords:
        return transcript

    # Pull the last user turn from history to give BM25 more signal
    for msg in reversed(history):
        if msg["role"] == "user":
            prior = msg["content"]
            # Strip injected blocks so we don't pollute the search query
            prior = re.sub(r"\[Reference\].*?\n\n", "", prior, flags=re.DOTALL)
            prior = re.sub(r"\[Cockpit state\].*?\n\n", "", prior, flags=re.DOTALL)
            prior = prior.replace("Pilot: ", "").strip()
            return f"{prior} {transcript}"

    return transcript


def _ask_ollama(
    transcript: str,
    state: dict | None,
    model: str,
    history: list[dict],
) -> str:
    in_jet    = state is not None
    sys_prompt = SYSTEM_PROMPT_INFLIGHT if in_jet else SYSTEM_PROMPT_GROUND

    ref_block = ""
    if _RAG_AVAILABLE:
        search_query = _build_search_query(transcript, history)
        hits = _rag_search(search_query, top_k=2)
        if hits:
            excerpts = "\n---\n".join(h["text"] for h in hits)
            ref_block = f"[Reference]\n{excerpts}\n\n"

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
            "num_predict": 80,
            "temperature": 0.35,
            "repeat_penalty": 1.1,
            "stop": ["\n1", "\n2", "\n-", "\n•", "\n*", "\nStep"],
        },
    }

    try:
        r = httpx.post(OLLAMA_URL, json=payload, timeout=15.0)
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
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
