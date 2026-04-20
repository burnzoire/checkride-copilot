#!/usr/bin/env python3
"""
LLM-first spike harness for Checkride Copilot.

Goal:
- Keep the existing STT/TTS transport loop.
- Use LLM tool-calling as the primary path for routing and context retrieval.
- Keep deterministic guards for procedure navigation commands only.

This is an experiment harness, not a production orchestrator.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger

from diagnostic.action_map import normalize_action
from diagnostic.engine import diagnose_action
from diagnostic.rules import get_rules_for_action
from retrieval.cockpit_lookup import format_location, lookup as cockpit_lookup, suggest as cockpit_suggest
from retrieval.keybinds import describe_switch
from retrieval.search import search as rag_search
from voice import stt, tts

COLLECTOR_URL = "http://127.0.0.1:7779/state"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"
STATE_FRESH_MS = 5000
MAX_HISTORY = 10
MAX_TOOL_CALLS = 2

TARGET_TOKENS_BY_CONF = {
    "HIGH": 60,
    "MEDIUM": 50,
    "LOW": 40,
}

MAX_TOKENS_BY_CONF = {
    "HIGH": 90,
    "MEDIUM": 80,
    "LOW": 70,
}

TEMP_BY_CONF = {
    "HIGH": 0.1,
    "MEDIUM": 0.2,
    "LOW": 0.15,
}

# Deterministic nav commands for active procedure mode.
_CONTINUE_RE = re.compile(
    r"^\s*(check|done|ok|okay|next|go\s+(on|ahead)|continue|ready|"
    r"roger|copy|affirm|affirmative|wilco|proceed|confirm(ed)?|yes|yep|go)\s*[.,!]?\s*$",
    re.IGNORECASE,
)

_REPEAT_RE = re.compile(
    r"^\s*(what|huh|say\s+again|come\s+again|repeat|again|"
    r"what\s+(did\s+you\s+(say|just\s+say)|was\s+that)|"
    r"repeat\s+(the\s+)?(last(\s+step)?|that)|"
    r"i\s+(didn.?t|did\s+not)\s+(hear|catch)\s+(that|you))\s*[?,!.]?\s*$",
    re.IGNORECASE,
)

_RESTART_RE = re.compile(
    r"\b(start\s+(from\s+)?(the\s+)?(start|beginning|top)|"
    r"restart|from\s+the\s+(top|start|beginning)|"
    r"back\s+to\s+(the\s+)?(start|beginning|top)|"
    r"again\s+from\s+the\s+(start|top|beginning))\b",
    re.IGNORECASE,
)

_CANCEL_PROC_RE = re.compile(
    r"^\s*(cancel|stop|forget\s+it|never\s*mind|that.?s\s+(all|it|fine|good))\s*[.,!]?\s*$",
    re.IGNORECASE,
)

_PROC_DIR = Path(__file__).parent.parent / "data" / "airframes" / "fa18c" / "procedures"
_PROC_INDEX: list[dict[str, Any]] = []
_PROC_CACHE: dict[str, dict[str, Any]] = {}


def _init_proc_index() -> None:
    index_path = _PROC_DIR / "index.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        _PROC_INDEX.extend(data.get("procedures", []))
    except Exception as e:
        logger.warning(f"Could not load procedures index: {e}")


_init_proc_index()


@dataclass
class Session:
    active_proc: str | None = None
    active_step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    last_confidence: str = "MEDIUM"
    last_reply_incomplete: bool = False


def _clean_step_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\u2013\u2014\u2018\u2019]", " ", text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_proc(key: str) -> dict[str, Any] | None:
    if key in _PROC_CACHE:
        return _PROC_CACHE[key]
    for entry in _PROC_INDEX:
        if entry.get("key") == key:
            proc_path = _PROC_DIR / entry["path"]
            try:
                proc = json.loads(proc_path.read_text(encoding="utf-8"))
                _PROC_CACHE[key] = proc
                return proc
            except Exception as e:
                logger.warning(f"Could not load procedure {key}: {e}")
                return None
    return None


def _validate_proc_key(key: str) -> bool:
    return any(p.get("key") == key for p in _PROC_INDEX)


def _maneuver_group_end(steps: list[dict[str, Any]], step_idx: int) -> int:
    if step_idx >= len(steps) or steps[step_idx].get("pace") != "maneuver":
        return step_idx
    i = step_idx
    while i + 1 < len(steps) and steps[i + 1].get("pace") == "maneuver":
        i += 1
    return i


def _format_step(proc_key: str, step_idx: int) -> str:
    proc = _load_proc(proc_key)
    if not proc:
        return "Procedure unavailable."

    steps = proc.get("steps", [])
    if not steps or step_idx >= len(steps):
        return f"{proc.get('canonical', proc_key)} complete."

    step = steps[step_idx]
    if step.get("pace") == "maneuver":
        parts = []
        i = step_idx
        while i < len(steps) and steps[i].get("pace") == "maneuver":
            parts.append(_clean_step_text(steps[i].get("voiced") or steps[i].get("action", "")))
            i += 1
        return " ".join(p for p in parts if p)

    return _clean_step_text(step.get("voiced") or step.get("action", ""))


def _format_proc_intro(proc_key: str) -> str:
    proc = _load_proc(proc_key)
    if not proc:
        return "Procedure unavailable."
    steps = proc.get("steps", [])
    if not steps:
        return f"{proc.get('canonical', proc_key)} has no steps."
    first = _clean_step_text(steps[0].get("voiced") or steps[0].get("action", ""))
    return f"{proc.get('canonical', proc_key)}. {len(steps)} steps total. First: {first}"


def _fetch_state() -> dict[str, Any] | None:
    try:
        r = httpx.get(COLLECTOR_URL, timeout=1.5)
        r.raise_for_status()
        state = r.json()
        if state.get("data_age_ms", 99999) > STATE_FRESH_MS:
            return None
        return state
    except Exception:
        return None


def _confidence_from_score(score: float) -> str:
    if score >= 18:
        return "HIGH"
    if score >= 14:
        return "MEDIUM"
    return "LOW"


def _clean_reply(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_incomplete(text: str, done_reason: str | None) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    # Ollama marks token-limited outputs explicitly.
    if (done_reason or "").lower() in {"length", "max_tokens"}:
        return True

    # If no clear sentence boundary, assume it may be cut.
    if not re.search(r"[.!?]['\")\]]?$", t):
        trailing = t.split()[-1].lower() if t.split() else ""
        if trailing in {
            "and",
            "or",
            "to",
            "for",
            "with",
            "if",
            "when",
            "because",
            "while",
            "during",
            "the",
            "a",
            "an",
            "your",
            "flight",
            "of",
            "in",
            "on",
        }:
            return True
        # If text is long and still no terminal punctuation, likely cut.
        if len(t.split()) >= 15:
            return True

    return False


def _stitch_reply(base: str, continuation: str) -> str:
    b = (base or "").strip()
    c = (continuation or "").strip()
    if not b:
        return c
    if not c:
        return b
    if c[0] in ".,;:!?":
        return (b + c).strip()
    return (b + " " + c).strip()


def _tool_search_natops(args: dict[str, Any], session: Session) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k", 3))
    min_score = float(args.get("min_score", 8.0))
    if not query:
        return {"ok": False, "error": "query is required"}

    hits = rag_search(query, top_k=max(1, min(top_k, 5)), min_score=min_score)
    top_score = float(hits[0]["score"]) if hits else 0.0
    conf = _confidence_from_score(top_score) if hits else "LOW"
    session.last_confidence = conf
    return {
        "ok": True,
        "query": query,
        "confidence": conf,
        "top_score": top_score,
        "hits": [
            {
                "section": h.get("section", ""),
                "page": h.get("page"),
                "score": round(float(h.get("score", 0.0)), 2),
                "text": h.get("text", "")[:1400],
            }
            for h in hits
        ],
    }


def _tool_lookup_switch(args: dict[str, Any], _session: Session) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is required"}

    hit = cockpit_lookup(query)
    if not hit:
        hint = cockpit_suggest(query)
        if hint:
            return {"ok": False, "found": False, "hint": format_location(hint)}
        return {"ok": False, "found": False, "error": "No switch matched"}

    kb = describe_switch(hit.get("dcs_actions") or [])
    return {
        "ok": True,
        "found": True,
        "canonical": hit.get("canonical"),
        "location": format_location(hit),
        "keybinds": kb,
    }


def _tool_get_cockpit_state(_args: dict[str, Any], _session: Session) -> dict[str, Any]:
    state = _fetch_state()
    if not state:
        return {"ok": False, "available": False}
    return {
        "ok": True,
        "available": True,
        "airframe": state.get("airframe_normalized"),
        "master_arm": state.get("master_arm"),
        "radar_mode": state.get("radar_mode"),
        "altitude_ft": state.get("altitude_ft"),
        "airspeed_kts": state.get("airspeed_kts"),
        "heading_deg": state.get("heading_deg"),
        "data_age_ms": state.get("data_age_ms"),
    }


def _tool_diagnose_action(args: dict[str, Any], _session: Session) -> dict[str, Any]:
    action = str(args.get("action", "")).strip()
    if not action:
        return {"ok": False, "error": "action is required"}

    state = _fetch_state()
    if not state:
        return {"ok": False, "error": "No fresh cockpit state available"}

    action_norm = normalize_action(action)
    rules = get_rules_for_action(action_norm)
    if not rules:
        return {"ok": False, "error": f"No rules for action '{action_norm}'"}

    result = diagnose_action(action_norm, state)
    blockers = result.get("blockers", [])
    warnings = result.get("warnings", [])
    return {
        "ok": True,
        "action": action_norm,
        "can_execute": bool(result.get("ok")),
        "blockers": blockers,
        "warnings": warnings,
    }


def _tool_list_procedures(args: dict[str, Any], _session: Session) -> dict[str, Any]:
    query = str(args.get("query", "")).strip().lower()
    limit = int(args.get("limit", 5))
    if not query:
        return {"ok": False, "error": "query is required"}

    tokens = set(re.findall(r"[a-z0-9]+", query))
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in _PROC_INDEX:
        text = " ".join([
            p.get("key", "").replace("_", " "),
            p.get("canonical", ""),
            " ".join(p.get("alternates", []) or []),
            str(p.get("source", {}).get("section", "")),
        ]).lower()
        words = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(tokens & words)
        if overlap > 0:
            scored.append((overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, p in scored[: max(1, min(limit, 8))]:
        out.append(
            {
                "score": score,
                "key": p.get("key"),
                "canonical": p.get("canonical"),
                "category": p.get("category"),
                "step_count": p.get("step_count"),
            }
        )
    return {"ok": True, "matches": out}


def _tool_start_procedure(args: dict[str, Any], session: Session) -> dict[str, Any]:
    proc_key = str(args.get("proc_key", "")).strip()
    if not proc_key:
        return {"ok": False, "error": "proc_key is required"}
    if not _validate_proc_key(proc_key):
        return {"ok": False, "error": f"Unknown proc_key '{proc_key}'"}

    session.active_proc = proc_key
    session.active_step = 0
    intro = _format_proc_intro(proc_key)
    return {"ok": True, "active_proc": proc_key, "intro": intro}


def _tool_get_active_procedure(_args: dict[str, Any], session: Session) -> dict[str, Any]:
    if not session.active_proc:
        return {"ok": True, "active": False}

    proc = _load_proc(session.active_proc)
    steps = proc.get("steps", []) if proc else []
    return {
        "ok": True,
        "active": True,
        "proc_key": session.active_proc,
        "canonical": proc.get("canonical") if proc else session.active_proc,
        "active_step": session.active_step + 1,
        "step_count": len(steps),
        "current_step_text": _format_step(session.active_proc, session.active_step),
    }


ToolFn = Callable[[dict[str, Any], Session], dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "search_natops": _tool_search_natops,
    "lookup_switch": _tool_lookup_switch,
    "get_cockpit_state": _tool_get_cockpit_state,
    "diagnose_action": _tool_diagnose_action,
    "list_procedures": _tool_list_procedures,
    "start_procedure": _tool_start_procedure,
    "get_active_procedure": _tool_get_active_procedure,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_natops",
            "description": "Search indexed NATOPS/procedure chunks. Use for tactical/procedural questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "min_score": {"type": "number", "default": 8.0},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_switch",
            "description": "Find exact cockpit switch location and keybind hints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cockpit_state",
            "description": "Get fresh cockpit state snapshot from the local collector.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_action",
            "description": "Diagnose blockers for a known action using live cockpit state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_procedures",
            "description": "Find candidate procedures from the indexed procedure catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_procedure",
            "description": "Activate a procedure by exact proc_key. Only call after list_procedures picks a clear match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proc_key": {"type": "string"},
                },
                "required": ["proc_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_procedure",
            "description": "Read current active procedure and step status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """
You are Checkride Copilot, an F/A-18C instructor voice assistant.

You are operating in an LLM-first tool-calling harness.
Rules:
1) Use tools before answering procedural or switch-location questions.
2) Never invent switch names, procedure keys, or cockpit states.
3) If confidence is low, ask one precise clarifying question.
4) Keep spoken answers concise: usually 1-2 sentences.
5) If start_procedure succeeds, confirm activation and brief first step.
6) For an active procedure, navigation words (next/repeat/restart/cancel) are handled externally.
""".strip()


def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    func = call.get("function", {}) if isinstance(call, dict) else {}
    name = func.get("name", "")
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _call_ollama_with_tools(transcript: str, session: Session, model: str) -> str:
    session.last_reply_incomplete = False
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session.history[-MAX_HISTORY * 2 :]

    proc_context = ""
    if session.active_proc:
        proc_context = (
            f"\nSession context: active procedure is {session.active_proc}, "
            f"currently at step {session.active_step + 1}."
        )

    messages.append({"role": "user", "content": transcript + proc_context})

    total_tool_calls = 0
    for _ in range(MAX_TOOL_CALLS + 1):
        conf = session.last_confidence if session.last_confidence in TARGET_TOKENS_BY_CONF else "MEDIUM"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": TOOL_SCHEMAS,
            "options": {
                # Soft target budget for first pass.
                "num_predict": TARGET_TOKENS_BY_CONF[conf],
                "temperature": TEMP_BY_CONF[conf],
                "repeat_penalty": 1.15,
                "stop": ["[Reference", "[Switch", "let me know", "if you have"],
            },
        }

        r = httpx.post(OLLAMA_URL, json=payload, timeout=25.0)
        r.raise_for_status()
        resp = r.json()
        msg = resp.get("message", {})
        done_reason = str(resp.get("done_reason", ""))
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            content = _clean_reply(msg.get("content", ""))
            if not content:
                return "Say again?"

            # If the response looks cut, ask for one continuation pass.
            if _looks_incomplete(content, done_reason):
                remaining = max(0, MAX_TOKENS_BY_CONF[conf] - TARGET_TOKENS_BY_CONF[conf])
                if remaining > 0:
                    cont_messages = messages + [
                        msg,
                        {
                            "role": "user",
                            "content": (
                                "Continue exactly where you stopped. "
                                "Do not repeat prior text. Finish the thought in one concise sentence."
                            ),
                        },
                    ]
                    cont_payload = {
                        "model": model,
                        "messages": cont_messages,
                        "stream": False,
                        "options": {
                            "num_predict": remaining,
                            "temperature": TEMP_BY_CONF[conf],
                            "repeat_penalty": 1.15,
                            "stop": ["[Reference", "[Switch", "let me know", "if you have"],
                        },
                    }
                    rc = httpx.post(OLLAMA_URL, json=cont_payload, timeout=20.0)
                    rc.raise_for_status()
                    cont_resp = rc.json()
                    cont_text = _clean_reply(cont_resp.get("message", {}).get("content", ""))
                    stitched = _stitch_reply(content, cont_text)
                    session.last_reply_incomplete = _looks_incomplete(stitched, str(cont_resp.get("done_reason", "")))
                    return stitched

                session.last_reply_incomplete = True
                return content

            return content

        messages.append(msg)

        for tc in tool_calls:
            if total_tool_calls >= MAX_TOOL_CALLS:
                break
            name, args = _parse_tool_call(tc)
            fn = TOOL_REGISTRY.get(name)
            if not fn:
                result = {"ok": False, "error": f"Unknown tool '{name}'"}
            else:
                try:
                    result = fn(args, session)
                except Exception as e:
                    result = {"ok": False, "error": f"Tool '{name}' failed: {e}"}

            total_tool_calls += 1
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                }
            )

        if total_tool_calls >= MAX_TOOL_CALLS:
            messages.append(
                {
                    "role": "user",
                    "content": "Tool call limit reached. Provide the best concise spoken answer now.",
                }
            )

    return "I do not have enough context yet. Say your request again with one more detail."


def _continue_last_answer(session: Session, model: str) -> str:
    """Continue the previous assistant reply when it was cut off."""
    conf = session.last_confidence if session.last_confidence in TARGET_TOKENS_BY_CONF else "MEDIUM"
    extra_budget = max(20, MAX_TOKENS_BY_CONF[conf] - TARGET_TOKENS_BY_CONF[conf])
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session.history[-MAX_HISTORY * 2 :]
    messages.append(
        {
            "role": "user",
            "content": (
                "Continue your previous answer exactly from where it stopped. "
                "Do not restart or repeat. Finish in one concise sentence."
            ),
        }
    )

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": extra_budget,
            "temperature": TEMP_BY_CONF[conf],
            "repeat_penalty": 1.15,
            "stop": ["[Reference", "[Switch", "let me know", "if you have"],
        },
    }
    r = httpx.post(OLLAMA_URL, json=payload, timeout=20.0)
    r.raise_for_status()
    resp = r.json()
    text = _clean_reply(resp.get("message", {}).get("content", ""))
    session.last_reply_incomplete = _looks_incomplete(text, str(resp.get("done_reason", "")))
    return text or "Say again?"


_PTT_VK: dict[str, int] = {
    "caps_lock": 0x14,
    "scroll_lock": 0x91,
    "num_lock": 0x90,
    **{f"f{n}": 0x6B + n for n in range(13, 25)},
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


def run(ptt_key: str, mic_device: int | None, speak: bool, model: str) -> None:
    logger.info("Pre-loading Whisper ...")
    stt._get_model()

    if speak:
        logger.info("Pre-loading TTS ...")
        tts.prewarm()

    print("\nCheckride Copilot - LLM-First Spike")
    print(f"PTT : {ptt_key.upper()}  |  Model : {model}  |  Audio : {'on' if speak else 'off'}")
    print(f"Hold [{ptt_key.upper()}] and speak.  Ctrl-C to quit.\n")

    session = Session()
    ptt_vk = _ptt_vk(ptt_key)

    while True:
        try:
            ctx_terms = None
            if session.active_proc:
                proc = _load_proc(session.active_proc)
                if proc:
                    ctx_terms = proc.get("terminology") or None

            transcript = stt.listen_once(ptt_key=ptt_key, device=mic_device, context_terms=ctx_terms)
            if not transcript:
                print("  (nothing heard)")
                continue

            print(f"  You  : {transcript!r}")
            stripped = transcript.strip()

            reply = ""

            # Deterministic nav guard for active procedures.
            if session.active_proc and _CANCEL_PROC_RE.fullmatch(stripped):
                session.active_proc = None
                session.active_step = 0
                reply = "Procedure cancelled."
                print("  [PROC] cancelled")
            elif session.active_proc and _REPEAT_RE.fullmatch(stripped):
                proc = _load_proc(session.active_proc)
                steps = proc.get("steps", []) if proc else []
                if steps:
                    reply = _format_step(session.active_proc, session.active_step)
                    print(f"  [PROC] repeat step {session.active_step + 1}/{len(steps)}")
                else:
                    reply = "Procedure unavailable."
            elif session.active_proc and _RESTART_RE.search(stripped):
                session.active_step = 0
                reply = _format_step(session.active_proc, 0)
                proc = _load_proc(session.active_proc)
                total = len(proc.get("steps", [])) if proc else "?"
                print(f"  [PROC] restart -> step 1/{total}")
            elif session.active_proc and _CONTINUE_RE.fullmatch(stripped):
                proc = _load_proc(session.active_proc)
                steps = proc.get("steps", []) if proc else []
                if not steps:
                    reply = "Procedure unavailable."
                else:
                    group_end = _maneuver_group_end(steps, session.active_step)
                    session.active_step = group_end + 1
                    if session.active_step >= len(steps):
                        reply = f"{proc.get('canonical', session.active_proc)} complete."
                        session.active_proc = None
                        session.active_step = 0
                        print("  [PROC] complete")
                    else:
                        reply = _format_step(proc.get("key", ""), session.active_step)
                        print(f"  [PROC] step {session.active_step + 1}/{len(steps)}")
            elif (not session.active_proc) and _CONTINUE_RE.fullmatch(stripped) and session.last_reply_incomplete:
                reply = _continue_last_answer(session, model)
            else:
                reply = _call_ollama_with_tools(transcript, session, model)

            print(f"  Reply: {reply}\n")

            session.history.append({"role": "user", "content": transcript, "ts": _time.time()})
            session.history.append({"role": "assistant", "content": reply, "ts": _time.time()})

            if speak:
                stop_tts = threading.Event()
                tts_done = threading.Event()

                def _interrupt_watch() -> None:
                    while not tts_done.is_set():
                        if ptt_vk and _key_held(ptt_vk):
                            stop_tts.set()
                            return
                        _time.sleep(0.04)

                threading.Thread(target=_interrupt_watch, daemon=True).start()
                try:
                    tts.speak(reply, stop_event=stop_tts)
                finally:
                    tts_done.set()

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            logger.error(f"Harness loop error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt", default="caps_lock", help="PTT key (default: caps_lock)")
    parser.add_argument("--mic", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--no-speak", action="store_true", help="Print reply without audio")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    run(ptt_key=args.ptt, mic_device=args.mic, speak=not args.no_speak, model=args.model)


if __name__ == "__main__":
    main()
