from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import httpx
from loguru import logger

from diagnostic.action_map import normalize_action
from diagnostic.engine import diagnose_action
from diagnostic.rules import get_rules_for_action
from orchestrator.session import Session
from retrieval.cockpit_lookup import format_location, lookup as cockpit_lookup, suggest as cockpit_suggest
from retrieval.keybinds import describe_switch
from retrieval.search import search as rag_search

COLLECTOR_URL = "http://127.0.0.1:7779/state"
STATE_FRESH_MS = 5000

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


def confidence_from_score(score: float) -> str:
    if score >= 18:
        return "HIGH"
    if score >= 14:
        return "MEDIUM"
    return "LOW"


def fetch_state() -> dict[str, Any] | None:
    try:
        r = httpx.get(COLLECTOR_URL, timeout=1.5)
        r.raise_for_status()
        state = r.json()
        if state.get("data_age_ms", 99999) > STATE_FRESH_MS:
            return None
        return state
    except Exception:
        return None


def clean_step_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\u2013\u2014\u2018\u2019]", " ", text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def load_proc(key: str) -> dict[str, Any] | None:
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


def validate_proc_key(key: str) -> bool:
    return any(p.get("key") == key for p in _PROC_INDEX)


def maneuver_group_end(steps: list[dict[str, Any]], step_idx: int) -> int:
    if step_idx >= len(steps) or steps[step_idx].get("pace") != "maneuver":
        return step_idx
    i = step_idx
    while i + 1 < len(steps) and steps[i + 1].get("pace") == "maneuver":
        i += 1
    return i


def format_step(proc_key: str, step_idx: int) -> str:
    proc = load_proc(proc_key)
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
            parts.append(clean_step_text(steps[i].get("voiced") or steps[i].get("action", "")))
            i += 1
        return " ".join(p for p in parts if p)

    return clean_step_text(step.get("voiced") or step.get("action", ""))


def format_proc_intro(proc_key: str) -> str:
    proc = load_proc(proc_key)
    if not proc:
        return "Procedure unavailable."
    steps = proc.get("steps", [])
    if not steps:
        return f"{proc.get('canonical', proc_key)} has no steps."
    first = clean_step_text(steps[0].get("voiced") or steps[0].get("action", ""))
    return f"{proc.get('canonical', proc_key)}. {len(steps)} steps total. First: {first}"


def tool_search_natops(args: dict[str, Any], session: Session) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k", 3))
    min_score = float(args.get("min_score", 8.0))
    if not query:
        return {"ok": False, "error": "query is required"}

    hits = rag_search(query, top_k=max(1, min(top_k, 5)), min_score=min_score)
    top_score = float(hits[0]["score"]) if hits else 0.0
    conf = confidence_from_score(top_score) if hits else "LOW"
    session.last_retrieval_confidence = conf
    session.turn_confidence = conf
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


def tool_lookup_switch(args: dict[str, Any], _session: Session) -> dict[str, Any]:
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


def tool_get_cockpit_state(_args: dict[str, Any], _session: Session) -> dict[str, Any]:
    state = fetch_state()
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


def tool_diagnose_action(args: dict[str, Any], _session: Session) -> dict[str, Any]:
    action = str(args.get("action", "")).strip()
    if not action:
        return {"ok": False, "error": "action is required"}

    state = fetch_state()
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


def tool_list_procedures(args: dict[str, Any], _session: Session) -> dict[str, Any]:
    query = str(args.get("query", "")).strip().lower()
    limit = int(args.get("limit", 5))
    if not query:
        return {"ok": False, "error": "query is required"}

    tokens = set(re.findall(r"[a-z0-9]+", query))
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in _PROC_INDEX:
        text = " ".join(
            [
                p.get("key", "").replace("_", " "),
                p.get("canonical", ""),
                " ".join(p.get("alternates", []) or []),
                str(p.get("source", {}).get("section", "")),
            ]
        ).lower()
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


def tool_start_procedure(args: dict[str, Any], session: Session) -> dict[str, Any]:
    proc_key = str(args.get("proc_key", "")).strip()
    if not proc_key:
        return {"ok": False, "error": "proc_key is required"}
    if not validate_proc_key(proc_key):
        return {"ok": False, "error": f"Unknown proc_key '{proc_key}'"}

    session.active_proc = proc_key
    session.active_step = 0
    intro = format_proc_intro(proc_key)
    return {"ok": True, "active_proc": proc_key, "intro": intro}


def tool_get_active_procedure(_args: dict[str, Any], session: Session) -> dict[str, Any]:
    if not session.active_proc:
        return {"ok": True, "active": False}

    proc = load_proc(session.active_proc)
    steps = proc.get("steps", []) if proc else []
    return {
        "ok": True,
        "active": True,
        "proc_key": session.active_proc,
        "canonical": proc.get("canonical") if proc else session.active_proc,
        "active_step": session.active_step + 1,
        "step_count": len(steps),
        "current_step_text": format_step(session.active_proc, session.active_step),
    }


ToolFn = Callable[[dict[str, Any], Session], dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "search_natops": tool_search_natops,
    "lookup_switch": tool_lookup_switch,
    "get_cockpit_state": tool_get_cockpit_state,
    "diagnose_action": tool_diagnose_action,
    "list_procedures": tool_list_procedures,
    "start_procedure": tool_start_procedure,
    "get_active_procedure": tool_get_active_procedure,
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
