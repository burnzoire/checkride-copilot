from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import httpx
from loguru import logger

from diagnostic.action_map import normalize_action
from diagnostic.engine import diagnose_action
from diagnostic.rules import get_rules_for_action
from mission.frequencies import resolve_frequency
from orchestrator.session import Session
from retrieval.cockpit_lookup import format_location, lookup as cockpit_lookup, suggest as cockpit_suggest
from retrieval.keybinds import describe_switch
from retrieval.search import search as rag_search

COLLECTOR_URL = os.getenv("CHECKRIDE_COLLECTOR_URL", "http://127.0.0.1:7779/state")
STATE_FRESH_MS = 5000

_PROC_DIR = Path(__file__).parent.parent / "data" / "airframes" / "fa18c" / "procedures"
_ACTIONS_PATH = Path(__file__).parent.parent / "data" / "airframes" / "fa18c" / "actions.json"
_AIRFIELDS_PATH = Path(__file__).parent.parent / "data" / "airfields_by_map.json"
_PROC_INDEX: list[dict[str, Any]] = []
_PROC_CACHE: dict[str, dict[str, Any]] = {}
_ACTIONS: list[dict[str, Any]] = []
_QUICK_ACTION_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "between",
    "can",
    "do",
    "for",
    "how",
    "i",
    "is",
    "my",
    "on",
    "the",
    "to",
    "what",
    "with",
}


def _query_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _procedure_intents(query: str) -> dict[str, bool]:
    q = query.lower()
    tokens = _query_tokens(query)
    tgp = (
        "targeting pod" in q
        or "tgp" in tokens
        or "atflir" in tokens
        or "litening" in tokens
        or "tpod" in tokens
    )
    return {
        "tgp": tgp,
        "power": bool(
            re.search(r"\b(power|turn|switch|set|start)\b", q)
            and re.search(r"\b(on|up)\b", q)
        ) or "power on" in q,
        "zoom": "zoom" in tokens or "fov" in tokens or "field of view" in q,
        "track": "point track" in q or ("point" in tokens and "track" in tokens),
        "use": "use" in tokens or "operate" in tokens,
    }


def _quick_action_intents(query: str) -> dict[str, bool]:
    q = query.lower()
    tokens = _query_tokens(query)
    wants_radio = bool(
        {
            "radio",
            "comms",
            "traffic",
            "tower",
            "awacs",
            "jtac",
            "tanker",
            "mayday",
            "say",
        }
        & tokens
    ) or "what to say" in q
    wants_landing = bool(
        {
            "land",
            "landing",
            "inbound",
            "airfield",
            "runway",
            "tower",
            "traffic",
        }
        & tokens
    )
    wants_refuel = bool({"refuel", "refueling", "tanker", "pre", "contact"} & tokens) or "pre-contact" in q

    return {
        "radio": wants_radio,
        "landing": wants_landing,
        "refuel": wants_refuel,
        "landing_radio": wants_radio and wants_landing,
    }


def _expand_procedure_query_tokens(query: str) -> set[str]:
    tokens = _query_tokens(query)
    intents = _procedure_intents(query)

    # Expand fused designators like gbu12/aim120/agm65 into component tokens so
    # they still match canonical hyphenated entries in the procedure index.
    fused = set(tokens)
    for tok in fused:
        m = re.match(r"^([a-z]{2,})(\d{1,3}[a-z]?)$", tok)
        if not m:
            continue
        prefix, suffix = m.group(1), m.group(2)
        tokens.add(prefix)
        tokens.add(suffix)
        tokens.add(f"{prefix}-{suffix}")

    if intents["tgp"]:
        tokens.update(
            {
                "tgp",
                "atflir",
                "litening",
                "targeting",
                "pod",
                "flir",
                "tpod",
            }
        )
    if intents["power"]:
        tokens.update({"power", "on", "stby", "standby", "opr", "operate", "sensor", "switch"})
    if intents["zoom"]:
        tokens.update({"zoom", "fov", "narrow", "wide", "field", "view"})
    if intents["track"]:
        tokens.update({"point", "track", "ptrk", "area", "slv", "reticle"})
    if intents["use"] and intents["tgp"]:
        tokens.update({"slew", "depress", "offset", "soi", "sensor", "control"})

    return tokens


def _procedure_search_text(entry: dict[str, Any], include_steps: bool = False) -> str:
    bits: list[str] = [
        entry.get("key", "").replace("_", " "),
        entry.get("canonical", ""),
        " ".join(entry.get("alternates", []) or []),
        str(entry.get("source", {}).get("section", "")),
        str(entry.get("path", "")),
        str(entry.get("category", "")),
    ]
    if include_steps:
        proc = load_proc(str(entry.get("key", "")))
        if proc:
            bits.extend(proc.get("terminology", []) or [])
            for step in proc.get("steps", []) or []:
                bits.append(step.get("action", ""))
                bits.append(step.get("voiced", ""))
    return " ".join(bits).lower()


def _init_proc_index() -> None:
    index_path = _PROC_DIR / "index.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        _PROC_INDEX.extend(data.get("procedures", []))
    except Exception as e:
        logger.warning(f"Could not load procedures index: {e}")


_init_proc_index()


def _init_actions() -> None:
    try:
        data = json.loads(_ACTIONS_PATH.read_text(encoding="utf-8"))
        _ACTIONS.extend(data.get("actions", []))
    except Exception as e:
        logger.warning(f"Could not load actions: {e}")


_init_actions()


@lru_cache(maxsize=1)
def _airfield_alias_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        data = json.loads(_AIRFIELDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return out

    for airfields in data.get("maps", {}).values():
        for entry in airfields:
            canonical = str(entry.get("canonical", "")).strip()
            if not canonical:
                continue
            out[canonical.lower()] = canonical
            for alias in entry.get("aliases", []) or []:
                alias_key = str(alias).strip().lower()
                if alias_key:
                    out[alias_key] = canonical
    return out


def _extract_airfield_name(query: str) -> str | None:
    q = query.lower()
    alias_map = _airfield_alias_lookup()
    if not alias_map:
        return None

    # Exact alias/canonical mention takes priority.
    best_alias = ""
    best_canonical = None
    for alias, canonical in alias_map.items():
        if len(alias) < 4:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q):
            if len(alias) > len(best_alias):
                best_alias = alias
                best_canonical = canonical
    if best_canonical:
        return best_canonical

    candidates = list(alias_map.keys())
    tokens = q.split()
    suffixes = {"airfield", "airport", "traffic", "tower"}
    for idx, token in enumerate(tokens):
        if token not in suffixes:
            continue
        for width in (3, 2, 1):
            start = max(0, idx - width)
            raw = " ".join(tokens[start:idx]).strip()
            raw = re.sub(r"^(?:at|to|near|nearby|the|a|an|for|from|frequency)\s+", "", raw).strip()
            if not raw:
                continue
            closest = get_close_matches(raw, candidates, n=1, cutoff=0.68)
            if closest:
                return alias_map[closest[0]]
    return None


def _render_quick_action(action: dict[str, Any], query: str, max_steps: int, session: Session | None = None) -> dict[str, Any]:
    steps = [clean_step_text(str(s)) for s in (action.get("steps", []) or [])][:max_steps]
    example = action.get("example")
    key = str(action.get("key", ""))

    if key == "radio_airfield_inbound":
        airfield = _extract_airfield_name(query) or "[airfield]"
        wants_final = bool(re.search(r"\b(final|base|downwind|pattern)\b", query.lower()))
        in_jet = fetch_state() is not None
        freq = resolve_frequency(airfield, in_jet=in_jet)
        
        # Track airfield in session for follow-up band queries
        if session and airfield != "[airfield]":
            session.last_airfield = airfield

        if wants_final:
            example = f"{airfield} traffic, Enfield 1-1, final runway [runway], full stop."
        else:
            example = f"{airfield} traffic, Enfield 1-1, 1-ship Hornet, west, 12 miles, 3000, inbound full stop."

        if isinstance(freq, dict) and freq.get("mhz"):
            mhz = float(freq["mhz"])
            example += f" Tune {mhz:.3f} MHz."

        rendered_steps = []
        for s in steps:
            rendered_steps.append(s.replace("[airfield]", airfield))
        if isinstance(freq, dict) and freq.get("mhz"):
            mhz = float(freq["mhz"])
            src = str(freq.get("source", "local"))
            rendered_steps.append(
                f"Frequency: {mhz:.3f} MHz ({src})."
            )
        steps = rendered_steps

    return {
        "ok": True,
        "found": True,
        "key": key,
        "title": action.get("title"),
        "description": action.get("description"),
        "example": example,
        "steps": [{"num": i, "text": t} for i, t in enumerate(steps, start=1)],
    }


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
        if not isinstance(state, dict):
            return None
        # Some local ad-hoc servers may omit freshness metadata.
        # Only reject staleness when data_age_ms is explicitly provided.
        age = state.get("data_age_ms")
        if isinstance(age, (int, float)) and age > STATE_FRESH_MS:
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


def step_more_info(proc_key: str, step_idx: int) -> str | None:
    proc = load_proc(proc_key)
    if not proc:
        return None

    steps = proc.get("steps", [])
    if not steps or step_idx >= len(steps):
        return None

    raw = steps[step_idx].get("more_info")
    if not raw:
        return None
    return clean_step_text(str(raw))


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
    min_score = float(args.get("min_score", 14.0))
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

    kb = describe_switch(hit.get("dcs_actions") or {})
    # Prefer concise spoken_location over verbose formatted_location if available
    location = hit.get("spoken_location") or format_location(hit)
    return {
        "ok": True,
        "found": True,
        "canonical": hit.get("canonical"),
        "location": location,
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

    tokens = _expand_procedure_query_tokens(query)
    intents = _procedure_intents(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in _PROC_INDEX:
        include_steps = False
        text = _procedure_search_text(p, include_steps=include_steps)
        words = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(tokens & words)

        bonus = 0
        if intents["tgp"] and re.search(r"\btgp|atflir|litening|targeting pod|flir\b", text):
            bonus += 3
        if intents["power"] and re.search(r"\bpower|stby|standby|opr|operate\b", text):
            bonus += 2
        if intents["zoom"] and re.search(r"\bzoom|fov|field of view|narrow|wide\b", text):
            bonus += 2
        if intents["track"] and re.search(r"\bpoint track|ptrk|area track\b", text):
            bonus += 2
        if "point track" in query and re.search(r"\bpoint\s*&?\s*area\s*track|\bpoint\s*track\b", text):
            bonus += 6
        if "turn on" in query or "power on" in query:
            if re.search(r"\bpower on quick start|quick start\b", text):
                bonus += 4
        if intents["tgp"] and p.get("category") == "weapons":
            bonus -= 2

        score = overlap + bonus
        if score > 0:
            scored.append((score, p))

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


def tool_get_quick_action(args: dict[str, Any], session: Session) -> dict[str, Any]:
    query = str(args.get("query", "")).strip().lower()
    if not query:
        return {"ok": False, "error": "query is required"}

    tokens = {t for t in _query_tokens(query) if t not in _QUICK_ACTION_STOPWORDS}
    intents = _procedure_intents(query)
    qa_intents = _quick_action_intents(query)

    if intents["tgp"] and intents["power"]:
        for action in _ACTIONS:
            if action.get("key") == "tgp_power_on":
                max_steps = int(args.get("max_steps", 4))
                max_steps = max(1, min(max_steps, 6))
                session.last_quick_action_key = "tgp_power_on"
                return _render_quick_action(action, query, max_steps, session)

    followup = bool(
        re.search(
            r"\b(another\s+example|what\s+about|on\s+final|on\s+base|downwind|pattern|what\s+is\s+it|say\s+it\s+again)\b",
            query,
        )
    )
    preferred_key = session.last_quick_action_key if followup else None

    best: dict[str, Any] | None = None
    best_score = 0

    for action in _ACTIONS:
        key = str(action.get("key", ""))
        text = " ".join(
            [
                key.replace("_", " "),
                action.get("title", ""),
                action.get("description", ""),
                " ".join(action.get("alternates", []) or []),
                " ".join(action.get("tags", []) or []),
                " ".join(action.get("steps", []) or []),
            ]
        ).lower()
        words = {w for w in _query_tokens(text) if w not in _QUICK_ACTION_STOPWORDS}
        overlap = len(tokens & words)
        bonus = 0
        if key == "carrier_call_the_ball" and re.search(r"\b(cull|coal|call)\s+the\s+(bull|ball)\b", query):
            bonus += 5
        if intents["tgp"] and "tgp" in words:
            bonus += 2
        if intents["power"] and re.search(r"\bpower|startup|stby|opr|on\b", text):
            bonus += 3
        if intents["zoom"] and re.search(r"\bzoom|fov\b", text):
            bonus += 3
        if intents["track"] and re.search(r"\bpoint|track|ptrk\b", text):
            bonus += 3

        if preferred_key and key == preferred_key:
            bonus += 8
        if re.search(r"\b(final|base|downwind|pattern)\b", query) and key == "radio_airfield_inbound":
            bonus += 5

        # Route "how do I say inbound/landing" requests to concise radio phrase quick actions.
        if qa_intents["landing_radio"]:
            if key == "radio_airfield_inbound":
                bonus += 10
            elif key.startswith("radio_"):
                bonus += 3
            if key == "radio_tanker_precontact":
                bonus -= 6
        elif qa_intents["radio"] and key.startswith("radio_"):
            bonus += 2

        if qa_intents["refuel"] and key == "radio_tanker_precontact":
            bonus += 6

        if re.search(r"\b(cobaltia|kobuleti)\b", query) and key == "radio_airfield_inbound":
            bonus += 4

        score = overlap + bonus
        if score > best_score:
            best = action
            best_score = score

    if not best or best_score < 2:
        return {"ok": True, "found": False}

    max_steps = int(args.get("max_steps", 4))
    max_steps = max(1, min(max_steps, 6))
    session.last_quick_action_key = str(best.get("key", "")) or None
    return _render_quick_action(best, query, max_steps, session)


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
        "current_step_more_info": step_more_info(session.active_proc, session.active_step),
    }


ToolFn = Callable[[dict[str, Any], Session], dict[str, Any]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "search_natops": tool_search_natops,
    "lookup_switch": tool_lookup_switch,
    "get_cockpit_state": tool_get_cockpit_state,
    "diagnose_action": tool_diagnose_action,
    "get_quick_action": tool_get_quick_action,
    "list_procedures": tool_list_procedures,
    "start_procedure": tool_start_procedure,
    "get_active_procedure": tool_get_active_procedure,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_natops",
            "description": "Search indexed NATOPS/procedure chunks. Last resort after quick actions and procedure lookup do not provide a usable result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "min_score": {"type": "number", "default": 14.0},
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
            "description": "Get fresh cockpit state snapshot from the local collector. Use for any question about current heading, altitude, airspeed, fuel, master arm, nav mode, radar mode, or any other live cockpit value.",
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
            "name": "get_quick_action",
            "description": "Find a short action card for a focused cockpit task (usually 2-6 steps).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_steps": {"type": "integer", "default": 4},
                },
                "required": ["query"],
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
