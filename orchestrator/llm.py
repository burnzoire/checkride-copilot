from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

import httpx

from orchestrator.session import Session
from orchestrator.tools import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    fetch_state,
    load_proc,
    tool_list_procedures,
    tool_start_procedure,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_HISTORY = 10
MAX_TOOL_CALLS = 2

# Tokens at which the model should stop generating boilerplate outros.
# Keep this list in sync — it applies to every Ollama payload in this module.
STOP_TOKENS = [
    "[Reference", "[Switch",
    "let me know", "if you have", "if you need", "feel free",
]

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

SYSTEM_PROMPT = """
You are Checkride Copilot, an F/A-18C instructor pilot (IP/ICP) voice assistant.

You are operating in an LLM-first tool-calling harness.
Rules:
1) Use tools before answering procedural or switch-location questions.
1a) For short focused requests or questions about simple actions (e.g., "turn on TGP", "how do I dispense chaff", "zoom TGP", "point track", "select amraam"), call get_quick_action first.
1b) Use full procedures only when the request is clearly multi-step and long-form (e.g., "walk me through startup", "pre-flight procedure").
1c) For questions about the user's configuration or keybinds (e.g., "what button have I assigned X to?"), call lookup_switch first.
1d) For any question about current flight state — heading, altitude, airspeed, fuel, master arm, nav mode, radar mode, or any other live cockpit value — call get_cockpit_state first, then answer directly from the result. Never ask the user to provide state you can fetch yourself.
1e) Tool priority order is strict: live sim state first, then quick actions/procedures, then broader guide/RAG search, and NATOPS/document search only as last resort.
2) Never invent switch names, procedure keys, or cockpit states.
3) If confidence is low, ask one precise clarifying question.
4) Keep spoken answers concise: usually 1-2 sentences. For lookup_switch results: if the user asks "where", give the location; if they ask "what button", give the keybind. Avoid redundant location descriptions (e.g., don't say "throttle grip and outboard throttle" — use one).
5) If start_procedure succeeds, confirm activation and brief first step.
6) For an active procedure, navigation words (next/repeat/restart/cancel) are handled externally.
7) If the user asks for details, example phrasing, or "more info" on a current step, call get_active_procedure and use current_step_more_info when available.
8) Never cite or mention manuals, pages, references, PDFs, or document sections in spoken replies.
9) Give direct instructor coaching tone. Do not say "the manual says" or "according to NATOPS".
10) Never guess. If no tool evidence is available, call a tool first or ask one precise clarifying question.
11) For closing or acknowledgment statements (e.g., "that's it", "thank you", "sounds good", "roger", "copy that", "all set"), respond briefly without calling tools or asking clarifying questions. Example responses: "You're welcome", "Roger that", "Good luck out there".
""".strip()


def _looks_action_or_procedure_query(text: str) -> bool:
    q = text.lower()
    return bool(
        re.search(
            r"\b(how\s+do\s+i|how\s+to|turn\s+on|power\s+on|set|select|fire|launch|drop|employ|use|"
            r"startup|start\s*up|procedure|steps?|walk\s+me\s+through|mk\s*-?\s*\d+|gbu\s*-?\s*\d+|"
            r"aim\s*-?\s*\d+|agm\s*-?\s*\d+|amraam|sidewinder|jdam|jsow|harm|maverick|bomb|missile)\b",
            q,
        )
    )


def _run_grounding_fallback_chain(
    transcript: str,
    session: Session,
    messages: list[dict[str, Any]],
    total_tool_calls: int,
) -> tuple[int, bool]:
    """Fallback grounding order: quick action -> procedure catalog -> NATOPS(last resort)."""
    used_any = False

    # For action/procedure-like requests, try quick actions first.
    if total_tool_calls < MAX_TOOL_CALLS and _looks_action_or_procedure_query(transcript):
        qa_fn = TOOL_REGISTRY.get("get_quick_action")
        if qa_fn:
            try:
                qa_result = qa_fn({"query": transcript, "max_steps": 5}, session)
            except Exception as e:
                qa_result = {"ok": False, "error": f"Quick action grounding failed: {e}"}
            total_tool_calls += 1
            used_any = True
            messages.append({"role": "tool", "name": "get_quick_action", "content": json.dumps(qa_result)})
            if isinstance(qa_result, dict) and qa_result.get("ok") is True and qa_result.get("found") is True:
                messages.append(
                    {
                        "role": "user",
                        "content": "Use the quick action result directly. Keep it concise and actionable.",
                    }
                )
                return total_tool_calls, used_any

    # Next, try indexed procedures before broad document search.
    if total_tool_calls < MAX_TOOL_CALLS and _looks_action_or_procedure_query(transcript):
        proc_fn = TOOL_REGISTRY.get("list_procedures")
        if proc_fn:
            try:
                proc_result = proc_fn({"query": transcript, "limit": 3}, session)
            except Exception as e:
                proc_result = {"ok": False, "error": f"Procedure grounding failed: {e}"}
            total_tool_calls += 1
            used_any = True
            messages.append({"role": "tool", "name": "list_procedures", "content": json.dumps(proc_result)})
            if isinstance(proc_result, dict) and proc_result.get("ok") is True and proc_result.get("matches"):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Use procedure matches first. If one match is clearly best, use it; "
                            "if not, ask one concise clarifying question."
                        ),
                    }
                )
                return total_tool_calls, used_any

    # Last resort only: NATOPS/RAG search.
    if total_tool_calls < MAX_TOOL_CALLS:
        docs_fn = TOOL_REGISTRY.get("search_natops")
        if docs_fn:
            try:
                docs_result = docs_fn({"query": transcript, "top_k": 3}, session)
            except Exception as e:
                docs_result = {"ok": False, "error": f"Grounding docs search failed: {e}"}
            total_tool_calls += 1
            used_any = True
            messages.append(
                {
                    "role": "tool",
                    "name": "search_natops",
                    "content": json.dumps(docs_result),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use only grounded tool output. "
                        "If evidence is insufficient, ask one precise clarifying question."
                    ),
                }
            )

    return total_tool_calls, used_any


def _is_cockpit_state_query(text: str) -> bool:
    """Detect questions about live cockpit values that should be answered from state, not docs."""
    q = text.lower()
    # Must be a question or current-state phrasing, not a how-to
    if not re.search(r"\b(what|what's|whats|tell me|current|am i|my|give me|show|read|reading)\b", q):
        return False
    # Exclude procedural how-to questions
    if re.search(r"\b(how\s+do|how\s+to|how\s+can|procedure|steps?|walk\s+me)\b", q):
        return False
    return bool(re.search(
        r"\b(heading|bearing|course|direction|altitude|asl|agl|elevation|height|"
        r"airspeed|speed|knots|kts|mach|fuel|bingo|fuel\s+state|"
        r"master\s+arm|radar\s+mode|nav\s+mode|selected\s+weapon|weapon\s+selected)\b",
        q
    ))


def _cockpit_state_fast_reply(transcript: str) -> str | None:
    """Call get_cockpit_state directly and return a grounded spoken answer."""
    from orchestrator.tools import fetch_state
    state = fetch_state()
    if not state:
        return "No cockpit state available right now — make sure DCS is running."

    q = transcript.lower()

    if re.search(r"\b(heading|bearing|course|direction)\b", q):
        hdg = state.get("heading_deg")
        if hdg is None:
            return "Heading not available in current state."
        return f"Your heading is {hdg:.0f} degrees magnetic."

    if re.search(r"\b(altitude|asl|agl|elevation|height)\b", q):
        alt = state.get("altitude_ft")
        if alt is None:
            return "Altitude not available in current state."
        return f"Altitude is {alt:.0f} feet."

    if re.search(r"\b(airspeed|speed|knots|kts|mach)\b", q):
        spd = state.get("airspeed_kts")
        if spd is None:
            return "Airspeed not available in current state."
        return f"Airspeed is {spd:.0f} knots."

    if re.search(r"\bmaster\s+arm\b", q):
        arm = state.get("master_arm")
        if arm is None:
            return "Master arm state not available."
        return f"Master arm is {arm}."

    if re.search(r"\bradar\s+mode\b", q):
        mode = state.get("radar_mode")
        if mode is None:
            return "Radar mode not available in current state."
        return f"Radar mode is {mode}."

    # Generic: return whatever we have
    parts: list[str] = []
    if state.get("heading_deg") is not None:
        parts.append(f"heading {state['heading_deg']:.0f}")
    if state.get("altitude_ft") is not None:
        parts.append(f"altitude {state['altitude_ft']:.0f} feet")
    if state.get("airspeed_kts") is not None:
        parts.append(f"airspeed {state['airspeed_kts']:.0f} knots")
    if not parts:
        return "Cockpit state is available but no flight parameters were found."
    return "Current state: " + ", ".join(parts) + "."


def _is_call_the_ball_query(text: str) -> bool:
    q = text.lower()
    if "call the ball" in q:
        return True
    return bool(
        re.search(r"\b(cull|coal|call)\s+the\s+(bull|ball)\b", q)
        or (
            re.search(r"\bcarrier|groove|approach|final\b", q)
            and re.search(r"\bball|bull\b", q)
        )
    )


def _is_full_procedure_query(text: str) -> bool:
    """Detect explicit multi-step requests that should start a procedure deterministically."""
    q = text.lower()
    asks_how = bool(re.search(r"\b(how\s+do\s+i|how\s+to|walk\s+me\s+through|procedure|steps?)\b", q))
    weapon_task = bool(
        re.search(r"\b(drop|dropping|release|employ|launch|fire|use)\b", q)
        and re.search(r"\b(gbu\s*-?\s*\d+\w*|mk\s*-?\s*\d+\w*|aim\s*-?\s*\d+\w*|agm\s*-?\s*\d+\w*|amraam|jdam|jsow|harm|maverick|paveway)\b", q)
    )
    return asks_how and weapon_task


def _extract_weapon_designator(text: str) -> str | None:
    m = re.search(r"\b(gbu|mk|aim|agm)\s*-?\s*(\d+[a-z]?(?:s)?)\b", text.lower())
    if not m:
        return None
    suffix = m.group(2)
    if suffix.endswith("s") and suffix[:-1].isdigit():
        suffix = suffix[:-1]
    return f"{m.group(1)}-{suffix}"


def _full_procedure_fast_reply(transcript: str, session: Session) -> str | None:
    """Start a clearly matched procedure for explicit multi-step requests."""
    result = tool_list_procedures({"query": transcript, "limit": 3}, session)
    if not isinstance(result, dict) or result.get("ok") is not True:
        return None

    matches = result.get("matches") or []
    if not matches:
        return None

    top = matches[0]
    top_score = int(top.get("score") or 0)
    second_score = int(matches[1].get("score") or 0) if len(matches) > 1 else -999

    # Require a clear winner to avoid false auto-starts.
    if top_score < 4 or (top_score - second_score) < 2:
        return None

    proc_key = str(top.get("key") or "").strip()
    if not proc_key:
        return None

    # If a specific weapon designator is present in the query, only auto-start
    # when the top candidate clearly contains that same designator.
    wanted = _extract_weapon_designator(transcript)
    if wanted:
        top_blob = " ".join(
            [
                str(top.get("key") or ""),
                str(top.get("canonical") or ""),
            ]
        ).lower().replace("_", "-")
        if wanted not in top_blob:
            return None

    started = tool_start_procedure({"proc_key": proc_key}, session)
    if not isinstance(started, dict) or started.get("ok") is not True:
        return None

    intro = str(started.get("intro") or "").strip()
    return intro or None


def _is_airfield_frequency_query(text: str) -> bool:
    q = text.lower()
    if re.search(r"\b(contact|call)\b", q) and re.search(r"\b(tower|approach|atc|traffic|airfield|airport)\b", q):
        return True
    # Any mention of frequency-related words (with optional tower/approach)
    if re.search(r"\b(freq|frequency|channel)\b", q):
        # Either paired with tower/approach keywords, or likely an airfield query
        if re.search(r"\b(tower|approach|atc|traffic|airfield|airport)\b", q):
            return True
        # Or if just asking "what's the frequency" (common phrasing)
        if re.search(r"\b(what|give|tell)\b.*\b(freq|frequency|channel)\b", q):
            return True
    # "Freak" is old radio slang for frequency
    if re.search(r"\bfreak\b", q):
        return True
    # Band-only queries like "UHF?" / "switch to uniform" / "What about VHF?"
    # These are likely follow-ups to an airfield context
    if re.search(r"\b(uhf|vhf|hf|uniform|victor|hotel|band)\b", q):
        # Only match if not asking about something else
        if not re.search(r"\b(mod|mode|setting|button|panel)\b", q):
            return True
    return False


def _extract_band_from_query(text: str) -> str | None:
    """Extract frequency band from query (UHF, VHF, VHF_HI, VHF_LOW, HF, or None for tower)."""
    q = text.lower()
    if re.search(r"\b(uhf|uniform)\b", q):
        return "UHF"
    if re.search(r"\bvhf\s*hi\b|\bvhf\s*high\b", q):
        return "VHF_HI"
    if re.search(r"\bvhf\s*low\b", q):
        return "VHF_LOW"
    if re.search(r"\b(vhf|victor)\b", q):
        return "VHF_HI"  # Default to VHF_HI if no qualifier
    if re.search(r"\b(hf|hotel)\b", q):
        return "HF"
    return None


def _last_user_turn_text(session: Session) -> str | None:
    for turn in reversed(session.history):
        if str(turn.get("role", "")).lower() == "user":
            content = str(turn.get("content", "")).strip()
            if content:
                return content
    return None


def _contextualize_transcript(transcript: str, session: Session) -> str:
    q = transcript.strip()
    if not q:
        return q

    tokens = re.findall(r"[a-z0-9_'-]+", q.lower())
    if len(tokens) > 7:
        return q

    follow_up_signal = bool(
        re.search(
            r"^(and\b|then\b|what\s+about\b|how\s+about\b|switch\b|set\b|tune\b|"
            r"go\s+to\b|use\b|that\b|this\b|it\b)",
            q.lower(),
        )
    )
    band_only_signal = bool(re.search(r"\b(uhf|vhf|hf|uniform|victor|hotel)\b", q.lower()))
    if not (follow_up_signal or band_only_signal):
        return q

    prev_user = _last_user_turn_text(session)
    if not prev_user:
        return q

    return f"{prev_user}. Follow-up: {q}"


def _is_miz_contact_query(text: str) -> bool:
    from mission.frequencies import _CVN_SHIP_NAMES_RE, _TANKER_CALLSIGN_RE, _query_fuzzy_matches_ship_name
    q = text.lower()
    if re.search(r"\b(tower|approach|atc|traffic|airfield|airport)\b", q):
        return False
    mentions_contact = bool(
        re.search(r"\b(cvn\s*-?\s*\d+|carrier|boat|tanker)\b", q)
        or _CVN_SHIP_NAMES_RE.search(q)
        or _TANKER_CALLSIGN_RE.search(q)
        or _query_fuzzy_matches_ship_name(q)
    )
    mentions_named_unit = bool(
        re.search(
            r"\b(?:[a-z]{2,}[ -]?\d+-\d+|unit\s*#?\s*\d+)\b",
            q,
        )
    )
    asks_channel = bool(re.search(r"\b(freq|frequency|channel|tacan|uhf|vhf|hf)\b", q))
    asks_presence = bool(re.search(r"\b(see|find|have|loaded|available|present)\b", q))
    asks_type = bool(re.search(r"\b(what\s+type|type\s+of\s+ship|ship\s+type|aircraft\s+type)\b", q))
    followup_contact = bool(re.search(r"\b(how\s+about|what\s+about|and)\b", q) and mentions_contact)
    lists_kind = _is_list_contacts_query(text)
    return (
        (mentions_contact and asks_channel)
        or bool(re.search(r"\btacan\b", q))
        or (mentions_contact and asks_presence)
        or followup_contact
        or lists_kind
        or bool(re.fullmatch(r"\s*(carrier|boat|tanker|cvn\s*-?\s*\d+)\s*", q))
        or bool(_TANKER_CALLSIGN_RE.fullmatch(q.strip()))
        or bool(re.fullmatch(r"\s*(unit\s*#?\s*\d+)\s*", q))
        or ((mentions_contact or mentions_named_unit) and (asks_type or asks_presence or asks_channel))
    )


def _contact_mode_from_query(text: str) -> str:
    return "tacan" if re.search(r"\btacan\b", text.lower()) else "radio"


def _is_list_contacts_query(text: str) -> bool:
    """Detect 'what carriers are in this mission?' / 'list the tankers' style queries.

    Must NOT fire for frequency/TACAN queries that happen to mention carriers.
    """
    q = text.lower()
    # Never treat a freq/tacan ask as a list query.
    if re.search(r"\b(freq|frequency|channel|tacan|uhf|vhf|hf|freak)\b", q):
        return False
    asks_list = bool(re.search(
        r"\b(what|which|list|show|tell\s+me|do\s+we\s+have|are\s+there|how\s+many)"
        r".{0,30}\b(carrier|carriers|tanker|tankers|boat|boats)\b",
        q,
    ))
    has_kind = bool(re.search(r"\b(carrier|carriers|tanker|tankers|boat|boats)\b", q))
    lists_all = bool(re.search(r"\b(available|in\s+(this|the)\s+mission|loaded|on\s+station)\b", q))
    return asks_list or (has_kind and lists_all)


def _list_contacts_reply(kind: str) -> str | None:
    """Build a spoken list of all unique carriers or tankers in the mission with their types."""
    from mission.frequencies import _CVN_NAMES, list_miz_named_contacts

    contacts = list_miz_named_contacts(kind=kind)
    if not contacts:
        return f"I don't see any {kind}s in the current mission."

    entries: list[str] = []
    seen_hull: set[str] = set()

    for c in contacts:
        name = str(c.get("name") or "").strip()
        platform_type = str(c.get("platform_type") or "").strip() or None

        if kind == "carrier":
            hull = _carrier_hull_from_contact_fields(name, platform_type)
            if not hull:
                continue
            if hull in seen_hull:
                continue
            seen_hull.add(hull)
            # Look up ship nickname for a human-readable label.
            hull_num = re.search(r"(\d+)", hull)
            nick: str | None = None
            if hull_num:
                nicks = _CVN_NAMES.get(hull_num.group(1), [])
                # Use the longest multi-word name as primary, e.g. 'George Washington'
                multi = [n for n in nicks if " " in n]
                nick = max(multi, key=len).title() if multi else (nicks[0].title() if nicks else None)
            label = f"{hull}" + (f" ({nick})" if nick else "")
            entries.append(label)
        else:
            # Tanker — deduplicate by platform_type + callsign_name
            pt = str(c.get("platform_type") or "").strip()
            cs = str(c.get("callsign_name") or "").strip()
            key = f"{pt}|{cs}".lower()
            if key in seen_hull:
                continue
            seen_hull.add(key)
            type_label = _tanker_type_label(platform_type) or pt or "unknown type"
            label = cs if cs else f"{type_label} tanker"
            entries.append(label)

    if not entries:
        return f"I don't see any {kind}s in the current mission."
    if len(entries) == 1:
        return f"There is one {kind} in this mission: {entries[0]}."
    items = ", ".join(entries[:-1]) + f", and {entries[-1]}"
    return f"There are {len(entries)} {kind}s in this mission: {items}."


def _is_contact_followup_query(text: str, session: Session) -> bool:
    if not session.last_contact:
        return False
    q = text.lower().strip()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(freq|frequency|channel|tacan|uhf|vhf|hf|type|ship|aircraft|present|available|loaded|see)\b",
            q,
        )
        or re.search(r"^(and\b|what\s+about\b|how\s+about\b)", q)
    )


def _carrier_label(contact: str, platform_type: str | None) -> str:
    known_hulls = {
        "stennis": "74",
        "john c stennis": "74",
        "roosevelt": "71",
        "theodore roosevelt": "71",
        "lincoln": "72",
        "abraham lincoln": "72",
        "washington": "73",
        "george washington": "73",
        "truman": "75",
        "harry s truman": "75",
        "vinson": "70",
        "carl vinson": "70",
        "nimitz": "68",
        "ford": "78",
    }

    pt = str(platform_type or "")
    m = re.search(r"\bCVN[_-]?(\d{1,3})\b", pt, re.I)
    if not m:
        m = re.search(r"\bCVN\s*-?\s*(\d{1,3})\b", contact, re.I)
    if m:
        return f"CVN-{m.group(1)}"

    blob = f"{contact} {pt}".lower()
    for key, hull in known_hulls.items():
        if re.search(rf"\b{re.escape(key)}\b", blob):
            return f"CVN-{hull}"

    return "carrier"


def _tanker_callsign(contact: str, callsign_name: str | None = None) -> str:
    if callsign_name:
        token = re.split(r"\s+", callsign_name.strip())[0]
        m = re.match(r"^([A-Za-z]+)\d{1,3}$", token)
        if m:
            return m.group(1).capitalize()
        if token:
            return token.capitalize()
    base = re.split(r"\s+\d+-\d+$", contact.strip())[0]
    base = re.split(r"-\d+-\d+$", base)[0]
    base = base.strip()
    if not base:
        return "tanker"
    token = re.split(r"\s+", base)[0]
    return token.capitalize() if token else "tanker"


def _tanker_type_label(platform_type: str | None) -> str | None:
    pt = str(platform_type or "").strip()
    if not pt:
        return None
    m = re.search(r"\b(KC[-_ ]?135|KC[-_ ]?130|IL[-_ ]?78)\w*\b", pt, re.I)
    if not m:
        return None
    return m.group(1).upper().replace("_", "-").replace(" ", "-")


def _contact_spoken_label(contact: str, kind: str, platform_type: str | None, callsign_name: str | None = None) -> str:
    k = (kind or "").lower()
    if k == "carrier":
        hull = _carrier_label(contact, platform_type)
        return f"{hull} carrier" if hull != "carrier" else "carrier"
    if k == "tanker":
        if callsign_name:
            return callsign_name
        t = _tanker_type_label(platform_type)
        if t:
            return f"{t} tanker"
        return "tanker"
    return "contact"


def _pending_contact_prompt(kind: str, mode: str) -> str:
    k = (kind or "contact").lower()
    if k == "carrier":
        suffix = "TACAN" if mode == "tacan" else "frequency"
        return f"I found multiple carriers in this mission. Say the hull number, for example CVN-74, for {suffix}."
    if k == "tanker":
        suffix = "TACAN" if mode == "tacan" else "frequency"
        return f"I found multiple tankers in this mission. Say which tanker for {suffix}."
    return "I found multiple contacts in this mission. Say the contact again with one more detail."


def _carrier_hull_from_text(text: str) -> str | None:
    m = re.search(r"\bCVN\s*-?\s*(\d{1,3})\b", str(text), re.I)
    if not m:
        return None
    return f"CVN-{m.group(1)}"


def _carrier_hull_from_contact_fields(contact: str, platform_type: str | None) -> str | None:
    label = _carrier_label(contact, platform_type)
    return label if re.fullmatch(r"CVN-\d{1,3}", label, re.I) else None


def _disambiguation_choices(kind: str, options: list[str], listed: list[dict[str, Any]] | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _push(val: str | None) -> None:
        v = str(val or "").strip()
        if not v:
            return
        k = v.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(v)

    if kind == "carrier":
        inferred_hulls: set[str] = set()
        for item in listed or []:
            name = str(item.get("name") or "").strip()
            platform_type = str(item.get("platform_type") or "").strip() or None
            hull = _carrier_hull_from_contact_fields(name, platform_type)
            if hull:
                inferred_hulls.add(hull)
        for opt in options:
            hull = _carrier_hull_from_text(opt)
            if hull:
                inferred_hulls.add(hull)

        # If we can infer hull IDs, present hulls only (no raw unit aliases).
        if inferred_hulls:
            return sorted(
                inferred_hulls,
                key=lambda h: int(re.search(r"(\d+)$", h).group(1)) if re.search(r"(\d+)$", h) else 999,
            )

        for item in listed or []:
            name = str(item.get("name") or "").strip()
            platform_type = str(item.get("platform_type") or "").strip() or None
            _push(_carrier_hull_from_contact_fields(name, platform_type) or name)
        for opt in options:
            _push(_carrier_hull_from_text(opt) or opt)
        return out

    if kind == "tanker":
        for item in listed or []:
            label = _contact_spoken_label(
                str(item.get("name") or "").strip(),
                str(item.get("kind") or "tanker"),
                str(item.get("platform_type") or "").strip() or None,
                str(item.get("callsign_name") or "").strip() or None,
            )
            _push(label)
        for opt in options:
            _push(opt)
        return out

    for opt in options:
        _push(opt)
    return out


def _pending_contact_prompt_with_options(kind: str, mode: str, choices: list[str]) -> str:
    k = (kind or "contact").lower()
    suffix = "TACAN" if mode == "tacan" else "frequency"
    if k == "carrier":
        if choices:
            return f"I found multiple carriers in this mission: {', '.join(choices)}. Say which one for {suffix}."
        return f"I found multiple carriers in this mission. Say the hull number, for example CVN-74, for {suffix}."
    if k == "tanker":
        if choices:
            return f"I found multiple tankers in this mission: {', '.join(choices)}. Say which one for {suffix}."
        return f"I found multiple tankers in this mission. Say which tanker for {suffix}."
    if choices:
        return f"I found multiple contacts in this mission: {', '.join(choices)}. Say which one."
    return "I found multiple contacts in this mission. Say the contact again with one more detail."


def _pick_pending_contact_option(
    query: str,
    options: list[str],
    *,
    kind: str | None = None,
    listed: list[dict[str, Any]] | None = None,
) -> str | None:
    q = query.strip().lower()
    if not q:
        return None

    if (kind or "").lower() == "carrier":
        desired_hull = _carrier_hull_from_text(query)
        if desired_hull:
            option_by_key = {str(opt).strip().lower(): str(opt).strip() for opt in options if str(opt).strip()}

            # Direct hull options (if present) should resolve immediately.
            for opt in options:
                raw = str(opt).strip()
                if not raw:
                    continue
                if (_carrier_hull_from_text(raw) or "").lower() == desired_hull.lower():
                    return raw

            # Otherwise map the requested hull to a concrete alias from listed contacts.
            for item in listed or []:
                name = str(item.get("name") or "").strip()
                platform_type = str(item.get("platform_type") or "").strip() or None
                if not name:
                    continue
                hull = _carrier_hull_from_contact_fields(name, platform_type)
                if hull and hull.lower() == desired_hull.lower() and name.lower() in option_by_key:
                    return option_by_key[name.lower()]

    best: str | None = None
    best_score = 0.0
    for opt in options:
        raw = str(opt).strip()
        if not raw:
            continue
        key = raw.lower()
        if key in q or q in key:
            return raw
        score = SequenceMatcher(None, q, key).ratio()
        if score > best_score:
            best = raw
            best_score = score
    if best and best_score >= 0.72:
        return best
    return None


def _miz_contact_fast_reply(transcript: str, session: Session) -> str | None:
    from mission.frequencies import find_miz_named_contact, list_miz_named_contacts, resolve_miz_named_contact

    # List-all query: "what carriers are in this mission?"
    if not session.pending_contact_options and _is_list_contacts_query(transcript):
        q = transcript.lower()
        kind = "tanker" if re.search(r"\b(tanker|tankers)\b", q) else "carrier"
        reply = _list_contacts_reply(kind)
        if reply:
            return reply

    if session.pending_contact_options:
        mode_hint = session.pending_contact_mode or _contact_mode_from_query(transcript)
        pending_kind = (session.pending_contact_kind or "contact").lower()
        listed = list_miz_named_contacts(kind=pending_kind) if pending_kind in {"carrier", "tanker"} else []
        picked = _pick_pending_contact_option(
            transcript,
            session.pending_contact_options,
            kind=pending_kind,
            listed=listed,
        )
        if picked:
            forced = resolve_miz_named_contact(picked, mode=mode_hint, fallback_name=session.last_contact)
            if forced and not forced.get("error"):
                session.pending_contact_kind = None
                session.pending_contact_mode = None
                session.pending_contact_options = []
                result = forced
            else:
                result = None
        else:
            result = None
        if result:
            contact = str(result.get("contact") or session.last_contact or "that contact")
            kind = str(result.get("kind") or "contact")
            platform_type = str(result.get("platform_type") or "").strip() or None
            callsign_name = str(result.get("callsign_name") or "").strip() or None
            spoken = _contact_spoken_label(contact, kind, platform_type, callsign_name=callsign_name)
            if result.get("error") == "missing":
                if mode_hint == "tacan":
                    return f"I can't provide TACAN for {spoken}."
                return f"I can't provide a radio frequency for {spoken}."
            session.last_contact = contact
            if mode_hint == "tacan":
                ch = int(result.get("channel"))
                band = str(result.get("band") or "X").upper()
                return f"{spoken} TACAN is {ch}{band}."
            mhz = float(result.get("mhz"))
            return f"{spoken} frequency is {mhz:.3f} MHz."

    presence_query = bool(re.search(r"\b(see|find|have|loaded|available|present)\b", transcript.lower()))
    type_query = bool(re.search(r"\b(what\s+type|type\s+of\s+ship|ship\s+type|aircraft\s+type)\b", transcript.lower()))

    if type_query:
        seen = find_miz_named_contact(transcript, fallback_name=session.last_contact)
        if not seen:
            return "I can't find that contact in the current mission."
        if seen.get("error") == "ambiguous":
            session.pending_contact_kind = str(seen.get("kind") or "contact")
            session.pending_contact_mode = "radio"
            session.pending_contact_options = [str(x).strip() for x in (seen.get("options") or []) if str(x).strip()]
            choices = _disambiguation_choices(session.pending_contact_kind or "contact", session.pending_contact_options)
            return _pending_contact_prompt_with_options(session.pending_contact_kind or "contact", "radio", choices)
        contact = str(seen.get("contact") or session.last_contact or "that contact")
        kind = str(seen.get("kind") or "contact")
        platform_type = str(seen.get("platform_type") or "").strip()
        callsign_name = str(seen.get("callsign_name") or "").strip() or None
        session.last_contact = contact
        label = _contact_spoken_label(contact, kind, platform_type, callsign_name=callsign_name)
        if platform_type:
            if re.search(r"^CVN[_-]?(\d+)$", platform_type, re.I):
                hull = re.search(r"(\d+)", platform_type).group(1)
                return f"{label} is a CVN-{hull} carrier."
            if re.search(r"^(KC[-_ ]?135|KC[-_ ]?130|IL[-_ ]?78)", platform_type, re.I):
                return f"{label} is a {platform_type.replace('_', '-') } tanker."
            return f"{label} is a {platform_type.replace('_', '-')}"
        return f"I can see {label} in the current mission, but I don't have a platform type for it."

    if presence_query:
        seen = find_miz_named_contact(transcript, fallback_name=session.last_contact)
        if not seen:
            return "I can't find that contact in the current mission."
        if seen.get("error") == "ambiguous":
            session.pending_contact_kind = str(seen.get("kind") or "contact")
            session.pending_contact_mode = "radio"
            session.pending_contact_options = [str(x).strip() for x in (seen.get("options") or []) if str(x).strip()]
            choices = _disambiguation_choices(session.pending_contact_kind or "contact", session.pending_contact_options)
            return _pending_contact_prompt_with_options(session.pending_contact_kind or "contact", "radio", choices)
        contact = str(seen.get("contact") or session.last_contact or "that contact")
        kind = str(seen.get("kind") or "contact")
        platform_type = str(seen.get("platform_type") or "").strip() or None
        callsign_name = str(seen.get("callsign_name") or "").strip() or None
        session.last_contact = contact
        label = _contact_spoken_label(contact, kind, platform_type, callsign_name=callsign_name)
        return f"I can see {label} in the current mission."

    mode = _contact_mode_from_query(transcript)
    result = resolve_miz_named_contact(transcript, mode=mode, fallback_name=session.last_contact)
    if not result:
        if mode == "tacan":
            return "I can't provide TACAN for that contact from the current mission."
        return "I can't provide a radio frequency for that contact from the current mission."

    if result.get("error") == "ambiguous":
        kind = str(result.get("kind") or "contact")
        options = [str(x).strip() for x in (result.get("options") or []) if str(x).strip()]
        listed: list[dict[str, Any]] = []
        if not options and kind in {"carrier", "tanker"}:
            listed = list_miz_named_contacts(kind=kind)
            options = [str(c.get("name") or "").strip() for c in listed if str(c.get("name") or "").strip()]
        else:
            listed = list_miz_named_contacts(kind=kind) if kind in {"carrier", "tanker"} else []

        # Auto-resolve only when there is a single distinct entity.
        # For carriers, distinctness is by hull number (CVN-XX), not contact alias.
        auto_target: str | None = None
        if kind == "carrier":
            hulls: dict[str, str] = {}
            for item in listed:
                name = str(item.get("name") or "").strip()
                platform_type = str(item.get("platform_type") or "").strip() or None
                hull = _carrier_hull_from_contact_fields(name, platform_type)
                if hull:
                    hulls.setdefault(hull, name)
            if len(hulls) == 1:
                # Resolve via a concrete alias/name to avoid ambiguous hull-only lookups.
                auto_target = next(iter(hulls.values()))
        elif kind == "tanker" and len(listed) == 1:
            auto_target = str(listed[0].get("name") or "").strip() or None

        if auto_target:
            single = resolve_miz_named_contact(auto_target, mode=mode, fallback_name=session.last_contact)
            if isinstance(single, dict):
                if single.get("error") == "missing":
                    contact = str(single.get("contact") or auto_target)
                    k = str(single.get("kind") or kind)
                    platform_type = str(single.get("platform_type") or "").strip() or None
                    callsign_name = str(single.get("callsign_name") or "").strip() or None
                    spoken = _contact_spoken_label(contact, k, platform_type, callsign_name=callsign_name)
                    session.last_contact = contact
                    session.pending_contact_kind = None
                    session.pending_contact_mode = None
                    session.pending_contact_options = []
                    if mode == "tacan":
                        return f"I can't provide TACAN for {spoken}."
                    return f"I can't provide a radio frequency for {spoken}."
                if single.get("error"):
                    single = None

            if isinstance(single, dict):
                contact = str(single.get("contact") or auto_target)
                k = str(single.get("kind") or kind)
                platform_type = str(single.get("platform_type") or "").strip() or None
                callsign_name = str(single.get("callsign_name") or "").strip() or None
                spoken = _contact_spoken_label(contact, k, platform_type, callsign_name=callsign_name)
                session.last_contact = contact
                session.pending_contact_kind = None
                session.pending_contact_mode = None
                session.pending_contact_options = []
                if mode == "tacan":
                    ch = int(single.get("channel"))
                    band = str(single.get("band") or "X").upper()
                    return f"{spoken} TACAN is {ch}{band}."
                mhz = float(single.get("mhz"))
                return f"{spoken} frequency is {mhz:.3f} MHz."

        session.pending_contact_kind = kind
        session.pending_contact_mode = mode
        session.pending_contact_options = options
        choices = _disambiguation_choices(kind, options, listed=listed)
        return _pending_contact_prompt_with_options(kind, mode, choices)

    contact = str(result.get("contact") or session.last_contact or "that contact")
    kind = str(result.get("kind") or "contact")
    platform_type = str(result.get("platform_type") or "").strip() or None
    callsign_name = str(result.get("callsign_name") or "").strip() or None
    spoken = _contact_spoken_label(contact, kind, platform_type, callsign_name=callsign_name)
    if result.get("error") == "missing":
        if mode == "tacan":
            return f"I can't provide TACAN for {spoken}."
        return f"I can't provide a radio frequency for {spoken}."

    session.last_contact = contact
    session.pending_contact_kind = None
    session.pending_contact_mode = None
    session.pending_contact_options = []
    if mode == "tacan":
        ch = int(result.get("channel"))
        band = str(result.get("band") or "X").upper()
        return f"{spoken} TACAN is {ch}{band}."

    mhz = float(result.get("mhz"))
    return f"{spoken} frequency is {mhz:.3f} MHz."


def _airfield_frequency_fast_reply(transcript: str, session: Session) -> str | None:
    from mission.frequencies import detect_theatre, find_active_miz_path, match_airfield_in_text, resolve_frequency
    
    band = _extract_band_from_query(transcript)
    theatre = detect_theatre()
    
    airfield = None
    explicit_airfield_query = bool(re.search(r"\b(tower|approach|airfield|airport|traffic)\b", transcript.lower()))
    should_try_airfield_match = bool(re.search(r"\b(freq|frequency|channel|freak)\b", transcript.lower())) or explicit_airfield_query
    if should_try_airfield_match:
        match = match_airfield_in_text(transcript, theatre=theatre) or match_airfield_in_text(transcript)
        if match and match.get("match_type") == "exact":
            airfield = str(match.get("canonical") or "") or None
        elif match and match.get("match_type") == "fuzzy":
            return "I couldn't confidently identify the airfield. Say the field again."
    
    # Fall back to session airfield if not found in query
    if not airfield and session.last_airfield:
        airfield = session.last_airfield
    
    if not airfield:
        if should_try_airfield_match or explicit_airfield_query or band is not None:
            return "I couldn't confidently identify the airfield. Say the field again."
        return None
    
    # Default to tower if no band specified
    if band is None:
        band = "tower"
    
    state = fetch_state()
    in_jet = state is not None or detect_theatre() is not None or find_active_miz_path() is not None
    freq_result = resolve_frequency(airfield, in_jet=in_jet, band=band)

    if isinstance(freq_result, dict) and freq_result.get("source") == "out-of-ao":
        airfield_name = str(freq_result.get("airfield") or airfield)
        return f"{airfield_name} is outside the current AO."
    
    if not freq_result or freq_result.get("mhz") is None:
        airfield_name = str(airfield or session.last_airfield or "that airfield")
        if band and band != "tower":
            return f"I can't provide a {band} frequency for {airfield_name}."
        return f"I can't provide a tower frequency for {airfield_name}."
    
    # Store for follow-up queries
    session.last_airfield = freq_result.get("airfield") or airfield
    
    mhz = freq_result.get("mhz")
    band_label = band if band != "tower" else "Tower"
    return f"{band_label} frequency is {mhz:.3f} MHz."


def _is_closing_statement(text: str) -> bool:
    """Detect acknowledgment/closing statements that don't require tool calls."""
    q = text.lower().strip()
    if "?" in q:
        return False
    if re.search(r"\b(what|how|why|when|where|which|who|can|could|would|should|do|does|did)\b", q):
        return False
    # Match common closing patterns
    closing_patterns = [
        r"\b(that's it|that is it|all set|we're good|sounds good|roger|copy|wilco|thanks|thank you|cheers|gotcha|got it)\b",
        r"^(yep|yup|sure|good|alright|thanks)(?:\s|$)",
        r"^(roger that|copy that|understood)$",
        r"^(yeah|yep|right|correct|exactly)\b.*\b(because|since)\b",
    ]
    return any(re.search(pattern, q) for pattern in closing_patterns)


def _closing_statement_reply() -> str:
    """Return a brief, instructor-appropriate closing acknowledgment."""
    replies = [
        "Roger that.",
        "Copy.",
        "You're welcome.",
        "Good luck out there.",
        "Let me know if you need anything else.",
    ]
    import random
    return random.choice(replies)


def _call_the_ball_fast_reply() -> str:
    proc = load_proc("carrier_landing_case_i_recovery")
    if not proc:
        return (
            "In the groove with wings level, make the ball call: side number, Hornet Ball, "
            "then fuel in thousands, for example, three-zero-one, Hornet Ball, three-point-two."
        )

    for step in proc.get("steps", []):
        action = str(step.get("action", "")).lower()
        voiced = str(step.get("voiced", ""))
        if "call the ball" in action or "call the ball" in voiced.lower():
            more = str(step.get("more_info", "")).strip()
            if more:
                return more
            if voiced:
                return voiced

    return (
        "In the groove with wings level, make the ball call: side number, Hornet Ball, "
        "then fuel in thousands, for example, three-zero-one, Hornet Ball, three-point-two."
    )


def _dedupe_immediate_phrase_repeats(text: str, min_words: int = 2, max_words: int = 8) -> str:
    """Remove immediate repeated word spans like 'ensure the communication Ensure the communication'."""
    tokens = text.split()
    if len(tokens) < min_words * 2:
        return text

    def _norm(tok: str) -> str:
        return re.sub(r"(^\W+|\W+$)", "", tok).lower()

    out: list[str] = []
    i = 0
    while i < len(tokens):
        max_n = min(max_words, (len(tokens) - i) // 2)
        matched = False
        for n in range(max_n, min_words - 1, -1):
            a = [_norm(t) for t in tokens[i : i + n]]
            b = [_norm(t) for t in tokens[i + n : i + 2 * n]]
            if not all(a) or not all(b):
                continue
            if a == b:
                out.extend(tokens[i : i + n])
                i += 2 * n
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1

    return " ".join(out)


def clean_reply(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    # Strip adjacent duplicate words — catches model stutters like "select select".
    text = re.sub(r"\b(\w+) \1\b", r"\1", text, flags=re.I)
    # Strip phrase-level duplication: "It's bound to X... It's bound to X..."
    # Matches multi-word phrases (3-11 words) that appear twice in succession.
    text = re.sub(r"(\S+(?:\s+\S+){2,10}?)\s+\1\b", r"\1", text, flags=re.I)
    text = _dedupe_immediate_phrase_repeats(text)
    # Strip adjacent duplicate full sentences.
    parts = re.split(r"(?<=[.!?])\s+", text)
    deduped: list[str] = []
    for p in parts:
        if deduped and p.strip().lower() == deduped[-1].strip().lower():
            continue
        deduped.append(p)
    text = " ".join(deduped)
    return text.strip()


def looks_incomplete(text: str, done_reason: str | None) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    if (done_reason or "").lower() in {"length", "max_tokens"}:
        return True

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
        if len(t.split()) >= 15:
            return True

    return False


def stitch_reply(base: str, continuation: str) -> str:
    b = (base or "").strip()
    c = (continuation or "").strip()
    if not b:
        return c
    if not c:
        return b
    if c[0] in ".,;:!?":
        return (b + c).strip()
    return (b + " " + c).strip()


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


def call_ollama_with_tools(transcript: str, session: Session, model: str) -> str:
    session.last_reply_incomplete = False
    query = _contextualize_transcript(transcript, session)
    contact_query = transcript if session.pending_contact_options else query

    # Deterministic phraseology path for critical carrier comms callouts.
    if _is_call_the_ball_query(query):
        return _call_the_ball_fast_reply()

    # Deterministic cockpit state path — heading, altitude, airspeed, master arm, radar mode.
    if _is_cockpit_state_query(query):
        state_fast = _cockpit_state_fast_reply(query)
        if state_fast:
            return state_fast

    # Deterministic full-procedure path for explicit multi-step weapon/task requests.
    if _is_full_procedure_query(query):
        proc_fast = _full_procedure_fast_reply(query, session)
        if proc_fast:
            return proc_fast

    # Deterministic mission contact path for carrier/tanker radio and TACAN.
    if session.pending_contact_options or _is_miz_contact_query(contact_query) or _is_contact_followup_query(contact_query, session):
        contact_fast = _miz_contact_fast_reply(contact_query, session)
        if contact_fast:
            return contact_fast

    # Deterministic radio-frequency path to avoid model drift/hallucinated channels.
    if _is_airfield_frequency_query(query):
        fast = _airfield_frequency_fast_reply(query, session)
        if fast:
            return fast

    # Detect and handle closing/acknowledgment statements without tool calls.
    if _is_closing_statement(query):
        return _closing_statement_reply()

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session.history[-MAX_HISTORY * 2 :]

    proc_context = ""
    if session.active_proc:
        proc_context = (
            f"\nSession context: active procedure is {session.active_proc}, "
            f"currently at step {session.active_step + 1}."
        )

    user_content = transcript
    if query != transcript:
        user_content = f"{transcript}\n\nContext hint from prior user turn: {query}"

    messages.append({"role": "user", "content": user_content + proc_context})

    total_tool_calls = 0
    for _ in range(MAX_TOOL_CALLS + 1):
        conf = session.turn_confidence if session.turn_confidence in TARGET_TOKENS_BY_CONF else "MEDIUM"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": TOOL_SCHEMAS,
            "options": {
                "num_predict": TARGET_TOKENS_BY_CONF[conf],
                "temperature": TEMP_BY_CONF[conf],
                "repeat_penalty": 1.15,
                "stop": STOP_TOKENS,
            },
        }

        r = httpx.post(OLLAMA_URL, json=payload, timeout=25.0)
        r.raise_for_status()
        resp = r.json()
        msg = resp.get("message", {})
        done_reason = str(resp.get("done_reason", ""))
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # Hard grounding guard: do not allow a direct answer before at least
            # one tool result has been collected for this turn.
            if total_tool_calls == 0:
                total_tool_calls, used_any = _run_grounding_fallback_chain(
                    query, session, messages, total_tool_calls
                )
                if used_any:
                    continue

            content = clean_reply(msg.get("content", ""))
            if not content:
                return "Say again?"

            if looks_incomplete(content, done_reason):
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
                            "stop": STOP_TOKENS,
                        },
                    }
                    rc = httpx.post(OLLAMA_URL, json=cont_payload, timeout=20.0)
                    rc.raise_for_status()
                    cont_resp = rc.json()
                    cont_text = clean_reply(cont_resp.get("message", {}).get("content", ""))
                    stitched = stitch_reply(content, cont_text)
                    session.last_reply_incomplete = looks_incomplete(stitched, str(cont_resp.get("done_reason", "")))
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
            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

            # If a short-action lookup fails, immediately fall back to docs search
            # so we still attempt to answer from indexed content before replying.
            if (
                name == "get_quick_action"
                and isinstance(result, dict)
                and result.get("ok") is True
                and result.get("found") is False
                and total_tool_calls < MAX_TOOL_CALLS
            ):
                proc_fn = TOOL_REGISTRY.get("list_procedures")
                if proc_fn:
                    try:
                        proc_result = proc_fn({"query": query, "limit": 3}, session)
                    except Exception as e:
                        proc_result = {"ok": False, "error": f"Fallback procedure search failed: {e}"}
                    total_tool_calls += 1
                    messages.append(
                        {
                            "role": "tool",
                            "name": "list_procedures",
                            "content": json.dumps(proc_result),
                        }
                    )

                if total_tool_calls < MAX_TOOL_CALLS:
                    docs_fn = TOOL_REGISTRY.get("search_natops")
                    if docs_fn:
                        try:
                            docs_result = docs_fn({"query": query, "top_k": 3}, session)
                        except Exception as e:
                            docs_result = {"ok": False, "error": f"Fallback docs search failed: {e}"}
                        total_tool_calls += 1
                        messages.append(
                            {
                                "role": "tool",
                                "name": "search_natops",
                                "content": json.dumps(docs_result),
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


def continue_last_answer(session: Session, model: str) -> str:
    conf = session.turn_confidence if session.turn_confidence in TARGET_TOKENS_BY_CONF else "MEDIUM"
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
            "stop": STOP_TOKENS,
        },
    }
    r = httpx.post(OLLAMA_URL, json=payload, timeout=20.0)
    r.raise_for_status()
    resp = r.json()
    text = clean_reply(resp.get("message", {}).get("content", ""))
    session.last_reply_incomplete = looks_incomplete(text, str(resp.get("done_reason", "")))
    return text or "Say again?"
