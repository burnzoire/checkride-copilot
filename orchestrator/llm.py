from __future__ import annotations

import json
import re
from typing import Any

import httpx

from orchestrator.session import Session
from orchestrator.tools import TOOL_REGISTRY, TOOL_SCHEMAS, load_proc

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


def _is_closing_statement(text: str) -> bool:
    """Detect acknowledgment/closing statements that don't require tool calls."""
    q = text.lower().strip()
    # Match common closing patterns
    closing_patterns = [
        r"\b(that's it|that is it|all set|we're good|sounds good|roger|copy|wilco|thanks|thank you|cheers|gotcha|got it|ok|okay)\b",
        r"^(yep|yup|sure|good|alright|thanks)(?:\s|$)",
        r"^(roger that|copy that|understood)$",
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


def clean_reply(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    # Strip adjacent duplicate words — catches model stutters like "select select".
    text = re.sub(r"\b(\w+) \1\b", r"\1", text, flags=re.I)
    # Strip phrase-level duplication: "It's bound to X... It's bound to X..."
    # Matches multi-word phrases (3-11 words) that appear twice in succession.
    text = re.sub(r"(\S+(?:\s+\S+){2,10}?)\s+\1\b", r"\1", text, flags=re.I)
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

    # Deterministic phraseology path for critical carrier comms callouts.
    if _is_call_the_ball_query(transcript):
        return _call_the_ball_fast_reply()

    # Detect and handle closing/acknowledgment statements without tool calls.
    if _is_closing_statement(transcript):
        return _closing_statement_reply()

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
                docs_fn = TOOL_REGISTRY.get("search_natops")
                if docs_fn:
                    try:
                        docs_result = docs_fn({"query": transcript, "top_k": 3}, session)
                    except Exception as e:
                        docs_result = {"ok": False, "error": f"Grounding docs search failed: {e}"}
                    total_tool_calls += 1
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
                docs_fn = TOOL_REGISTRY.get("search_natops")
                if docs_fn:
                    try:
                        docs_result = docs_fn({"query": transcript, "top_k": 3}, session)
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
