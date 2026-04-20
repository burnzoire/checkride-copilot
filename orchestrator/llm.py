from __future__ import annotations

import json
import re
from typing import Any

import httpx

from orchestrator.session import Session
from orchestrator.tools import TOOL_REGISTRY, TOOL_SCHEMAS

OLLAMA_URL = "http://localhost:11434/api/chat"
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


def clean_reply(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
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
                            "stop": ["[Reference", "[Switch", "let me know", "if you have"],
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
            "stop": ["[Reference", "[Switch", "let me know", "if you have"],
        },
    }
    r = httpx.post(OLLAMA_URL, json=payload, timeout=20.0)
    r.raise_for_status()
    resp = r.json()
    text = clean_reply(resp.get("message", {}).get("content", ""))
    session.last_reply_incomplete = looks_incomplete(text, str(resp.get("done_reason", "")))
    return text or "Say again?"
