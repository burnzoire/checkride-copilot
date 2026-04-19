"""
Instructor Co-Pilot (ICP) agent.

A separate LLM persona from the operational copilot. Where the copilot delivers
pinned steps and switch locations, the ICP explains *why*, gives technique
context, discusses common errors, and can rewrite procedure steps on demand.

Usage from voice_demo:
    from orchestrator.icp import ask_icp, rewrite_step
    reply = ask_icp("Why do we cool the Maverick seeker before arming?")
    voiced = rewrite_step("Master Arm switch - ARM (UP) Alignment Time Remaining")
"""

import re
import httpx
from loguru import logger

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"

# ─────────────────────────────────────────────────────────────────────────────
# Instructor Co-Pilot system prompt
# ─────────────────────────────────────────────────────────────────────────────
# Tone: authoritative but approachable, like a 2000-hr Hornet IP running a
# ground brief. Explains the WHY, flags common student errors, uses correct
# aviation phraseology. Not a manual — a teacher.
# ─────────────────────────────────────────────────────────────────────────────

ICP_SYSTEM = """\
You are the Checkride Copilot — an F/A-18C Hornet instructor pilot (IP) with
2,000+ hours in the jet. You train, you do not recite. Explain the WHY, flag
common errors, speak the way a real IP would in a brief or on hot mic.

Direct, calm, authoritative. Never encyclopaedic, never apologetic.
Never mention manuals or documents — you are the knowledge.
Short sentences. Under 40 words unless depth is genuinely required.
"""

REWRITE_SYSTEM = """\
You are an F/A-18C Hornet instructor pilot rewriting a procedure step for
spoken delivery in a cockpit training session.

Rules:
1. Action-first, imperative: "Set", "Select", "Press" — not "You should" or
   "Go in".
2. One sentence. Keep the primary action; drop secondary clauses.
3. No parenthetical acronym expansions. Remove "(Stores Management System)" etc.
4. Strip DCS keybind notation inside angle brackets entirely.
5. Never substitute a control name. "Cage/Uncage Button" stays exactly that.
6. No "Note:" annotations — omit them.
7. Under 20 words unless the action genuinely requires more.
8. Output ONLY the rewritten text. No preamble, quotes, or numbering.
"""


def _call(system: str, prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model":   model,
        "system":  system,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.3, "num_predict": 200},
    }
    r = httpx.post(OLLAMA_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["response"].strip()


def ask_icp(
    question:  str,
    history:   list[dict] | None = None,
    model:     str = DEFAULT_MODEL,
) -> str:
    """
    Answer a question from the Instructor Co-Pilot perspective.

    history: optional list of {"role": "user"|"assistant", "content": str}
             used to build a multi-turn context string.
    """
    ctx = ""
    if history:
        for turn in history[-6:]:   # last 3 exchanges
            role = "Student" if turn["role"] == "user" else "IP"
            ctx += f"{role}: {turn['content']}\n"
    prompt = f"{ctx}Student: {question}\nIP:"
    try:
        return _call(ICP_SYSTEM, prompt, model)
    except Exception as e:
        logger.warning(f"ICP LLM error: {e}")
        return "Stand by — comms issue."


def rewrite_step(action: str, model: str = DEFAULT_MODEL) -> str:
    """
    Rewrite a single raw procedure step in instructor voice.
    Returns the voiced text, or the original on error.
    """
    # Strip OCR-noise characters before sending
    clean = re.sub(r"[^\x20-\x7E]", " ", action)
    clean = re.sub(r"\s+", " ", clean).strip()
    try:
        return _call(REWRITE_SYSTEM, f"Rewrite this step:\n{clean}", model)
    except Exception as e:
        logger.warning(f"ICP rewrite error: {e}")
        return action
