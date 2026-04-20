from __future__ import annotations

import enum
import re

from orchestrator.session import Session


class NavCommand(str, enum.Enum):
    NONE = "none"
    NEXT = "next"
    REPEAT = "repeat"
    RESTART = "restart"
    CANCEL = "cancel"
    CONTINUE_LAST_REPLY = "continue_last_reply"


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


def check(transcript: str, session: Session) -> NavCommand:
    stripped = transcript.strip()

    # Active procedure commands always have priority.
    if session.active_proc:
        if _CANCEL_PROC_RE.fullmatch(stripped):
            return NavCommand.CANCEL
        if _REPEAT_RE.fullmatch(stripped):
            return NavCommand.REPEAT
        if _RESTART_RE.search(stripped):
            return NavCommand.RESTART
        if _CONTINUE_RE.fullmatch(stripped):
            return NavCommand.NEXT

    # No active procedure: only allow continuation of an incomplete answer.
    if session.last_reply_incomplete and _CONTINUE_RE.fullmatch(stripped):
        return NavCommand.CONTINUE_LAST_REPLY

    return NavCommand.NONE
