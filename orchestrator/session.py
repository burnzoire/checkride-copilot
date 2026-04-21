from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    active_proc: str | None = None
    active_step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    last_confidence: str = "MEDIUM"
    last_retrieval_confidence: str = "MEDIUM"
    turn_confidence: str = "MEDIUM"
    last_reply_incomplete: bool = False
    last_quick_action_key: str | None = None
    last_airfield: str | None = None

    def begin_turn(self) -> None:
        self.turn_confidence = "MEDIUM"

    def add_user_turn(self, content: str, ts: float) -> None:
        self.history.append({"role": "user", "content": content, "ts": ts})

    def add_assistant_turn(self, content: str, ts: float) -> None:
        self.history.append({"role": "assistant", "content": content, "ts": ts})
