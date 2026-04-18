"""
Rule-based diagnostic engine.

diagnose_action(action_id, live_state) → dict with can_execute, blockers, warnings.
All logic is deterministic — no LLM involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiagnosticRule:
    condition_id: str
    description: str
    check_field: str
    required_value: Any
    comparator: str       # "eq" | "ne" | "gte" | "lte" | "in" | "not_in"
    fix_instruction: str
    severity: str         # "blocking" | "warning"


def check_condition(rule: DiagnosticRule, live_state: dict) -> bool:
    """Return True if the condition is SATISFIED (not a blocker)."""
    val = live_state.get(rule.check_field)

    # Unknown field → conservatively treat as blocking
    if val is None or val == "unknown":
        return False

    c = rule.comparator
    if c == "eq":
        return val == rule.required_value
    if c == "ne":
        return val != rule.required_value
    if c == "gte":
        try:
            return float(val) >= float(rule.required_value)
        except (TypeError, ValueError):
            return False
    if c == "lte":
        try:
            return float(val) <= float(rule.required_value)
        except (TypeError, ValueError):
            return False
    if c == "in":
        return val in rule.required_value
    if c == "not_in":
        return val not in rule.required_value
    return False


def diagnose_action(action_id: str, live_state: dict) -> dict:
    """
    Run all rules for the given action against live_state.

    Returns:
        {
            "action":       <normalized action id>,
            "can_execute":  <bool | None if no rules>,
            "blockers":     [{"condition", "description", "required_value",
                              "current_value", "fix", "severity"}, ...],
            "warnings":     [<fix_instruction str>, ...],
            "state_age_ms": <int>,
        }
    """
    from diagnostic.action_map import normalize_action
    from diagnostic.rules import get_rules_for_action

    normalized = normalize_action(action_id, live_state)
    rules = get_rules_for_action(normalized)

    if not rules:
        return {
            "action":       normalized,
            "can_execute":  None,
            "blockers":     [],
            "warnings":     [f"No diagnostic rules defined for action '{normalized}'."],
            "state_age_ms": live_state.get("data_age_ms", -1),
        }

    blockers: list[dict] = []
    warnings: list[str] = []

    for rule in rules:
        if not check_condition(rule, live_state):
            entry = {
                "condition":      rule.condition_id,
                "description":    rule.description,
                "required_value": str(rule.required_value),
                "current_value":  str(live_state.get(rule.check_field, "unknown")),
                "fix":            rule.fix_instruction,
                "severity":       rule.severity,
            }
            if rule.severity == "blocking":
                blockers.append(entry)
            else:
                warnings.append(rule.fix_instruction)

    return {
        "action":       normalized,
        "can_execute":  len(blockers) == 0,
        "blockers":     blockers,
        "warnings":     warnings,
        "state_age_ms": live_state.get("data_age_ms", -1),
    }
