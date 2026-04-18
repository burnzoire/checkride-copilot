"""
Maps natural-language pilot queries to canonical action IDs.
Handles aliases and uses live state to resolve ambiguous weapon references.
"""

from __future__ import annotations

from difflib import get_close_matches

# str  → canonical action_id
# list → ambiguous; resolved by live_state selected_weapon
ACTION_ALIASES: dict[str, str | list[str]] = {
    # Generic fire → resolved by selected weapon
    "fire":               ["fire_aim120", "fire_aim9x", "fire_gun"],
    "shoot":              ["fire_aim120", "fire_aim9x", "fire_gun"],
    "fire missile":       ["fire_aim120", "fire_aim9x"],
    "shoot missile":      ["fire_aim120", "fire_aim9x"],
    "launch missile":     ["fire_aim120", "fire_aim9x"],

    # AIM-120 / AMRAAM
    "fire aim-120":       "fire_aim120",
    "fire amraam":        "fire_aim120",
    "fire aim120":        "fire_aim120",
    "shoot amraam":       "fire_aim120",
    "fire_aim120":        "fire_aim120",

    # AIM-9
    "fire aim-9":         "fire_aim9x",
    "fire sidewinder":    "fire_aim9x",
    "fire aim9":          "fire_aim9x",
    "fire_aim9x":         "fire_aim9x",

    # GBU-12 / LGB
    "drop bomb":          "release_gbu12",
    "release bomb":       "release_gbu12",
    "drop gbu":           "release_gbu12",
    "release gbu":        "release_gbu12",
    "drop gbu-12":        "release_gbu12",
    "lase bomb":          "release_gbu12",
    "release_gbu12":      "release_gbu12",

    # TGP
    "track target":       "tgp_track",
    "tgp track":          "tgp_track",
    "designate target":   "tgp_track",
    "slew tgp":           "tgp_track",
    "tgp_track":          "tgp_track",

    # Laser
    "lase":               "lase_target",
    "lase target":        "lase_target",
    "fire laser":         "lase_target",
    "laser":              "lase_target",
    "lase_target":        "lase_target",

    # Engine start
    "start engine":       "start_engine",
    "engine start":       "start_engine",
    "start engines":      "start_engine",
    "start_engine":       "start_engine",
}

# Ordered fallback preference when weapon is ambiguous
_WEAPON_TO_ACTION: list[tuple[str, str]] = [
    ("AIM-120", "fire_aim120"),
    ("AMRAAM",  "fire_aim120"),
    ("AIM-9",   "fire_aim9x"),
    ("SIDEWINDER", "fire_aim9x"),
    ("GBU",     "release_gbu12"),
    ("BOMB",    "release_gbu12"),
]


def normalize_action(raw: str, live_state: dict | None = None) -> str:
    """
    Return the canonical action_id for a raw pilot query or alias string.

    Ambiguous aliases (list values) are resolved using live_state.selected_weapon.
    Falls back to fuzzy string matching if no exact alias is found.
    """
    key = raw.lower().strip()
    result = ACTION_ALIASES.get(key)

    if result is None:
        matches = get_close_matches(key, ACTION_ALIASES.keys(), n=1, cutoff=0.6)
        result = ACTION_ALIASES.get(matches[0]) if matches else key

    if isinstance(result, list):
        if live_state:
            selected = (live_state.get("selected_weapon") or "").upper()
            for fragment, action_id in _WEAPON_TO_ACTION:
                if fragment in selected:
                    return action_id
        return result[0]

    return result if isinstance(result, str) else key
