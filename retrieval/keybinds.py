"""
DCS keybind reader for F/A-18C.

Parses the player's joystick .diff.lua files and returns what physical
button each DCS action is bound to, so location responses can include
"You have it bound to VKB Buttons 11, 10, 12, 13 for AMRAAM, Sparrow,
Gun, Sidewinder respectively."
"""

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

_DCS_INPUT_ROOT = Path.home() / "Saved Games" / "DCS" / "Config" / "Input"

_DEVICE_PATTERNS: list[tuple[str, str]] = [
    (r"VKB.*Gunfighter", "VKB Gunfighter"),
    (r"VKB.*Gladiator",  "VKB Gladiator"),
    (r"S-TECS",       "S-TECS throttle"),
    (r"WINWING.*UFC", "WinWing UFC"),
    (r"WINWING.*ICP", "WinWing ICP"),
    (r"WINWING.*MFD", "WinWing MFD"),
    (r"WINWING",      "WinWing"),
    (r"MFG",          "MFG Crosswind"),
    (r"T-Rudder",     "T-Rudder"),
    (r"TWCS",         "TWCS throttle"),
]


def _device_short(filename: str) -> str:
    """Extract a short device name from a diff.lua filename."""
    name = re.sub(r"\s*\{[^}]+\}.*$", "", filename).strip()
    name = re.sub(r"\.diff\.lua$", "", name, flags=re.IGNORECASE).strip()
    for pattern, short in _DEVICE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return short
    return name.strip() or filename


def _parse_diff(path: Path) -> dict[str, list[str]]:
    """Return {action_name: [bound_key, ...]} from one .diff.lua file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    kd_pos = text.find('"keyDiffs"')
    if kd_pos == -1:
        return {}
    try:
        outer_open = text.index("{", kd_pos)
    except ValueError:
        return {}

    name_re = re.compile(r'\["name"\]\s*=\s*"([^"]+)"')
    key_re  = re.compile(r'\["key"\]\s*=\s*"([^"]+)"')
    bindings: dict[str, list[str]] = {}

    i, depth, block_start = outer_open + 1, 1, None
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            if depth == 1:
                block_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 1 and block_start is not None:
                block = text[block_start : i + 1]
                m = name_re.search(block)
                if m:
                    keys = key_re.findall(block)
                    if keys:
                        bindings.setdefault(m.group(1), []).extend(keys)
                block_start = None
        i += 1

    return bindings


@lru_cache(maxsize=4)
def load(airframe: str = "FA-18C_hornet") -> dict[str, list[tuple[str, str]]]:
    """Return {action_name: [(device_short, key), ...]} across all diff files."""
    merged: dict[str, list[tuple[str, str]]] = {}
    input_dir = _DCS_INPUT_ROOT / airframe / "joystick"
    if not input_dir.exists():
        return merged
    for f in sorted(input_dir.glob("*.diff.lua")):
        device = _device_short(f.name)
        for action, keys in _parse_diff(f).items():
            entries = merged.setdefault(action, [])
            for key in keys:
                pair = (device, key)
                if pair not in entries:
                    entries.append(pair)
    return merged


def _fmt_key(key: str) -> str:
    if m := re.match(r"JOY_BTN(\d+)$", key, re.IGNORECASE):
        return f"Button {m.group(1)}"
    if m := re.match(r"JOY_([A-Z0-9_]+)$", key, re.IGNORECASE):
        return m.group(1).replace("_", " ").title()
    return key


def describe_switch(dcs_actions: dict[str, str], airframe: str = "FA-18C_hornet") -> str | None:
    """
    Given {label: dcs_action_name}, return a spoken keybind sentence.

    Format: "You have it bound to VKB Buttons 11, 10, 12, 13 for AMRAAM,
             Sparrow, Gun, Sidewinder respectively."
    Returns None if no bindings found.
    """
    kb = load(airframe)
    if not kb:
        return None

    # Collect first (device, key) per label.
    label_device: dict[str, str] = {}
    label_key:    dict[str, str] = {}
    for label, action in dcs_actions.items():
        entries = kb.get(action, [])
        if entries:
            device, key = entries[0]
            label_device[label] = device
            label_key[label]    = _fmt_key(key)

    if not label_key:
        return None

    # Group labels by device; pick the device with the most complete set.
    by_device: dict[str, list[str]] = defaultdict(list)
    for label, device in label_device.items():
        by_device[device].append(label)
    best = max(by_device, key=lambda d: len(by_device[d]))
    labels = by_device[best]
    keys   = [label_key[l] for l in labels]

    # "Buttons 11, 10, 12, 13 for AMRAAM, Sparrow, Gun, Sidewinder respectively"
    btn_nums = [re.match(r"Button (\d+)$", k) for k in keys]
    if len(keys) == 1:
        return f"You have it bound to {best} {keys[0]} for {labels[0]}."
    if all(btn_nums):
        nums = ", ".join(m.group(1) for m in btn_nums)
        cmds = ", ".join(labels)
        return f"You have it bound to {best} Buttons {nums} for {cmds} respectively."
    parts = ", ".join(f"{k} for {l}" for k, l in zip(keys, labels))
    return f"You have it bound to {best} — {parts}."
