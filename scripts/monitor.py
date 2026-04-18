#!/usr/bin/env python3
"""
Live DCS state monitor for Phase 0 cockpit parameter discovery.

Listens for UDP packets from Export.lua and prints a real-time diff of the
raw CockpitParams table. Toggle switches in the cockpit and watch which
indices change — those are the indices to map in the normalizer.

Usage:
    python scripts/monitor.py                  # live monitor
    python scripts/monitor.py --capture        # also save every packet to captures/
    python scripts/monitor.py --diff           # only print changed indices (quieter)

Captures are saved to scripts/captures/packet_<n>.json for offline analysis.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

COLLECTOR_URL = "http://127.0.0.1:7779/state"
POLL_INTERVAL = 0.25   # seconds
CAPTURES = Path(__file__).parent / "captures"

# Fields we care most about for Phase 0 — highlight these prominently
PRIORITY_FIELDS = {
    "master_arm", "battery_switch", "external_power",
    "radar_powered", "radar_mode",
    "tgp_powered", "tgp_mode", "tgp_tdc_priority", "tgp_tracking", "tgp_lasing",
    "selected_weapon", "laser_armed",
    "engine_left_state", "engine_right_state",
    "gear_position", "flaps_position", "hook_position",
    "autopilot_engaged",
}


def _fmt_val(v) -> str:
    if v is True:  return "TRUE"
    if v is False: return "false"
    if isinstance(v, float): return f"{v:.1f}"
    return str(v)


def _print_state(state: dict, prev_state: dict, cur_cp: dict, diff_only: bool) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    age = state.get("data_age_ms", -1)
    airframe = state.get("airframe_normalized", "?")
    print(f"\n{'─'*60}")
    print(f"  {ts}  |  {airframe}  |  age={age}ms")
    print(f"{'─'*60}")

    # Always show all priority fields; mark changed ones with <<
    for field in sorted(PRIORITY_FIELDS):
        val = state.get(field)
        prev_val = prev_state.get(field)
        label = "??" if val in (None, "unknown") else _fmt_val(val)
        changed_marker = "  <<" if val != prev_val and prev_state else ""
        unknown_marker = "  [unknown]" if val in (None, "unknown") else ""
        print(f"  {field:<30} {label}{changed_marker}{unknown_marker}")

    # Raw CockpitParams index diff — for discovering unmapped fields
    changed = {k: v for k, v in cur_cp.items() if prev_cp.get(k) != v}
    if changed:
        print(f"\n  [Raw index changes — {len(changed)} indices]")
        for idx in sorted(changed.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            old = prev_cp.get(idx, "<new>")
            new = changed[idx]
            print(f"    [{idx:>5}]  {_fmt_val(old):>12}  ->  {_fmt_val(new)}")
    elif cur_cp:
        print(f"\n  [Raw indices: {len(cur_cp)} total, no changes]")
    else:
        print(f"\n  [Raw indices: none (LoGetCockpitParams may be empty)]")


def run(diff_only: bool, capture: bool) -> None:
    if capture:
        CAPTURES.mkdir(exist_ok=True)

    print(f"Polling {COLLECTOR_URL} at {1/POLL_INTERVAL:.0f} Hz  (Ctrl-C to stop)")
    print("Start DCS and get into the FA-18C cockpit.")
    if diff_only:
        print("Mode: DIFF — only printing changed CockpitParam indices.\n")
    else:
        print("Mode: FULL — printing all priority fields + changed indices.\n")

    prev_cp: dict = {}
    prev_state: dict = {}
    prev_state_hash: str = ""
    packet_count = 0
    waiting = False

    while True:
        try:
            try:
                r = httpx.get(COLLECTOR_URL, timeout=1.0)
                r.raise_for_status()
                state = r.json()
                waiting = False
            except httpx.HTTPStatusError:
                # 503 = no data yet
                if not waiting:
                    print("Waiting for DCS...", end="", flush=True)
                    waiting = True
                else:
                    print(".", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue
            except Exception:
                if not waiting:
                    print("Waiting for collector...", end="", flush=True)
                    waiting = True
                else:
                    print(".", end="", flush=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Also fetch raw CockpitParams for index discovery
            cur_cp: dict = {}
            try:
                rr = httpx.get(COLLECTOR_URL.replace("/state", "/raw_cp"), timeout=1.0)
                if rr.status_code == 200:
                    cur_cp = {str(k): v for k, v in rr.json().items()}
            except Exception:
                pass

            # Deduplicate — only print when something changes
            state_hash = json.dumps(state, sort_keys=True) + json.dumps(cur_cp, sort_keys=True)
            if state_hash == prev_state_hash:
                time.sleep(POLL_INTERVAL)
                continue
            prev_state_hash = state_hash
            packet_count += 1

            if capture:
                path = CAPTURES / f"state_{packet_count:05d}.json"
                path.write_text(json.dumps({"state": state, "raw_cp": cur_cp}, indent=2))

            _print_state(state, prev_state, cur_cp, diff_only)
            prev_state = state
            prev_cp = cur_cp

        except KeyboardInterrupt:
            break

        time.sleep(POLL_INTERVAL)

    print(f"\nDone. {packet_count} state changes observed.")
    if capture:
        print(f"Captures saved to: {CAPTURES}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff",    action="store_true", help="Only show changed CockpitParam indices")
    parser.add_argument("--capture", action="store_true", help="Save every packet to scripts/captures/")
    args = parser.parse_args()
    run(diff_only=args.diff, capture=args.capture)
