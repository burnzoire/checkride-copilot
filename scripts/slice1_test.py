#!/usr/bin/env python3
"""
First coding slice: DCS live state → diagnose → speak. No LLM required.

Validates:
  - DCS export collector reachability (falls back to sample state if offline)
  - Diagnostic engine correctness
  - Piper TTS audio playback

Usage:
    python scripts/slice1_test.py
    python scripts/slice1_test.py --action tgp_track
    python scripts/slice1_test.py --state tests/fixtures/sample_state_ready.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from diagnostic.engine import diagnose_action
from voice.tts import speak

COLLECTOR_URL = "http://localhost:7779/state"

SAMPLE_STATE = {
    "airframe": "FA-18C_hornet",
    "airframe_normalized": "fa18c",
    "battery_switch": "ON",
    "master_arm": "SAFE",
    "selected_weapon": "AIM-120C",
    "radar_mode": "RWS",
    "radar_powered": True,
    "tgp_powered": False,
    "tgp_mode": "STBY",
    "tgp_tdc_priority": False,
    "tgp_tracking": False,
    "laser_armed": False,
    "altitude_ft": 15000.0,
    "engine_left_state": "running",
    "engine_right_state": "running",
    "data_age_ms": 200,
}


def get_state(state_path: str | None) -> dict:
    if state_path:
        data = json.loads(Path(state_path).read_text())
        print(f"[state] Loaded from {state_path}")
        return data

    if HAS_HTTPX:
        try:
            r = httpx.get(COLLECTOR_URL, timeout=1.0)
            r.raise_for_status()
            print("[state] Live state from collector.")
            return r.json()
        except Exception as e:
            print(f"[state] Collector not reachable ({e}) — using sample state.")
    else:
        print("[state] httpx not installed — using sample state.")

    return SAMPLE_STATE


def format_for_speech(result: dict) -> str:
    stale = result.get("state_age_ms", 0) > 2000
    suffix = " Note: data may be out of date." if stale else ""

    if result.get("can_execute") is None:
        return f"No rules found for this action.{suffix}"

    if result["can_execute"] and not result.get("warnings"):
        return f"Clear to execute.{suffix}"

    if result["can_execute"] and result.get("warnings"):
        w = result["warnings"][0]
        extra = f" Plus {len(result['warnings']) - 1} more." if len(result["warnings"]) > 1 else ""
        return f"Clear with caution. {w}{extra}{suffix}"

    blockers = result["blockers"]
    fix = blockers[0]["fix"]
    extra = f" Plus {len(blockers) - 1} more blockers." if len(blockers) > 1 else ""
    return f"Blocked. {fix}{extra}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice-1: state → diagnose → speak")
    parser.add_argument("--action", default="fire_aim120", help="Action to diagnose")
    parser.add_argument("--state", default=None, help="Path to a state JSON fixture")
    parser.add_argument("--no-speak", action="store_true", help="Skip TTS playback")
    args = parser.parse_args()

    print("=== DCS Copilot: Slice 1 ===\n")

    # 1 — Get state
    state = get_state(args.state)
    print(f"[state] master_arm={state.get('master_arm')}  "
          f"radar={state.get('radar_mode')}  "
          f"tgp_powered={state.get('tgp_powered')}  "
          f"data_age={state.get('data_age_ms')}ms\n")

    # 2 — Diagnose
    t0 = time.monotonic()
    result = diagnose_action(args.action, state)
    diag_ms = int((time.monotonic() - t0) * 1000)

    print(f"[diag] action={result['action']}  can_execute={result['can_execute']}  ({diag_ms}ms)")
    for b in result["blockers"]:
        print(f"  BLOCKING  {b['condition']}: {b['fix']}")
    for w in result.get("warnings", []):
        print(f"  WARNING   {w}")

    # 3 — TTS
    text = format_for_speech(result)
    print(f"\n[tts]  \"{text}\"")

    if not args.no_speak:
        t1 = time.monotonic()
        speak(text)
        tts_ms = int((time.monotonic() - t1) * 1000)
        print(f"\n[timing] diag={diag_ms}ms  tts={tts_ms}ms  total={diag_ms + tts_ms}ms")
    else:
        print("\n[tts] Playback skipped (--no-speak).")


if __name__ == "__main__":
    main()
