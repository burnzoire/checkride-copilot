#!/usr/bin/env python3
"""
Instructor Co-Pilot rewriter.

For each procedure step, calls the local Ollama LLM with a strict
instructor-voice system prompt and writes the result into a "voiced"
field alongside the original "action" text.

"voiced" is what the TTS reads. "action" is preserved for RAG / search.

Run:
    python -m data.airframes.fa18c.voice_procedures [--model qwen2.5:7b] [--dry-run]
    python -m data.airframes.fa18c.voice_procedures --proc agm_65f_missile_ir_seeker_only

Pass --proc <key> to rewrite a single procedure for testing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

PROC_DIR   = Path("data/airframes/fa18c/procedures")
INDEX_PATH = PROC_DIR / "index.json"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ── Instructor Co-Pilot system prompt ────────────────────────────────────────
#
# These guidelines define the voice. They apply to EVERY step rewrite.
# Edit here to tune the persona — the script re-uses this prompt for all procs.
#
SYSTEM_PROMPT = """You are an F/A-18C Hornet instructor pilot giving real-time
spoken guidance to a student in the cockpit. Rewrite a single procedure step
in your voice.

Rules — follow all of them exactly:
1. Action-first, imperative mood. Start with the verb: "Set", "Select",
   "Press", "Check", "Confirm" — not "You should", "Go in", "Make sure".
2. One sentence max. If the original has multiple actions, keep the primary
   one only.
3. No parenthetical acronym expansions. Remove "(Stores Management System)"
   and similar inline definitions entirely.
4. Strip DCS keybind notation: remove anything inside (◄ ... ►) brackets.
   Replace with the plain action, e.g. "press Weapon Release".
5. Preserve all control names exactly as given. Never substitute one button
   or switch for another. If it says "Cage/Uncage Button", say "Cage/Uncage
   Button" — do not replace it with any other control name.
6. No "Note:" annotations — omit them entirely.
7. Under 20 words unless the action genuinely requires more.
8. Output ONLY the rewritten step text. No preamble, no quotes, no numbering.
"""

USER_TEMPLATE = "Rewrite this step:\n{action}"


def _call_ollama(action: str, model: str) -> str:
    payload = {
        "model":  model,
        "system": SYSTEM_PROMPT,
        "prompt": USER_TEMPLATE.format(action=action),
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
    }
    r = httpx.post(OLLAMA_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["response"].strip()


def _rewrite_proc(proc_path: Path, model: str, dry_run: bool) -> int:
    proc = json.loads(proc_path.read_text(encoding="utf-8"))
    changed = 0
    for step in proc["steps"]:
        action = step["action"]
        try:
            voiced = _call_ollama(action, model)
        except Exception as e:
            print(f"    [WARN] Ollama error on step {step['num']}: {e}", file=sys.stderr)
            voiced = action  # fall back to original
        step["voiced"] = voiced
        changed += 1
        print(f"    {step['num']:2d}. {voiced}")
    if not dry_run:
        proc_path.write_text(json.dumps(proc, indent=2, ensure_ascii=False), encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="qwen2.5:7b")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print rewrites without saving")
    ap.add_argument("--review",   action="store_true",
                    help="Pause for y/n approval before writing each procedure")
    ap.add_argument("--proc",     default=None,
                    help="Rewrite a single procedure by key")
    ap.add_argument("--category", default=None,
                    help="Rewrite one category only (e.g. weapons, normal_ops)")
    args = ap.parse_args()

    index  = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    entries = index["procedures"]

    if args.proc:
        entries = [e for e in entries if e["key"] == args.proc]
        if not entries:
            sys.exit(f"Procedure '{args.proc}' not found in index.")
    elif args.category:
        entries = [e for e in entries if e["category"] == args.category]
        if not entries:
            sys.exit(f"No procedures in category '{args.category}'.")

    total_steps = 0
    for entry in entries:
        proc_path = PROC_DIR / entry["path"]
        print(f"\n{entry['canonical']} ({entry['step_count']} steps)")
        t0 = time.time()
        n = _rewrite_proc(proc_path, args.model, dry_run=True)   # always preview
        elapsed = time.time() - t0

        if args.dry_run:
            print(f"  (dry run, {elapsed:.1f}s)")
        else:
            if args.review:
                ans = input("  Write? [y/N] ").strip().lower()
                if ans != "y":
                    print("  Skipped.")
                    continue
            # Write for real
            _rewrite_proc(proc_path, args.model, dry_run=False)
            print(f"  Written. ({elapsed:.1f}s)")
        total_steps += n

    print(f"\nTotal: {total_steps} steps across {len(entries)} procedures.")
    if args.dry_run:
        print("(Dry run — no files written. Re-run without --dry-run to save.)")


if __name__ == "__main__":
    main()
