#!/usr/bin/env python3
"""
Checkride Copilot installer.

Appends the Export.lua snippet to the user's DCS export chain.
Safe to run multiple times — will not double-install.

Usage:
    python scripts/install.py
    python scripts/install.py --dcs-variant openbeta
    python scripts/install.py --dry-run
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SNIPPET   = REPO_ROOT / "dcs" / "Export.lua"
MARKER    = "-- Checkride Copilot"

# Saved Games locations to try, in order
DCS_VARIANTS = ["DCS", "DCS.openbeta", "DCS.release"]


def find_saved_games() -> Path:
    candidates = [
        Path.home() / "Saved Games",
        Path.home() / "OneDrive" / "Saved Games",       # iCloud/OneDrive redirect
        Path.home() / "iCloudDrive" / "Saved Games",
    ]
    for base in candidates:
        if base.exists():
            return base
    raise FileNotFoundError(
        "Could not find Saved Games folder. "
        "Use --saved-games to specify the path explicitly."
    )


def find_export_lua(saved_games: Path, variant: str | None) -> Path:
    variants = [variant] if variant else DCS_VARIANTS
    for v in variants:
        candidate = saved_games / v / "Scripts" / "export.lua"
        if candidate.exists():
            return candidate
    # If none found, use the first variant as the install target
    fallback = saved_games / (variant or DCS_VARIANTS[0]) / "Scripts" / "export.lua"
    return fallback


def already_installed(export_lua: Path) -> bool:
    if not export_lua.exists():
        return False
    return MARKER in export_lua.read_text(encoding="utf-8", errors="replace")


def backup(export_lua: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = export_lua.with_suffix(f".lua.backup_{ts}")
    shutil.copy2(export_lua, backup_path)
    return backup_path


def install(export_lua: Path, dry_run: bool) -> None:
    snippet = SNIPPET.read_text(encoding="utf-8")

    if already_installed(export_lua):
        print(f"Already installed in:\n  {export_lua}")
        return

    if dry_run:
        print(f"[dry-run] Would append to: {export_lua}")
        print(f"[dry-run] Snippet ({len(snippet)} chars) starts with: {snippet[:80]!r}")
        return

    # Ensure the Scripts directory exists (first-time DCS install, no export.lua yet)
    export_lua.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if export_lua.exists():
        bk = backup(export_lua)
        print(f"Backed up existing export.lua -> {bk.name}")
        existing = export_lua.read_text(encoding="utf-8")

    with export_lua.open("w", encoding="utf-8") as f:
        if existing:
            f.write(existing.rstrip())
            f.write("\n\n")
        f.write(snippet)

    print(f"Installed to:\n  {export_lua}")
    print("Restart DCS (or reload the mission) for the change to take effect.")


def uninstall(export_lua: Path, dry_run: bool) -> None:
    if not export_lua.exists():
        print("export.lua not found — nothing to uninstall.")
        return

    text = export_lua.read_text(encoding="utf-8")
    if MARKER not in text:
        print("Checkride Copilot snippet not found in export.lua.")
        return

    # Strip everything from our marker to the end (our block is always appended last)
    idx = text.index(MARKER)
    # Step back to the preceding newline to clean up whitespace
    cleaned = text[:idx].rstrip() + "\n"

    if dry_run:
        print(f"[dry-run] Would remove {len(text) - len(cleaned)} chars from {export_lua}")
        return

    bk = backup(export_lua)
    print(f"Backed up -> {bk.name}")
    export_lua.write_text(cleaned, encoding="utf-8")
    print(f"Removed Checkride Copilot snippet from:\n  {export_lua}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkride Copilot DCS export installer")
    parser.add_argument("--uninstall",   action="store_true", help="Remove the snippet")
    parser.add_argument("--dry-run",     action="store_true", help="Show what would happen, don't write")
    parser.add_argument("--dcs-variant", default=None,        help="DCS variant folder (e.g. 'DCS.openbeta')")
    parser.add_argument("--saved-games", default=None,        help="Override Saved Games path")
    args = parser.parse_args()

    try:
        saved_games = Path(args.saved_games) if args.saved_games else find_saved_games()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    export_lua = find_export_lua(saved_games, args.dcs_variant)
    print(f"DCS export.lua: {export_lua}")

    if args.uninstall:
        uninstall(export_lua, args.dry_run)
    else:
        install(export_lua, args.dry_run)


if __name__ == "__main__":
    main()
