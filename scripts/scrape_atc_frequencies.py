#!/usr/bin/env python3
"""Scrape DCS terrain Radio.lua files into data/airfield_frequencies_by_map.json.

Usage:
  python -m scripts.scrape_atc_frequencies
  python -m scripts.scrape_atc_frequencies --dcs-root "D:/DCS World"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mission.frequencies import write_scraped_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcs-root", default=None, help="Path to DCS World root")
    args = parser.parse_args()

    root = Path(args.dcs_root) if args.dcs_root else None
    data = write_scraped_index(dcs_root=root)
    maps = data.get("maps", {})
    total = sum(len(v or []) for v in maps.values())
    print(f"Scraped {total} airfield radio entries across {len(maps)} maps.")


if __name__ == "__main__":
    main()
