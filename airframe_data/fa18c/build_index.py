"""
Build fa18c/cockpit_index.json from the NeoEngress cockpit layout image.

Pipeline:
  1. OCR the image with EasyOCR to get (text, bounding_box) pairs
  2. Normalise coordinates to 0-1 range
  3. Assign each label to a console area based on its x position
  4. Cluster nearby labels into panels using simple proximity grouping
  5. Emit cockpit_index.json — areas > panels > labels

Run:
    python -m airframe_data.fa18c.build_index

Output: airframe_data/fa18c/cockpit_index_raw.json   (OCR dump, for inspection)
        airframe_data/fa18c/cockpit_index.json        (curated index used by search)
"""

import json
import sys
from pathlib import Path

IMAGE_PATH = Path(__file__).parent.parent.parent / ".training" / "(2023) NeoEngress FA-18C Full Layout - 4220x5200.jpg"
RAW_OUT    = Path(__file__).parent / "cockpit_index_raw.json"
INDEX_OUT  = Path(__file__).parent / "cockpit_index.json"

# ── Spatial area boundaries (normalised x: 0=left edge, 1=right edge) ──────
# The cockpit image has the instrument panel in the centre, left console on the
# left, right console on the right.  These cut-offs are approximate — tune
# them after inspecting cockpit_index_raw.json if labels land in the wrong area.
AREA_BOUNDS = [
    ("left_console",       0.00, 0.32),
    ("instrument_panel",   0.32, 0.68),
    ("right_console",      0.68, 1.00),
]

# Within the instrument_panel area, use y position to distinguish sub-areas
PANEL_Y_BOUNDS = [
    ("hud_area",           0.00, 0.15),
    ("ddi_area",           0.15, 0.45),
    ("center_console",     0.45, 0.75),
    ("lower_instrument",   0.75, 1.00),
]


def _cx(box) -> float:
    """Centre-x of an EasyOCR bounding box (list of 4 [x,y] points)."""
    xs = [p[0] for p in box]
    return (min(xs) + max(xs)) / 2


def _cy(box) -> float:
    ys = [p[1] for p in box]
    return (min(ys) + max(ys)) / 2


def _area(nx: float, ny: float) -> str:
    for name, lo, hi in AREA_BOUNDS:
        if lo <= nx < hi:
            if name == "instrument_panel":
                for pname, ylo, yhi in PANEL_Y_BOUNDS:
                    if ylo <= ny < yhi:
                        return pname
            return name
    return "unknown"


def _panel_key(nx: float, ny: float, grid: int = 12) -> str:
    """Coarse grid cell used to cluster nearby labels into the same panel."""
    col = int(nx * grid)
    row = int(ny * grid)
    return f"{col}_{row}"


def ocr_image(image_path: Path) -> list[dict]:
    import easyocr
    from PIL import Image

    print(f"Loading image ({image_path.stat().st_size // 1024} KB) ...")
    img = Image.open(image_path)
    w, h = img.size
    print(f"  {w}x{h} px")

    print("Running EasyOCR (this may take 30-60 s on first run) ...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    results = reader.readtext(str(image_path), detail=1, paragraph=False)
    print(f"  {len(results)} text regions detected")

    records = []
    for box, text, conf in results:
        text = text.strip()
        if not text or conf < 0.3:
            continue
        cx = _cx(box) / w
        cy = _cy(box) / h
        records.append({
            "text":  text,
            "conf":  round(conf, 3),
            "nx":    round(cx, 4),
            "ny":    round(cy, 4),
            "area":  _area(cx, cy),
            "panel": _panel_key(cx, cy),
        })

    records.sort(key=lambda r: (r["area"], r["ny"], r["nx"]))
    return records


def build_index(records: list[dict]) -> dict:
    """Group OCR records into areas > panels > label list."""
    from collections import defaultdict

    areas: dict[str, dict] = defaultdict(lambda: {"panels": defaultdict(list)})
    for r in records:
        areas[r["area"]]["panels"][r["panel"]].append(r["text"])

    # Convert to plain dict for JSON serialisation
    out: dict = {}
    for area, data in areas.items():
        out[area] = {}
        for panel_key, labels in data["panels"].items():
            # Deduplicate while preserving order
            seen: set = set()
            unique = [l for l in labels if not (l in seen or seen.add(l))]  # type: ignore[func-returns-value]
            out[area][panel_key] = unique
    return out


def main() -> None:
    if not IMAGE_PATH.exists():
        sys.exit(f"Image not found: {IMAGE_PATH}")

    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)

    records = ocr_image(IMAGE_PATH)
    RAW_OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Raw OCR dump -> {RAW_OUT}")

    index = build_index(records)
    INDEX_OUT.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Cockpit index -> {INDEX_OUT}")

    # Summary
    for area, panels in index.items():
        total = sum(len(v) for v in panels.values())
        print(f"  {area}: {len(panels)} panels, {total} labels")


if __name__ == "__main__":
    main()
