"""
Extract and chunk the F/A-18C guide PDF into retrieval/chunks.json.

Improvements over v1:
- Section headers are detected and prepended to every chunk within that section.
  This anchors each chunk to its procedure (e.g. "2.7.2 AGM-65F/G MAVERICK (IR)"),
  preventing cross-weapon contamination in BM25 scoring.
- Chunk size raised to 600 chars (better procedure step coverage).
- Section name stored in chunk metadata for disambiguation.

Run once (or whenever the source PDF changes):
    python -m retrieval.ingest
"""

import json
import re
import sys
from pathlib import Path

PDF_PATH   = Path(__file__).parent.parent / ".training" / "DCS FA-18C Hornet Guide.pdf"
OUT_PATH   = Path(__file__).parent / "chunks.json"
CHUNK_SIZE = 600   # target chars per chunk
OVERLAP    = 120   # overlap between adjacent chunks

# Matches section headers like "2.7.2 – AGM-65F/G MAVERICK (IR-MAVF, ...)"
# or "1 2.8 – AGM-65E MAVERICK (Laser-Guided MAV)"
_SECTION_RE = re.compile(
    r"^(\d+\.\d+(?:\.\d+)?)"                 # section number at line start e.g. 2.7.2
    r"[ \t]*[–\-\u2013]?[ \t]*"              # optional separator (may be on same or next line)
    r"([A-Z][A-Z0-9 \-/\(\),]{4,70})$",      # UPPERCASE title (weapons/systems sections)
    re.MULTILINE,
)


def _clean(text: str) -> str:
    text = re.sub(r"\(\s*left\s*click[^)]*\)",   "(left-click)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*right\s*click[^)]*\)",  "(right-click)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*middle\s*click[^)]*\)", "(middle-click)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*mouse\s*wheel\s*up[^)]*\)",   "(scroll up)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*mouse\s*wheel\s*down[^)]*\)", "(scroll down)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*mouse\s*wheel[^)]*\)",         "(scroll wheel)", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[LR](ALT|CTRL|SHIFT)\+\S+", "", text)
    # Collapse horizontal whitespace only — preserve newlines for section detection
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)   # max two consecutive blank lines
    return text.strip()


def _extract_section(text: str) -> str | None:
    """Return 'section_num: title' if this page starts a new section, else None."""
    m = _SECTION_RE.search(text)
    if m:
        num   = m.group(1).strip()
        title = m.group(2).strip()
        # Filter out TOC lines (very short titles, or lines that look like page refs)
        if len(title) >= 8 and not re.match(r"^\d+$", title):
            return f"{num}: {title}"
    return None


def _chunk_text(text: str, page: int, section: str | None) -> list[dict]:
    # Flatten to single line for BM25 tokenisation
    flat = re.sub(r"\s+", " ", text).strip()
    prefix = f"[{section}] " if section else ""
    chunks = []
    start = 0
    while start < len(flat):
        end  = start + CHUNK_SIZE
        body = flat[start:end].strip()
        if body:
            chunks.append({
                "page":    page,
                "section": section or "",
                "text":    (prefix + body) if start == 0 else f"[{section}] {body}" if section else body,
            })
        start += CHUNK_SIZE - OVERLAP
    return chunks


def ingest() -> None:
    try:
        import pypdf
    except ImportError:
        sys.exit("pypdf not installed — run: pip install pypdf")

    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    print(f"Reading {PDF_PATH.name} ...")
    reader = pypdf.PdfReader(str(PDF_PATH))
    all_chunks: list[dict] = []
    current_section: str | None = None

    for i, page in enumerate(reader.pages, start=1):
        raw  = page.extract_text() or ""
        text = _clean(raw)
        if not text:
            continue

        detected = _extract_section(text)
        if detected:
            current_section = detected

        all_chunks.extend(_chunk_text(text, page=i, section=current_section))

    OUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks -> {OUT_PATH}")

    # Summary of sections found
    sections = sorted({c["section"] for c in all_chunks if c["section"]})
    print(f"  {len(sections)} sections detected")
    for s in sections[-10:]:  # show last 10 as a sanity check
        print(f"    {s}")


if __name__ == "__main__":
    ingest()
