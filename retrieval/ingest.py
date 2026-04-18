"""
Extract and chunk the F/A-18C guide PDF into retrieval/chunks.json.

Run once (or whenever the source PDF changes):
    python -m retrieval.ingest

Chunks are plain text paragraphs of ~400 chars with 100-char overlap, tagged
with the source page number so answers can cite page references if needed.
"""

import json
import re
import sys
from pathlib import Path

PDF_PATH   = Path(__file__).parent.parent / ".training" / "DCS FA-18C Hornet Guide.pdf"
OUT_PATH   = Path(__file__).parent / "chunks.json"
CHUNK_SIZE = 400   # target chars per chunk
OVERLAP    = 100   # overlap between adjacent chunks


def _clean(text: str) -> str:
    # Normalise DCS cockpit interaction cues to unambiguous prose so the LLM
    # can relay them without misreading "click" as a physical sensation.
    text = re.sub(r"\(\s*left\s*click[^)]*\)",   "(left-click)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*right\s*click[^)]*\)",  "(right-click)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*middle\s*click[^)]*\)", "(middle-click)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*mouse\s*wheel\s*up[^)]*\)",   "(scroll up)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*mouse\s*wheel\s*down[^)]*\)", "(scroll down)", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*mouse\s*wheel[^)]*\)",         "(scroll wheel)", text, flags=re.IGNORECASE)
    # Strip bare keyboard shortcuts — not useful spoken aloud
    text = re.sub(r"\b[LR](ALT|CTRL|SHIFT)\+\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, page: int) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"page": page, "text": chunk.strip()})
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

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _clean(raw)
        if text:
            all_chunks.extend(_chunk_text(text, page=i))

    OUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    ingest()
