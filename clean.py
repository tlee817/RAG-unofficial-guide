"""
clean.py — Load raw documents, clean them, and save to documents/cleaned/

Run this before ingest.py. Cleaned files are what the pipeline actually uses.

What gets removed (Bruinwalk-specific UI chrome):
  - "Show More" / "Show Less" lines
  - "Helpful?" lines
  - Lines containing only a number (vote counts like " 0", " 3")
  - "Verified Reviewer" badge lines
  - HTML entities (&amp; &nbsp; &lt; &gt; &#39; &quot;)
  - Excess blank lines (3+ collapsed to 1)

What gets kept:
  - Quarter / Grade / date header lines (useful context for RAG)
  - All review text, opinions, and advice
"""

import html
import re
from pathlib import Path

RAW_DIR = Path("documents")
CLEAN_DIR = Path("documents/cleaned")

# Lines that are pure Bruinwalk UI — match the whole stripped line
_JUNK_LINES = re.compile(
    r"^\s*(Show (?:More|Less)|Helpful\?|Verified Reviewer|\d+)\s*$",
    re.IGNORECASE,
)


def clean_document(text: str) -> str:
    # 1. Decode HTML entities (&amp; → &, &nbsp; → space, etc.)
    text = html.unescape(text)

    # 2. Remove junk lines
    lines = text.splitlines()
    kept = [line for line in lines if not _JUNK_LINES.match(line)]

    # 3. Collapse runs of 3+ blank lines into a single blank line
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))

    return cleaned.strip()


def clean_all() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(p for p in RAW_DIR.iterdir() if p.suffix == ".txt")
    if not txt_files:
        print(f"No .txt files found in {RAW_DIR}/")
        return

    print(f"Cleaning {len(txt_files)} file(s) → {CLEAN_DIR}/\n")
    for path in txt_files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_document(raw)

        out_path = CLEAN_DIR / path.name
        out_path.write_text(cleaned, encoding="utf-8")

        raw_lines = len(raw.splitlines())
        clean_lines = len(cleaned.splitlines())
        removed = raw_lines - clean_lines
        print(f"  {path.name}: {raw_lines} → {clean_lines} lines  ({removed} removed)")

    # Print one full cleaned document so you can visually verify it
    sample_path = CLEAN_DIR / txt_files[0].name
    sample = sample_path.read_text(encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"SAMPLE — {txt_files[0].name}")
    print(f"{'='*60}\n")
    print(sample[:3000])
    if len(sample) > 3000:
        print(f"\n... ({len(sample) - 3000} more characters)")


if __name__ == "__main__":
    clean_all()
