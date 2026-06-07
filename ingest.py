"""
ingest.py — Load, clean, and chunk documents from documents/cleaned/

Run clean.py first to produce the cleaned source files.

Chunking strategy (matches planning.md with one refinement — see MAX_REVIEW_WORDS):
  - Bruinwalk files (detected by "Quarter:" markers) use review-atomic mode:
      short reviews (≤ MAX_REVIEW_WORDS) → one chunk per review
      long reviews  (> MAX_REVIEW_WORDS) → sliding window within the review,
        each sub-chunk prefixed with the review header (Quarter/Grade/date)
        so retrieval context is preserved.
  - All other files use sliding window: CHUNK_SIZE words, CHUNK_OVERLAP overlap.
"""

import re
from pathlib import Path

DOCUMENTS_DIR = Path("documents/cleaned")  # run clean.py first

CHUNK_SIZE = 150       # words (≈ tokens for all-MiniLM-L6-v2)
CHUNK_OVERLAP = 25     # words
MAX_REVIEW_WORDS = 190 # reviews longer than this get windowed to avoid
                       # silent truncation by the 256-token model limit


# --- header pattern: Quarter / Grade / date lines at the top of a review ---
_HEADER = re.compile(
    r"^(Quarter:[^\n]*\nGrade:[^\n]*\n(?:[^\n]*\n)?)",
    re.MULTILINE,
)


def sliding_window(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping windows that end on sentence boundaries.
    Target ~size words per chunk; never cut mid-sentence.
    """
    # split into sentences on . ! ? followed by whitespace or end-of-string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_words = [], [], 0
    for sentence in sentences:
        sent_words = len(sentence.split())
        if current_words + sent_words > size and current:
            chunks.append(" ".join(current))
            # carry back `overlap` words into the next chunk
            carry_back = []
            carry_words = 0
            for sent in reversed(current):
                w = len(sent.split())
                if carry_words + w > overlap:
                    break
                carry_back.insert(0, sent)
                carry_words += w
            current = carry_back
            current_words = carry_words
        current.append(sentence)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_by_review(text: str) -> list[str]:
    """Split a Bruinwalk file on 'Quarter:' headers."""
    blocks = re.split(r"(?=^Quarter:)", text, flags=re.MULTILINE)
    return [b.strip() for b in blocks if b.strip()]


def chunk_review(review: str) -> list[str]:
    """
    Single review → one or more chunks.

    Short reviews stay atomic. Long reviews are windowed, but each window
    keeps the Quarter/Grade/date header so the chunk is self-contained.
    """
    if len(review.split()) <= MAX_REVIEW_WORDS:
        return [review]

    # extract the 2-3 line header to prepend to every sub-chunk
    m = _HEADER.match(review)
    header = m.group(1).rstrip() + "\n" if m else ""
    body = review[len(header):]

    return [header + window for window in sliding_window(body)]


def load_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        print(f"  [skip] pdfplumber not installed — cannot read {path.name}")
        return ""
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def chunk_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = load_pdf(path)
    elif suffix == ".txt":
        raw = path.read_text(encoding="utf-8", errors="ignore")
    else:
        return []

    if not raw.strip():
        return []

    # determine mode by content, not filename
    is_bruinwalk = bool(re.search(r"^Quarter:", raw, re.MULTILINE))

    if is_bruinwalk:
        reviews = split_by_review(raw)
        raw_chunks = []
        for r in reviews:
            raw_chunks.extend(chunk_review(r))
        mode = "review-atomic+window"
    else:
        raw_chunks = sliding_window(raw)
        mode = "sliding-window"

    # professor and course from filename: bruinwalk_[prof]_[course].txt
    parts = path.stem.lower().split("_")
    professor = parts[1] if len(parts) > 1 else "unknown"
    course    = parts[2] if len(parts) > 2 else "unknown"

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text.split()) < 5:
            continue
        chunks.append({
            "text":        chunk_text,
            "source":      path.name,
            "professor":   professor,
            "course":      course,
            "chunk_index": i,
            "chunk_mode":  mode,
        })

    return chunks


def ingest_all() -> list[dict]:
    all_chunks = []
    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.suffix.lower() not in (".txt", ".pdf"):
            continue
        chunks = chunk_file(path)
        mode = chunks[0]["chunk_mode"] if chunks else "n/a"
        print(f"  {path.name}: {len(chunks)} chunks  [{mode}]")
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    print(f"Ingesting from {DOCUMENTS_DIR}/\n")
    chunks = ingest_all()

    lengths = [len(c["text"].split()) for c in chunks]
    print(f"\nTotal chunks : {len(chunks)}")
    print(f"Word counts  : min={min(lengths)}  max={max(lengths)}"
          f"  median={sorted(lengths)[len(lengths)//2]}  mean={sum(lengths)//len(lengths)}")
    over = sum(1 for l in lengths if l > MAX_REVIEW_WORDS)
    print(f"Over {MAX_REVIEW_WORDS} words : {over}  (should be 0 after windowing)")

    print("\n--- Spot check: first chunk from each file ---")
    seen: set[str] = set()
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            preview = c["text"][:300].replace("\n", " ")
            print(f"\n[{c['source']}]  professor={c['professor']}  course={c['course']}")
            print(f"  {preview}")
