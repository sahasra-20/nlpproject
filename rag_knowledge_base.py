"""
Builds and manages the RAG knowledge base from the KCC dataset.
We format every row as:

    "Crop: {crop}. Q: {question} A: {answer}"   (when crop is known)
    "Q: {question} A: {answer}"                 (when crop is null/unknown)
    # builds rag_chunks.json
"""

import json
import re
import pandas as pd


# ── constants ──────────────────────────────────────────────────────────────────
CHUNK_FILE   = "rag_chunks.json"
DATA_FILE    = "data.csv"

# KCC answers are already short
# if an answer exceeds MAX_CHUNK_WORDS do we split it with overlap.
MAX_CHUNK_WORDS   = 200
OVERLAP_WORDS     = 30


# ── helpers ────────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    """Basic whitespace normalisation."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _split_with_overlap(text: str, max_words: int, overlap: int) -> list[str]:
    """
    Split a long text into chunks of ≤ max_words words with `overlap`-word
    overlap between consecutive chunks.  Never splits mid-sentence.
    """
    # Try sentence-aware splitting first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent_words = sent.split()
        if current_len + len(sent_words) > max_words and current:
            chunks.append(" ".join(current))
            # keep the last `overlap` words for context continuity
            current     = current[-overlap:] if overlap else []
            current_len = len(current)
        current.extend(sent_words)
        current_len += len(sent_words)

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def format_chunk(row: pd.Series) -> str:
    """
    Converts one CSV row into a RAG chunk string.
    Format (crop known):   "Crop: {crop}. Q: {question} A: {answer}"
    Format (crop unknown): "Q: {question} A: {answer}"
    """
    raw_crop = row.get("crop", None)
    crop     = _clean(str(raw_crop)) if raw_crop is not None else ""
    q        = _clean(str(row.get("question", "")))
    a        = _clean(str(row.get("answer",   "")))

    # Only prepend crop name when it is a real, non-empty, non-'unknown' value
    if crop and crop.lower() not in ("", "unknown", "nan", "none"):
        return f"Crop: {crop}. Q: {q} A: {a}"
    return f"Q: {q} A: {a}"


def build_knowledge_base(
    csv_file: str = DATA_FILE,
    chunk_file: str = CHUNK_FILE,
    verbose: bool = True,
) -> list[dict]:
    """
    Reads KCC CSV, format each row as a RAG chunk, split overlong answers,
    and write the result to `chunk_file`.

    Returns the list of chunk dicts:
        [{"text": str}, ...]                        (crop unknown)
        [{"text": str, "crop": str}, ...]           (crop known)
    """
    df = pd.read_csv(csv_file).dropna(subset=["question", "answer"])
    if verbose:
        print(f"Loaded {len(df):,} rows from {csv_file}")

    chunks = []
    for _, row in df.iterrows():
        text  = format_chunk(row)
        words = text.split()

        if len(words) <= MAX_CHUNK_WORDS:
            # Short enough — use as-is
            sub_chunks = [text]
        else:
            # Split overlong chunk with overlap
            sub_chunks = _split_with_overlap(text, MAX_CHUNK_WORDS, OVERLAP_WORDS)

        for sc in sub_chunks:
            raw_crop = row.get("crop", None)
            crop_val = _clean(str(raw_crop)) if raw_crop is not None else ""
            chunk = {"text": sc}
            # Only store crop key when it is a real, known value
            if crop_val and crop_val.lower() not in ("", "unknown", "nan", "none"):
                chunk["crop"] = crop_val
            chunks.append(chunk)

    if verbose:
        print(f"Created {len(chunks):,} chunks  (from {len(df):,} rows)")

    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Saved to {chunk_file}")

    return chunks


def load_knowledge_base(chunk_file: str = CHUNK_FILE) -> list[dict]:
    """Load previously built chunks from disk."""
    with open(chunk_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ── self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found — run from the project root.")
        sys.exit(1)

    chunks = build_knowledge_base(verbose=True)

    # Sanity checks
    assert len(chunks) > 0, "No chunks produced!"
    assert all("text" in c for c in chunks[:10]), "Bad chunk format"
    # crop key is optional — only present when crop is a real known value
    has_crop = sum(1 for c in chunks if "crop" in c)
    print(f"Chunks with crop metadata: {has_crop:,} / {len(chunks):,}")

    print("\nSample chunk:")
    print("-" * 60)
    print(chunks[0]["text"][:300])
    print("-" * 60)
    print("Knowledge base build: OK")
